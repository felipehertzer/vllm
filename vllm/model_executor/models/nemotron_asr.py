# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only NVIDIA Nemotron 3.5 ASR RNN-T model."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ParakeetEncoder
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import BaseModelOutput

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.inputs import (
    ExplicitEncoderDecoderPrompt,
    MultiModalDataDict,
    PromptType,
    TextPrompt,
    TokensPrompt,
)
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsRealtime,
    SupportsTranscription,
)
from vllm.model_executor.models.parakeet import ParakeetExtractor
from vllm.model_executor.models.transducer_asr import (
    TransducerASRForcedDecoderState,
    TransducerDecodeConfig,
    TransducerHypothesis,
    TransducerPredictionDecoder,
    greedy_decode_transducer_batch,
    greedy_decode_transducer_batch_with_timestamps,
    strip_asr_special_tokens,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_asr import (
    NEMOTRON_ASR_LANGUAGE_ALIASES,
    NemotronASRConfig,
)

logger = init_logger(__name__)

NEMOTRON_ASR_SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mt": "Maltese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class NemotronASRProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> NemotronASRConfig:
        return self.ctx.get_hf_config(NemotronASRConfig)

    def get_feature_extractor(self) -> ParakeetExtractor:
        return ParakeetExtractor(self.get_hf_config().encoder_config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=self.get_hf_config().sample_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_num_audio_tokens(self, num_samples: int) -> int:
        return self.get_feature_extractor().audio_token_count(num_samples)


class NemotronASRDummyInputsBuilder(BaseDummyInputsBuilder[NemotronASRProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        sample_rate = self.info.get_hf_config().sample_rate
        audio_overrides = mm_options.get("audio")

        return {
            "audio": self._get_dummy_audios(
                length=30 * sample_rate,
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }


class NemotronASRMultiModalProcessor(
    EncDecMultiModalProcessor[NemotronASRProcessingInfo]
):
    skip_decoder_start_token: bool = True

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [self.info.get_hf_config().blank_token_id]

    def _extract_audio_features(
        self,
        audios: Sequence[np.ndarray],
        *,
        device: str = "cpu",
    ) -> BatchFeature:
        extractor = self.info.get_feature_extractor()
        raw_speech = [
            torch.as_tensor(audio, device=device, dtype=torch.float32)
            for audio in audios
        ]

        for i, speech in enumerate(raw_speech):
            if len(speech.shape) > 1:
                logger.warning(
                    "Only mono-channel audio is supported for Nemotron ASR. "
                    "Averaging channels to mono."
                )
                raw_speech[i] = speech.mean(-1)

        audio_lengths = torch.tensor(
            [len(speech) for speech in raw_speech],
            dtype=torch.long,
            device=device,
        )
        max_length = max(len(speech) for speech in raw_speech)
        input_features = extractor._pad_raw_speech(raw_speech, max_length, device)
        input_features = extractor._apply_preemphasis(input_features, audio_lengths)
        input_features = extractor._torch_extract_fbank_features(input_features, device)
        input_features, attention_mask = extractor._normalize_mel_features(
            input_features,
            audio_lengths,
        )

        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask,
            },
            tensor_type="pt",
        )

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del hf_processor_mm_kwargs
        mm_data, passthrough_data = self._get_hf_mm_data(mm_items)
        if not mm_data:
            return BatchFeature(
                data={"input_ids": [[0]], **passthrough_data},
                tensor_type="pt",
            )

        raw_audios = mm_data.get("audios")
        if isinstance(raw_audios, np.ndarray):
            audios = [raw_audios]
        elif isinstance(raw_audios, Sequence):
            audios = list(raw_audios)
        else:
            raise ValueError("Nemotron ASR expects audio inputs.")

        inputs = self._extract_audio_features(audios)
        inputs["input_ids"] = [[0]]
        inputs.update(passthrough_data)
        return inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            input_features=MultiModalFieldConfig.batched("audio"),
            attention_mask=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def get_audio_replacement(item_idx: int) -> list[int]:
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio_len = audios.get_audio_length(item_idx)
            num_tokens = self.info.get_num_audio_tokens(num_samples=audio_len)
            return [0] * num_tokens

        return [
            PromptReplacement(
                modality="audio",
                target=[0],
                replacement=get_audio_replacement,
            )
        ]


class NemotronASRJoint(nn.Module):
    def __init__(self, config: NemotronASRConfig) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        self.head = nn.Linear(config.decoder_hidden_size, config.vocab_size)

    def forward(
        self,
        encoder_states: torch.Tensor,
        decoder_states: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.activation(encoder_states + decoder_states)
        return self.head(hidden_states)


def _nemotron_rel_shift(attention_scores: torch.Tensor) -> torch.Tensor:
    batch_size, num_heads, query_length, position_length = attention_scores.shape
    attention_scores = F.pad(attention_scores, pad=(1, 0))
    attention_scores = attention_scores.view(batch_size, num_heads, -1, query_length)
    attention_scores = attention_scores[:, :, 1:].view(
        batch_size,
        num_heads,
        query_length,
        position_length,
    )
    return attention_scores


def _nemotron_eager_attention(
    query_states_with_bias_u: torch.Tensor,
    query_states_with_bias_v: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    relative_key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    matrix_ac = query_states_with_bias_u @ key_states.transpose(-2, -1)
    matrix_bd = query_states_with_bias_v @ relative_key_states.transpose(-2, -1)
    matrix_bd = _nemotron_rel_shift(matrix_bd)
    matrix_bd = matrix_bd[..., : query_states_with_bias_u.shape[-2]]
    attention_scores = (matrix_ac + matrix_bd) * scaling
    if attention_mask is not None:
        attention_scores = attention_scores.masked_fill(
            attention_mask.logical_not(),
            float("-inf"),
        )
    attention_probs = torch.softmax(attention_scores.float(), dim=-1).to(
        value_states.dtype
    )
    return attention_probs @ value_states


def _nemotron_sdpa_attention(
    query_states_with_bias_u: torch.Tensor,
    query_states_with_bias_v: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    relative_key_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    matrix_bd = query_states_with_bias_v @ relative_key_states.transpose(-2, -1)
    matrix_bd = _nemotron_rel_shift(matrix_bd)
    matrix_bd = matrix_bd[..., : query_states_with_bias_u.shape[-2]]
    matrix_bd = matrix_bd * scaling
    if attention_mask is not None:
        matrix_bd = matrix_bd.masked_fill(attention_mask.logical_not(), float("-inf"))
    return F.scaled_dot_product_attention(
        query_states_with_bias_u,
        key_states,
        value_states,
        attn_mask=matrix_bd,
        dropout_p=0.0,
        scale=scaling,
    )


class NemotronASRAttention(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.config = attention.config
        self.head_dim = attention.head_dim
        self.scaling = attention.scaling
        self.attention_dropout = attention.attention_dropout
        self.q_proj = attention.q_proj
        self.k_proj = attention.k_proj
        self.v_proj = attention.v_proj
        self.o_proj = attention.o_proj
        self.relative_k_proj = attention.relative_k_proj
        self.bias_u = attention.bias_u
        self.bias_v = attention.bias_v
        self._compiled_cuda_attention = None

    @staticmethod
    def _cuda_compile_enabled() -> bool:
        setting = os.environ.get(
            "NEMOTRON_ASR_CUDA_COMPILE_ATTENTION",
            "1",
        ).lower()
        return setting not in {"0", "false", "off", "no"}

    def _attention(
        self,
        query_states_with_bias_u: torch.Tensor,
        query_states_with_bias_v: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        relative_key_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            query_states_with_bias_u.device.type == "cuda"
            and self._cuda_compile_enabled()
        ):
            if self._compiled_cuda_attention is None:
                self._compiled_cuda_attention = torch.compile(
                    _nemotron_sdpa_attention,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
            return self._compiled_cuda_attention(
                query_states_with_bias_u,
                query_states_with_bias_v,
                key_states,
                value_states,
                relative_key_states,
                attention_mask,
                self.scaling,
            )
        return _nemotron_eager_attention(
            query_states_with_bias_u,
            query_states_with_bias_v,
            key_states,
            value_states,
            relative_key_states,
            attention_mask,
            self.scaling,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        del kwargs
        input_shape = hidden_states.shape[:-1]
        batch_size, seq_length = input_shape
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states_with_bias_u = query_states + self.bias_u.view(
            1,
            self.config.num_attention_heads,
            1,
            self.head_dim,
        )
        query_states_with_bias_v = query_states + self.bias_v.view(
            1,
            self.config.num_attention_heads,
            1,
            self.head_dim,
        )

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(
            batch_size,
            -1,
            self.config.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)

        attn_output = self._attention(
            query_states_with_bias_u,
            query_states_with_bias_v,
            key_states,
            value_states,
            relative_key_states,
            attention_mask,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class NemotronASRCausalConv2D(nn.Conv2d):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
        )
        self.left_padding = kernel_size - 1
        self.right_padding = stride - 1

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded = F.pad(
            input,
            pad=(
                self.left_padding,
                self.right_padding,
                self.left_padding,
                self.right_padding,
            ),
        )
        return super().forward(padded)


class NemotronASRCausalConv1D(nn.Conv1d):
    def __init__(self, conv: nn.Conv1d) -> None:
        kernel_size = conv.kernel_size[0]
        super().__init__(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=0,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
        )
        self.left_padding = kernel_size - 1
        self.right_padding = 0

    def _uses_manual_depthwise_kernel(self, device_type: str) -> bool:
        return (
            device_type != "cuda"
            and self.bias is None
            and self.groups == self.in_channels == self.out_channels
            and self.stride == (1,)
            and self.dilation == (1,)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded = F.pad(input, pad=(self.left_padding, self.right_padding))
        if self._uses_manual_depthwise_kernel(input.device.type):
            output = input.new_zeros(input.shape)
            weights = self.weight[:, 0, :]
            for kernel_idx in range(self.kernel_size[0]):
                output = output + (
                    padded[:, :, kernel_idx : kernel_idx + input.shape[-1]]
                    * weights[:, kernel_idx][None, :, None]
                )
            return output
        return super().forward(padded)


class NemotronASRCausalSubsamplingConv2D(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.kernel_size = config.subsampling_conv_kernel_size
        self.stride = config.subsampling_conv_stride
        self.channels = config.subsampling_conv_channels
        self.num_layers = int(math.log2(config.subsampling_factor))
        self.layers = nn.ModuleList()
        self.layers.append(
            NemotronASRCausalConv2D(
                in_channels=1,
                out_channels=self.channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
            )
        )
        self.layers.append(nn.ReLU())
        for _ in range(self.num_layers - 1):
            self.layers.append(
                NemotronASRCausalConv2D(
                    in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    groups=self.channels,
                )
            )
            self.layers.append(nn.Conv2d(self.channels, self.channels, kernel_size=1))
            self.layers.append(nn.ReLU())

        out_length = self._get_output_feature_length(int(config.num_mel_bins))
        self.linear = nn.Linear(
            config.subsampling_conv_channels * out_length,
            config.hidden_size,
            bias=True,
        )

    def _get_output_feature_length(self, input_length: int) -> int:
        length = float(input_length)
        add_pad = self.kernel_size - 1 + self.stride - 1 - self.kernel_size
        for _ in range(self.num_layers):
            length = math.floor((length + add_pad) / self.stride + 1.0)
        return int(length)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        hidden_states = input_features.unsqueeze(1)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[2],
            -1,
        )
        return self.linear(hidden_states)


class NemotronASRBatchNorm1dNoStats(nn.Module):
    def __init__(self, num_features: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean = input.mean(dim=(0, 2), keepdim=True)
        variance = input.var(dim=(0, 2), unbiased=False, keepdim=True)
        output = (input - mean) * torch.rsqrt(variance + self.eps)
        return output * self.weight[None, :, None] + self.bias[None, :, None]


class NemotronASRConvLayerNorm(nn.Module):
    def __init__(self, num_features: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(
            input.transpose(1, 2),
            (input.shape[1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.transpose(1, 2)


class NemotronASREncoder(ParakeetEncoder):
    def __init__(self, config) -> None:
        super().__init__(config)
        if bool(getattr(config, "causal_downsampling", False)):
            self.subsampling = NemotronASRCausalSubsamplingConv2D(config)
        self._patch_nemo_convolution_modules()

    @staticmethod
    def _conv1d_without_bias(conv: nn.Conv1d) -> nn.Conv1d:
        return nn.Conv1d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=False,
            padding_mode=conv.padding_mode,
        )

    def _patch_nemo_attention_modules(self) -> None:
        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is None or isinstance(attention, NemotronASRAttention):
                continue
            layer.self_attn = NemotronASRAttention(attention)

    def _patch_nemo_convolution_modules(self) -> None:
        for layer in self.layers:
            conv_module = getattr(layer, "conv", None)
            if conv_module is None:
                continue
            for conv_name in (
                "pointwise_conv1",
                "depthwise_conv",
                "pointwise_conv2",
            ):
                conv = getattr(conv_module, conv_name, None)
                if isinstance(conv, nn.Conv1d) and conv.bias is not None:
                    setattr(conv_module, conv_name, self._conv1d_without_bias(conv))

            depthwise_conv = getattr(conv_module, "depthwise_conv", None)
            if bool(getattr(self.config, "conv_causal", False)) and isinstance(
                depthwise_conv, nn.Conv1d
            ):
                conv_module.depthwise_conv = NemotronASRCausalConv1D(depthwise_conv)

            norm = getattr(conv_module, "norm", None)
            conv_norm_type = getattr(self.config, "conv_norm_type", "batch_norm")
            if conv_norm_type == "layer_norm" and isinstance(norm, nn.BatchNorm1d):
                conv_module.norm = NemotronASRConvLayerNorm(
                    norm.num_features,
                    eps=norm.eps,
                )
            elif isinstance(norm, nn.BatchNorm1d):
                conv_module.norm = NemotronASRBatchNorm1dNoStats(
                    norm.num_features,
                    eps=norm.eps,
                )

    def _get_nemo_attention_mask(
        self,
        attention_mask: torch.Tensor,
        *,
        target_length: int,
    ) -> torch.Tensor:
        valid_mask = self._get_output_attention_mask(
            attention_mask,
            target_length=target_length,
        )
        pairwise_mask = valid_mask[:, None, :] & valid_mask[:, :, None]

        context_mask = torch.ones(
            target_length,
            target_length,
            dtype=torch.bool,
            device=attention_mask.device,
        )
        att_context_left = int(getattr(self.config, "att_context_left", -1))
        att_context_right = int(getattr(self.config, "att_context_right", -1))
        att_context_style = getattr(self.config, "att_context_style", "regular")

        if att_context_style == "regular":
            if att_context_left >= 0:
                context_mask = context_mask.triu(diagonal=-att_context_left)
            if att_context_right >= 0:
                context_mask = context_mask.tril(diagonal=att_context_right)
        elif att_context_style == "chunked_limited":
            if att_context_right == -1:
                if att_context_left >= 0:
                    context_mask = context_mask.triu(diagonal=-att_context_left)
            else:
                chunk_size = att_context_right + 1
                left_chunks = (
                    att_context_left // chunk_size if att_context_left >= 0 else 10000
                )
                chunk_idx = torch.div(
                    torch.arange(
                        target_length,
                        dtype=torch.int,
                        device=attention_mask.device,
                    ),
                    chunk_size,
                    rounding_mode="trunc",
                )
                diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
                context_mask = (diff_chunks <= left_chunks) & (diff_chunks >= 0)

        return (pairwise_mask & context_mask.unsqueeze(0)).unsqueeze(1)

    def _get_subsampling_output_length(self, input_lengths: torch.Tensor):
        if not bool(getattr(self.config, "causal_downsampling", False)):
            return super()._get_subsampling_output_length(input_lengths)

        kernel_size = self.config.subsampling_conv_kernel_size
        stride = self.config.subsampling_conv_stride
        num_layers = int(math.log2(self.config.subsampling_factor))
        all_padding = (kernel_size - 1) + (stride - 1)
        add_padding = all_padding - kernel_size
        lengths = input_lengths.to(dtype=torch.float)
        for _ in range(num_layers):
            lengths = (
                torch.div(
                    lengths + add_padding,
                    stride,
                    rounding_mode="floor",
                )
                + 1
            )
        return lengths.clamp_min(0).to(dtype=torch.int)

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> BaseModelOutput:
        hidden_states = self.subsampling(input_features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.encode_positions(hidden_states)

        hidden_states = nn.functional.dropout(
            hidden_states,
            p=self.dropout,
            training=self.training,
        )
        position_embeddings = nn.functional.dropout(
            position_embeddings,
            p=self.dropout_positions,
            training=self.training,
        )

        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = self._get_nemo_attention_mask(
                attention_mask,
                target_length=hidden_states.shape[1],
            )

        for encoder_layer in self.layers:
            to_drop = False
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    to_drop = True

            if not to_drop:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask=encoder_attention_mask,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

        return BaseModelOutput(last_hidden_state=hidden_states)


class NemotronASRModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: NemotronASRConfig = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype
        self._use_cuda_cpu_decoder = self._should_use_cuda_cpu_decoder()
        self._use_mps_encoder_hybrid = self._should_use_mps_encoder_hybrid(self.dtype)
        self._mps_encoder_dtype = self._get_mps_encoder_dtype()
        self._mps_encoder_hybrid_ready = False

        self.encoder = NemotronASREncoder(config.encoder_config).to(self.dtype)
        self.prompt_kernel = (
            nn.Sequential(
                nn.Linear(
                    config.encoder_config.hidden_size + config.prompt_dim,
                    config.encoder_config.hidden_size * 2,
                ),
                nn.ReLU(),
                nn.Linear(
                    config.encoder_config.hidden_size * 2,
                    config.encoder_config.hidden_size,
                ),
            )
            if config.prompt_dim > 0
            else None
        )
        if self.prompt_kernel is not None:
            self.prompt_kernel = self.prompt_kernel.to(self.dtype)
        self.encoder_projector = nn.Linear(
            config.encoder_config.hidden_size,
            config.decoder_hidden_size,
        ).to(self.dtype)
        self.decoder = TransducerPredictionDecoder(
            vocab_size=config.vocab_size,
            hidden_size=config.decoder_hidden_size,
            num_layers=config.num_decoder_layers,
        ).to(self.dtype)
        self.joint = NemotronASRJoint(config).to(self.dtype)

    @staticmethod
    def _should_use_cuda_cpu_decoder() -> bool:
        setting = os.environ.get("NEMOTRON_ASR_CUDA_DECODER_DEVICE", "auto").lower()
        if setting in {"cuda", "gpu", "0", "false", "off", "no"}:
            return False
        if setting in {"cpu", "1", "true", "on", "yes", "auto"}:
            return True
        logger.warning(
            "Unknown NEMOTRON_ASR_CUDA_DECODER_DEVICE=%r; using auto.",
            setting,
        )
        return True

    @staticmethod
    def _should_use_mps_encoder_hybrid(dtype: torch.dtype) -> bool:
        setting = os.environ.get("NEMOTRON_ASR_MPS_ENCODER", "auto").lower()
        if setting in {"0", "false", "off", "no"}:
            return False
        if setting not in {"1", "true", "on", "yes", "auto"}:
            logger.warning(
                "Unknown NEMOTRON_ASR_MPS_ENCODER=%r; using auto.",
                setting,
            )
        if dtype != torch.float32:
            return False
        mps_backend = getattr(torch.backends, "mps", None)
        return bool(
            mps_backend is not None
            and mps_backend.is_available()
            and torch.backends.mps.is_built()
        )

    @staticmethod
    def _get_mps_encoder_dtype() -> torch.dtype:
        setting = os.environ.get("NEMOTRON_ASR_MPS_ENCODER_DTYPE", "float16").lower()
        if setting in {"fp16", "float16", "half"}:
            return torch.float16
        if setting in {"fp32", "float32", "full"}:
            return torch.float32
        logger.warning(
            "Unknown NEMOTRON_ASR_MPS_ENCODER_DTYPE=%r; using float16.",
            setting,
        )
        return torch.float16

    def _decoder_device(self) -> torch.device:
        return self.encoder_projector.weight.device

    def _move_decoder_stack(self, device: torch.device) -> None:
        if self.prompt_kernel is not None:
            self.prompt_kernel.to(device)
        self.encoder_projector.to(device)
        self.decoder.to(device)
        self.joint.to(device)

    def _prepare_encoder_decode_devices(
        self,
        input_device: torch.device,
    ) -> tuple[torch.device, torch.device]:
        if input_device.type == "cuda":
            self.encoder._patch_nemo_attention_modules()
            encoder_device = next(self.encoder.parameters()).device
            if self._use_cuda_cpu_decoder:
                decoder_device = torch.device("cpu")
                if self._decoder_device().type != decoder_device.type:
                    logger.info(
                        "Using Nemotron ASR CUDA encoder with CPU RNN-T decoder."
                    )
                    self._move_decoder_stack(decoder_device)
                return encoder_device, decoder_device
            if self._decoder_device().type != encoder_device.type:
                self._move_decoder_stack(encoder_device)
            return encoder_device, self._decoder_device()
        if not self._use_mps_encoder_hybrid:
            encoder_device = next(self.encoder.parameters()).device
            return encoder_device, self._decoder_device()

        encoder_device = torch.device("mps")
        decoder_device = torch.device("cpu")
        encoder_current_device = next(self.encoder.parameters()).device
        if (
            not self._mps_encoder_hybrid_ready
            or encoder_current_device.type != encoder_device.type
            or next(self.encoder.parameters()).dtype != self._mps_encoder_dtype
            or self._decoder_device().type != decoder_device.type
        ):
            logger.info(
                "Using Nemotron ASR hybrid local execution: encoder on MPS "
                "%s, RNN-T decoder on CPU.",
                str(self._mps_encoder_dtype).replace("torch.", ""),
            )
            self.encoder.to(device=encoder_device, dtype=self._mps_encoder_dtype)
            self._move_decoder_stack(decoder_device)
            self._mps_encoder_hybrid_ready = True
        return encoder_device, decoder_device

    def get_encoder_outputs(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        encoder_device, decoder_device = self._prepare_encoder_decode_devices(
            input_features.device
        )
        encoder_dtype = (
            self._mps_encoder_dtype
            if self._use_mps_encoder_hybrid and encoder_device.type == "mps"
            else self.dtype
        )
        input_features = input_features.to(device=encoder_device, dtype=encoder_dtype)
        attention_mask = attention_mask.to(device=encoder_device)
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )
        hidden_states = encoder_outputs.last_hidden_state
        output_mask = getattr(encoder_outputs, "attention_mask", None)
        if output_mask is None and attention_mask is not None:
            output_mask = self.encoder._get_output_attention_mask(
                attention_mask,
                target_length=hidden_states.shape[1],
            )

        if output_mask is None:
            if hidden_states.device != decoder_device:
                hidden_states = hidden_states.to(
                    device=decoder_device,
                    dtype=self.dtype,
                )
            return list(hidden_states)

        if hidden_states.device != decoder_device:
            hidden_states = hidden_states.to(
                device=decoder_device,
                dtype=self.dtype,
            )
            output_mask = output_mask.to(decoder_device)

        return [
            hidden_state[mask.to(dtype=torch.bool)]
            for hidden_state, mask in zip(hidden_states, output_mask)
        ]

    def _encoder_with_language_prompt(
        self,
        encoder_outputs: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = encoder_outputs[0].device
        padded_outputs = nn.utils.rnn.pad_sequence(
            list(encoder_outputs), batch_first=True
        )
        if prompt_ids is None:
            prompt_id = self.config.prompt_id_for_language(None)
            prompt_ids = torch.full(
                (len(encoder_outputs),),
                prompt_id,
                dtype=torch.long,
                device=device,
            )
        if self.prompt_kernel is None or self.config.prompt_dim <= 0:
            return self.encoder_projector(padded_outputs)
        prompt_one_hot = torch.nn.functional.one_hot(
            prompt_ids.to(device=device),
            num_classes=self.config.prompt_dim,
        ).to(dtype=padded_outputs.dtype)
        prompt_vectors = prompt_one_hot[:, None, :].expand(
            -1,
            padded_outputs.shape[1],
            -1,
        )
        prompted = self.prompt_kernel(
            torch.cat((padded_outputs, prompt_vectors), dim=-1)
        )
        return self.encoder_projector(prompted)

    def _joint_logits(
        self,
        encoder_state: torch.Tensor,
        decoder_state: torch.Tensor,
    ) -> torch.Tensor:
        return self.joint(encoder_state, decoder_state)

    def greedy_decode_batch(
        self,
        encoder_outputs: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:
        if not encoder_outputs:
            return []

        device = encoder_outputs[0].device
        cfg = self.config
        lengths = torch.tensor(
            [int(encoder_output.shape[0]) for encoder_output in encoder_outputs],
            dtype=torch.long,
            device=device,
        )
        encoder_projected = self._encoder_with_language_prompt(
            encoder_outputs,
            prompt_ids=prompt_ids,
        )
        decode_config = TransducerDecodeConfig(
            vocab_size=cfg.vocab_size,
            blank_token_id=cfg.blank_token_id,
            eos_token_id=cfg.eos_token_id,
            max_symbols_per_step=cfg.max_symbols_per_step,
        )
        return greedy_decode_transducer_batch(
            encoder_projected=encoder_projected,
            lengths=lengths,
            decoder=self.decoder,
            joint_logits=self._joint_logits,
            config=decode_config,
        )

    def greedy_decode_batch_with_timestamps(
        self,
        encoder_outputs: Sequence[torch.Tensor],
        prompt_ids: torch.Tensor | None = None,
    ) -> list[TransducerHypothesis]:
        if not encoder_outputs:
            return []

        device = encoder_outputs[0].device
        cfg = self.config
        lengths = torch.tensor(
            [int(encoder_output.shape[0]) for encoder_output in encoder_outputs],
            dtype=torch.long,
            device=device,
        )
        encoder_projected = self._encoder_with_language_prompt(
            encoder_outputs,
            prompt_ids=prompt_ids,
        )
        decode_config = TransducerDecodeConfig(
            vocab_size=cfg.vocab_size,
            blank_token_id=cfg.blank_token_id,
            eos_token_id=cfg.eos_token_id,
            max_symbols_per_step=cfg.max_symbols_per_step,
        )
        return greedy_decode_transducer_batch_with_timestamps(
            encoder_projected=encoder_projected,
            lengths=lengths,
            decoder=self.decoder,
            joint_logits=self._joint_logits,
            config=decode_config,
        )


@MULTIMODAL_REGISTRY.register_processor(
    NemotronASRMultiModalProcessor,
    info=NemotronASRProcessingInfo,
    dummy_inputs=NemotronASRDummyInputsBuilder,
)
@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": -1, "forced_decoder_ids": 0}
)
class NemotronASRForRNNT(
    nn.Module,
    SupportsTranscription,
    SupportsMultiModal,
    SupportsRealtime,
):
    supports_transcription_only = True
    supported_languages = NEMOTRON_ASR_SUPPORTED_LANGUAGES
    no_space_languages: set[str] = {"ja", "zh"}
    realtime_max_tokens = 256
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "model.encoder.",
            "prompt_kernel.": "model.prompt_kernel.",
            "encoder_projector.": "model.encoder_projector.",
            "decoder.": "model.decoder.",
            "joint.": "model.joint.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.config: NemotronASRConfig = vllm_config.model_config.hf_config
        self.dtype = vllm_config.model_config.dtype

        with self._mark_tower_model(vllm_config, "audio"):
            self.model = NemotronASRModel(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return "<audio>"

        raise ValueError("Only audio modality is supported")

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    @staticmethod
    def get_model_state_cls():
        from vllm.v1.worker.gpu.model_states.nemotron_asr import NemotronASRModelState

        return NemotronASRModelState

    @classmethod
    def validate_language(cls, language: str | None) -> str | None:
        if language in (None, "auto"):
            return language
        if language in cls.supported_languages:
            return language
        if language in NEMOTRON_ASR_LANGUAGE_ALIASES:
            return language
        raise ValueError(
            f"Unsupported language: {language!r}. Must be 'auto' or one of "
            f"{list(cls.supported_languages.keys())}."
        )

    @classmethod
    def update_speech_to_text_sampling_params(
        cls,
        *,
        sampling_params: object,
        model_config: ModelConfig,
        language: str | None,
    ) -> None:
        hf_config: NemotronASRConfig = model_config.hf_config
        extra_args = dict(getattr(sampling_params, "extra_args", None) or {})
        extra_args["transducer_asr_prompt_id"] = hf_config.prompt_id_for_language(
            language
        )
        sampling_params.extra_args = extra_args

    def _parse_and_validate_audio_input(
        self,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_features = kwargs.pop("input_features", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if isinstance(input_features, Sequence):
            input_features = nn.utils.rnn.pad_sequence(
                list(input_features),
                batch_first=True,
            )
        if isinstance(attention_mask, Sequence):
            attention_mask = nn.utils.rnn.pad_sequence(
                list(attention_mask),
                batch_first=True,
            )

        if not isinstance(input_features, torch.Tensor):
            raise ValueError("Nemotron ASR requires input_features.")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("Nemotron ASR requires attention_mask.")

        return input_features, attention_mask

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        input_features, attention_mask = self._parse_and_validate_audio_input(**kwargs)
        return self.model.get_encoder_outputs(input_features, attention_mask)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        forced_decoder_ids: torch.Tensor | None = None,
        forced_decoder_sequences: Sequence[Sequence[int]] | None = None,
        forced_decoder_request_indices: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del intermediate_tensors, kwargs

        if input_ids is None:
            raise ValueError("Nemotron ASR forward requires input_ids.")

        batch_size = input_ids.shape[0]
        device = input_ids.device
        if forced_decoder_ids is not None:
            forced_token_ids = forced_decoder_ids.to(device=device, dtype=torch.long)
        else:
            if forced_decoder_sequences is None and encoder_outputs:
                forced_decoder_sequences = self.model.greedy_decode_batch(
                    encoder_outputs
                )

            if forced_decoder_sequences:
                forced_decoder_state = TransducerASRForcedDecoderState(
                    eos_token_id=self.config.eos_token_id
                )
                forced_decoder_state.set_sequences(forced_decoder_sequences)
                forced_token_ids = forced_decoder_state.get_forced_token_ids(
                    positions=positions,
                    device=device,
                    request_indices=forced_decoder_request_indices,
                )
            else:
                forced_token_ids = torch.full(
                    (batch_size,),
                    self.config.eos_token_id,
                    dtype=torch.long,
                    device=device,
                )

        logits = torch.full(
            (batch_size, self.config.vocab_size),
            -1.0e9,
            dtype=torch.float32,
            device=device,
        )
        logits.scatter_(1, forced_token_ids.unsqueeze(1), 0.0)
        return logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_speech_to_text_config(
        cls,
        model_config: ModelConfig,
        task_type: str,
    ) -> SpeechToTextConfig:
        hf_config = model_config.hf_config
        return SpeechToTextConfig(
            sample_rate=hf_config.sample_rate,
            max_audio_clip_s=30,
            min_energy_split_window_size=None,
        )

    @classmethod
    def get_generation_prompt(
        cls,
        stt_params: SpeechToTextParams,
    ) -> PromptType:
        audio = stt_params.audio
        sample_rate = stt_params.stt_config.sample_rate
        hf_config = stt_params.model_config.hf_config
        hf_config.resolve_prompt_language(stt_params.language)

        return ExplicitEncoderDecoderPrompt(
            encoder_prompt=TextPrompt(
                prompt="",
                multi_modal_data={"audio": (audio, sample_rate)},
            ),
            decoder_prompt=TokensPrompt(prompt_token_ids=[hf_config.blank_token_id]),
        )

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        del input_stream
        hf_config = model_config.hf_config
        sample_rate = int(hf_config.sample_rate)
        chunk_samples = max(
            1, int(round(sample_rate * hf_config.realtime_chunk_ms / 1000))
        )
        buffer = np.empty(0, dtype=np.float32)

        async for audio_chunk in audio_stream:
            buffer = np.concatenate(
                (buffer, audio_chunk.astype(np.float32, copy=False))
            )
            while len(buffer) >= chunk_samples:
                segment = buffer[:chunk_samples].copy()
                buffer = buffer[chunk_samples:]
                yield cls.get_generation_prompt(
                    SpeechToTextParams(
                        audio=segment,
                        stt_config=SpeechToTextConfig(
                            sample_rate=sample_rate,
                            max_audio_clip_s=None,
                            min_energy_split_window_size=None,
                        ),
                        model_config=model_config,
                        language=None,
                    )
                )

        if len(buffer) > 0:
            yield cls.get_generation_prompt(
                SpeechToTextParams(
                    audio=buffer.copy(),
                    stt_config=SpeechToTextConfig(
                        sample_rate=sample_rate,
                        max_audio_clip_s=None,
                        min_energy_split_window_size=None,
                    ),
                    model_config=model_config,
                    language=None,
                )
            )

    @classmethod
    def get_num_audio_tokens(
        cls,
        audio_duration_s: float,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
    ) -> int | None:
        hf_config = model_config.hf_config.encoder_config
        extractor = ParakeetExtractor(hf_config)
        num_samples = int(audio_duration_s * stt_config.sample_rate)
        return extractor.audio_token_count(num_samples)

    @classmethod
    def post_process_output(cls, text: str) -> str:
        return strip_asr_special_tokens(text)


__all__ = [
    "NEMOTRON_ASR_SUPPORTED_LANGUAGES",
    "NemotronASRForRNNT",
    "NemotronASRModel",
]
