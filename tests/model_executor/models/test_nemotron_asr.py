# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from transformers import ParakeetEncoderConfig
from transformers.feature_extraction_utils import BatchFeature

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    NemotronASRForRNNTConfig,
)
from vllm.model_executor.models.nemotron_asr import (
    NemotronASRAttention,
    NemotronASRCausalConv1D,
    NemotronASRCausalSubsamplingConv2D,
    NemotronASRConvLayerNorm,
    NemotronASREncoder,
    NemotronASRForRNNT,
    NemotronASRModel,
    NemotronASRMultiModalProcessor,
)
from vllm.model_executor.models.transducer_asr import (
    TransducerDecodeConfig,
    TransducerPredictionDecoder,
    greedy_decode_transducer_batch,
    greedy_decode_transducer_batch_with_timestamps,
    strip_asr_special_tokens,
)
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.nemotron_asr import NemotronASRConfig


class _FakeDecoder:
    def __init__(self):
        self.calls = 0

    def predict_batch(self, token_ids, state, rows):
        self.calls += 1
        batch = token_ids.shape[0]
        del state, rows
        pred_state = torch.zeros(batch, 4)
        next_state = (
            torch.zeros(1, batch, 4),
            torch.zeros(1, batch, 4),
        )
        return pred_state, next_state


def test_nemotron_processor_bypasses_transformers_audio_api():
    processor = object.__new__(NemotronASRMultiModalProcessor)
    audio = np.zeros(160, dtype=np.float32)
    processor._get_hf_mm_data = Mock(  # type: ignore[method-assign]
        return_value=({"audios": [audio]}, {"passthrough": [torch.tensor(1)]})
    )
    processor._extract_audio_features = Mock(  # type: ignore[method-assign]
        return_value=BatchFeature(
            data={
                "input_features": torch.zeros(1, 2, 3),
                "attention_mask": torch.ones(1, 2),
            },
            tensor_type="pt",
        )
    )

    result = processor._apply_hf_processor_main(SimpleNamespace(), {})

    processor._extract_audio_features.assert_called_once()
    assert result["input_ids"] == [[0]]
    assert result["passthrough"][0].item() == 1


def test_nemotron_config_resolves_default_and_alias_language():
    config = NemotronASRConfig(
        vocab_size=4,
        blank_token_id=3,
        prompt_language_to_id={"auto": 0, "en-US": 1, "pt-BR": 2},
    )

    assert config.resolve_prompt_language(None) == "en-US"
    assert config.prompt_id_for_language("en") == 1
    assert config.prompt_id_for_language("pt") == 2


def test_nemotron_config_rejects_unknown_prompt_language():
    config = NemotronASRConfig(
        vocab_size=4,
        blank_token_id=3,
        prompt_language_to_id={"auto": 0, "en-US": 1},
    )

    try:
        config.prompt_id_for_language("xx")
    except ValueError as exc:
        assert "Unsupported Nemotron ASR language" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_nemotron_model_accepts_auto_and_rejects_unknown_language():
    assert NemotronASRForRNNT.validate_language("auto") == "auto"
    assert NemotronASRForRNNT.validate_language("pt") == "pt"

    with pytest.raises(ValueError, match="Unsupported language"):
        NemotronASRForRNNT.validate_language("xx")


def test_nemotron_sampling_params_carry_prompt_id():
    hf_config = NemotronASRConfig(
        vocab_size=4,
        blank_token_id=3,
        prompt_language_to_id={"auto": 0, "en-US": 1, "pt-BR": 2},
    )
    sampling_params = SamplingParams(max_tokens=4)

    NemotronASRForRNNT.update_speech_to_text_sampling_params(
        sampling_params=sampling_params,
        model_config=SimpleNamespace(hf_config=hf_config),
        language="pt",
    )

    assert sampling_params.extra_args == {"transducer_asr_prompt_id": 2}


def test_nemotron_mps_encoder_hybrid_uses_auto_when_available(monkeypatch):
    monkeypatch.delenv("NEMOTRON_ASR_MPS_ENCODER", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    assert NemotronASRModel._should_use_mps_encoder_hybrid(torch.float32)


def test_nemotron_mps_encoder_hybrid_can_be_disabled(monkeypatch):
    monkeypatch.setenv("NEMOTRON_ASR_MPS_ENCODER", "0")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)

    assert not NemotronASRModel._should_use_mps_encoder_hybrid(torch.float32)


def test_nemotron_mps_encoder_dtype_defaults_to_float16(monkeypatch):
    monkeypatch.delenv("NEMOTRON_ASR_MPS_ENCODER_DTYPE", raising=False)

    assert NemotronASRModel._get_mps_encoder_dtype() is torch.float16


def test_nemotron_mps_encoder_dtype_accepts_float32(monkeypatch):
    monkeypatch.setenv("NEMOTRON_ASR_MPS_ENCODER_DTYPE", "float32")

    assert NemotronASRModel._get_mps_encoder_dtype() is torch.float32


def test_nemotron_cuda_decoder_defaults_to_cpu(monkeypatch):
    monkeypatch.delenv("NEMOTRON_ASR_CUDA_DECODER_DEVICE", raising=False)

    assert NemotronASRModel._should_use_cuda_cpu_decoder()


def test_nemotron_cuda_decoder_can_stay_on_cuda(monkeypatch):
    monkeypatch.setenv("NEMOTRON_ASR_CUDA_DECODER_DEVICE", "cuda")

    assert not NemotronASRModel._should_use_cuda_cpu_decoder()


def test_nemotron_causal_subsampling_matches_nemo_flatten_width():
    config = ParakeetEncoderConfig(
        hidden_size=1024,
        num_mel_bins=128,
        subsampling_conv_channels=256,
        subsampling_factor=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
    )
    subsampling = NemotronASRCausalSubsamplingConv2D(config)

    output = subsampling(torch.zeros(1, 101, 128))

    assert subsampling.linear.in_features == 256 * 17
    assert output.shape == (1, 14, 1024)


def test_nemotron_encoder_causal_subsampling_length_matches_nemo():
    config = ParakeetEncoderConfig(
        hidden_size=1024,
        num_mel_bins=128,
        subsampling_conv_channels=256,
        subsampling_factor=8,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        causal_downsampling=True,
    )
    encoder = NemotronASREncoder(config)

    output_lengths = encoder._get_subsampling_output_length(torch.tensor([2999]))

    assert output_lengths.tolist() == [376]


def test_nemotron_encoder_wraps_attention_for_cuda_compile():
    config = ParakeetEncoderConfig(
        hidden_size=8,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_mel_bins=128,
        subsampling_conv_channels=4,
        subsampling_factor=8,
    )

    encoder = NemotronASREncoder(config)
    assert not isinstance(encoder.layers[0].self_attn, NemotronASRAttention)

    encoder._patch_nemo_attention_modules()

    assert isinstance(encoder.layers[0].self_attn, NemotronASRAttention)
    assert hasattr(encoder.layers[0].self_attn, "q_proj")


def test_nemotron_encoder_state_dict_matches_nemo_conv_layout():
    config = ParakeetEncoderConfig(
        hidden_size=8,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_mel_bins=128,
        subsampling_conv_channels=4,
        subsampling_factor=8,
        causal_downsampling=True,
        conv_causal=True,
        conv_norm_type="layer_norm",
    )

    encoder = NemotronASREncoder(config)
    conv = encoder.layers[0].conv

    assert conv.pointwise_conv1.bias is None
    assert conv.depthwise_conv.bias is None
    assert conv.pointwise_conv2.bias is None
    assert isinstance(conv.depthwise_conv, NemotronASRCausalConv1D)
    assert isinstance(conv.norm, NemotronASRConvLayerNorm)
    assert set(conv.norm.state_dict()) == {"weight", "bias"}


def test_nemotron_causal_conv1d_matches_manual_left_padding():
    torch.manual_seed(0)
    source = torch.nn.Conv1d(
        in_channels=4,
        out_channels=4,
        kernel_size=5,
        groups=4,
        bias=False,
    )
    causal = NemotronASRCausalConv1D(source)
    causal.weight.data.copy_(source.weight)
    inputs = torch.randn(2, 4, 11)

    expected = source(F.pad(inputs, pad=(4, 0)))

    torch.testing.assert_close(causal(inputs), expected)
    assert causal._uses_manual_depthwise_kernel("cpu")
    assert causal._uses_manual_depthwise_kernel("mps")
    assert not causal._uses_manual_depthwise_kernel("cuda")


def test_nemotron_encoder_uses_chunked_limited_attention_mask():
    config = ParakeetEncoderConfig(
        hidden_size=8,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_mel_bins=128,
        subsampling_conv_channels=4,
        subsampling_factor=8,
        causal_downsampling=True,
        att_context_style="chunked_limited",
        att_context_left=2,
        att_context_right=1,
    )
    encoder = NemotronASREncoder(config)

    mask = encoder._get_nemo_attention_mask(
        torch.ones(1, 48, dtype=torch.bool),
        target_length=6,
    )

    assert mask.shape == (1, 1, 6, 6)
    assert bool(mask[0, 0, 0, 1])
    assert not bool(mask[0, 0, 1, 2])
    assert bool(mask[0, 0, 4, 2])
    assert not bool(mask[0, 0, 4, 0])


def test_nemotron_config_updates_runtime_metadata():
    model_config = SimpleNamespace(
        enforce_eager=False,
        hf_config=SimpleNamespace(eos_token_id=3),
        hf_text_config=SimpleNamespace(eos_token_id=3),
        override_generation_config={},
    )
    vllm_config = SimpleNamespace(model_config=model_config)

    assert MODELS_CONFIG_MAP["NemotronASRForRNNT"] is NemotronASRForRNNTConfig
    NemotronASRForRNNTConfig.verify_and_update_config(vllm_config)

    assert model_config.override_generation_config == {"eos_token_id": 3}


def test_nemotron_stt_config_uses_chunk_duration():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            sample_rate=16000,
            eos_token_id=3,
            chunk_ms=1120,
        )
    )

    stt_config = NemotronASRForRNNT.get_speech_to_text_config(
        model_config,
        task_type="transcribe",
    )

    assert stt_config.sample_rate == 16000
    assert stt_config.max_audio_clip_s == 30
    assert stt_config.min_energy_split_window_size is None


def test_rnnt_greedy_decode_batch_emits_until_blank():
    decoder = _FakeDecoder()
    calls = 0

    def joint_logits(encoder_state, pred_state):
        nonlocal calls
        del encoder_state, pred_state
        calls += 1
        if calls == 1:
            return torch.tensor([[9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        return torch.tensor([[0.0, 0.0, 9.0], [0.0, 0.0, 9.0]])

    outputs = greedy_decode_transducer_batch(
        encoder_projected=torch.zeros(2, 1, 4),
        lengths=torch.tensor([1, 1]),
        decoder=decoder,
        joint_logits=joint_logits,
        config=TransducerDecodeConfig(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            max_symbols_per_step=4,
        ),
    )

    assert outputs == [[0, 1], [0, 1]]


def test_rnnt_greedy_decode_preserves_emission_frames():
    decoder = _FakeDecoder()

    def joint_logits(encoder_state, pred_state):
        del pred_state
        frame = int(encoder_state[0, 0].item())
        if frame in {0, 2}:
            return torch.tensor([[9.0, 0.0, 0.0]])
        return torch.tensor([[0.0, 0.0, 9.0]])

    hypotheses = greedy_decode_transducer_batch_with_timestamps(
        encoder_projected=torch.tensor([[[0.0] * 4, [1.0] * 4, [2.0] * 4]]),
        lengths=torch.tensor([3]),
        decoder=decoder,
        joint_logits=joint_logits,
        config=TransducerDecodeConfig(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            max_symbols_per_step=1,
        ),
    )

    assert hypotheses[0].token_ids == [0, 0, 1]
    assert hypotheses[0].timesteps == [0, 2]


def test_rnnt_greedy_decode_batch_advances_at_symbol_limit():
    decoder = _FakeDecoder()

    outputs = greedy_decode_transducer_batch(
        encoder_projected=torch.zeros(1, 1, 4),
        lengths=torch.tensor([1]),
        decoder=decoder,
        joint_logits=lambda encoder_state, pred_state: torch.tensor([[9.0, 0.0, 0.0]]),
        config=TransducerDecodeConfig(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            max_symbols_per_step=2,
        ),
    )

    assert outputs == [[0, 0, 1]]


def test_rnnt_greedy_decode_batch_reuses_prediction_on_blank_frames():
    decoder = _FakeDecoder()

    outputs = greedy_decode_transducer_batch(
        encoder_projected=torch.zeros(1, 3, 4),
        lengths=torch.tensor([3]),
        decoder=decoder,
        joint_logits=lambda encoder_state, pred_state: torch.tensor([[0.0, 0.0, 9.0]]),
        config=TransducerDecodeConfig(
            vocab_size=3,
            blank_token_id=2,
            eos_token_id=1,
            max_symbols_per_step=4,
        ),
    )

    assert outputs == [[1]]
    assert decoder.calls == 1


def test_strip_asr_special_tokens_preserves_words():
    assert strip_asr_special_tokens("<en-US> Forward [noise] pocket") == (
        "Forward pocket"
    )


def test_cpu_prediction_gate_cache_matches_lstm_states_and_weight_updates():
    """Frozen input projections must preserve recurrence and reload semantics."""
    torch.manual_seed(713)
    decoder = TransducerPredictionDecoder(
        vocab_size=11, hidden_size=16, num_layers=2
    ).eval()
    rows = torch.tensor([0])
    reference_state = None
    optimized_state = None
    with torch.inference_mode():
        for token in (10, 3, 3, 7, 1):
            labels = torch.tensor([token])
            reference, reference_state = decoder.lstm(
                decoder.embedding(labels[:, None]), reference_state
            )
            reference = decoder.decoder_projector(reference)[:, 0, :]
            actual, optimized_state = decoder.predict_batch(
                labels, optimized_state, rows
            )
            torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
            for actual_state, expected_state in zip(
                optimized_state, reference_state, strict=True
            ):
                torch.testing.assert_close(
                    actual_state, expected_state, atol=1e-6, rtol=1e-5
                )
        previous_table = decoder._cpu_input_gate_cache
        decoder.embedding.weight.add_(0.25)
        labels = torch.tensor([2])
        reference, _ = decoder.lstm(decoder.embedding(labels[:, None]))
        actual, _ = decoder.predict_batch(labels, None, rows)
        torch.testing.assert_close(
            actual,
            decoder.decoder_projector(reference)[:, 0, :],
            atol=1e-6,
            rtol=1e-5,
        )
        assert decoder._cpu_input_gate_cache is not previous_table


def test_prediction_gate_cache_does_not_interfere_with_training():
    decoder = TransducerPredictionDecoder(vocab_size=11, hidden_size=16, num_layers=2)
    output, _ = decoder.predict_batch(torch.tensor([2]), None, torch.tensor([0]))
    output.sum().backward()
    assert decoder.embedding.weight.grad is not None
    assert decoder._cpu_input_gate_cache is None
