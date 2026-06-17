# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from transformers import ParakeetEncoderConfig

from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    NemotronASRForRNNTConfig,
)
from vllm.model_executor.models.nemotron_asr import (
    NemotronASRCausalSubsamplingConv2D,
    NemotronASREncoder,
    NemotronASRForRNNT,
)
from vllm.model_executor.models.transducer_asr import (
    TransducerDecodeConfig,
    greedy_decode_transducer_batch,
    strip_asr_special_tokens,
)
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs.nemotron_asr import NemotronASRConfig


class _FakeDecoder:
    def predict_batch(self, token_ids, state, rows):
        batch = token_ids.shape[0]
        del state, rows
        pred_state = torch.zeros(batch, 4)
        next_state = (
            torch.zeros(1, batch, 4),
            torch.zeros(1, batch, 4),
        )
        return pred_state, next_state


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
    assert output.shape == (1, 11, 1024)


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
    )

    encoder = NemotronASREncoder(config)
    conv = encoder.layers[0].conv

    assert conv.pointwise_conv1.bias is None
    assert conv.depthwise_conv.bias is None
    assert conv.pointwise_conv2.bias is None
    assert set(conv.norm.state_dict()) == {"weight", "bias"}


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


def test_strip_asr_special_tokens_preserves_words():
    assert strip_asr_special_tokens("<en-US> Forward [noise] pocket") == (
        "Forward pocket"
    )
