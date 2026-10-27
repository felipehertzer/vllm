# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.transformers_utils.nemotron_asr_conversion import (
    build_nemotron_asr_config,
    normalize_nemotron_asr_weight_name,
    normalize_state_dict,
)


def test_build_nemotron_asr_config_for_prompt_streaming_layout():
    config = build_nemotron_asr_config(
        {
            "sample_rate": 16000,
            "model_defaults": {
                "enc_hidden": 1024,
                "pred_hidden": 640,
                "initialize_prompt_feature": True,
                "num_prompts": 128,
                "prompt_dictionary": {"auto": 0, "en-US": 1, "pt-BR": 2},
            },
            "preprocessor": {"features": 128},
            "encoder": {
                "n_layers": 24,
                "d_model": 1024,
                "n_heads": 8,
                "ff_expansion_factor": 4,
                "subsampling_factor": 8,
                "subsampling_conv_channels": 256,
                "conv_kernel_size": 9,
                "dropout": 0.1,
                "dropout_att": 0.1,
                "pos_emb_max_len": 5000,
                "att_context_style": "chunked_limited",
                "att_context_size": [56, 3],
                "causal_downsampling": True,
                "conv_norm_type": "layer_norm",
                "conv_context_size": "causal",
            },
            "decoder": {
                "vocab_size": 1024,
                "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
            },
            "joint": {
                "num_classes": 1024,
                "jointnet": {"activation": "relu", "joint_hidden": 640},
            },
            "decoding": {"greedy": {"max_symbols": 10}},
        }
    )

    assert config["architectures"] == ["NemotronASRForRNNT"]
    assert config["model_type"] == "nemotron_asr"
    assert config["vocab_size"] == 1025
    assert config["blank_token_id"] == 1024
    assert config["prompt_dim"] == 128
    assert config["prompt_language_to_id"]["pt-BR"] == 2
    assert config["encoder_config"]["att_context_style"] == "chunked_limited"
    assert config["encoder_config"]["att_context_left"] == 56
    assert config["encoder_config"]["att_context_right"] == 3
    assert config["encoder_config"]["conv_norm_type"] == "layer_norm"
    assert config["encoder_config"]["conv_causal"] is True


def test_build_nemotron_asr_config_supports_english_only_without_prompt():
    config = build_nemotron_asr_config(
        {
            "sample_rate": 16000,
            "model_defaults": {
                "enc_hidden": 1024,
                "pred_hidden": 640,
                "joint_hidden": 640,
            },
            "preprocessor": {"features": 128},
            "encoder": {
                "n_layers": 24,
                "d_model": 1024,
                "n_heads": 8,
                "ff_expansion_factor": 4,
                "subsampling_factor": 8,
            },
            "decoder": {
                "vocab_size": 1024,
                "prednet": {"pred_hidden": 640, "pred_rnn_layers": 2},
            },
            "joint": {
                "num_classes": 1024,
                "jointnet": {"activation": "relu", "joint_hidden": 640},
            },
        }
    )

    assert config["prompt_dim"] == 0
    assert config["prompt_language_to_id"]["en-US"] == 0


def test_normalize_nemotron_asr_weight_name_maps_nemo_keys_to_hf_layout():
    examples = {
        "preprocessor.featurizer.window": None,
        "ctc_decoder.decoder_layers.0.weight": None,
        "encoder.pre_encode.out.weight": "encoder.subsampling.linear.weight",
        "encoder.pre_encode.conv.0.weight": "encoder.subsampling.layers.0.weight",
        "encoder.layers.0.conv.batch_norm.weight": "encoder.layers.0.conv.norm.weight",
        "encoder.layers.0.self_attn.linear_q.weight": (
            "encoder.layers.0.self_attn.q_proj.weight"
        ),
        "decoder.prediction.embed.weight": "decoder.embedding.weight",
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": "decoder.lstm.weight_ih_l0",
        "prompt_kernel.0.weight": "prompt_kernel.0.weight",
        "prompt_kernel.2.bias": "prompt_kernel.2.bias",
        "joint.pred.weight": "decoder.decoder_projector.weight",
        "joint.enc.bias": "encoder_projector.bias",
        "joint.joint_net.2.weight": "joint.head.weight",
    }

    for nemo_name, hf_name in examples.items():
        assert normalize_nemotron_asr_weight_name(nemo_name) == hf_name


def test_normalize_state_dict_requires_nemotron_asr_prefixes():
    state_dict = {
        "encoder.pre_encode.out.weight": torch.empty(1),
        "encoder.layers.0.self_attn.linear_q.weight": torch.empty(1),
        "prompt_kernel.0.weight": torch.empty(1),
        "decoder.prediction.embed.weight": torch.empty(1),
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": torch.empty(1),
        "joint.pred.weight": torch.empty(1),
        "joint.enc.weight": torch.empty(1),
        "joint.joint_net.2.weight": torch.empty(1),
    }

    converted = normalize_state_dict(state_dict)

    assert "encoder.subsampling.linear.weight" in converted
    assert "encoder.layers.0.self_attn.q_proj.weight" in converted
    assert "prompt_kernel.0.weight" in converted
    assert "decoder.embedding.weight" in converted
    assert "decoder.lstm.weight_ih_l0" in converted
    assert "decoder.decoder_projector.weight" in converted
    assert "encoder_projector.weight" in converted
    assert "joint.head.weight" in converted


def test_normalize_state_dict_allows_english_only_without_prompt_kernel():
    state_dict = {
        "encoder.pre_encode.out.weight": torch.empty(1),
        "encoder.layers.0.self_attn.linear_q.weight": torch.empty(1),
        "decoder.prediction.embed.weight": torch.empty(1),
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": torch.empty(1),
        "joint.pred.weight": torch.empty(1),
        "joint.enc.weight": torch.empty(1),
        "joint.joint_net.2.weight": torch.empty(1),
    }

    converted = normalize_state_dict(state_dict)

    assert "prompt_kernel.0.weight" not in converted
    assert "encoder_projector.weight" in converted
