# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from tools.convert_parakeet_tdt_nemo_to_hf import (
    build_parakeet_tdt_config,
    normalize_nemo_weight_name,
    normalize_state_dict,
)


def test_build_parakeet_tdt_config_for_v2_nemo_layout():
    config = build_parakeet_tdt_config(
        {
            "sample_rate": 16000,
            "model_defaults": {
                "enc_hidden": 1024,
                "pred_hidden": 640,
                "tdt_durations": [0, 1, 2, 3, 4],
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

    assert config["architectures"] == ["ParakeetForTDT"]
    assert config["model_type"] == "parakeet_tdt"
    assert config["vocab_size"] == 1025
    assert config["blank_token_id"] == 1024
    assert config["eos_token_id"] == 1024
    assert config["decoder_hidden_size"] == 640
    assert config["encoder_config"]["hidden_size"] == 1024
    assert config["encoder_config"]["intermediate_size"] == 4096


def test_normalize_nemo_weight_name_maps_v2_keys_to_hf_layout():
    examples = {
        "preprocessor.featurizer.window": None,
        "encoder.pre_encode.out.weight": "encoder.subsampling.linear.weight",
        "encoder.pre_encode.conv.0.weight": "encoder.subsampling.layers.0.weight",
        "encoder.layers.0.conv.batch_norm.weight": "encoder.layers.0.conv.norm.weight",
        "encoder.layers.0.self_attn.pos_bias_u": "encoder.layers.0.self_attn.bias_u",
        "encoder.layers.0.self_attn.linear_q.weight": (
            "encoder.layers.0.self_attn.q_proj.weight"
        ),
        "encoder.layers.0.self_attn.linear_pos.weight": (
            "encoder.layers.0.self_attn.relative_k_proj.weight"
        ),
        "decoder.prediction.embed.weight": "decoder.embedding.weight",
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": "decoder.lstm.weight_ih_l0",
        "joint.pred.weight": "decoder.decoder_projector.weight",
        "joint.enc.bias": "encoder_projector.bias",
        "joint.joint_net.2.weight": "joint.head.weight",
    }

    for nemo_name, hf_name in examples.items():
        assert normalize_nemo_weight_name(nemo_name) == hf_name


def test_normalize_state_dict_requires_parakeet_tdt_prefixes():
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

    assert "encoder.subsampling.linear.weight" in converted
    assert "encoder.layers.0.self_attn.q_proj.weight" in converted
    assert "decoder.embedding.weight" in converted
    assert "decoder.lstm.weight_ih_l0" in converted
    assert "decoder.decoder_projector.weight" in converted
    assert "encoder_projector.weight" in converted
    assert "joint.head.weight" in converted
