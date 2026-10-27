# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert NVIDIA Nemotron ASR NeMo checkpoints to vLLM/HF layout."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from transformers.convert_slow_tokenizer import SentencePieceExtractor

from vllm.transformers_utils.configs.nemotron_asr import NEMOTRON_ASR_DEFAULT_LANGUAGE

DEFAULT_ENCODER_CONFIG = {
    "activation_dropout": 0.1,
    "attention_bias": False,
    "attention_dropout": 0.1,
    "conv_kernel_size": 9,
    "convolution_bias": False,
    "dropout": 0.1,
    "dropout_positions": 0.0,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layerdrop": 0.0,
    "max_position_embeddings": 5000,
    "model_type": "parakeet_encoder",
    "num_attention_heads": 8,
    "num_hidden_layers": 24,
    "num_key_value_heads": 8,
    "num_mel_bins": 128,
    "scale_input": False,
    "subsampling_conv_channels": 256,
    "subsampling_conv_kernel_size": 3,
    "subsampling_conv_stride": 2,
    "subsampling_factor": 8,
}


def _safe_extract(archive: tarfile.TarFile, target: Path) -> None:
    target = target.resolve()
    for member in archive.getmembers():
        member_path = (target / member.name).resolve()
        if target != member_path and target not in member_path.parents:
            raise ValueError(f"Refusing to extract unsafe path: {member.name}")
    try:
        archive.extractall(target, filter="data")
    except TypeError:
        archive.extractall(target)


def extract_nemo_archive(nemo_path: Path, work_dir: Path) -> Path:
    extract_dir = work_dir / "nemo"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(nemo_path, "r:*") as archive:
        _safe_extract(archive, extract_dir)
    return extract_dir


def _find_one(root: Path, patterns: tuple[str, ...]) -> Path:
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find any of: {', '.join(patterns)}")


def _get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = cfg
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _pick(cfg: Mapping[str, Any], paths: tuple[str, ...], default: Any) -> Any:
    for path in paths:
        value = _get(cfg, path)
        if value is not None:
            return value
    return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _activation_name(value: Any, default: str = "silu") -> str:
    if not isinstance(value, str):
        return default
    if value.lower() in {"swish", "silu"}:
        return "silu"
    return value.lower()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _prompt_dictionary(
    nemo_config: Mapping[str, Any],
) -> tuple[int, dict[str, int], str]:
    model_defaults = _get(nemo_config, "model_defaults", {}) or {}
    prompt_dictionary = _get(model_defaults, "prompt_dictionary", {}) or {}
    if not prompt_dictionary:
        if not bool(_get(model_defaults, "initialize_prompt_feature", False)):
            return (
                0,
                {"auto": 0, "en": 0, NEMOTRON_ASR_DEFAULT_LANGUAGE: 0},
                NEMOTRON_ASR_DEFAULT_LANGUAGE,
            )
        return (
            128,
            {"auto": 0, NEMOTRON_ASR_DEFAULT_LANGUAGE: 1},
            NEMOTRON_ASR_DEFAULT_LANGUAGE,
        )

    language_to_id = {str(key): int(value) for key, value in prompt_dictionary.items()}
    default_language = (
        NEMOTRON_ASR_DEFAULT_LANGUAGE
        if NEMOTRON_ASR_DEFAULT_LANGUAGE in language_to_id
        else next(iter(language_to_id))
    )
    num_prompts = _as_int(
        _get(model_defaults, "num_prompts"), max(language_to_id.values()) + 1
    )
    return num_prompts, language_to_id, default_language


def _streaming_config(nemo_config: Mapping[str, Any]) -> dict[str, Any]:
    encoder = _get(nemo_config, "encoder", {})
    att_context_size = _get(encoder, "att_context_size", [-1, -1]) or [-1, -1]
    if att_context_size and not isinstance(att_context_size[0], (int, float)):
        att_context_size = list(att_context_size[0])

    streaming = _get(encoder, "streaming_cfg", {}) or {}

    def int_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [int(item) for item in value]
        return [int(value)]

    return {
        "att_context_left": _as_int(att_context_size[0], -1),
        "att_context_right": (
            _as_int(att_context_size[1], -1) if len(att_context_size) > 1 else -1
        ),
        "att_context_style": str(_get(encoder, "att_context_style", "chunked_limited")),
        "causal_downsampling": bool(_get(encoder, "causal_downsampling", True)),
        "conv_causal": _get(encoder, "conv_context_size") == "causal",
        "streaming": {
            "chunk_size": int_list(_get(streaming, "chunk_size")),
            "shift_size": int_list(_get(streaming, "shift_size")),
            "pre_encode_cache_size": int_list(_get(streaming, "pre_encode_cache_size")),
            "cache_drop_size": _as_int(_get(streaming, "cache_drop_size"), 0),
            "last_channel_cache_size": _as_int(
                _get(streaming, "last_channel_cache_size"),
                0,
            ),
            "valid_out_len": _as_int(_get(streaming, "valid_out_len"), 0),
            "drop_extra_pre_encoded": _as_int(
                _get(streaming, "drop_extra_pre_encoded"),
                0,
            ),
        },
    }


def build_nemotron_asr_config(nemo_config: Mapping[str, Any]) -> dict[str, Any]:
    encoder = _get(nemo_config, "encoder", {})
    prednet = _get(nemo_config, "decoder.prednet", {})
    jointnet = _get(nemo_config, "joint.jointnet", {})
    preprocessor = _get(nemo_config, "preprocessor", {})
    greedy = _get(nemo_config, "decoding.greedy", {}) or {}

    token_vocab_size = _as_int(
        _pick(nemo_config, ("joint.num_classes", "decoder.vocab_size"), 8192),
        8192,
    )
    vocab_size = token_vocab_size + 1
    blank_token_id = vocab_size - 1
    sample_rate = _as_int(
        _pick(nemo_config, ("sample_rate", "preprocessor.sample_rate"), 16000),
        16000,
    )
    decoder_hidden_size = _as_int(
        _pick(
            nemo_config,
            ("model_defaults.pred_hidden", "decoder.prednet.pred_hidden"),
            640,
        ),
        640,
    )
    encoder_hidden_size = _as_int(
        _pick(nemo_config, ("model_defaults.enc_hidden", "encoder.d_model"), 1024),
        1024,
    )
    num_prompts, language_to_id, default_language = _prompt_dictionary(nemo_config)

    encoder_config = DEFAULT_ENCODER_CONFIG.copy()
    encoder_config.update(
        {
            "attention_bias": bool(_pick(nemo_config, ("encoder.use_bias",), False)),
            "attention_dropout": _as_float(_get(encoder, "dropout_att"), 0.1),
            "conv_kernel_size": _as_int(_get(encoder, "conv_kernel_size"), 9),
            "conv_norm_type": str(_get(encoder, "conv_norm_type", "batch_norm")),
            "convolution_bias": bool(_get(encoder, "use_bias", False)),
            "dropout": _as_float(_get(encoder, "dropout"), 0.1),
            "dropout_positions": _as_float(_get(encoder, "dropout_emb"), 0.0),
            "hidden_act": _activation_name(_get(encoder, "activation"), "silu"),
            "hidden_size": encoder_hidden_size,
            "intermediate_size": encoder_hidden_size
            * _as_int(_get(encoder, "ff_expansion_factor"), 4),
            "max_position_embeddings": _as_int(_get(encoder, "pos_emb_max_len"), 5000),
            "num_attention_heads": _as_int(_get(encoder, "n_heads"), 8),
            "num_hidden_layers": _as_int(_get(encoder, "n_layers"), 24),
            "num_key_value_heads": _as_int(_get(encoder, "n_heads"), 8),
            "num_mel_bins": _as_int(_get(preprocessor, "features"), 128),
            "normalize": str(_get(preprocessor, "normalize", "NA")),
            "scale_input": bool(_get(encoder, "xscaling", True)),
            "subsampling_conv_channels": _as_int(
                _get(encoder, "subsampling_conv_channels"),
                256,
            ),
            "subsampling_factor": _as_int(_get(encoder, "subsampling_factor"), 8),
        }
    )
    encoder_config.update(_streaming_config(nemo_config))

    max_symbols = _pick(
        nemo_config,
        (
            "decoding.greedy.max_symbols",
            "decoding.greedy.max_symbols_per_step",
        ),
        _get(greedy, "max_symbols_per_step", 10),
    )

    return {
        "architectures": ["NemotronASRForRNNT"],
        "blank_token_id": blank_token_id,
        "decoder_hidden_size": decoder_hidden_size,
        "encoder_config": encoder_config,
        "hidden_act": _activation_name(_get(jointnet, "activation"), "relu"),
        "is_encoder_decoder": True,
        "max_symbols_per_step": _as_int(max_symbols, 10),
        "model_type": "nemotron_asr",
        "num_decoder_layers": _as_int(_get(prednet, "pred_rnn_layers"), 1),
        "pad_token_id": 0,
        "eos_token_id": blank_token_id,
        "sample_rate": sample_rate,
        "vocab_size": vocab_size,
        "prompt_dim": num_prompts,
        "prompt_default_language": default_language,
        "prompt_language_to_id": language_to_id,
        "chunk_ms": 1120,
        "realtime_chunk_ms": 320,
        "frame_shift_seconds": 0.08,
    }


def normalize_nemotron_asr_weight_name(name: str) -> str | None:
    for prefix in ("model.", "module."):
        if name.startswith(prefix):
            name = name[len(prefix) :]

    if name.startswith("preprocessor."):
        return None
    if name.startswith("ctc_decoder."):
        return None

    replacements = (
        ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
        ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
        (".conv.batch_norm.", ".conv.norm."),
        (".self_attn.pos_bias_u", ".self_attn.bias_u"),
        (".self_attn.pos_bias_v", ".self_attn.bias_v"),
        (".self_attn.linear_q.", ".self_attn.q_proj."),
        (".self_attn.linear_k.", ".self_attn.k_proj."),
        (".self_attn.linear_v.", ".self_attn.v_proj."),
        (".self_attn.linear_out.", ".self_attn.o_proj."),
        (".self_attn.linear_pos.", ".self_attn.relative_k_proj."),
        ("decoder.prediction.embed.", "decoder.embedding."),
        ("decoder.prediction.dec_rnn.lstm.", "decoder.lstm."),
        ("joint.pred.", "decoder.decoder_projector."),
        ("joint.enc.", "encoder_projector."),
        ("joint.joint_net.2.", "joint.head."),
    )

    for old, new in replacements:
        name = name.replace(old, new)
    return name


def normalize_state_dict(state_dict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        new_name = normalize_nemotron_asr_weight_name(name)
        if new_name is None:
            continue
        converted[new_name] = tensor.contiguous()

    required_prefixes = [
        "encoder.",
        "encoder_projector.",
        "decoder.embedding.",
        "decoder.lstm.",
        "decoder.decoder_projector.",
        "joint.head.",
    ]
    if any(name.startswith("prompt_kernel.") for name in converted):
        required_prefixes.append("prompt_kernel.")
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in converted)
    ]
    if missing:
        raise ValueError(
            "Converted checkpoint is missing expected prefixes: "
            + ", ".join(missing)
            + ". Check whether this Nemotron checkpoint uses a new layout."
        )

    return converted


def load_nemo_config(extract_dir: Path) -> dict[str, Any]:
    config_path = _find_one(extract_dir, ("model_config.yaml", "model_config.yml"))
    with config_path.open() as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"{config_path} did not contain a YAML mapping")
    return loaded


def load_nemo_state_dict(extract_dir: Path) -> Mapping[str, Any]:
    weights_path = _find_one(extract_dir, ("model_weights.ckpt", "*.ckpt"))
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, Mapping) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"{weights_path} did not contain a PyTorch state dict")
    return state_dict


def build_tokenizer(tokenizer_model: Path, blank_token_id: int) -> Tokenizer:
    vocab, merges = SentencePieceExtractor(str(tokenizer_model)).extract()
    vocab["<blank>"] = blank_token_id
    tokenizer = Tokenizer(
        BPE(vocab=vocab, merges=merges, unk_token="<unk>", fuse_unk=False)
    )
    tokenizer.normalizer = normalizers.Sequence([normalizers.Nmt(), normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁",
        prepend_scheme="always",
    )
    tokenizer.add_special_tokens(["<unk>", "<blank>"])
    return tokenizer


def write_tokenizer_artifacts(
    extract_dir: Path, output_dir: Path, config: Mapping[str, Any]
) -> None:
    tokenizer_model = _find_one(extract_dir, ("*tokenizer.model", "tokenizer.model"))
    tokenizer = build_tokenizer(tokenizer_model, int(config["blank_token_id"]))
    tokenizer.save(str(output_dir / "tokenizer.json"))
    shutil.copyfile(tokenizer_model, output_dir / "tokenizer.model")

    _write_json(
        output_dir / "tokenizer_config.json",
        {
            "clean_up_tokenization_spaces": False,
            "eos_token": "<blank>",
            "model_max_length": 1000000000000000019884624838656,
            "pad_token": "<unk>",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
        },
    )


def write_model_artifacts(
    *,
    config: Mapping[str, Any],
    weights: Mapping[str, torch.Tensor],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "generation_config.json",
        {
            "_from_model_config": True,
            "decoder_start_token_id": config["blank_token_id"],
            "eos_token_id": config["eos_token_id"],
            "output_attentions": False,
            "output_hidden_states": False,
            "pad_token_id": config["pad_token_id"],
        },
    )
    _write_json(
        output_dir / "processor_config.json",
        {
            "blank_token": "<blank>",
            "feature_extractor": {
                "feature_extractor_type": "ParakeetFeatureExtractor",
                "feature_size": config["encoder_config"]["num_mel_bins"],
                "hop_length": 160,
                "n_fft": 512,
                "padding_side": "right",
                "padding_value": 0.0,
                "preemphasis": 0.97,
                "return_attention_mask": True,
                "sampling_rate": config["sample_rate"],
                "win_length": 400,
            },
            "processor_class": "ParakeetProcessor",
        },
    )
    save_file(weights, output_dir / "model.safetensors")


def convert_nemo_to_hf(nemo_path: Path, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="nemotron-asr-nemo-") as tmp:
        extract_dir = extract_nemo_archive(nemo_path, Path(tmp))
        nemo_config = load_nemo_config(extract_dir)
        config = build_nemotron_asr_config(nemo_config)
        state_dict = load_nemo_state_dict(extract_dir)
        weights = normalize_state_dict(state_dict)
        write_model_artifacts(config=config, weights=weights, output_dir=output_dir)
        write_tokenizer_artifacts(extract_dir, output_dir, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Nemotron 3.5 ASR .nemo archive to vLLM/HF format."
    )
    parser.add_argument(
        "nemo_path",
        type=Path,
        help="Path to nemotron-3.5-asr-streaming .nemo archive",
    )
    parser.add_argument("output_dir", type=Path, help="Output HF model directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_nemo_to_hf(args.nemo_path, args.output_dir)
    print(f"Wrote converted Nemotron ASR model to {args.output_dir}")


__all__ = [
    "build_nemotron_asr_config",
    "convert_nemo_to_hf",
    "extract_nemo_archive",
    "load_nemo_config",
    "load_nemo_state_dict",
    "main",
    "normalize_nemotron_asr_weight_name",
    "normalize_state_dict",
    "parse_args",
]
