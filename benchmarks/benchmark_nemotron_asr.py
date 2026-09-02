# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark Nemotron ASR local model stages.

Examples:

    PYTHONPATH=. python benchmarks/benchmark_nemotron_asr.py \
      --model-dir /path/to/nemotron-vllm-hf \
      --audio-file /path/to/audio.mp3 \
      --mode mps-hybrid --runs 12 --warmup 3

    PYTHONPATH=. python benchmarks/benchmark_nemotron_asr.py \
      --model-dir /path/to/nemotron-vllm-hf \
      --audio-file /path/to/audio.mp3 \
      --mode cuda --batch-size 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import safe_open
from transformers import PreTrainedTokenizerFast

from vllm.config.compilation import CompilationConfig, CompilationMode
from vllm.model_executor.models.nemotron_asr import NemotronASRForRNNT
from vllm.model_executor.models.parakeet import ParakeetExtractor
from vllm.multimodal.media.audio import load_audio
from vllm.transformers_utils.configs.nemotron_asr import NemotronASRConfig


class _MMConfig:
    def get_limit_per_prompt(self, modality: str) -> int:
        del modality
        return 1


def _stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = round((len(ordered) - 1) * 0.95)
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
        "mean": statistics.mean(values),
        "max": max(values),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _mode_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this machine.")
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(
    model_dir: Path,
    *,
    mode: str,
) -> tuple[NemotronASRConfig, NemotronASRForRNNT, float]:
    os.environ["NEMOTRON_ASR_MPS_ENCODER"] = "auto" if mode == "mps-hybrid" else "0"
    hf_config = NemotronASRConfig.from_pretrained(model_dir)
    multimodal_config = _MMConfig()
    model_config = SimpleNamespace(
        hf_config=hf_config,
        dtype=torch.float32,
        multimodal_config=multimodal_config,
    )
    model_config.get_multimodal_config = lambda: multimodal_config
    vllm_config = SimpleNamespace(
        model_config=model_config,
        compilation_config=CompilationConfig(mode=CompilationMode.NONE),
    )

    started = time.perf_counter()
    model = NemotronASRForRNNT(vllm_config=vllm_config)
    with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as f:
        weight_keys = f.keys()
        model.load_weights((key, f.get_tensor(key)) for key in weight_keys)
    if mode == "cuda":
        model.to(_mode_device(mode))
    model.eval()
    return hf_config, model, time.perf_counter() - started


def _load_audio(audio_file: Path, *, sample_rate: int, seconds: float) -> np.ndarray:
    audio, _ = load_audio(audio_file, sr=sample_rate, mono=True)
    audio = audio.astype(np.float32, copy=False)
    max_samples = int(sample_rate * seconds)
    return audio[:max_samples]


def _make_features(
    hf_config: NemotronASRConfig,
    audio: np.ndarray,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    lengths = torch.tensor([len(audio)] * batch_size, dtype=torch.long)
    raw = [torch.from_numpy(audio) for _ in range(batch_size)]
    extractor = ParakeetExtractor(hf_config.encoder_config)

    started = time.perf_counter()
    features = extractor._pad_raw_speech(raw, len(audio), "cpu")
    features = extractor._apply_preemphasis(features, lengths)
    features = extractor._torch_extract_fbank_features(features, "cpu")
    features, attention_mask = extractor._normalize_mel_features(features, lengths)
    return features, attention_mask, time.perf_counter() - started


def _run_once(
    model: NemotronASRForRNNT,
    hf_config: NemotronASRConfig,
    audio: np.ndarray,
    *,
    mode: str,
    batch_size: int,
) -> tuple[float, float, float, list[list[int]]]:
    features, attention_mask, feature_s = _make_features(
        hf_config,
        audio,
        batch_size=batch_size,
    )
    input_device = _mode_device(mode)
    features = features.to(input_device)
    attention_mask = attention_mask.to(input_device)

    if mode == "mps-hybrid":
        torch.mps.synchronize()
    else:
        _sync(input_device)
    started = time.perf_counter()
    encoder_outputs = model.model.get_encoder_outputs(features, attention_mask)
    if mode == "mps-hybrid":
        torch.mps.synchronize()
    else:
        _sync(input_device)
    encoder_s = time.perf_counter() - started

    decode_device = encoder_outputs[0].device
    _sync(decode_device)
    started = time.perf_counter()
    token_ids = model.model.greedy_decode_batch(encoder_outputs)
    _sync(decode_device)
    decode_s = time.perf_counter() - started
    return feature_s, encoder_s, decode_s, token_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--audio-file", type=Path, required=True)
    parser.add_argument("--mode", choices=["cpu", "mps-hybrid", "cuda"], required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=12)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    hf_config, model, load_s = _load_model(args.model_dir, mode=args.mode)
    audio = _load_audio(
        args.audio_file,
        sample_rate=int(hf_config.sample_rate),
        seconds=args.seconds,
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_dir)

    last_tokens: list[list[int]] = []
    for _ in range(args.warmup):
        _, _, _, last_tokens = _run_once(
            model,
            hf_config,
            audio,
            mode=args.mode,
            batch_size=args.batch_size,
        )

    rows: list[tuple[float, float, float]] = []
    output_hashes: list[str] = []
    mps_memory_rows: list[tuple[int, int]] = []
    for _ in range(args.runs):
        feature_s, encoder_s, decode_s, last_tokens = _run_once(
            model,
            hf_config,
            audio,
            mode=args.mode,
            batch_size=args.batch_size,
        )
        rows.append((feature_s, encoder_s, decode_s))
        output_hashes.append(
            hashlib.sha256(
                json.dumps(last_tokens, separators=(",", ":")).encode()
            ).hexdigest()
        )
        if args.mode == "mps-hybrid":
            mps_memory_rows.append(
                (
                    torch.mps.current_allocated_memory(),
                    torch.mps.driver_allocated_memory(),
                )
            )

    text = NemotronASRForRNNT.post_process_output(
        tokenizer.decode(last_tokens[0], skip_special_tokens=True)
    )
    infer_ms = _stats([sum(row) * 1000.0 for row in rows])
    mps_memory = None
    if args.mode == "mps-hybrid":
        mps_memory = {
            "current_allocated_bytes": torch.mps.current_allocated_memory(),
            "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            "max_observed_allocated_bytes": max(row[0] for row in mps_memory_rows),
            "max_observed_driver_bytes": max(row[1] for row in mps_memory_rows),
        }
    print(
        json.dumps(
            {
                "mode": args.mode,
                "encoder_device": "mps" if args.mode == "mps-hybrid" else args.mode,
                "decoder_device": "cpu" if args.mode == "mps-hybrid" else args.mode,
                "mps_encoder_dtype": str(model.model._mps_encoder_dtype),
                "batch_size": args.batch_size,
                "seconds": args.seconds,
                "load_s": load_s,
                "feature_ms": _stats([row[0] * 1000.0 for row in rows]),
                "encoder_ms": _stats([row[1] * 1000.0 for row in rows]),
                "decode_ms": _stats([row[2] * 1000.0 for row in rows]),
                "infer_ms": infer_ms,
                "realtime_factor_x": args.seconds / (infer_ms["median"] / 1000.0),
                "outputs_identical": len(set(output_hashes)) == 1,
                "output_sha256": output_hashes[-1],
                "mps_memory": mps_memory,
                "tokens_first": len(last_tokens[0]),
                "tokens_total": sum(len(tokens) for tokens in last_tokens),
                "text_first": text,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
