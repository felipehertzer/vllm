# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utilities for preparing Nemotron ASR NeMo checkpoints for vLLM."""

from __future__ import annotations

from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

NEMOTRON_ASR_HF_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_ASR_NEMO_FILE = "nemotron-3.5-asr-streaming-0.6b.nemo"


def _converted_dir_for_nemo(nemo_path: Path) -> Path:
    return nemo_path.parent / f"{nemo_path.stem}-vllm-hf"


def _is_converted_model_dir(path: Path) -> bool:
    return (path / "config.json").exists() and (path / "model.safetensors").exists()


def _convert_nemo_if_needed(nemo_path: Path) -> Path:
    output_dir = _converted_dir_for_nemo(nemo_path)
    if _is_converted_model_dir(output_dir):
        return output_dir

    from vllm.transformers_utils.nemotron_asr_conversion import convert_nemo_to_hf

    logger.info(
        "Converting Nemotron ASR NeMo checkpoint %s to %s", nemo_path, output_dir
    )
    convert_nemo_to_hf(nemo_path, output_dir)
    return output_dir


def maybe_prepare_nemotron_asr_model(
    model: str,
    *,
    revision: str | None,
    token: str | None,
) -> str:
    """Return a vLLM/HF model directory for Nemotron ASR when needed."""

    model_path = Path(model)
    if model_path.exists():
        if model_path.is_dir():
            if _is_converted_model_dir(model_path):
                return str(model_path)
            nemo_files = sorted(model_path.glob("*.nemo"))
            if not nemo_files:
                return model
            return str(_convert_nemo_if_needed(nemo_files[0]))
        if model_path.suffix == ".nemo":
            return str(_convert_nemo_if_needed(model_path))
        return model

    if model != NEMOTRON_ASR_HF_ID:
        return model

    from huggingface_hub import hf_hub_download

    nemo_path = Path(
        hf_hub_download(
            repo_id=NEMOTRON_ASR_HF_ID,
            filename=NEMOTRON_ASR_NEMO_FILE,
            revision=revision,
            token=token,
        )
    )
    return str(_convert_nemo_if_needed(nemo_path))


__all__ = [
    "NEMOTRON_ASR_HF_ID",
    "NEMOTRON_ASR_NEMO_FILE",
    "maybe_prepare_nemotron_asr_model",
]
