#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert NVIDIA Nemotron 3.5 ASR NeMo checkpoints to vLLM/HF layout."""

from vllm.transformers_utils.nemotron_asr_conversion import (
    build_nemotron_asr_config,
    convert_nemo_to_hf,
    main,
    normalize_nemotron_asr_weight_name,
    normalize_state_dict,
)

__all__ = [
    "build_nemotron_asr_config",
    "convert_nemo_to_hf",
    "main",
    "normalize_nemotron_asr_weight_name",
    "normalize_state_dict",
]


if __name__ == "__main__":
    main()
