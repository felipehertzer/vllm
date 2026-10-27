# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.configs.nemotron_asr import NemotronASRConfig


def test_get_config_registers_nemotron_asr(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nemotron_asr"}),
        encoding="utf-8",
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, NemotronASRConfig)
    assert config.model_type == "nemotron_asr"


def test_only_nemotron_asr_counts_the_final_stft_frame():
    import torch
    from transformers import ParakeetEncoderConfig

    from vllm.model_executor.models.parakeet import ParakeetExtractor

    def valid_frames(encoder_config) -> int:
        extractor = ParakeetExtractor(encoder_config)
        features = torch.ones(1, 12, encoder_config.num_mel_bins)
        _, mask = extractor._normalize_mel_features(features, torch.tensor([1600]))
        return int(mask.sum())

    # 1600 samples at hop 160: Parakeet masks the last centered frame,
    # Nemotron ASR keeps it.
    assert valid_frames(ParakeetEncoderConfig(sampling_rate=16000)) == 10
    assert valid_frames(NemotronASRConfig().encoder_config) == 11
