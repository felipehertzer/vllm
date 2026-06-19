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
