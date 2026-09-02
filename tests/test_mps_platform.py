# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.config.compilation import CompilationMode
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import cpu_platform_plugin, mps_platform_plugin
from vllm.platforms.mps import MpsPlatform
from vllm.usage.usage_lib import UsageContext
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_mps_platform_plugin_requires_target_device(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    with patch.object(envs, "VLLM_TARGET_DEVICE", "cpu", create=True):
        assert mps_platform_plugin() is None


def test_mps_platform_plugin_activates_on_darwin_mps(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "mps")
    with (
        patch.object(envs, "VLLM_TARGET_DEVICE", "mps", create=True),
        patch("sys.platform", "darwin"),
        patch.object(torch.backends.mps, "is_built", return_value=True),
        patch.object(torch.backends.mps, "is_available", return_value=True),
    ):
        assert mps_platform_plugin() == "vllm.platforms.mps.MpsPlatform"
        assert cpu_platform_plugin() is None


@pytest.mark.parametrize(
    "quantization",
    ["auto_gptq", "compressed-tensors", "gptq", "gptq_marlin"],
)
def test_mps_platform_accepts_gptq_quantization(quantization: str):
    MpsPlatform.verify_quantization(quantization)


def test_mps_platform_rejects_cuda_only_quantization():
    with pytest.raises(ValueError, match="not supported on MPS"):
        MpsPlatform.verify_quantization("awq")


def test_mps_platform_reports_current_device():
    assert MpsPlatform.current_device() == torch.device("mps")


def test_mps_platform_supports_hybrid_kv_cache():
    assert MpsPlatform.support_hybrid_kv_cache() is True


def test_mps_platform_disables_cuda_only_runtime_features():
    config = SimpleNamespace(
        model_config=SimpleNamespace(disable_cascade_attn=False),
        attention_config=SimpleNamespace(backend=None),
        scheduler_config=SimpleNamespace(async_scheduling=True),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL,
            cudagraph_capture_sizes=[1],
            cudagraph_mm_encoder=True,
            mode=CompilationMode.VLLM_COMPILE,
        ),
        parallel_config=SimpleNamespace(worker_cls="auto", enable_dbo=True),
    )

    MpsPlatform.check_and_update_config(config)

    assert config.model_config.disable_cascade_attn is True
    assert config.attention_config.backend == AttentionBackendEnum.MPS_ATTN
    assert config.scheduler_config.async_scheduling is False
    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.compilation_config.cudagraph_capture_sizes == []
    assert config.compilation_config.cudagraph_mm_encoder is False
    assert config.compilation_config.mode == CompilationMode.NONE
    assert config.parallel_config.worker_cls == "vllm.v1.worker.mps_worker.MPSWorker"
    assert config.parallel_config.enable_dbo is False


def test_mps_batch_defaults_are_small_enough_for_metal_kv_cache():
    with patch("vllm.engine.arg_utils.current_platform") as platform:
        platform.get_device_total_memory.return_value = 48 * 1024**3
        platform.get_device_name.return_value = "Apple Metal GPU"
        platform.is_tpu.return_value = False
        platform.is_cpu.return_value = False
        platform.is_mps.return_value = True
        platform.is_rocm.return_value = False

        default_tokens, default_seqs = EngineArgs.get_batch_defaults(world_size=1)

    assert default_tokens[UsageContext.LLM_CLASS] == 4096
    assert default_tokens[UsageContext.OPENAI_API_SERVER] == 2048
    assert default_seqs[UsageContext.LLM_CLASS] == 4
    assert default_seqs[UsageContext.OPENAI_API_SERVER] == 4


def test_mps_cleanup_skips_unsupported_host_cache_flush():
    with (
        patch("vllm.platforms.current_platform") as platform,
        patch.object(torch.accelerator, "empty_cache") as empty_cache,
        patch.object(torch.accelerator, "empty_host_cache") as empty_host_cache,
    ):
        platform.is_cpu.return_value = False
        platform.is_mps.return_value = True
        platform.is_rocm.return_value = False

        cleanup_dist_env_and_memory()

    empty_cache.assert_called_once_with()
    empty_host_cache.assert_not_called()
