# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import platform
import sys
from datetime import timedelta
from typing import TYPE_CHECKING

import psutil
import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from torch.distributed import PrefixStore, ProcessGroup

    from vllm.config import VllmConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)


class MpsPlatform(Platform):
    _enum = PlatformEnum.MPS
    device_name = "mps"
    device_type = "mps"
    dispatch_key = "MPS"
    dist_backend = "gloo"
    device_control_env_var = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if selected_backend and selected_backend != AttentionBackendEnum.MPS_ATTN:
            logger.info("Cannot use %s backend on MPS.", selected_backend)
        if attn_selector_config.use_mla:
            raise NotImplementedError("MLA is not supported on MPS.")
        if attn_selector_config.use_sparse:
            raise NotImplementedError("Sparse Attention is not supported on MPS.")
        if attn_selector_config.use_mm_prefix:
            raise NotImplementedError("MM prefix attention is not supported on MPS.")
        if attn_selector_config.use_per_head_quant_scales:
            raise NotImplementedError("Per-head quant scales are not supported on MPS.")
        return AttentionBackendEnum.MPS_ATTN.get_path()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        machine = platform.machine() or "unknown"
        return f"Apple Metal GPU ({machine})"

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"mps:{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        if hasattr(torch.mps, "recommended_max_memory"):
            return int(torch.mps.recommended_max_memory())
        return psutil.virtual_memory().total

    @classmethod
    def mem_get_info(cls, device: torch.types.Device | None = None) -> tuple[int, int]:
        total = cls.get_device_total_memory()
        used = int(torch.mps.driver_allocated_memory())
        free = max(total - used, 0)
        return free, total

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        if device.type != "mps":
            raise ValueError(f"MPS platform cannot set non-MPS device {device}.")

    @classmethod
    def current_device(cls) -> torch.device:
        return torch.device("mps")

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.mps.manual_seed(seed)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config import CUDAGraphMode
        from vllm.config.compilation import CompilationMode

        model_config = vllm_config.model_config
        if model_config is not None:
            model_config.disable_cascade_attn = True

        attention_config = vllm_config.attention_config
        if attention_config.backend is None:
            attention_config.backend = AttentionBackendEnum.MPS_ATTN

        scheduler_config = vllm_config.scheduler_config
        scheduler_config.async_scheduling = False

        compilation_config = vllm_config.compilation_config
        compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        compilation_config.cudagraph_capture_sizes = []
        compilation_config.cudagraph_mm_encoder = False
        if compilation_config.mode == CompilationMode.VLLM_COMPILE:
            logger.warning(
                "vLLM compile/CUDA graph mode is not supported on MPS yet; "
                "using eager PyTorch execution."
            )
            compilation_config.mode = CompilationMode.NONE

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.mps_worker.MPSWorker"
        if parallel_config.enable_dbo:
            logger.warning("Dual-Batch Overlap is not supported on MPS, disabled.")
            parallel_config.enable_dbo = False

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        if quant in {"auto_gptq", "compressed-tensors", "gptq", "gptq_marlin"}:
            return
        raise ValueError(
            f"{quant} quantization is not supported on MPS. Use an unquantized "
            "PyTorch/safetensors checkpoint for Apple Metal execution."
        )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        return float(torch.mps.driver_allocated_memory())

    @classmethod
    def device_count(cls) -> int:
        return 1 if torch.backends.mps.is_available() else 0

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return os.cpu_count() or 1

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype not in cls().supported_dtypes:
            supported = ", ".join(str(dtype) for dtype in cls().supported_dtypes)
            raise ValueError(f"MPS does not support {dtype}; use one of {supported}.")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def import_kernels(cls) -> None:
        # MPS uses PyTorch-native kernels in this fork path.
        return None

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: "PrefixStore",
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> "ProcessGroup":
        raise NotImplementedError("MPS uses the regular gloo init path.")

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return False

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def is_mps_available(cls) -> bool:
        return sys.platform.startswith("darwin") and torch.backends.mps.is_available()
