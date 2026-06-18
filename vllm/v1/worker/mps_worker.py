# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
from contextlib import nullcontext
from math import ceil

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.mps_model_runner import MPSModelRunner
from vllm.v1.worker.worker_base import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger(__name__)


class MPSWorker(Worker):
    """Worker for Apple Metal/MPS execution."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        assert self.device_config.device_type == "mps"
        assert current_platform.is_mps()
        self.requested_memory = 0
        self.available_kv_cache_memory_bytes = 0

    def _get_target_kv_cache_memory_bytes(self) -> int | None:
        kv_cache_specs = self.model_runner.get_kv_cache_spec()
        if not kv_cache_specs:
            return None

        block_sizes = {spec.block_size for spec in kv_cache_specs.values()}
        if len(block_sizes) != 1:
            logger.debug(
                "Skipping MPS KV cache cap for mixed block sizes: %s", block_sizes
            )
            return None

        block_size = block_sizes.pop()
        bytes_per_block = sum(spec.page_size_bytes for spec in kv_cache_specs.values())
        target_tokens = max(
            self.model_config.max_model_len * self.scheduler_config.max_num_seqs,
            self.scheduler_config.max_num_batched_tokens,
        )
        target_blocks = ceil(target_tokens / block_size)
        reserve_blocks = max(16, ceil(target_blocks * 0.25))
        return (target_blocks + reserve_blocks) * bytes_per_block

    def _maybe_get_memory_pool_context(self, tag: str):
        return nullcontext()

    def _set_mps_allocator_memory_fraction(self) -> None:
        set_memory_fraction = getattr(
            torch.mps, "set_per_process_memory_fraction", None
        )
        if not callable(set_memory_fraction):
            return

        fraction = float(self.cache_config.gpu_memory_utilization)
        set_memory_fraction(fraction)
        logger.info(
            "Set MPS per-process memory fraction to %.3f "
            "(based on gpu_memory_utilization).",
            fraction,
        )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on MPS, ignore it.")

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on MPS, ignore it.")

    def init_device(self):
        self.device = torch.device("mps")
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        set_random_seed(self.model_config.seed)
        gc.collect()
        torch.mps.empty_cache()
        self._set_mps_allocator_memory_fraction()

        free_memory, total_memory = current_platform.mem_get_info(self.device)
        self.requested_memory = int(
            total_memory * self.cache_config.gpu_memory_utilization
        )
        if free_memory < self.requested_memory:
            raise ValueError(
                f"Free memory on MPS ({format_gib(free_memory)}/"
                f"{format_gib(total_memory)} GiB) is less than requested "
                f"GPU memory utilization ({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB)."
            )

        num_ubatches = 2 if self.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)
        self.model_runner = MPSModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            report_usage_stats(self.vllm_config)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            logger.info(
                "Explicitly reserving %s GiB for KV cache on MPS.",
                format_gib(kv_cache_memory_bytes),
            )
            return kv_cache_memory_bytes

        gc.collect()
        torch.mps.empty_cache()
        before = int(torch.mps.driver_allocated_memory())
        self.model_runner.profile_run()
        torch.mps.synchronize()
        gc.collect()
        torch.mps.empty_cache()
        after = int(torch.mps.driver_allocated_memory())
        profiled_usage = max(before, after)

        available = self.requested_memory - profiled_usage
        if available <= 0:
            free_memory, total_memory = current_platform.mem_get_info(self.device)
            raise ValueError(
                "MPS profiling left no space for KV cache. "
                f"Profiled usage: {format_gib(profiled_usage)} GiB, requested: "
                f"{format_gib(self.requested_memory)} GiB, free/total: "
                f"{format_gib(free_memory)}/{format_gib(total_memory)} GiB. "
                "Increase --gpu-memory-utilization or reduce model/context size."
            )

        target_kv_cache_memory = self._get_target_kv_cache_memory_bytes()
        if target_kv_cache_memory is not None and target_kv_cache_memory < available:
            logger.info(
                "Capping MPS KV cache memory from %s GiB to %s GiB "
                "for max_model_len=%d, max_num_seqs=%d, max_num_batched_tokens=%d.",
                format_gib(available),
                format_gib(target_kv_cache_memory),
                self.model_config.max_model_len,
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )
            available = target_kv_cache_memory

        self.available_kv_cache_memory_bytes = int(available)
        logger.info(
            "Available MPS KV cache memory: %s GiB",
            format_gib(self.available_kv_cache_memory_bytes),
        )
        return self.available_kv_cache_memory_bytes

    def compile_or_warm_up_model(self) -> CompilationTimes:
        self.model_runner.profile_run()
        set_random_seed(self.model_config.seed)
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        raise RuntimeError("Torch profiler is not wired for the MPS worker yet.")

    def determine_num_available_blocks(self) -> tuple[int, int]:
        return super().determine_num_available_blocks()
