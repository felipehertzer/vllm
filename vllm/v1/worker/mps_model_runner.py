# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


class MPSModelRunner(GPUModelRunner):
    """GPUModelRunner variant that keeps tensors on Apple Metal/MPS."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        with _torch_mps_cuda_compat():
            super().__init__(vllm_config, device)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False
        self._postprocess_triton()

    def _init_device_properties(self) -> None:
        self.num_sms = 1

    def _postprocess_triton(self) -> None:
        import vllm.v1.worker.block_table

        vllm.v1.worker.block_table._COMPUTE_SLOT_MAPPING_KERNEL.kernel = (
            mps_compute_slot_mapping_kernel
        )

    def _sync_device(self) -> None:
        torch.mps.synchronize()

    def _get_or_create_async_output_copy_stream(self):
        if self.async_output_copy_stream is None:
            self.async_output_copy_stream = _MPSStream()
        return self.async_output_copy_stream

    def _init_kv_zero_meta(self) -> None:
        return None

    def _zero_block_ids(self, block_ids: list[int]) -> None:
        if not block_ids:
            return
        ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        seen: set[int] = set()
        for kv_cache in self.kv_caches:
            if not isinstance(kv_cache, torch.Tensor):
                continue
            ptr = kv_cache.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            kv_cache.index_fill_(0, ids, 0)

    def _copy_draft_token_ids_to_cpu(
        self, scheduler_output: "SchedulerOutput", zeros_only: bool = False
    ) -> None:
        if torch.is_tensor(self._draft_token_ids):
            self.prev_num_spec_tokens = self._draft_token_ids.shape[1]
        if self.use_async_scheduling and not (
            scheduler_output.has_structured_output_requests
            or self.input_batch.sampling_metadata.output_token_ids
        ):
            return
        self._draft_token_req_ids = self.input_batch.req_ids.copy()

        draft_token_ids = self._draft_token_ids
        if not torch.is_tensor(draft_token_ids) or self.draft_token_ids_cpu is None:
            return
        num_reqs, num_spec_tokens = draft_token_ids.shape
        if zeros_only:
            self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens] = 0
        else:
            self.draft_token_ids_cpu[:num_reqs, :num_spec_tokens].copy_(
                draft_token_ids.cpu()
            )

    def _get_draft_token_ids_cpu(self) -> tuple[list[list[int]], list[str]]:
        if isinstance(self._draft_token_ids, list):
            return self._draft_token_ids, self.input_batch.req_ids
        req_ids = self._draft_token_req_ids
        if req_ids is None or self.draft_token_ids_cpu is None:
            return [], []
        assert isinstance(self._draft_token_ids, torch.Tensor)
        num_spec_tokens = self._draft_token_ids.shape[1]
        return (
            self.draft_token_ids_cpu[: len(req_ids), :num_spec_tokens].tolist(),
            req_ids,
        )

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_cpu is None:
            return
        counts = valid_sampled_tokens_count.cpu()
        self.valid_sampled_token_count_cpu[: counts.shape[0]].copy_(counts)
        if self.use_async_spec_decode:
            self.valid_sampled_token_count_gpu = valid_sampled_tokens_count
        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        counts_cpu = self.valid_sampled_token_count_cpu
        if prev_sampled_token_ids is None or counts_cpu is None:
            return []
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        return sampled_token_ids.cpu().tolist()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if not self.num_spec_tokens or not self._draft_token_req_ids:
            return None
        draft_token_ids, req_ids = self._get_draft_token_ids_cpu()
        return DraftTokenIds(req_ids, draft_token_ids)


class _MPSStream:
    def wait_stream(self, stream) -> None:
        return None

    def record_event(self, event=None) -> None:
        return None

    def synchronize(self) -> None:
        torch.mps.synchronize()


class _MPSEvent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def record(self, stream=None) -> None:
        return None

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def wait(self, stream=None) -> None:
        return None

    def query(self) -> bool:
        return True


@contextmanager
def _torch_mps_cuda_compat():
    cuda_stream = torch.cuda.Stream
    cuda_event = torch.cuda.Event
    cuda_current_stream = torch.cuda.current_stream
    cuda_default_stream = torch.cuda.default_stream
    cuda_stream_ctx = torch.cuda.stream
    cuda_mem_get_info = torch.cuda.mem_get_info
    try:
        torch.cuda.Stream = lambda *args, **kwargs: _MPSStream()  # type: ignore[assignment]
        torch.cuda.Event = lambda *args, **kwargs: _MPSEvent()  # type: ignore[assignment]
        torch.cuda.current_stream = lambda *args, **kwargs: _MPSStream()
        torch.cuda.default_stream = lambda *args, **kwargs: _MPSStream()
        torch.cuda.stream = lambda stream: nullcontext()
        torch.cuda.mem_get_info = lambda *args, **kwargs: _mps_mem_get_info()
        yield
    finally:
        torch.cuda.Stream = cuda_stream
        torch.cuda.Event = cuda_event
        torch.cuda.current_stream = cuda_current_stream
        torch.cuda.default_stream = cuda_default_stream
        torch.cuda.stream = cuda_stream_ctx
        torch.cuda.mem_get_info = cuda_mem_get_info


def _mps_mem_get_info() -> tuple[int, int]:
    total = int(torch.mps.recommended_max_memory())
    used = int(torch.mps.driver_allocated_memory())
    return max(total - used, 0), total


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *args, **kwargs) -> Callable:
        return self.func


def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    KV_CACHE_BLOCK_SIZE: int,
    BLOCKS_PER_KV_BLOCK: int,
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int = PAD_SLOT_ID,
    BLOCK_SIZE: int | None = None,
) -> None:
    del block_table_stride, BLOCK_SIZE
    if max_num_tokens > num_tokens:
        slot_mapping[num_tokens:max_num_tokens].fill_(PAD_ID)
    if num_tokens == 0:
        return

    device = positions.device
    token_positions = positions[:num_tokens].to(dtype=torch.long)
    lengths = query_start_loc[1:].to(device=device, dtype=torch.long) - query_start_loc[
        :-1
    ].to(device=device, dtype=torch.long)
    req_indices = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=device, dtype=torch.long), lengths
    )[:num_tokens]

    virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    virtual_block_indices = token_positions // virtual_block_size

    virtual_block_offsets = token_positions - virtual_block_indices * virtual_block_size
    is_local = (
        virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % TOTAL_CP_WORLD_SIZE == TOTAL_CP_RANK
    local_block_offsets = (
        virtual_block_offsets // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
    )

    block_indices = (
        virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size
    )
    block_numbers = block_table[req_indices, block_indices].to(dtype=torch.long)
    slot_offsets = local_block_offsets % block_size
    slot_ids = block_numbers * block_size + slot_offsets
    pad_ids = torch.full_like(slot_ids, PAD_ID)
    slot_mapping[:num_tokens].copy_(torch.where(is_local, slot_ids, pad_ids))


mps_compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)
