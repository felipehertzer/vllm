# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.cache import CacheDType
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_MPS_ATTN_PROFILE_TOTALS: dict[str, float] = {}
_MPS_ATTN_PROFILE_COUNTS: dict[str, int] = {}
_BATCHED_DECODE_MIN_REQUESTS = 16


def _profile_active(device: torch.device) -> bool:
    return device.type == "mps" and envs.VLLM_MPS_PROFILE


def _profile_start(device: torch.device) -> float:
    torch.mps.synchronize()
    return time.perf_counter()


def _profile_mark(name: str, started: float, device: torch.device) -> float:
    torch.mps.synchronize()
    now = time.perf_counter()
    _MPS_ATTN_PROFILE_TOTALS[name] = (
        _MPS_ATTN_PROFILE_TOTALS.get(name, 0.0) + now - started
    )
    _MPS_ATTN_PROFILE_COUNTS[name] = _MPS_ATTN_PROFILE_COUNTS.get(name, 0) + 1
    return now


def _profile_log(prefix: str) -> None:
    if not _MPS_ATTN_PROFILE_TOTALS:
        return
    parts = [
        f"{name}={total * 1000:.2f}ms/{_MPS_ATTN_PROFILE_COUNTS[name]}"
        for name, total in sorted(_MPS_ATTN_PROFILE_TOTALS.items())
    ]
    logger.info("MPS profile %s: %s", prefix, ", ".join(parts))
    _MPS_ATTN_PROFILE_TOTALS.clear()
    _MPS_ATTN_PROFILE_COUNTS.clear()


class MPSAttentionBackend(AttentionBackend):
    """PyTorch-native paged attention backend for Apple Metal/MPS."""

    forward_includes_kv_cache_update = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16, 32, 64, 128]

    @staticmethod
    def get_name() -> str:
        return "MPS_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MPSAttentionBackendImpl"]:
        return MPSAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["MPSAttentionMetadataBuilder"]:
        return MPSAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return num_blocks, block_size, num_kv_heads, 2, head_size

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MPSAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool | torch.Tensor = True
    query_start_loc_list: list[int] | None = None
    seq_lens_list: list[int] | None = None
    block_table_cpu: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.query_start_loc_list is None:
            self.query_start_loc_list = self.query_start_loc_cpu.tolist()
        if self.seq_lens_list is None:
            self.seq_lens_list = self.seq_lens_cpu.tolist()
        if self.block_table_cpu is None:
            self.block_table_cpu = self.block_table.detach().cpu()


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MPSAttentionMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_cpu = common_attn_metadata.block_table_tensor.detach().cpu()
        return MPSAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            query_start_loc_list=query_start_loc_cpu.tolist(),
            seq_lens=common_attn_metadata.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_list=seq_lens_cpu.tolist(),
            block_table=common_attn_metadata.block_table_tensor,
            block_table_cpu=block_table_cpu,
            slot_mapping=common_attn_metadata.slot_mapping,
            causal=common_attn_metadata.causal,
        )


class MPSAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("MPS_ATTN currently supports decoder attention.")
        if alibi_slopes is not None:
            raise NotImplementedError("MPS_ATTN does not support ALiBi yet.")
        if sinks is not None:
            raise NotImplementedError("MPS_ATTN does not support attention sinks yet.")
        if logits_soft_cap not in (None, 0):
            raise NotImplementedError("MPS_ATTN does not support logits soft cap yet.")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads}) for MPS_ATTN."
            )

        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.sliding_window = -1 if sliding_window is None else sliding_window
        self.kv_cache_dtype = kv_cache_dtype

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MPSAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("MPS_ATTN does not support output quantization.")
        if attn_metadata is None:
            return output

        profile = _profile_active(query.device)
        if profile:
            started = _profile_start(query.device)
        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        if profile:
            started = _profile_mark("setup", started, query.device)

        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            self._write_kv_cache(
                kv_cache,
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                attn_metadata,
            )
        if profile:
            started = _profile_mark("write_kv", started, query.device)

        flat_cache = kv_cache.reshape(-1, self.num_kv_heads, 2, self.head_size)
        query_start_loc = cast(list[int], attn_metadata.query_start_loc_list)
        seq_lens = cast(list[int], attn_metadata.seq_lens_list)
        block_size = kv_cache.shape[1]
        if profile:
            started = _profile_mark("reshape_cache", started, query.device)

        if self._can_use_batched_single_token_decode(
            attn_metadata,
            query_start_loc,
            seq_lens,
        ):
            self._batched_single_token_decode(
                query=query,
                flat_cache=flat_cache,
                block_size=block_size,
                attn_metadata=attn_metadata,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                output=output,
                profile=profile,
            )
            if profile:
                _profile_log(
                    f"mps_attn batched_decode reqs={len(seq_lens)} "
                    f"seq={max(seq_lens) if seq_lens else 0}"
                )
            return output

        for req_idx, seq_len in enumerate(seq_lens):
            q_start = query_start_loc[req_idx]
            q_end = query_start_loc[req_idx + 1]
            q_len = q_end - q_start
            if q_len <= 0 or seq_len <= 0:
                continue

            if self._can_use_current_kv_for_prefill(
                key, value, q_start, q_end, q_len, seq_len
            ):
                key_seq = key[q_start:q_end]
                value_seq = value[q_start:q_end]
            else:
                slots = self._get_request_slots(
                    attn_metadata,
                    attn_metadata.block_table,
                    req_idx,
                    seq_len,
                    block_size,
                    query.device,
                )
                if isinstance(slots, slice):
                    kv_seq = flat_cache[slots]
                else:
                    kv_seq = flat_cache.index_select(0, slots)
                key_seq = kv_seq[:, :, 0, :]
                value_seq = kv_seq[:, :, 1, :]
            if profile:
                started = _profile_mark("fetch_kv", started, query.device)
            use_gqa = self.num_queries_per_kv != 1 and self._can_use_native_gqa(
                q_len, seq_len
            )
            if self.num_queries_per_kv != 1 and not use_gqa:
                key_seq = key_seq.repeat_interleave(self.num_queries_per_kv, dim=1)
                value_seq = value_seq.repeat_interleave(self.num_queries_per_kv, dim=1)
            if profile:
                started = _profile_mark("repeat_gqa", started, query.device)

            q = query[q_start:q_end].transpose(0, 1).unsqueeze(0)
            k = key_seq.transpose(0, 1).unsqueeze(0)
            v = value_seq.transpose(0, 1).unsqueeze(0)
            attn_mask = None
            is_causal = self._can_use_sdpa_causal(q_len, seq_len, attn_metadata.causal)
            if (
                not self._can_skip_mask_for_single_token_decode(
                    q_len, attn_metadata.causal
                )
                and not is_causal
            ):
                attn_mask = self._build_mask(
                    q_len=q_len,
                    seq_len=seq_len,
                    causal=attn_metadata.causal,
                    device=query.device,
                )
            if profile:
                started = _profile_mark("prepare_sdpa", started, query.device)
            attn_output = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=is_causal,
                use_gqa=use_gqa,
            )
            if profile:
                started = _profile_mark("sdpa", started, query.device)
            attn_output = attn_output.squeeze(0).transpose(0, 1)
            if output.dim() == 3:
                output[q_start:q_end].copy_(attn_output)
            else:
                output[q_start:q_end].copy_(attn_output.reshape(q_len, -1))
            if profile:
                started = _profile_mark("copy_output", started, query.device)

        if profile:
            _profile_log(
                f"mps_attn q={attn_metadata.max_query_len} "
                f"seq={max(seq_lens) if seq_lens else 0}"
            )
        return output

    def _can_use_current_kv_for_prefill(
        self,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        q_start: int,
        q_end: int,
        q_len: int,
        seq_len: int,
    ) -> bool:
        return (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
            and q_len == seq_len
            and q_start >= 0
            and q_end <= key.shape[0]
        )

    def _can_use_batched_single_token_decode(
        self,
        attn_metadata: MPSAttentionMetadata,
        query_start_loc: list[int],
        seq_lens: list[int],
    ) -> bool:
        if not seq_lens:
            return False
        if isinstance(attn_metadata.causal, torch.Tensor):
            return False
        if self.sliding_window != -1:
            return False
        if len(seq_lens) < _BATCHED_DECODE_MIN_REQUESTS:
            return False
        if len(set(seq_lens)) != 1:
            return False
        for req_idx, seq_len in enumerate(seq_lens):
            q_len = query_start_loc[req_idx + 1] - query_start_loc[req_idx]
            if q_len != 1 or seq_len <= 0:
                return False
        return True

    def _batched_single_token_decode(
        self,
        query: torch.Tensor,
        flat_cache: torch.Tensor,
        block_size: int,
        attn_metadata: MPSAttentionMetadata,
        query_start_loc: list[int],
        seq_lens: list[int],
        output: torch.Tensor,
        profile: bool,
    ) -> None:
        started = _profile_start(query.device) if profile else 0.0
        groups: dict[int, list[int]] = {}
        for req_idx, seq_len in enumerate(seq_lens):
            groups.setdefault(seq_len, []).append(req_idx)

        if profile:
            started = _profile_mark("batch_group", started, query.device)

        use_gqa = self.num_queries_per_kv != 1
        for seq_len, req_indices in groups.items():
            kv_seqs = []
            q_indices = []
            for req_idx in req_indices:
                q_indices.append(query_start_loc[req_idx])
                slots = self._get_request_slots(
                    attn_metadata,
                    attn_metadata.block_table,
                    req_idx,
                    seq_len,
                    block_size,
                    query.device,
                )
                if isinstance(slots, slice):
                    kv_seqs.append(flat_cache[slots])
                else:
                    kv_seqs.append(flat_cache.index_select(0, slots))
            if profile:
                started = _profile_mark("batch_fetch_kv", started, query.device)

            kv_batch = torch.stack(kv_seqs, dim=0)
            key_seq = kv_batch[:, :, :, 0, :]
            value_seq = kv_batch[:, :, :, 1, :]
            q = query[q_indices].unsqueeze(2)
            k = key_seq.transpose(1, 2)
            v = value_seq.transpose(1, 2)
            if self.num_queries_per_kv != 1 and not use_gqa:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            if profile:
                started = _profile_mark("batch_prepare_sdpa", started, query.device)

            attn_output = self._scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=False,
                use_gqa=use_gqa,
            )
            if profile:
                started = _profile_mark("batch_sdpa", started, query.device)

            attn_output = attn_output[:, :, 0, :]
            if output.dim() == 3:
                output[q_indices] = attn_output
            else:
                output[q_indices] = attn_output.reshape(len(req_indices), -1)
            if profile:
                started = _profile_mark("batch_copy_output", started, query.device)

    def _write_kv_cache(
        self,
        kv_cache: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MPSAttentionMetadata,
    ) -> None:
        flat_cache = kv_cache.reshape(-1, self.num_kv_heads, 2, self.head_size)
        block_size = kv_cache.shape[1]
        write_slices = self._get_contiguous_kv_write_slices(
            attn_metadata,
            block_size,
            key.shape[0],
        )
        if write_slices is not None:
            for q_start, q_end, cache_slice in write_slices:
                flat_cache[cache_slice, :, 0, :] = key[q_start:q_end]
                flat_cache[cache_slice, :, 1, :] = value[q_start:q_end]
            return

        slot_mapping = attn_metadata.slot_mapping[: key.shape[0]]
        slot_mapping = slot_mapping.to(dtype=torch.long, device=key.device)
        if slot_mapping.numel() == 0:
            return
        if slot_mapping.numel() == key.shape[0]:
            flat_cache[slot_mapping, :, 0, :] = key
            flat_cache[slot_mapping, :, 1, :] = value
            return

        valid = slot_mapping >= 0
        if not valid.any():
            return
        slots = slot_mapping[valid]
        flat_cache[slots, :, 0, :] = key[valid]
        flat_cache[slots, :, 1, :] = value[valid]

    def _get_contiguous_kv_write_slices(
        self,
        attn_metadata: MPSAttentionMetadata,
        block_size: int,
        num_tokens: int,
    ) -> list[tuple[int, int, slice]] | None:
        """Return per-request KV cache slices when physical blocks are contiguous."""
        cache = getattr(attn_metadata, "_mps_kv_write_slices_cache", None)
        if cache is None:
            cache = {}
            attn_metadata._mps_kv_write_slices_cache = cache
        cache_key = (block_size, num_tokens)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        if num_tokens == 0:
            cache[cache_key] = []
            return []

        query_start_loc = cast(list[int], attn_metadata.query_start_loc_list)
        seq_lens = cast(list[int], attn_metadata.seq_lens_list)
        block_table_cpu = cast(torch.Tensor, attn_metadata.block_table_cpu)
        write_slices: list[tuple[int, int, slice]] = []

        for req_idx, seq_len in enumerate(seq_lens):
            q_start = query_start_loc[req_idx]
            q_end = query_start_loc[req_idx + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            if q_end > num_tokens or q_len > seq_len:
                cache[cache_key] = None
                return None

            num_blocks = ceil(seq_len / block_size)
            if num_blocks <= 0:
                cache[cache_key] = None
                return None
            blocks = block_table_cpu[req_idx, :num_blocks]
            if blocks.numel() > 1 and not bool(torch.all(blocks.diff() == 1).item()):
                cache[cache_key] = None
                return None

            first_block = int(blocks[0].item())
            cache_start = first_block * block_size + (seq_len - q_len)
            write_slices.append(
                (q_start, q_end, slice(cache_start, cache_start + q_len))
            )

        cache[cache_key] = write_slices
        return write_slices

    def _get_request_slots(
        self,
        attn_metadata: MPSAttentionMetadata,
        block_table: torch.Tensor,
        req_idx: int,
        seq_len: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor | slice:
        cache = getattr(attn_metadata, "_mps_request_slots_cache", None)
        if cache is None:
            cache = {}
            attn_metadata._mps_request_slots_cache = cache
        cache_key = (req_idx, seq_len, block_size)
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        num_blocks = (seq_len + block_size - 1) // block_size
        block_table_cpu = cast(torch.Tensor, attn_metadata.block_table_cpu)
        blocks_cpu = block_table_cpu[req_idx, :num_blocks].to(dtype=torch.long)
        if blocks_cpu.numel() > 0 and bool(torch.all(blocks_cpu.diff() == 1).item()):
            start = int(blocks_cpu[0].item()) * block_size
            slots = slice(start, start + seq_len)
            cache[cache_key] = slots
            return slots

        blocks = block_table[req_idx, :num_blocks].to(dtype=torch.long, device=device)
        offsets = torch.arange(block_size, dtype=torch.long, device=device)
        slots = (blocks[:, None] * block_size + offsets[None, :]).reshape(-1)[:seq_len]
        cache[cache_key] = slots
        return slots

    def _can_skip_mask_for_single_token_decode(
        self, q_len: int, causal: bool | torch.Tensor
    ) -> bool:
        return q_len == 1 and causal is True and self.sliding_window == -1

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        use_gqa: bool,
    ) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=use_gqa,
        )

    def _can_use_native_gqa(self, q_len: int, seq_len: int) -> bool:
        return self.sliding_window == -1

    def _can_use_sdpa_causal(
        self,
        q_len: int,
        seq_len: int,
        causal: bool | torch.Tensor,
    ) -> bool:
        return (
            q_len == seq_len
            and q_len > 1
            and causal is True
            and self.sliding_window == -1
        )

    def _build_mask(
        self,
        q_len: int,
        seq_len: int,
        causal: bool | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor | None:
        if isinstance(causal, torch.Tensor):
            raise NotImplementedError("MPS_ATTN does not support dynamic causal masks.")
        if not causal and self.sliding_window == -1:
            return None

        key_pos = torch.arange(seq_len, device=device)
        query_pos = torch.arange(seq_len - q_len, seq_len, device=device)
        mask = torch.ones((q_len, seq_len), dtype=torch.bool, device=device)
        if causal:
            mask &= key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
        if self.sliding_window != -1:
            mask &= key_pos.unsqueeze(0) > (
                query_pos.unsqueeze(1) - self.sliding_window
            )
        return mask
