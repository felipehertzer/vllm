# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.mps_attn import (
    MPSAttentionBackendImpl,
    MPSAttentionMetadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.mps_model_runner import _compute_slot_mapping_kernel_impl

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)


class _Layer:
    _k_scale_float = 1.0
    _v_scale_float = 1.0


@pytest.mark.parametrize("output_layout", ["flat", "heads"])
def test_mps_attention_matches_torch_sdpa_for_single_prefill(output_layout):
    device = torch.device("mps")
    dtype = torch.float32
    num_heads = 4
    num_kv_heads = 2
    head_size = 8
    block_size = 4
    num_tokens = 5
    scale = head_size**-0.5
    torch.manual_seed(0)

    query = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    key = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_size, device=device, dtype=dtype)
    kv_cache = torch.zeros(
        2,
        block_size,
        num_kv_heads,
        2,
        head_size,
        device=device,
        dtype=dtype,
    )
    metadata = MPSAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], device=device, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, num_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([num_tokens], device=device, dtype=torch.int32),
        seq_lens_cpu=torch.tensor([num_tokens], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]], device=device, dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, device=device, dtype=torch.int64),
        causal=True,
    )
    output_shape = (
        (num_tokens, num_heads, head_size)
        if output_layout == "heads"
        else (num_tokens, num_heads * head_size)
    )
    output = torch.empty(*output_shape, device=device, dtype=dtype)
    impl = MPSAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    result = impl.forward(_Layer(), query, key, value, kv_cache, metadata, output)

    repeated_key = key.repeat_interleave(num_heads // num_kv_heads, dim=1)
    repeated_value = value.repeat_interleave(num_heads // num_kv_heads, dim=1)
    mask = torch.ones((num_tokens, num_tokens), dtype=torch.bool, device=device).tril()
    expected = F.scaled_dot_product_attention(
        query.transpose(0, 1).unsqueeze(0),
        repeated_key.transpose(0, 1).unsqueeze(0),
        repeated_value.transpose(0, 1).unsqueeze(0),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    expected = expected.squeeze(0).transpose(0, 1)
    if output_layout == "flat":
        expected = expected.reshape(num_tokens, -1)

    torch.mps.synchronize()
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
    assert bool((kv_cache.abs().sum() > 0).item())


@pytest.mark.parametrize("output_layout", ["flat", "heads"])
def test_mps_attention_batched_single_token_decode_matches_torch_sdpa(
    output_layout,
):
    device = torch.device("mps")
    dtype = torch.float32
    num_heads = 4
    num_kv_heads = 2
    head_size = 8
    block_size = 8
    seq_lens = [5] * 16
    num_tokens = len(seq_lens)
    scale = head_size**-0.5
    torch.manual_seed(1)

    query = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=dtype)
    full_keys = [
        torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        for seq_len in seq_lens
    ]
    full_values = [
        torch.randn(seq_len, num_kv_heads, head_size, device=device, dtype=dtype)
        for seq_len in seq_lens
    ]
    key = torch.stack([request_key[-1] for request_key in full_keys], dim=0)
    value = torch.stack([request_value[-1] for request_value in full_values], dim=0)
    kv_cache = torch.zeros(
        len(seq_lens),
        block_size,
        num_kv_heads,
        2,
        head_size,
        device=device,
        dtype=dtype,
    )
    for req_idx, seq_len in enumerate(seq_lens):
        kv_cache[req_idx, :seq_len, :, 0, :] = full_keys[req_idx]
        kv_cache[req_idx, :seq_len, :, 1, :] = full_values[req_idx]

    query_start_loc_cpu = torch.arange(num_tokens + 1, dtype=torch.int32)
    slot_mapping = torch.tensor(
        [
            seq_len - 1 + req_idx * block_size
            for req_idx, seq_len in enumerate(seq_lens)
        ],
        device=device,
        dtype=torch.int64,
    )
    metadata = MPSAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=1,
        query_start_loc=query_start_loc_cpu.to(device),
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.arange(
            len(seq_lens), device=device, dtype=torch.int32
        ).unsqueeze(1),
        slot_mapping=slot_mapping,
        causal=True,
    )
    output_shape = (
        (num_tokens, num_heads, head_size)
        if output_layout == "heads"
        else (num_tokens, num_heads * head_size)
    )
    output = torch.empty(*output_shape, device=device, dtype=dtype)
    impl = MPSAttentionBackendImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    assert impl._can_use_batched_single_token_decode(
        metadata,
        metadata.query_start_loc_list,
        metadata.seq_lens_list,
    )
    result = impl.forward(_Layer(), query, key, value, kv_cache, metadata, output)

    expected_rows = []
    for req_idx, seq_len in enumerate(seq_lens):
        repeated_key = full_keys[req_idx].repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        repeated_value = full_values[req_idx].repeat_interleave(
            num_heads // num_kv_heads, dim=1
        )
        expected = F.scaled_dot_product_attention(
            query[req_idx : req_idx + 1].transpose(0, 1).unsqueeze(0),
            repeated_key.transpose(0, 1).unsqueeze(0),
            repeated_value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        expected_rows.append(expected.squeeze(0).transpose(0, 1).squeeze(0))
    expected = torch.stack(expected_rows, dim=0)
    if output_layout == "flat":
        expected = expected.reshape(num_tokens, -1)

    torch.mps.synchronize()
    assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)


def test_mps_slot_mapping_matches_block_table_layout():
    device = torch.device("mps")
    query_start_loc = torch.tensor([0, 3, 7], device=device, dtype=torch.int32)
    positions = torch.tensor([0, 1, 5, 0, 3, 4, 7], device=device, dtype=torch.int64)
    block_table = torch.tensor(
        [
            [2, 3, 0],
            [5, 6, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    slot_mapping = torch.full((10,), -123, device=device, dtype=torch.int64)

    _compute_slot_mapping_kernel_impl(
        num_tokens=positions.shape[0],
        max_num_tokens=slot_mapping.shape[0],
        query_start_loc=query_start_loc,
        positions=positions,
        block_table=block_table,
        block_table_stride=block_table.stride(0),
        block_size=4,
        slot_mapping=slot_mapping,
        KV_CACHE_BLOCK_SIZE=8,
        BLOCKS_PER_KV_BLOCK=2,
        TOTAL_CP_WORLD_SIZE=1,
        TOTAL_CP_RANK=0,
        CP_KV_CACHE_INTERLEAVE_SIZE=1,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=1024,
    )

    expected = torch.tensor(
        [
            8,
            9,
            13,
            20,
            23,
            24,
            27,
            PAD_SLOT_ID,
            PAD_SLOT_ID,
            PAD_SLOT_ID,
        ],
        device=device,
        dtype=torch.int64,
    )
    torch.mps.synchronize()
    assert torch.equal(slot_mapping, expected)
