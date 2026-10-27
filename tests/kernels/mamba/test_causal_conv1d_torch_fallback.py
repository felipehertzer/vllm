# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


def _causal_conv_ref(
    seq_x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_state: torch.Tensor,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_len = weight.shape[1] - 1
    conv_input = torch.cat([initial_state, seq_x], dim=-1)
    out = F.conv1d(
        conv_input.to(weight.dtype),
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=weight.shape[0],
    )[..., -seq_x.shape[-1] :]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    return out.to(seq_x.dtype), conv_input[..., -state_len:].squeeze(0)


def test_causal_conv1d_torch_fallback_prefill_updates_state():
    torch.manual_seed(0)
    dim = 6
    width = 4
    seqlens = [3, 2, 4]
    x = torch.randn(dim, sum(seqlens))
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_states = torch.randn(5, dim, width - 1)
    conv_states_ref = conv_states.clone()
    query_start_loc = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    cache_indices = torch.tensor([1, NULL_BLOCK_ID, 3], dtype=torch.int32)
    has_initial_state = torch.tensor([True, True, False])

    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    expected = torch.zeros_like(x)
    for seq_idx, slot in enumerate(cache_indices.tolist()):
        start = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        if slot == NULL_BLOCK_ID:
            continue
        seq_x = x[:, start:end].unsqueeze(0)
        initial = (
            conv_states_ref[slot].unsqueeze(0)
            if bool(has_initial_state[seq_idx].item())
            else torch.zeros_like(conv_states_ref[slot]).unsqueeze(0)
        )
        seq_out, final_state = _causal_conv_ref(
            seq_x, weight, bias, initial, activation="silu"
        )
        expected[:, start:end] = seq_out.squeeze(0)
        conv_states_ref[slot, :, : width - 1] = final_state

    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(conv_states, conv_states_ref, rtol=1e-5, atol=1e-5)


def test_causal_conv1d_torch_fallback_decode_updates_indexed_state():
    torch.manual_seed(1)
    batch = 3
    dim = 5
    width = 3
    x = torch.randn(batch, dim)
    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    conv_state = torch.randn(5, dim, width - 1)
    conv_state_ref = conv_state.clone()
    conv_state_indices = torch.tensor([2, NULL_BLOCK_ID, 4], dtype=torch.int32)

    out = causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
    )

    expected = torch.zeros_like(x)
    for seq_idx, slot in enumerate(conv_state_indices.tolist()):
        if slot == NULL_BLOCK_ID:
            continue
        seq_x = x[seq_idx : seq_idx + 1].unsqueeze(-1)
        initial = conv_state_ref[slot].unsqueeze(0)
        seq_out, final_state = _causal_conv_ref(
            seq_x, weight, bias, initial, activation="silu"
        )
        expected[seq_idx] = seq_out.squeeze(0).squeeze(-1)
        conv_state_ref[slot, :, : width - 1] = final_state

    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-5)
    assert torch.allclose(conv_state, conv_state_ref, rtol=1e-5, atol=1e-5)
