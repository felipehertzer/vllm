# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)


def test_fused_recurrent_packed_decode_torch_fallback_matches_reference():
    torch.manual_seed(0)
    B = 4
    H = 2
    HV = 4
    K = 8
    V = 6
    qkv_dim = 2 * (H * K) + (HV * V)
    scale = K**-0.5

    mixed_qkv = torch.randn((B, qkv_dim), dtype=torch.float32)
    a = torch.randn((B, HV), dtype=torch.float32)
    b = torch.randn((B, HV), dtype=torch.float32)
    A_log = torch.randn((HV,), dtype=torch.float32)
    dt_bias = torch.randn((HV,), dtype=torch.float32)
    ssm_state_indices = torch.tensor([1, 0, 3, -1], dtype=torch.int32)
    state = torch.randn((5, HV, V, K), dtype=torch.float32)
    state_ref = state.clone()
    out = torch.empty((B, 1, HV, V), dtype=torch.float32)
    out_ref = torch.zeros_like(out)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = F.normalize(q.reshape(B, H, K), p=2, dim=-1, eps=1e-6)
    k = F.normalize(k.reshape(B, H, K), p=2, dim=-1, eps=1e-6)
    v = v.reshape(B, HV, V)
    head_indices = (torch.arange(HV) // (HV // H)).clamp(max=H - 1)
    q = q.index_select(1, head_indices) * scale
    k = k.index_select(1, head_indices)
    g = -torch.exp(A_log) * F.softplus(a + dt_bias, beta=1.0, threshold=20.0)
    beta = torch.sigmoid(b)

    for seq_idx, state_idx in enumerate(ssm_state_indices.tolist()):
        if state_idx <= 0:
            continue
        seq_state = state_ref[state_idx]
        seq_state.mul_(torch.exp(g[seq_idx]).view(HV, 1, 1))
        pred_v = torch.sum(seq_state * k[seq_idx, :, None, :], dim=-1)
        delta_v = (v[seq_idx] - pred_v) * beta[seq_idx].view(HV, 1)
        seq_state.add_(delta_v[:, :, None] * k[seq_idx, :, None, :])
        out_ref[seq_idx, 0] = torch.sum(seq_state * q[seq_idx, :, None, :], dim=-1)
        state_ref[state_idx] = seq_state

    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(state, state_ref)
