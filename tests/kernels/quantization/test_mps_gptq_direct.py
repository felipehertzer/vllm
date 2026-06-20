# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig,
    AutoGPTQLinearMethod,
)


def _pack_int4_rows(unpacked: torch.Tensor) -> torch.Tensor:
    packed = torch.zeros(
        unpacked.shape[0] // 8,
        unpacked.shape[1],
        dtype=torch.int32,
    )
    for shift in range(8):
        packed |= unpacked[shift::8] << (shift * 4)
    return packed


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS GPTQ direct kernel requires Apple Metal",
)
def test_mps_gptq_direct_gemv_matches_dense_and_falls_back_for_batches():
    torch.manual_seed(123)
    input_size = 256
    output_size = 8192
    group_size = 128
    groups = input_size // group_size

    unpacked_qweight = torch.randint(
        0,
        16,
        (input_size, output_size),
        dtype=torch.int32,
    )
    qweight = _pack_int4_rows(unpacked_qweight).to("mps")
    scales = (torch.rand(groups, output_size, dtype=torch.float16) * 0.02).to("mps")
    qzeros = torch.full(
        (groups, output_size // 8),
        0x77777777,
        dtype=torch.int32,
        device="mps",
    )
    g_idx = torch.arange(input_size, dtype=torch.int32, device="mps") // group_size

    layer = torch.nn.Module()
    layer.params_dtype = torch.float16
    layer.register_parameter(
        "qweight", torch.nn.Parameter(qweight, requires_grad=False)
    )
    layer.register_parameter("scales", torch.nn.Parameter(scales, requires_grad=False))
    layer.register_parameter("qzeros", torch.nn.Parameter(qzeros, requires_grad=False))
    layer.register_parameter("g_idx", torch.nn.Parameter(g_idx, requires_grad=False))

    config = AutoGPTQConfig(
        weight_bits=4,
        group_size=group_size,
        desc_act=True,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={
            "bits": 4,
            "group_size": group_size,
            "desc_act": True,
            "sym": True,
        },
    )
    method = AutoGPTQLinearMethod(config)
    method._use_mps_dequant = True
    method.process_weights_after_loading(layer)

    single = torch.randn(1, input_size, device="mps", dtype=torch.float16) * 0.1
    direct = method.apply(layer, single)
    dense = torch.nn.functional.linear(single, layer.weight)
    assert method._use_mps_direct_gemv is True
    assert torch.allclose(direct, dense, atol=5e-3, rtol=5e-3)

    batched = torch.randn(2, input_size, device="mps", dtype=torch.float16) * 0.1
    assert method._apply_mps_direct_gemv(layer, batched, bias=None) is None
    assert torch.equal(
        method.apply(layer, batched),
        torch.nn.functional.linear(batched, layer.weight),
    )
