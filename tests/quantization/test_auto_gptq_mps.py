# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.model_executor.layers.quantization.auto_gptq import (
    _dequantize_gptq_weight,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    gptq_pack,
    gptq_quantize_weights,
    pack_cols,
)
from vllm.scalar_type import scalar_types


@pytest.mark.parametrize("act_order", [False, True])
def test_mps_gptq_dequant_matches_reference(act_order: bool):
    torch.manual_seed(0)
    input_size = 64
    output_size = 16
    group_size = 32
    weight = torch.randn(input_size, output_size, dtype=torch.float32)
    test_perm = torch.randperm(input_size) if act_order else None
    weight_ref, qweight_unpacked, scales, g_idx, _ = gptq_quantize_weights(
        weight,
        scalar_types.uint4b8,
        group_size,
        act_order,
        test_perm=test_perm,
    )
    qweight = gptq_pack(qweight_unpacked, 4, input_size, output_size)
    qzeros_unpacked = torch.full(
        (input_size // group_size, output_size),
        scalar_types.uint4b8.bias - 1,
        dtype=torch.int32,
    )
    qzeros = pack_cols(qzeros_unpacked, 4, input_size // group_size, output_size)

    dequant_weight = _dequantize_gptq_weight(
        qweight=qweight,
        scales=scales,
        qzeros=qzeros,
        g_idx=g_idx,
        bits=4,
        group_size=group_size,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert torch.allclose(dequant_weight.t(), weight_ref, atol=1e-6, rtol=0)
