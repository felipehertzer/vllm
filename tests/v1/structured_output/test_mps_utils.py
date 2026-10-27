# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.structured_output.utils import _apply_grammar_bitmask_torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is not available"
)


def test_apply_grammar_bitmask_torch_on_mps():
    device = torch.device("mps")
    logits = torch.zeros((2, 8), dtype=torch.float16, device=device)
    bitmask = torch.tensor(
        [
            [(1 << 1) | (1 << 3) | (1 << 7)],
            [(1 << 0) | (1 << 2) | (1 << 4)],
        ],
        dtype=torch.int32,
        device=device,
    )

    _apply_grammar_bitmask_torch(logits, bitmask)

    result = logits.cpu()
    assert torch.isneginf(result[0, [0, 2, 4, 5, 6]]).all()
    assert torch.equal(result[0, [1, 3, 7]], torch.zeros(3, dtype=torch.float16))
    assert torch.isneginf(result[1, [1, 3, 5, 6, 7]]).all()
    assert torch.equal(result[1, [0, 2, 4]], torch.zeros(3, dtype=torch.float16))
