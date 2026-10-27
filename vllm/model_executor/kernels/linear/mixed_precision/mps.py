# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
    CPUWNA16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

_MPSWNA16_SUPPORTED_QUANT_TYPES = (scalar_types.uint4, scalar_types.uint4b8)


class MPSWNA16LinearKernel(CPUWNA16LinearKernel):
    """MPS backend for compressed-tensors WNA16 linear layers.

    Apple Metal does not expose native int4 GEMM through PyTorch today, so this
    backend converts packed WNA16 weights to dense weights once after loading and
    runs the actual matmul on MPS with the model activation dtype.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_mps():
            return False, "MPSWNA16 only supported on MPS"

        if c.weight_type not in _MPSWNA16_SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "MPSWNA16, supported types are: "
                f"{_MPSWNA16_SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size != -1 and c.group_size % 2 != 0:
            return (
                False,
                f"Group size ({c.group_size}) not supported by "
                "MPSWNA16, supported group sizes are multiples of 2",
            )

        if c.partition_weight_shape[0] % 32 != 0:
            return (
                False,
                f"Input size ({c.partition_weight_shape[0]}) not supported by "
                "MPSWNA16, supported sizes are multiples of 32",
            )

        if c.partition_weight_shape[1] % 32 != 0:
            return (
                False,
                f"Output size ({c.partition_weight_shape[1]}) not supported by "
                "MPSWNA16, supported sizes are multiples of 32",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        self._process_gptq_weights_torch(layer)
