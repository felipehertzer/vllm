# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.mps import (
    MPSWNA16LinearKernel,
)
from vllm.platforms.interface import PlatformEnum
from vllm.scalar_type import scalar_types


def _config(
    *,
    weight_type=scalar_types.uint4b8,
    act_type=torch.bfloat16,
    group_size: int = 128,
    zero_points: bool = False,
    shape: tuple[int, int] = (2560, 2560),
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=shape,
        partition_weight_shape=shape,
        weight_type=weight_type,
        act_type=act_type,
        group_size=group_size,
        zero_points=zero_points,
        has_g_idx=False,
    )


def test_choose_mp_linear_kernel_picks_mps_wna16():
    platform = SimpleNamespace(
        _enum=PlatformEnum.MPS,
        get_device_capability=lambda: None,
        is_mps=lambda: True,
    )

    with (
        patch("vllm.model_executor.kernels.linear.current_platform", platform),
        patch(
            "vllm.model_executor.kernels.linear.mixed_precision.mps.current_platform",
            platform,
        ),
    ):
        kernel_type = choose_mp_linear_kernel(_config())

    assert kernel_type is MPSWNA16LinearKernel


@pytest.mark.parametrize(
    ("config", "expected_reason"),
    [
        (_config(weight_type=scalar_types.uint8b128), "Quant type"),
        (_config(group_size=127), "Group size"),
        (_config(shape=(2559, 2560)), "Input size"),
        (_config(shape=(2560, 2559)), "Output size"),
    ],
)
def test_mps_wna16_rejects_unsupported_configs(
    config: MPLinearLayerConfig, expected_reason: str
):
    platform = SimpleNamespace(is_mps=lambda: True)

    with patch(
        "vllm.model_executor.kernels.linear.mixed_precision.mps.current_platform",
        platform,
    ):
        ok, reason = MPSWNA16LinearKernel.can_implement(config)

    assert ok is False
    assert reason is not None
    assert expected_reason in reason
