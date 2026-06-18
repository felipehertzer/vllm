# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.v1.worker.mps_worker import MPSWorker


def _worker_for_kv_specs(
    *,
    block_size: int = 16,
    page_size_bytes: int = 1024,
    num_layers: int = 2,
    max_model_len: int = 2048,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 2048,
) -> MPSWorker:
    worker = MPSWorker.__new__(MPSWorker)
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {
            f"layer.{idx}": SimpleNamespace(
                block_size=block_size,
                page_size_bytes=page_size_bytes,
            )
            for idx in range(num_layers)
        }
    )
    worker.model_config = SimpleNamespace(max_model_len=max_model_len)
    worker.scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    return worker


def test_mps_target_kv_cache_memory_uses_configured_capacity_with_reserve():
    worker = _worker_for_kv_specs()

    assert worker._get_target_kv_cache_memory_bytes() == 640 * 2 * 1024


def test_mps_target_kv_cache_memory_has_minimum_reserve_blocks():
    worker = _worker_for_kv_specs(
        max_model_len=64,
        max_num_seqs=1,
        max_num_batched_tokens=64,
    )

    assert worker._get_target_kv_cache_memory_bytes() == 20 * 2 * 1024


def test_mps_target_kv_cache_memory_skips_mixed_block_sizes():
    worker = MPSWorker.__new__(MPSWorker)
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {
            "layer.0": SimpleNamespace(block_size=16, page_size_bytes=1024),
            "layer.1": SimpleNamespace(block_size=32, page_size_bytes=2048),
        }
    )

    assert worker._get_target_kv_cache_memory_bytes() is None


def test_mps_worker_sets_allocator_memory_fraction(monkeypatch: pytest.MonkeyPatch):
    worker = MPSWorker.__new__(MPSWorker)
    worker.cache_config = SimpleNamespace(gpu_memory_utilization=0.73)
    calls: list[float] = []

    monkeypatch.setattr(
        "vllm.v1.worker.mps_worker.torch.mps.set_per_process_memory_fraction",
        calls.append,
    )

    worker._set_mps_allocator_memory_fraction()

    assert calls == [0.73]
