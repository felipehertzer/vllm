# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a prototype packed GPTQ linear kernel on Apple MPS.

This script is intentionally not wired into model execution. It captures the
current gap between the working MPS dequantize-to-BF16 path and a first packed
Metal shader prototype, so future MPS GPTQ kernel work has a reproducible target.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.quantization.auto_gptq import (
    _dequantize_gptq_weight,
)

DEFAULT_LAYERS = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
)

MPS_GPTQ_QMV_SHADER = r"""
#include <metal_stdlib>
using namespace metal;

kernel void qmv64x4(
    device bfloat* out [[buffer(0)]],
    device const bfloat* x [[buffer(1)]],
    device const int* qweight_i [[buffer(2)]],
    device const bfloat* scales [[buffer(3)]],
    device const int* qzeros_i [[buffer(4)]],
    device const int* g_idx [[buffer(5)]],
    constant int& M [[buffer(6)]],
    constant int& K [[buffer(7)]],
    constant int& N [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  threadgroup float scratch[4][64];
  uint lane = tid.x;
  uint col = tid.y;
  uint n = tg.x * 4u + col;
  uint m = tg.y;
  float acc = 0.0f;

  if (m < (uint)M && n < (uint)N) {
    for (uint k = lane; k < (uint)K; k += 64u) {
      uint packed_w = as_type<uint>(qweight_i[(int)(k >> 3) * N + (int)n]);
      uint q = (packed_w >> ((k & 7u) * 4u)) & 0xFu;
      int group = g_idx[k];
      uint packed_z =
          as_type<uint>(qzeros_i[group * ((N + 7) >> 3) + ((int)n >> 3)]);
      uint z = ((packed_z >> (((int)n & 7) * 4)) & 0xFu) + 1u;
      float scale = float(scales[group * N + (int)n]);
      float input = float(x[m * (uint)K + k]);
      acc += input * ((float(q) - float(z)) * scale);
    }
  }

  scratch[col][lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 32u) scratch[col][lane] += scratch[col][lane + 32u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 16u) scratch[col][lane] += scratch[col][lane + 16u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 8u) scratch[col][lane] += scratch[col][lane + 8u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 4u) scratch[col][lane] += scratch[col][lane + 4u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 2u) scratch[col][lane] += scratch[col][lane + 2u];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u && m < (uint)M && n < (uint)N) {
    scratch[col][0] += scratch[col][1];
    out[m * (uint)N + n] = bfloat(scratch[col][0]);
  }
}
"""


def _time_ms(fn, *, warmup: int, samples: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    timings = []
    for _ in range(samples):
        started = time.perf_counter()
        fn()
        torch.mps.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _summary(timings: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(timings), 4),
        "median_ms": round(statistics.median(timings), 4),
        "min_ms": round(min(timings), 4),
        "max_ms": round(max(timings), 4),
    }


def _load_layer_tensors(
    safetensors_path: Path,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with safe_open(safetensors_path, framework="pt", device="cpu") as tensors:
        return (
            tensors.get_tensor(f"{layer_name}.qweight"),
            tensors.get_tensor(f"{layer_name}.scales"),
            tensors.get_tensor(f"{layer_name}.qzeros"),
            tensors.get_tensor(f"{layer_name}.g_idx"),
        )


def _benchmark_layer(
    shader,
    safetensors_path: Path,
    layer_name: str,
    *,
    warmup: int,
    samples: int,
) -> dict[str, object]:
    qweight, scales, qzeros, g_idx = _load_layer_tensors(
        safetensors_path,
        layer_name,
    )
    input_size = g_idx.numel()
    output_size = qweight.shape[1]
    batch = 1

    qweight_mps = qweight.to("mps")
    scales_mps = scales.to("mps", dtype=torch.bfloat16)
    qzeros_mps = qzeros.to("mps")
    g_idx_mps = g_idx.to("mps")
    x = torch.randn((batch, input_size), device="mps", dtype=torch.bfloat16)
    packed_out = torch.empty(
        (batch, output_size),
        device="mps",
        dtype=torch.bfloat16,
    )

    dense_weight = _dequantize_gptq_weight(
        qweight,
        scales,
        qzeros,
        g_idx,
        bits=4,
        group_size=128,
        dtype=torch.bfloat16,
        device=torch.device("mps"),
    )

    threads = (math.ceil(output_size / 4) * 64, batch * 4, 1)
    group_size = (64, 4, 1)

    def packed_run() -> None:
        shader.qmv64x4(
            packed_out,
            x,
            qweight_mps,
            scales_mps,
            qzeros_mps,
            g_idx_mps,
            batch,
            input_size,
            output_size,
            threads=threads,
            group_size=group_size,
        )

    def dense_run() -> None:
        F.linear(x, dense_weight)

    packed_run()
    dense_ref = F.linear(x, dense_weight)
    torch.mps.synchronize()
    max_error = (packed_out.float() - dense_ref.float()).abs().max().item()
    mean_error = (packed_out.float() - dense_ref.float()).abs().mean().item()

    packed_timings = _time_ms(packed_run, warmup=warmup, samples=samples)
    dense_timings = _time_ms(dense_run, warmup=warmup, samples=samples)
    packed_summary = _summary(packed_timings)
    dense_summary = _summary(dense_timings)

    return {
        "layer": layer_name,
        "input_size": input_size,
        "output_size": output_size,
        "max_error": max_error,
        "mean_error": mean_error,
        "packed": packed_summary,
        "dense_dequant": dense_summary,
        "packed_vs_dense_median": round(
            packed_summary["median_ms"] / dense_summary["median_ms"],
            4,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "safetensors_path",
        type=Path,
        help="Path to the GPTQ model.safetensors file.",
    )
    parser.add_argument("--layer", action="append", dest="layers")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine.", file=sys.stderr)
        return 1

    layers = tuple(args.layers) if args.layers else DEFAULT_LAYERS
    shader = torch.mps.compile_shader(MPS_GPTQ_QMV_SHADER)
    results = [
        _benchmark_layer(
            shader,
            args.safetensors_path,
            layer,
            warmup=args.warmup,
            samples=args.samples,
        )
        for layer in layers
    ]
    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
