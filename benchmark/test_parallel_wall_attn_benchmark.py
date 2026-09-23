# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Official-contract Wall-Attention benchmark against upstream FLA."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from functools import lru_cache

import pytest
import torch
import triton

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:
    from recording import benchmark_metric, record_benchmark_result

from flag_attn import parallel_wall_attn
from flag_attn.FLA.wall_attn import select_route
from flag_attn.utils import has_triton_tle

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
SHAPE_FAMILIES = (("mha", 8), ("gqa", 2))
WARMUP_CALLS = 30
MEASUREMENT_MS = 200
REPEATS = 7


@lru_cache(maxsize=1)
def _load_fla_provider():
    try:
        from fla.ops.wall_attn import parallel_wall_attn as fla_wall_attn
    except Exception as exc:
        raise RuntimeError(f"official FLA Wall-Attention is unavailable: {exc}") from exc
    return fla_wall_attn


def _make_inputs(t: int, h: int, dtype: torch.dtype):
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    q = torch.randn(1, t, 8, 64, device="cuda", dtype=dtype)
    k = torch.randn(1, t, h, 64, device="cuda", dtype=dtype)
    v = torch.randn(1, t, h, 64, device="cuda", dtype=dtype)
    g = (-torch.randn(1, t, 8, 64, device="cuda").abs() * 0.05).to(dtype)
    return q, k, v, g


def _bench_ms(fn: Callable[[], torch.Tensor]) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=0,
            rep=MEASUREMENT_MS,
            return_mode="median",
        )
    )


def _summary(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "std_ms": statistics.pstdev(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


@torch.inference_mode()
def _run_benchmark(dtype_name, dtype, record_property=None):
    fla_wall_attn = _load_fla_provider()
    metrics = []
    speedups = []

    print("dtype family T FLA_ms Final_ms speedup")
    for family, h in SHAPE_FAMILIES:
        for t in SEQUENCE_LENGTHS:
            inputs = _make_inputs(t, h, dtype)
            scale = 64**-0.5
            providers = {
                "fla": lambda: fla_wall_attn(*inputs, scale=scale),
                "optimized": lambda: parallel_wall_attn(*inputs, scale=scale),
            }
            expected_route = "tle_bf16" if dtype is torch.bfloat16 else "tle_fp16"
            assert select_route(*inputs, scale=scale) == expected_route

            outputs = {name: fn() for name, fn in providers.items()}
            torch.cuda.synchronize()
            for name, output in outputs.items():
                assert torch.isfinite(output).all(), f"{name} produced NaN/Inf"
            torch.testing.assert_close(
                outputs["optimized"],
                outputs["fla"],
                atol=0.05,
                rtol=0.05,
            )

            provider_names = tuple(providers)
            for warmup_index in range(WARMUP_CALLS):
                order = provider_names if warmup_index % 2 == 0 else provider_names[::-1]
                for name in order:
                    providers[name]()
            torch.cuda.synchronize()

            samples = {name: [] for name in provider_names}
            for repeat in range(REPEATS):
                order = provider_names if repeat % 2 == 0 else provider_names[::-1]
                for name in order:
                    samples[name].append(_bench_ms(providers[name]))

            latency = {name: _summary(values) for name, values in samples.items()}
            fla_ms = float(latency["fla"]["median_ms"])
            optimized_ms = float(latency["optimized"]["median_ms"])
            speedup = fla_ms / optimized_ms
            speedups.append(speedup)
            print(
                dtype_name,
                family,
                t,
                f"{fla_ms:.6f}",
                f"{optimized_ms:.6f}",
                f"{speedup:.4f}",
                flush=True,
            )
            metrics.append(
                benchmark_metric(
                    shape_detail={
                        "B": 1,
                        "T": t,
                        "H": h,
                        "HQ": 8,
                        "D": 64,
                        "family": family,
                    },
                    latency_base=fla_ms,
                    latency=optimized_ms,
                    speedup=speedup,
                    provider_route=expected_route,
                    samples=latency,
                )
            )

    geomean_speedup = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    print(f"geomean {dtype_name}: {geomean_speedup:.4f}x vs FLA", flush=True)
    record_benchmark_result(
        record_property,
        op_name="parallel_wall_attn",
        dtype=dtype_name,
        result=metrics,
        baseline="official_fla",
        phase="forward",
        hardware="H100/SM90",
        input_contract="gate_scale=0.05, seed=0",
        warmup_calls=WARMUP_CALLS,
        measurement_ms=MEASUREMENT_MS,
        repeats=REPEATS,
        geomean_speedup_vs_fla=geomean_speedup,
    )


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "dtype_name,dtype",
    (("bf16", torch.bfloat16), ("fp16", torch.float16)),
    ids=("bf16", "fp16"),
)
@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) and has_triton_tle()),
    reason="Wall-Attention performance benchmark requires H100/SM90 with TLE",
)
def test_parallel_wall_attn_benchmark(dtype_name, dtype, record_property):
    _run_benchmark(dtype_name, dtype, record_property)


def main():
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) and has_triton_tle()):
        raise RuntimeError("Wall-Attention performance benchmark requires H100/SM90 with TLE")
    for dtype_name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        _run_benchmark(dtype_name, dtype)


if __name__ == "__main__":
    main()
