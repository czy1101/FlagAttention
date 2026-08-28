# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure absolute latency of the public MetaX chunk_kda operator."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import triton


_BACKEND_ENV = "FLAG_ATTN_CHUNK_KDA_BACKEND"
_DEFAULT_SEQUENCE_LENGTHS = (256, 1024, 2048, 4096, 6144, 8192)
_BATCH = 1
_HEADS = 96
_DIM = 128
_CHUNK_SIZE = 16
_LOWER_BOUND = -5.0

ChunkKDA = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


@contextlib.contextmanager
def _public_auto_backend() -> Iterator[None]:
    previous = os.environ.get(_BACKEND_ENV)
    os.environ[_BACKEND_ENV] = "auto"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_BACKEND_ENV, None)
        else:
            os.environ[_BACKEND_ENV] = previous


def _load_public_operator() -> ChunkKDA:
    from flag_attn import chunk_kda

    return chunk_kda


def _make_inputs(
    sequence_length: int,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    if sequence_length <= 0:
        raise ValueError("sequence length must be positive")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    shape = (_BATCH, sequence_length, _HEADS, _DIM)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    g = torch.randn(shape, device=device, dtype=dtype)
    beta = torch.randn(
        (_BATCH, sequence_length, _HEADS),
        device=device,
        dtype=dtype,
    )
    a_log = torch.log(
        torch.empty(_HEADS, device=device, dtype=torch.float32).uniform_(1, 16)
    )
    dt_bias = torch.randn(
        (_HEADS, _DIM),
        device=device,
        dtype=torch.float32,
    )

    kwargs: dict[str, Any] = {
        "scale": 1.0 / math.sqrt(_DIM),
        "initial_state": None,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": _LOWER_BOUND,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state_v_first": True,
        "cu_seqlens": None,
        "chunk_size": _CHUNK_SIZE,
    }
    return (q, k, v, g, beta), kwargs


def _input_versions(
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any],
) -> list[tuple[torch.Tensor, int]]:
    tensors = list(args)
    tensors.extend(
        value for value in kwargs.values() if isinstance(value, torch.Tensor)
    )
    return [(tensor, tensor._version) for tensor in tensors]


def _inputs_unchanged(
    snapshots: list[tuple[torch.Tensor, int]],
) -> bool:
    return all(tensor._version == version for tensor, version in snapshots)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: list[float], seed: int) -> dict[str, float | int | str]:
    mean = statistics.fmean(values)
    median = statistics.median(values)
    p20 = _percentile(values, 0.20)
    p80 = _percentile(values, 0.80)
    raw_cv = statistics.pstdev(values) / mean if mean else 0.0

    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    robust_cv = 1.4826 * mad / median if median else 0.0

    midpoint = len(values) // 2
    first_median = statistics.median(values[:midpoint])
    second_median = statistics.median(values[midpoint:])
    half_drift = (
        abs(first_median - second_median) / median
        if median
        else 0.0
    )

    outlier_cutoff = 3.0 * 1.4826 * mad
    mad_outliers = (
        sum(abs(value - median) > outlier_cutoff for value in values)
        if outlier_cutoff
        else 0
    )

    generator = random.Random(seed)
    bootstrap_medians = []
    for _ in range(2000):
        sample = [
            values[generator.randrange(len(values))]
            for _ in range(len(values))
        ]
        bootstrap_medians.append(statistics.median(sample))

    ci_low = _percentile(bootstrap_medians, 0.025)
    ci_high = _percentile(bootstrap_medians, 0.975)
    ci_relative_half_width = (ci_high - ci_low) / (2.0 * median)

    if (
        robust_cv <= 0.03
        and ci_relative_half_width <= 0.02
        and half_drift <= 0.01
    ):
        status = "ROBUST_STABLE"
    elif (
        robust_cv <= 0.05
        and ci_relative_half_width <= 0.03
        and half_drift <= 0.01
    ):
        status = "ACCEPT_WITH_TAIL"
    else:
        status = "TARGETED_REVIEW"

    return {
        "p20": p20,
        "p50": median,
        "p80": p80,
        "mean": mean,
        "raw_cv": raw_cv,
        "robust_cv": robust_cv,
        "p80_over_p20": p80 / p20,
        "half_drift": half_drift,
        "mad": mad,
        "mad_outliers": mad_outliers,
        "median_ci95_low": ci_low,
        "median_ci95_high": ci_high,
        "ci_relative_half_width": ci_relative_half_width,
        "status": status,
    }


def _timed_block(
    operator: ChunkKDA,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, Any],
    inner: int,
) -> tuple[float, float, tuple[torch.Tensor, torch.Tensor | None]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start.record()

    result = None
    for _ in range(inner):
        result = operator(*args, **kwargs)

    end.record()
    end.synchronize()

    assert result is not None
    event_ms = float(start.elapsed_time(end)) / inner
    wall_ms = (time.perf_counter() - wall_start) * 1000.0 / inner
    return event_ms, wall_ms, result


@torch.inference_mode()
def _run_shape(
    operator: ChunkKDA,
    sequence_length: int,
    warmup: int,
    blocks: int,
    target_block_ms: float,
    max_inner: int,
    seed: int,
    print_samples: bool,
) -> dict[str, Any]:
    with torch.inference_mode(False):
        args, kwargs = _make_inputs(sequence_length, seed)
    versions = _input_versions(args, kwargs)

    original_backend = os.environ.get(_BACKEND_ENV)

    with _public_auto_backend():
        result = None
        for _ in range(warmup):
            result = operator(*args, **kwargs)
        torch.cuda.synchronize()

        pilot_ms, _, result = _timed_block(
            operator,
            args,
            kwargs,
            inner=1,
        )
        if pilot_ms <= 0:
            raise RuntimeError(f"invalid pilot latency: {pilot_ms}")

        inner = max(
            1,
            min(max_inner, math.ceil(target_block_ms / pilot_ms)),
        )

        for _ in range(2):
            for _ in range(inner):
                result = operator(*args, **kwargs)
            torch.cuda.synchronize()

        event_samples = []
        wall_samples = []

        for block_index in range(blocks):
            event_ms, wall_ms, result = _timed_block(
                operator,
                args,
                kwargs,
                inner,
            )
            event_samples.append(event_ms)
            wall_samples.append(wall_ms)

            if print_samples:
                print(
                    f"T={sequence_length} block={block_index} "
                    f"inner={inner} event_ms={event_ms:.6f} "
                    f"wall_ms={wall_ms:.6f}",
                    flush=True,
                )

    if os.environ.get(_BACKEND_ENV) != original_backend:
        raise AssertionError(f"{_BACKEND_ENV} was not restored")

    assert result is not None
    output, final_state = result

    expected_output_shape = (
        _BATCH,
        sequence_length,
        _HEADS,
        _DIM,
    )
    expected_state_shape = (
        _BATCH,
        _HEADS,
        _DIM,
        _DIM,
    )

    if output.shape != expected_output_shape:
        raise AssertionError((output.shape, expected_output_shape))
    if final_state is None:
        raise AssertionError("final state is missing")
    if final_state.shape != expected_state_shape:
        raise AssertionError((final_state.shape, expected_state_shape))
    if output.dtype != torch.bfloat16:
        raise AssertionError(output.dtype)
    if final_state.dtype != torch.float32:
        raise AssertionError(final_state.dtype)
    if not torch.isfinite(output).all().item():
        raise AssertionError("output contains non-finite values")
    if not torch.isfinite(final_state).all().item():
        raise AssertionError("final state contains non-finite values")
    if not _inputs_unchanged(versions):
        raise AssertionError("input tensor was modified in place")

    event_stats = _statistics(
        event_samples,
        seed=seed + 100_000,
    )
    wall_stats = _statistics(
        wall_samples,
        seed=seed + 200_000,
    )

    return {
        "shape": (
            f"B{_BATCH} T{sequence_length} H{_HEADS} "
            f"K{_DIM} V{_DIM} BF16 initial=None"
        ),
        "batch": _BATCH,
        "sequence_length": sequence_length,
        "heads": _HEADS,
        "key_dim": _DIM,
        "value_dim": _DIM,
        "dtype": "torch.bfloat16",
        "initial_state": "none",
        "output_final_state": True,
        "dispatch": "public_auto",
        "warmup": warmup,
        "blocks": blocks,
        "pilot_ms": pilot_ms,
        "inner": inner,
        "event_samples_ms": event_samples,
        "wall_samples_ms": wall_samples,
        "event": event_stats,
        "wall": wall_stats,
        "inputs_unchanged": True,
        "finite": True,
        "status": event_stats["status"],
    }


def _print_table(results: list[dict[str, Any]]) -> None:
    print()
    print(
        "| Shape | Event p20 ms | Event p50 ms | Event p80 ms | "
        "Robust CV | CI half-width | Drift | Wall p50 ms | Status |"
    )
    print(
        "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"
    )

    for result in results:
        event = result["event"]
        wall = result["wall"]
        print(
            f"| {result['shape']} "
            f"| {event['p20']:.6f} "
            f"| {event['p50']:.6f} "
            f"| {event['p80']:.6f} "
            f"| {event['robust_cv']:.4%} "
            f"| {event['ci_relative_half_width']:.4%} "
            f"| {event['half_drift']:.4%} "
            f"| {wall['p50']:.6f} "
            f"| {result['status']} |"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure public MetaX chunk_kda absolute latency.",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=list(_DEFAULT_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--blocks", type=int, default=21)
    parser.add_argument("--target-block-ms", type=float, default=40.0)
    parser.add_argument("--max-inner", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--print-samples", action="store_true")
    parser.add_argument("--list-default-shapes", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_default_shapes:
        for sequence_length in _DEFAULT_SEQUENCE_LENGTHS:
            print(
                f"B{_BATCH} T{sequence_length} H{_HEADS} "
                f"K{_DIM} V{_DIM} BF16 initial=None"
            )
        return

    if args.warmup < 1:
        raise ValueError("warmup must be positive")
    if args.blocks < 9:
        raise ValueError("blocks must be at least 9")
    if args.target_block_ms <= 0:
        raise ValueError("target block latency must be positive")
    if args.max_inner < 1:
        raise ValueError("max inner count must be positive")
    if any(length <= 0 for length in args.sequence_lengths):
        raise ValueError("all sequence lengths must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("chunk_kda benchmark requires a CUDA-compatible device")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("chunk_kda benchmark requires exactly one visible device")

    device = torch.cuda.get_device_name(0)
    if device != "MetaX C550":
        raise RuntimeError(f"expected MetaX C550, got {device}")

    properties = torch.cuda.get_device_properties(0)
    operator = _load_public_operator()
    original_backend = os.environ.get(_BACKEND_ENV)

    metadata = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "device": device,
        "uuid": str(getattr(properties, "uuid", "unknown")),
        "provider": "public_auto",
        "measurement": "device_event_steady_state_amortized",
        "sequence_lengths": list(args.sequence_lengths),
        "warmup": args.warmup,
        "blocks": args.blocks,
        "target_block_ms": args.target_block_ms,
        "max_inner": args.max_inner,
        "seed": args.seed,
    }

    print("===== chunk_kda absolute benchmark identity =====")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    results = []
    for sequence_length in args.sequence_lengths:
        print(f"benchmarking: T={sequence_length}", flush=True)
        result = _run_shape(
            operator=operator,
            sequence_length=sequence_length,
            warmup=args.warmup,
            blocks=args.blocks,
            target_block_ms=args.target_block_ms,
            max_inner=args.max_inner,
            seed=args.seed + sequence_length,
            print_samples=args.print_samples,
        )
        results.append(result)
        gc.collect()
        torch.cuda.empty_cache()

    if os.environ.get(_BACKEND_ENV) != original_backend:
        raise AssertionError(f"{_BACKEND_ENV} was not restored")

    _print_table(results)

    statuses = [result["status"] for result in results]
    if all(status == "ROBUST_STABLE" for status in statuses):
        overall_status = "ROBUST_STABLE"
    elif all(
        status in {"ROBUST_STABLE", "ACCEPT_WITH_TAIL"}
        for status in statuses
    ):
        overall_status = "ACCEPT_WITH_TAIL"
    else:
        overall_status = "COMPLETE_WITH_REVIEW_SHAPES"

    payload = {
        "metadata": metadata,
        "results": results,
        "overall_status": overall_status,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json_output={args.json_output}")

    print(f"overall_status={overall_status}")
    print("KDA_PUBLIC_AUTO_ABSOLUTE_BENCHMARK: COMPLETE")


if __name__ == "__main__":
    main()
