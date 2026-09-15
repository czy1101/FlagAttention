# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Pytest-only DiffKV decode benchmark.

Run from the FlagAttention repository root with::

    pytest -q -s benchmark/diffkv_attention_benchmark.py

Benchmark settings are defined in ``BenchmarkConfig`` below.  The TLE and
standard Triton paths must be measured in separate pytest processes when they
are compared; Triton caches JIT kernels per process and should not be mixed in
one timing loop.

The benchmark allocates all paged-cache/workspace tensors before timing and
uses one CUDA event pair per operator call, with Triton's benchmark-cache
clear between calls.  It reports one row per shape with locally measured FA3
(when the optional vLLM FA3 extension is available), TLE or non-TLE
Triton 2D/3D, best speedup, and timing credibility diagnostics.  A
vLLM benchmark CSV can still be supplied as a fallback/reference with
``--fa3-csv``.  The FA3 extension is loaded directly with
``torch.ops.load_library`` from the active Python environment's
``site-packages/fa3_runtime`` directory.  The benchmark does not search the
repository, checkout ancestors, or external overlay directories for FA3.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import csv
import math
import os
import pathlib
import statistics
import sys
import time

import pytest
import torch
import triton

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fixed benchmark configuration used by the pytest entry point."""

    mode: str = "both"
    paths: tuple[str, ...] = ("2d", "3d")
    batches: tuple[int, ...] = (1, 8, 16, 32)
    seq_lens: tuple[int, ...] = (512, 2048, 8192, 32768)
    window_size: int = 128
    dtype: str = "bfloat16"
    warmup: int = 20
    iterations: int = 30
    samples: int = 9
    stability_cv_pct: float = 5.0
    backend: str = "auto"
    fa3_csv: tuple[pathlib.Path, ...] | None = None
    fa3: str = "auto"
    csv: pathlib.Path | None = None


def fa3_key(
    mode: str,
    batch: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
) -> tuple[str, int, int, int, int, int, int]:
    return (
        mode,
        batch,
        seq_len,
        num_query_heads,
        num_kv_heads,
        head_size_qk,
        head_size_v,
    )


def load_fa3_reference(paths: list[pathlib.Path] | None):
    """Load FA3 p50 values from vLLM benchmark CSV files, if provided."""
    reference: dict[tuple[str, int, int, int, int, int, int], float] = {}
    for path in paths or []:
        with path.open(newline="", encoding="utf-8") as input_file:
            for row in csv.DictReader(input_file):
                if "fa3" not in row.get("provider", "").lower():
                    continue
                try:
                    key = fa3_key(
                        row.get("mode", "decode"),
                        int(row.get("batch_size", row.get("batch", "0"))),
                        int(row["seq_len"]),
                        int(row.get("num_query_heads", "64")),
                        int(row.get("num_kv_heads", "4")),
                        int(row.get("head_size_qk", "192")),
                        int(row.get("head_size_v", "128")),
                    )
                    value = float(row["latency_p50_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    reference[key] = value
    return reference


def _fa3_module():
    """Import the in-repository FA3 adapter lazily after CUDA is available."""
    return importlib.import_module("flag_attn.diffkv_attention.FA3")


def load_fa3_provider(requested: str):
    """Load FA3 exclusively from the active Python environment."""
    return _fa3_module().load_fa3_provider(requested)


def make_inputs(
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    num_kv_heads: int,
):
    hq, hkv, dqk, dv, block_size = 64, num_kv_heads, 192, 128, 16
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    query = torch.randn(batch, hq, dqk, device="cuda", dtype=dtype)
    # Keep the same packed HND storage layout as vLLM's DiffKV backend, then
    # expose NHD K/V views.  FA3 is sensitive to cache strides (and this also
    # makes the Triton-vs-FA3 comparison use the production layout).
    kv_cache = torch.randn(
        batch * blocks_per_seq,
        hkv,
        block_size,
        dqk + dv,
        device="cuda",
        dtype=dtype,
    )
    kv_cache_nhd = kv_cache.transpose(1, 2)
    key_cache = kv_cache_nhd[..., :dqk]
    value_cache = kv_cache_nhd[..., dqk:]
    block_tables = torch.arange(
        batch * blocks_per_seq, device="cuda", dtype=torch.int32
    ).reshape(batch, blocks_per_seq)
    context_lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    return query, key_cache, value_cache, context_lens, block_tables


def build_runner(inputs, path: str, window_size: int, diffkv_impl):
    """Build a runner with the unified TLE/standard backend dispatcher."""
    query, key_cache, value_cache, context_lens, block_tables = inputs
    batch, hq, dqk = query.shape
    hkv = key_cache.shape[2]
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    scale = dqk**-0.5
    if path not in {"2d", "3d"}:
        raise ValueError(f"unsupported path: {path}")
    use_3d = path == "3d"

    num_q_per_kv = hq // hkv
    block_m = 16 if num_q_per_kv <= 16 else triton.next_power_of_2(num_q_per_kv)
    block_q = block_m // num_q_per_kv
    total_num_q_blocks = query.shape[0] // block_q + batch
    num_sms = torch.cuda.get_device_properties(query.device).multi_processor_count

    # Backend selection is environment-based: the benchmark keeps TLE and
    # standard Triton as explicit, reproducible comparison modes.
    use_optimized = diffkv_impl.USE_TLE
    impl_name = "TLE" if use_optimized else "non-TLE"
    selected_backend = diffkv_impl.SELECTED_BACKEND
    if use_3d:
        if use_optimized:
            num_segments = diffkv_impl.get_num_par_softmax_segments(
                seq_len,
                batch,
                True,
                total_num_q_blocks=total_num_q_blocks,
                num_kv_heads=hkv,
                num_sms=num_sms,
                block_size=block_size,
            )
        else:
            num_segments = diffkv_impl.get_num_par_softmax_segments(
                seq_len, batch, True
            )
        padded_v = 128
        segm_output = torch.empty(
            batch,
            hq,
            num_segments,
            padded_v,
            device="cuda",
            dtype=query.dtype if use_optimized else torch.float32,
        )
        segm_max = torch.empty(
            batch, hq, num_segments, device="cuda", dtype=torch.float32
        )
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None

    triton_window = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    out = torch.empty(batch, hq, 128, device="cuda", dtype=query.dtype)
    extra_kwargs = {}
    if use_optimized and use_3d and diffkv_impl.should_use_tle_fused_reducer(
        dqk, 128, 1, seq_len, batch, block_size, True
    ):
        extra_kwargs["fused_reducer_counter"] = torch.zeros(
            batch * hq, device="cuda", dtype=torch.int32
        )

    def run():
        diffkv_impl.unified_attention_diffkv(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=context_lens,
            softmax_scale=scale,
            causal=True,
            window_size=triton_window,
            block_table=block_tables,
            softcap=0.0,
            max_seqlen_q=1,
            seq_threshold_3D=threshold,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=segm_output,
            softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_expsum,
            max_seqlen_k=seq_len,
            backend=selected_backend,
            path=path,
            **extra_kwargs,
        )

    return run, impl_name


def build_fa3_runner(inputs, window_size: int):
    """Build a direct FA3 runner from the in-repository adapter."""
    return _fa3_module().build_fa3_runner(inputs, window_size)


def measure(fn, warmup: int, iterations: int, samples: int):
    """Measure per-call GPU latency using Triton's do_bench protocol.

    Recording one event around a long Python loop can include host submission
    gaps for microsecond kernels.  vLLM's benchmark records an event pair for
    every invocation and clears the benchmark cache before each one; mirror
    that protocol here so standalone FlagAttention numbers are comparable.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    cache = None
    try:
        driver = triton.runtime.driver.active
        cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()
    except Exception:
        driver = None
    values = []
    for _ in range(samples):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
        for start, end in zip(starts, ends):
            if driver is not None and cache is not None:
                driver.clear_cache(cache)
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        per_call = [start.elapsed_time(end) * 1000.0
                    for start, end in zip(starts, ends)]
        values.append(statistics.median(per_call))
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv_pct = 100.0 * statistics.pstdev(values) / mean if mean else 0.0
    return median, min(values), max(values), cv_pct


def parse_dtype(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_diffkv_backend(name: str):
    """Load exactly one backend before Triton JIT compilation starts."""
    os.environ["FLAG_ATTN_DIFFKV_BACKEND"] = name
    # Import the submodule explicitly so this benchmark remains independent
    # of the package's top-level exports.
    diffkv_impl = importlib.import_module("flag_attn.diffkv_attention.diffkv_attention")

    if name == "tle" and not diffkv_impl.HAS_TLE:
        detail = diffkv_impl.get_diffkv_backend_info()["tle_error"]
        raise RuntimeError(
            "TLE backend was requested but is unavailable. Install a Triton "
            "build exposing triton.experimental.tle.language. "
            f"Import error: {detail}"
        )
    return diffkv_impl


def run_benchmark(config: BenchmarkConfig | None = None):
    config = BenchmarkConfig() if config is None else config
    if not torch.cuda.is_available():
        raise RuntimeError("DiffKV benchmark requires CUDA")
    diffkv_impl = load_diffkv_backend(config.backend)
    dtype = parse_dtype(config.dtype)
    fa3_reference = load_fa3_reference(config.fa3_csv)
    fa3_status = load_fa3_provider(config.fa3)
    modes = ["full", "swa"] if config.mode == "both" else [config.mode]
    rows = []
    print(f"Device: {torch.cuda.get_device_name()} dtype={config.dtype}")
    print(
        "DiffKV backend: "
        f"requested={config.backend}, selected={diffkv_impl.SELECTED_BACKEND}, "
        f"HAS_TLE={diffkv_impl.is_tle_available()}"
    )
    if not fa3_status["available"]:
        print(
            "FA3 direct measurement unavailable; "
            "a supplied --fa3-csv may still be used as a reference. "
            f"Reason: {fa3_status['error']}"
        )
    if not diffkv_impl.is_tle_available():
        print(
            "TLE unavailable; using standard non-TLE Triton path. "
            f"Reason: {diffkv_impl.get_diffkv_backend_info()['tle_error']}"
        )
    print(
        "Timing: per-call CUDA events around preallocated operators; "
        "Triton benchmark cache cleared between calls"
    )
    for mode in modes:
        window = -1 if mode == "full" else config.window_size
        hkv = 4 if mode == "full" else 8
        section_rows = []
        for batch in config.batches:
            for seq_len in config.seq_lens:
                inputs = make_inputs(batch, seq_len, dtype, hkv)
                measurements = {}
                path_impls = {}
                for path in config.paths:
                    runner, impl_name = build_runner(
                        inputs, path, window, diffkv_impl
                    )
                    p50, pmin, pmax, cv_pct = measure(
                        runner,
                        config.warmup, config.iterations, config.samples,
                    )
                    measurements[path] = (p50, pmin, pmax, cv_pct)
                    path_impls[path] = impl_name
                fa3_measurement = None
                fa3_error = None
                if fa3_status["available"]:
                    try:
                        fa3_measurement = measure(
                            build_fa3_runner(inputs, window),
                            config.warmup,
                            config.iterations,
                            config.samples,
                        )
                    except Exception as exc:  # pragma: no cover - host-specific
                        fa3_error = f"{type(exc).__name__}: {exc}"
                best_path = min(measurements, key=lambda path: measurements[path][0])
                best = measurements[best_path]
                reference_fa3_us = fa3_reference.get(
                    fa3_key(
                        "decode",
                        batch,
                        seq_len,
                        64,
                        hkv,
                        192,
                        128,
                    )
                )
                if fa3_measurement is not None:
                    fa3_us = fa3_measurement[0]
                    fa3_source = "local"
                else:
                    fa3_us = reference_fa3_us
                    fa3_source = "reference-csv" if fa3_us is not None else "n/a"
                best_speedup = fa3_us / best[0] if fa3_us is not None else None
                credibility = (
                    f"range=[{best[1] / 1000:.6f},{best[2] / 1000:.6f}]ms "
                    f"CV={best[3]:.2f}% samples={config.samples} "
                    f"{'stable' if best[3] <= config.stability_cv_pct else 'noisy'}"
                )
                if fa3_us is None:
                    credibility += " FA3=n/a"
                else:
                    if fa3_measurement is not None:
                        credibility += (
                            f" FA3=local CV={fa3_measurement[3]:.2f}% "
                            f"{'stable' if fa3_measurement[3] <= config.stability_cv_pct else 'noisy'}"
                        )
                    else:
                        credibility += " FA3=reference-CSV"
                if fa3_error is not None:
                    credibility += f" FA3-error={fa3_error}"
                shape_text = (
                    f"B={batch} Q=1 KV={seq_len} HQ=64 HKV={hkv} "
                    f"DQK=192 DV=128 {config.dtype}"
                )
                section_rows.append(
                    (
                        shape_text,
                        f"{fa3_us / 1000:.6f}" if fa3_us is not None else "n/a",
                        f"{measurements['2d'][0] / 1000:.6f}"
                        if "2d" in measurements else "n/a",
                        f"{measurements['3d'][0] / 1000:.6f}"
                        if "3d" in measurements else "n/a",
                        path_impls.get("2d", "n/a"),
                        path_impls.get("3d", "n/a"),
                        f"{best_speedup:.3f}x" if best_speedup is not None else "n/a",
                        best_path.upper(),
                        credibility,
                    )
                )
                for path, measurement in measurements.items():
                    rows.append({
                        "mode": mode,
                        "batch": batch,
                        "seq_len": seq_len,
                        "path": path,
                        "implementation": path_impls[path],
                        "dtype": config.dtype,
                        "backend": diffkv_impl.SELECTED_BACKEND,
                        "fa3_source": fa3_source,
                        "fa3_p50_us": ""
                        if fa3_us is None else f"{fa3_us:.3f}",
                        "fa3_min_us": ""
                        if fa3_measurement is None else f"{fa3_measurement[1]:.3f}",
                        "fa3_max_us": ""
                        if fa3_measurement is None else f"{fa3_measurement[2]:.3f}",
                        "fa3_sample_cv_pct": ""
                        if fa3_measurement is None
                        else f"{fa3_measurement[3]:.3f}",
                        "fa3_error": fa3_error or "",
                        "p50_us": f"{measurement[0]:.3f}",
                        "min_us": f"{measurement[1]:.3f}",
                        "max_us": f"{measurement[2]:.3f}",
                        "sample_cv_pct": f"{measurement[3]:.3f}",
                        "best_path": best_path,
                        "best_speedup": ""
                        if best_speedup is None else f"{best_speedup:.6f}",
                    })
        headers = (
            "shape (B,Q,KV,HQ,HKV,DQK,DV,dtype)",
            "FA3 p50(ms)",
            "Triton-2D p50(ms)",
            "Triton-3D p50(ms)",
            "2D impl",
            "3D impl",
            "best speedup",
            "best path",
            "data credibility",
        )
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in section_rows))
            for index in range(len(headers))
        ]
        separator_length = sum(widths) + 2 * (len(widths) - 1)
        title = "Full Attention" if mode == "full" else "SWA"
        print("\n" + "=" * separator_length)
        print(f"[{title}]")
        print("=" * separator_length)
        print("  ".join(value.ljust(widths[index])
                         for index, value in enumerate(headers)))
        print("  ".join("-" * width for width in widths))
        for row in section_rows:
            print("  ".join(value.ljust(widths[index])
                             for index, value in enumerate(row)))
        print("-" * separator_length)
    if config.csv is not None:
        config.csv.parent.mkdir(parents=True, exist_ok=True)
        with config.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {config.csv}")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DiffKV benchmark requires CUDA",
)
def test_perf_diffkv_attention():
    """Run the default DiffKV benchmark under pytest."""
    run_benchmark()
