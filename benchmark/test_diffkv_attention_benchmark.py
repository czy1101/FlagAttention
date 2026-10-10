# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""DiffKV Full Attention and SWA decode benchmarks, organized like chunk GLA.

Run with pytest -v -s -m diffkv_attention from the repository root. There is
an optional FA3 comparison baseline imported through the same ordinary Python
import used by the reference attention benchmarks. If FA3 is unavailable,
the benchmark still runs and reports FlagAttention-only timings.

Generic 2D/3D paths retain preallocated buffers and the original timing
protocol. Registered vendor implementations use their public API and report
full-call latency separately. FLAG_ATTN_DIFFKV_BACKEND selects the generic
TLE/standard Triton implementation; runtime selects the device vendor.
"""

from collections.abc import Callable
import math
import statistics

import pytest
import torch
import triton

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn.diffkv_attention import api as diffkv_impl
from flag_attn.diffkv_attention.api import DEFAULT_LAYOUT, LAUNCH, OP_NAME, SUPPORTED_PATHS


DEFAULT_WARMUP = 100
DEFAULT_REP = 500
DEFAULT_SAMPLES = 9
STABILITY_CV_THRESHOLD_PCT = 5.0
DEFAULT_SEED = 0
PAGE_TABLE = "random"
WINDOW_SIZE = 128
TABLE_WIDTH = 105

# Preserve the original Full Attention/SWA matrix and production head layout.
_SHAPES = [(batch, length) for batch in (1, 8, 16, 32) for length in (512, 2048, 8192, 32768)]
_DTYPES = [torch.bfloat16]
_MODES = ("full", "swa")
_PATHS = ("2d", "3d")


# FA3 is optional. Keep this as a normal Python import: the benchmark does not
# scan the filesystem, load a shared library, or require vLLM. The flash-attn-3
# wheel exposes both the public wrapper and the low-level op; the latter accepts
# a preallocated ``out`` tensor without triggering the PyTorch alias checker.
try:
    from flash_attn_interface import (
        _flash_attn_forward as _fa3_forward,
        flash_attn_3_gpu as _fa3_gpu,
    )

    _fa3_direct_forward = getattr(_fa3_gpu, "fwd", None)
    _FA3_ERROR = None
except Exception as exc:  # pragma: no cover - depends on the host install
    _fa3_forward = None
    _fa3_direct_forward = None
    _FA3_ERROR = f"{type(exc).__name__}: {exc}"


def _fa3_available() -> bool:
    return callable(_fa3_forward) and callable(_fa3_direct_forward)


def shape_seed(base_seed: int, mode: str, batch: int, seq_len: int, num_kv_heads: int) -> int:
    """Derive a stable per-shape seed without relying on Python's hash()."""
    mode_offset = 0 if mode == "full" else 1_000_003
    return (base_seed + mode_offset + batch * 1_009 + seq_len * 9_176 + num_kv_heads * 65_537) % (2**63 - 1)


def make_inputs(
    batch: int,
    seq_len: int,
    dtype: torch.dtype,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
    block_size: int,
    *,
    seed: int,
    page_table: str,
    device: str,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    query = torch.randn(
        batch,
        num_query_heads,
        head_size_qk,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    # Preserve production packed HND storage, exposed as strided NHD views.
    kv_cache = torch.randn(
        batch * blocks_per_seq,
        num_kv_heads,
        block_size,
        head_size_qk + head_size_v,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    kv_cache_nhd = kv_cache.transpose(1, 2)
    key_cache = kv_cache_nhd[..., :head_size_qk]
    value_cache = kv_cache_nhd[..., head_size_qk:]
    if page_table == "identity":
        physical_pages = torch.arange(batch * blocks_per_seq, device=device, dtype=torch.int32)
    else:
        physical_pages = torch.randperm(
            batch * blocks_per_seq,
            device=device,
            dtype=torch.int64,
            generator=generator,
        ).to(torch.int32)
    block_tables = physical_pages.reshape(batch, blocks_per_seq)
    context_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return query, key_cache, value_cache, context_lens, block_tables


def build_runner(inputs, path: str, window_size: int, diffkv_impl):
    """Build a runner with the unified TLE/standard backend dispatcher."""
    query, key_cache, value_cache, context_lens, block_tables = inputs
    if test_backend.has_specialization("diffkv_attention"):
        from flag_attn import diffkv_attention

        def run_public():
            return diffkv_attention(*inputs, window_size=window_size, path=path)

        return run_public, f"{test_backend.runtime.device.vendor_name} public API (full call)"
    batch, hq, dqk = query.shape
    hkv = key_cache.shape[2]
    dv = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    scale = dqk**-0.5
    if path not in SUPPORTED_PATHS:
        raise ValueError(f"unsupported path: {path}")
    use_3d = path == "3d"

    num_q_per_kv = hq // hkv
    block_m = LAUNCH.query_block_m if num_q_per_kv <= LAUNCH.query_block_m else triton.next_power_of_2(num_q_per_kv)
    block_q = block_m // num_q_per_kv
    total_num_q_blocks = query.shape[0] // block_q + batch
    properties = test_backend.device_fn.get_device_properties(query.device)
    num_sms = getattr(properties, "multi_processor_count", None)

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
            num_segments = diffkv_impl.get_num_par_softmax_segments(seq_len, batch, True)
        padded_v = triton.next_power_of_2(value_cache.shape[-1])
        segm_output = torch.empty(
            batch,
            hq,
            num_segments,
            padded_v,
            device=test_backend.device,
            dtype=query.dtype if use_optimized else torch.float32,
        )
        segm_max = torch.empty(batch, hq, num_segments, device=test_backend.device, dtype=torch.float32)
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None

    triton_window = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    out = torch.empty(batch, hq, value_cache.shape[-1], device=test_backend.device, dtype=query.dtype)
    extra_kwargs = {}
    if (
        use_optimized
        and use_3d
        and diffkv_impl.should_use_tle_fused_reducer(dqk, dv, 1, seq_len, batch, block_size, True)
    ):
        extra_kwargs["fused_reducer_counter"] = torch.zeros(batch * hq, device=test_backend.device, dtype=torch.int32)

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


def _build_fa3_runner(inputs, window_size: int):
    """Build one preallocated FA3 paged-cache callable, if FA3 is installed."""
    if not _fa3_available():
        return None

    query, key_cache, value_cache, context_lens, block_tables = inputs
    batch, _, head_size_qk = query.shape
    head_size_v = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    query_4d = query.unsqueeze(1)
    output = torch.empty(
        batch,
        1,
        DEFAULT_LAYOUT.num_query_heads,
        head_size_v,
        device=query.device,
        dtype=query.dtype,
    )
    left, right = (window_size - 1, 0) if window_size > 0 else (-1, -1)

    def run():
        _fa3_direct_forward(
            query_4d,
            key_cache,
            value_cache,
            out=output,
            seqused_k=context_lens,
            max_seqlen_q=1,
            max_seqlen_k=seq_len,
            page_table=block_tables,
            softmax_scale=head_size_qk**-0.5,
            is_causal=True,
            window_size_left=left,
            window_size_right=right,
            num_splits=0,
            is_rotary_interleaved=True,
        )

    return run


def _bench_ms(fn, warmup_ms: int, rep_ms: int, samples: int):
    """Measure a callable with the selected device timing helper.

    ``do_bench`` owns warmup, CUDA-event placement, synchronization, adaptive
    repetition count, and benchmark-cache clearing.  We invoke it once per
    sample and reduce each returned list of per-repetition millisecond values
    to a sample median.  The outer sample distribution is retained for the
    stability CV reported by this benchmark.
    """
    values = []
    for _ in range(samples):
        timings_ms = test_backend.do_bench(
            fn,
            warmup=warmup_ms,
            rep=rep_ms,
            return_mode="all",
        )
        if not isinstance(timings_ms, (list, tuple)):
            timings_ms = [timings_ms]
        values.append(statistics.median(float(value) for value in timings_ms))
    median = statistics.median(values)
    mean = statistics.fmean(values)
    cv_pct = 100.0 * statistics.pstdev(values) / mean if mean else 0.0
    return median, min(values), max(values), cv_pct


def _print_header(
    mode: str,
    scope: str,
    warmup: int,
    rep: int,
    samples: int,
    fa3_available: bool,
) -> None:
    title = "Full Attention" if mode == "full" else "SWA"
    print(f"\n{'=' * TABLE_WIDTH}")
    print(f"  diffkv_attention benchmark - {title}")
    print(f"  device={test_backend.get_device_name()} scope={scope}")
    baseline = "FA3 (flash_attn_interface)" if fa3_available else "unavailable"
    print(f"  warmup={warmup}ms rep={rep}ms samples={samples}; baseline: {baseline}")
    print(f"{'=' * TABLE_WIDTH}")
    print(
        f"{'B':>3} {'KV':>6} {'HQ':>4} {'HKV':>4} {'DQK':>4} {'DV':>4} "
        f"{'dtype':>8} {'FA3(ms)':>11} {'2D(ms)':>11} {'3D(ms)':>11} "
        f"{'speedup':>9} {'path':>5}"
    )


def _print_row(
    batch,
    length,
    kv_heads,
    dtype,
    fa3_measurement,
    path_measurements,
    best_path,
) -> None:
    fa3_latency = fa3_measurement[0] if fa3_measurement is not None else None
    path_latency = {
        path: measurement[0] for path, measurement in path_measurements.items()
    }
    best_latency = path_latency[best_path]
    speedup = fa3_latency / best_latency if fa3_latency is not None else None
    fa3_text = f"{fa3_latency:.6f}" if fa3_latency is not None else "n/a"
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "n/a"
    print(
        f"{batch:>3} {length:>6} {DEFAULT_LAYOUT.num_query_heads:>4} {kv_heads:>4} "
        f"{DEFAULT_LAYOUT.head_size_qk:>4} {DEFAULT_LAYOUT.head_size_v:>4} "
        f"{str(dtype).removeprefix('torch.'):>8} {fa3_text:>11} "
        f"{path_latency['2d']:>11.6f} {path_latency['3d']:>11.6f} "
        f"{speedup_text:>9} {best_path.upper():>5}",
        flush=True,
    )
    for path, measurement in path_measurements.items():
        cv_pct = measurement[3]
        if cv_pct > STABILITY_CV_THRESHOLD_PCT:
            print(
                f"[WARN unstable] {path.upper()} sample CV={cv_pct:.3f}% "
                f"exceeds {STABILITY_CV_THRESHOLD_PCT:.1f}%"
            )


@torch.inference_mode()
def run_benchmark(
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REP,
    record_property: Callable[[str, object], None] | None = None,
    *,
    samples: int = DEFAULT_SAMPLES,
) -> None:
    if not test_backend.supports_operator(OP_NAME):
        raise RuntimeError(test_backend.skip_reason(OP_NAME))
    if warmup < 0 or rep <= 0 or samples <= 0:
        raise ValueError("warmup must be non-negative; rep and samples must be positive")

    specialized = test_backend.has_specialization(OP_NAME)
    scope = "public_api_forward" if specialized else "preallocated_forward_kernel"
    mode_name = "api" if specialized else "kernel"
    fa3 = _fa3_available()
    if specialized:
        implementation_name = test_backend.runtime.device.vendor_name
    else:
        implementation_name = diffkv_impl.SELECTED_BACKEND
        print(f"DiffKV implementation: {implementation_name}")
    if fa3:
        print("FA3 baseline: flash_attn_interface (optional Python import)")
    else:
        print(f"FA3 baseline unavailable; FlagAttention-only ({_FA3_ERROR})")

    # Full and SWA are both decode workloads. Submit one record per dtype so
    # summary readers that group by op_name/dtype retain every measured row.
    recorders = {
        dtype: BenchmarkRecorder(
            record_property,
            op_name=OP_NAME,
            dtype=str(dtype),
            mode=mode_name,
            baseline="FA3" if fa3 else None,
            phase="decode",
            benchmark_scope=scope,
            vendor=test_backend.runtime.device.vendor_name,
        )
        for dtype in _DTYPES
    }

    for mode in _MODES:
        window = -1 if mode == "full" else WINDOW_SIZE
        kv_heads = diffkv_impl.num_kv_heads_for_mode(mode)
        _print_header(mode, scope, warmup, rep, samples, fa3)
        for dtype in _DTYPES:
            recorder = recorders[dtype]
            for batch, length in _SHAPES:
                inputs = make_inputs(
                    batch,
                    length,
                    dtype,
                    DEFAULT_LAYOUT.num_query_heads,
                    kv_heads,
                    DEFAULT_LAYOUT.head_size_qk,
                    DEFAULT_LAYOUT.head_size_v,
                    DEFAULT_LAYOUT.block_size,
                    seed=shape_seed(DEFAULT_SEED, mode, batch, length, kv_heads),
                    page_table=PAGE_TABLE,
                    device=test_backend.device,
                )
                fa3_runner = _build_fa3_runner(inputs, window) if fa3 else None
                fa3_measurement = (
                    _bench_ms(fa3_runner, warmup, rep, samples)
                    if fa3_runner is not None
                    else None
                )
                path_measurements = {}
                for path in _PATHS:
                    runner, _ = build_runner(inputs, path, window, diffkv_impl)
                    measurement = _bench_ms(runner, warmup, rep, samples)
                    latency, minimum, maximum, cv_pct = measurement
                    if not math.isfinite(latency) or latency <= 0:
                        raise RuntimeError(f"Invalid DiffKV latency for {mode} B={batch} KV={length} {path}")
                    path_measurements[path] = measurement
                    recorder.add(
                        shape_detail={
                            "mode": mode,
                            "batch": batch,
                            "query_length": 1,
                            "kv_length": length,
                            "num_query_heads": DEFAULT_LAYOUT.num_query_heads,
                            "num_kv_heads": kv_heads,
                            "head_size_qk": DEFAULT_LAYOUT.head_size_qk,
                            "head_size_v": DEFAULT_LAYOUT.head_size_v,
                            "path": path,
                        },
                        latency=latency,
                        latency_base=(
                            fa3_measurement[0] if fa3_measurement is not None else None
                        ),
                        min_ms=minimum,
                        max_ms=maximum,
                        sample_cv_pct=cv_pct,
                        implementation=implementation_name,
                    )
                best_path = min(path_measurements, key=lambda path: path_measurements[path][0])
                _print_row(
                    batch,
                    length,
                    kv_heads,
                    dtype,
                    fa3_measurement,
                    path_measurements,
                    best_path,
                )
                del inputs, runner, fa3_runner
        print("-" * TABLE_WIDTH)
    for recorder in recorders.values():
        recorder.record()
    print("\nDiffKV benchmark complete.")


@pytest.mark.diffkv_attention
@pytest.mark.skipif(not test_backend.supports_operator(OP_NAME), reason=test_backend.skip_reason(OP_NAME))
def test_diffkv_attention_benchmark(record_property) -> None:
    run_benchmark(record_property=record_property)
