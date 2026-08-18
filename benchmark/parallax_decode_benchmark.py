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

"""Benchmark the FlagAttention Parallax TLE decode kernel.

When the external ``parallax-kernel`` package with the ``[cutedsl]`` extra
(https://github.com/Yifei-Zuo/Parallax) is importable, its CuTe SM90 decode
baseline is benchmarked alongside FlagAttention for comparison; otherwise only
the FlagAttention kernel is timed.

Examples::

    CUDA_VISIBLE_DEVICES=0 python benchmark/parallax_decode_benchmark.py
    CUDA_VISIBLE_DEVICES=0 python benchmark/parallax_decode_benchmark.py \
        --shape 1,2048,8,8,128 --shape 2,8192,8,2,128 --dtype float16

Each shape is checked against the fp32 reference before it is timed. Providers
are captured into separate CUDA graphs with multiple calls per graph, then
timed by graph replay in mirrored provider order. The reported speedup is
``CuTe_ms / TLE_ms``; values above 1 mean TLE is faster.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics

import torch

from flag_attn.parallax import HAS_TLE, parallax_decode as parallax_decode_tle

try:
    import parallax as parallax_kernel

    if not parallax_kernel.cute_decode_available:
        raise ImportError(
            "parallax-kernel is installed without the [cutedsl] extra; "
            "install it with: pip install 'parallax-kernel[cutedsl]'"
        )
    from parallax import parallax_decode as parallax_decode_cute
except Exception as exc:  # parallax-kernel is an optional benchmark baseline.
    parallax_decode_cute = None
    CUTE_IMPORT_ERROR = exc
else:
    CUTE_IMPORT_ERROR = None


DEFAULT_SHAPES = [
    (1, 512, 8, 8, 64),
    (1, 2048, 8, 8, 128),
    (2, 2048, 8, 2, 128),
    (1, 8192, 8, 2, 128),
]


def _parse_shape(value: str):
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 5:
        raise argparse.ArgumentTypeError("shape must be B,L,HQ,HKV,D")
    return values


def _capture_graph(fn, calls_per_graph: int) -> torch.cuda.CUDAGraph:
    """Capture ``calls_per_graph`` stable-buffer operator invocations."""
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        # Finish lazy backend work before capture on this side stream.
        for _ in range(3):
            fn()
        capture_stream.synchronize()
        with torch.cuda.graph(graph, stream=capture_stream):
            for _ in range(calls_per_graph):
                fn()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    return graph


def _graph_sample_ms(graph: torch.cuda.CUDAGraph, calls_per_graph: int) -> float:
    """Time one graph replay and return per-operator device latency."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / calls_per_graph


def _median_and_mad_percent(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return median, 100.0 * mad / max(median, 1e-12)


def _sample_median_and_mad(
    graphs: dict[str, torch.cuda.CUDAGraph],
    calls_per_graph: int,
    samples: int,
) -> dict[str, tuple[float, float]]:
    """Replay captured graphs in mirrored order and report median and MAD%."""
    names = list(graphs)
    runs: dict[str, list[float]] = {name: [] for name in names}
    if len(names) == 1:
        # A single provider has no ordering bias to cancel out.
        name = names[0]
        runs[name] = [
            _graph_sample_ms(graphs[name], calls_per_graph) for _ in range(samples)
        ]
    else:
        first, second = names
        orders = (
            (first, second, second, first),
            (second, first, first, second),
        )
        round_id = 0
        while any(len(runs[name]) < samples for name in names):
            for provider in orders[round_id % 2]:
                if len(runs[provider]) < samples:
                    runs[provider].append(
                        _graph_sample_ms(graphs[provider], calls_per_graph)
                    )
            round_id += 1
    return {name: _median_and_mad_percent(values) for name, values in runs.items()}


def _rel_err(actual, expected):
    denominator = max(expected.float().abs().max().item(), 1e-6)
    return (actual.float() - expected.float()).abs().max().item() / denominator


def _decode_reference(q, r, k, v, scale, window_size_left=-1):
    """FP32 Q=1 oracle without the degenerate cuBLAS einsum path."""
    HQ, H = q.shape[2], k.shape[2]
    repeat = HQ // H
    qf = q[:, 0].float()
    rf = r[:, 0].float()
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    if repeat > 1:
        kf = kf.repeat_interleave(repeat, dim=1)
        vf = vf.repeat_interleave(repeat, dim=1)
    s1 = (qf[:, :, None, :] * kf).sum(dim=-1) * scale
    s2 = (rf[:, :, None, :] * kf).sum(dim=-1)
    if window_size_left >= 0:
        first = max(k.shape[1] - window_size_left, 0)
        valid = torch.arange(k.shape[1], device=k.device) >= first
        s1 = s1.masked_fill(~valid[None, None, :], float("-inf"))
    pivot = s1.amax(dim=-1, keepdim=True)
    pivot_safe = torch.where(torch.isfinite(pivot), pivot, torch.zeros_like(pivot))
    p1 = torch.exp(s1 - pivot_safe)
    p2 = p1 * s2
    d1 = p1.sum(dim=-1, keepdim=True)
    d2 = p2.sum(dim=-1, keepdim=True)
    o1 = (p1[..., None] * vf).sum(dim=2)
    o2 = (p2[..., None] * vf).sum(dim=2)
    inv_d1 = torch.where(d1 > 0, d1.reciprocal(), torch.zeros_like(d1))
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    return result[:, None].contiguous()


def _make_inputs(B, L, HQ, H, D, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator)
    r = torch.randn(B, 1, HQ, D, device="cuda", dtype=dtype, generator=generator) * 0.5
    k = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    v = torch.randn(B, L, H, D, device="cuda", dtype=dtype, generator=generator)
    return q, r, k, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=_parse_shape,
                        help="B,L,HQ,HKV,D; repeatable")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"),
                        default="bfloat16")
    parser.add_argument("--window", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--iterations", type=int, default=100,
        help="operator calls captured inside each CUDA graph",
    )
    parser.add_argument(
        "--samples", type=int, default=12,
        help="timed graph replays per provider",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1 or args.samples < 1:
        parser.error("warmup, iterations and samples must all be >= 1")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not HAS_TLE:
        raise SystemExit(
            "FlagTree TLE is unavailable: expected triton.experimental.tle.language"
        )
    run_cute = parallax_decode_cute is not None
    if run_cute and torch.cuda.get_device_capability()[0] != 9:
        raise SystemExit("the CuTe baseline requires an SM90 GPU")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    shapes = args.shape or DEFAULT_SHAPES
    print(f"GPU: {torch.cuda.get_device_name()} | dtype={dtype} | window={args.window}")
    print(
        f"mode=CUDA-graph, eager-warmup={args.warmup}, "
        f"calls/graph={args.iterations}, replays/provider={args.samples}"
    )
    if run_cute:
        print(
            "speedup = CuTe/TLE (>1 means TLE is faster); "
            "measurement_order = mirrored ABBA/BAAB"
        )
    else:
        print(
            "CuTe baseline unavailable; timing FlagAttention TLE only. Install "
            "'parallax-kernel[cutedsl]' from "
            "https://github.com/Yifei-Zuo/Parallax to enable the comparison. "
            f"Underlying import error: {CUTE_IMPORT_ERROR!r}"
        )
    print(
        "TLE config: "
        f"BN={os.environ.get('PARALLAX_TLE_BLOCK_N', 'auto')} "
        f"warps={os.environ.get('PARALLAX_TLE_NUM_WARPS', '4')} "
        f"stages={os.environ.get('PARALLAX_TLE_NUM_STAGES', '3')} "
        f"max_splits={os.environ.get('PARALLAX_TLE_MAX_SPLITS', '32')} "
        f"forced_splits={os.environ.get('PARALLAX_TLE_DECODE_SPLITS', 'auto')}"
    )
    if "PARALLAX_TLE_BLOCK_N" not in os.environ:
        print("TLE BN auto policy: BN=64 for D=128,L<=512; BN=128 otherwise")
    header = (
        f"{'B':>3} {'L':>7} {'HQ':>4} {'HKV':>4} {'D':>4} "
        f"{'TLE(ms)':>11} {'TLEMAD%':>9} {'TLE-ref':>10}"
    )
    if run_cute:
        header = (
            f"{'B':>3} {'L':>7} {'HQ':>4} {'HKV':>4} {'D':>4} "
            f"{'CuTe(ms)':>11} {'TLE(ms)':>11} {'speedup':>9} "
            f"{'CuTeMAD%':>9} {'TLEMAD%':>9} "
            f"{'CuTe-ref':>10} {'TLE-ref':>10} {'TLE-CuTe':>10}"
        )
    print(header)

    for case_id, (B, L, HQ, H, D) in enumerate(shapes):
        q, r, k, v = _make_inputs(B, L, HQ, H, D, dtype, seed=2026 + case_id)
        scale = 1.0 / math.sqrt(D)
        out_tle = torch.empty_like(q)
        tle_fn = lambda: parallax_decode_tle(
            q, r, k, v, scale, window_size_left=args.window, out=out_tle
        )
        out_cute = torch.empty_like(q) if run_cute else None
        cute_fn = (
            lambda: parallax_decode_cute(
                q, r, k, v, scale, window_size_left=args.window, out=out_cute
            )
            if run_cute
            else None
        )

        # Compile outside the timed region and reject numerically invalid rows.
        tle_fn()
        if run_cute:
            cute_fn()
        torch.cuda.synchronize()
        reference = _decode_reference(q, r, k, v, scale, args.window)
        tle_ref_error = _rel_err(out_tle, reference)
        errors = [tle_ref_error]
        if run_cute:
            cute_ref_error = _rel_err(out_cute, reference)
            cross_error = _rel_err(out_tle, out_cute)
            errors += [cute_ref_error, cross_error]
        if max(errors) >= 1e-2:
            raise RuntimeError(f"correctness check failed for {(B, L, HQ, H, D)}")

        # Warm every lazy path before capture. Stable out/workspace tensors are
        # supplied by the closures, so replay performs no allocator work.
        for _ in range(args.warmup):
            tle_fn()
            if run_cute:
                cute_fn()
        torch.cuda.synchronize()
        graphs = {"tle": _capture_graph(tle_fn, args.iterations)}
        if run_cute:
            graphs["cute"] = _capture_graph(cute_fn, args.iterations)
        for _ in range(3):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()

        timings = _sample_median_and_mad(graphs, args.iterations, args.samples)
        tle_ms, tle_mad = timings["tle"]
        row = (
            f"{B:3d} {L:7d} {HQ:4d} {H:4d} {D:4d} "
            f"{tle_ms:11.6f} {tle_mad:9.3f} {tle_ref_error:10.3e}"
        )
        if run_cute:
            cute_ms, cute_mad = timings["cute"]
            row = (
                f"{B:3d} {L:7d} {HQ:4d} {H:4d} {D:4d} "
                f"{cute_ms:11.6f} {tle_ms:11.6f} {cute_ms / tle_ms:8.3f}x "
                f"{cute_mad:9.3f} {tle_mad:9.3f} "
                f"{cute_ref_error:10.3e} {tle_ref_error:10.3e} {cross_error:10.3e}"
            )
        print(row)


if __name__ == "__main__":
    main()
