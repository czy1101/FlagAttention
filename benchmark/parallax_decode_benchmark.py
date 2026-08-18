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

Examples::

    CUDA_VISIBLE_DEVICES=0 python benchmark/parallax_decode_benchmark.py
    CUDA_VISIBLE_DEVICES=0 python benchmark/parallax_decode_benchmark.py \
        --shape 1,2048,8,8,128 --shape 2,8192,8,2,128 --dtype float16

Each shape is checked against the fp32 reference before it is timed. The local
FlagAttention TLE operator is captured into a CUDA graph with multiple calls
per graph, then timed by graph replay. No external operator package is
imported.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics

import torch

from flag_attn.parallax.decode import (
    HAS_TLE,
    parallax_decode as parallax_decode_tle,
)


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
    graph: torch.cuda.CUDAGraph,
    calls_per_graph: int,
    samples: int,
) -> tuple[float, float]:
    """Replay the local TLE graph and report median latency and MAD%."""
    values = [
        _graph_sample_ms(graph, calls_per_graph)
        for _ in range(samples)
    ]
    return _median_and_mad_percent(values)


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
        help="timed graph replays for the local TLE operator",
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
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    shapes = args.shape or DEFAULT_SHAPES
    print(f"GPU: {torch.cuda.get_device_name()} | dtype={dtype} | window={args.window}")
    print(
        f"mode=CUDA-graph, eager-warmup={args.warmup}, "
        f"calls/graph={args.iterations}, replays={args.samples}"
    )
    print("provider=FlagAttention TLE; reference=local fp32 PyTorch oracle")
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
    print(
        f"{'B':>3} {'L':>7} {'HQ':>4} {'HKV':>4} {'D':>4} "
        f"{'TLE(ms)':>11} {'TLEMAD%':>9} {'TLE-ref':>10}"
    )

    for case_id, (B, L, HQ, H, D) in enumerate(shapes):
        q, r, k, v = _make_inputs(B, L, HQ, H, D, dtype, seed=2026 + case_id)
        scale = 1.0 / math.sqrt(D)
        out_tle = torch.empty_like(q)
        tle_fn = lambda: parallax_decode_tle(
            q, r, k, v, scale, window_size_left=args.window, out=out_tle
        )

        # Compile outside the timed region and reject numerically invalid rows.
        tle_fn()
        torch.cuda.synchronize()
        reference = _decode_reference(q, r, k, v, scale, args.window)
        tle_ref_error = _rel_err(out_tle, reference)
        if tle_ref_error >= 1e-2:
            raise RuntimeError(f"correctness check failed for {(B, L, HQ, H, D)}")

        # Warm every lazy path before capture. Stable out/workspace tensors are
        # supplied by the closures, so replay performs no allocator work.
        for _ in range(args.warmup):
            tle_fn()
        torch.cuda.synchronize()
        graph = _capture_graph(tle_fn, args.iterations)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()

        tle_ms, tle_mad = _sample_median_and_mad(
            graph, args.iterations, args.samples
        )
        print(
            f"{B:3d} {L:7d} {HQ:4d} {H:4d} {D:4d} "
            f"{tle_ms:11.6f} {tle_mad:9.3f} {tle_ref_error:10.3e}"
        )


if __name__ == "__main__":
    main()
