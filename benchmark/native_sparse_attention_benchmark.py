# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Forward-only MetaX NSA absolute-latency benchmark."""

import os
import statistics
from dataclasses import dataclass

import torch

from flag_attn.runtime.backend._metax import (
    parallel_nsa,
    parallel_nsa_compression,
)


@dataclass(frozen=True)
class Shape:
    batch: int
    tokens: int
    query_heads: int
    kv_heads: int
    dim: int


COMPREHENSIVE_SHAPES = (
    Shape(1, 16384, 64, 4, 64),
    Shape(1, 8192, 256, 16, 64),
    Shape(1, 16384, 256, 16, 64),
    Shape(1, 65536, 256, 16, 64),
    Shape(1, 16384, 512, 32, 64),
    Shape(1, 16384, 256, 16, 128),
    Shape(4, 8192, 256, 16, 64),
)

SMOKE_SHAPES = (Shape(1, 128, 16, 1, 64),)

BLOCK_SIZE = 64
SELECTED_BLOCKS = 16
WARMUP = int(os.environ.get("NSA_BENCH_WARMUP", "10"))
CALLS = int(os.environ.get("NSA_BENCH_CALLS", "5"))
SAMPLES = int(os.environ.get("NSA_BENCH_SAMPLES", "30"))

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def selected_block_indices(shape: Shape) -> torch.Tensor:
    current = (
        torch.arange(shape.tokens, device="cuda", dtype=torch.long)
        // BLOCK_SIZE
    )
    history = torch.arange(
        SELECTED_BLOCKS - 1,
        -1,
        -1,
        device="cuda",
        dtype=torch.long,
    )
    indices = (current[:, None] - history[None, :]).clamp_min_(0)
    return (
        indices[None, :, None, :]
        .expand(
            shape.batch,
            shape.tokens,
            shape.kv_heads,
            SELECTED_BLOCKS,
        )
        .contiguous()
    )


def profile(function):
    with torch.no_grad():
        for _ in range(WARMUP):
            function()
        torch.cuda.synchronize()

        samples = []
        for _ in range(SAMPLES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(CALLS):
                function()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / CALLS)

    return {
        "min_ms": min(samples),
        "p50_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
    }


def benchmark_selected(shape: Shape, dtype: torch.dtype):
    q = torch.randn(
        shape.batch,
        shape.tokens,
        shape.query_heads,
        shape.dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        shape.batch,
        shape.tokens,
        shape.kv_heads,
        shape.dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    indices = selected_block_indices(shape)
    scale = shape.dim**-0.5

    def run():
        return parallel_nsa(
            q=q,
            k=k,
            v=v,
            block_indices=indices,
            block_counts=SELECTED_BLOCKS,
            block_size=BLOCK_SIZE,
            scale=scale,
            cu_seqlens=None,
        )

    with torch.no_grad():
        output = run()
    torch.cuda.synchronize()
    assert tuple(output.shape) == (
        shape.batch,
        shape.tokens,
        shape.query_heads,
        shape.dim,
    )
    assert torch.isfinite(output).all()

    return profile(run)


def benchmark_compression(shape: Shape, dtype: torch.dtype):
    compressed_tokens = (
        shape.tokens + BLOCK_SIZE - 1
    ) // BLOCK_SIZE

    q = torch.randn(
        shape.batch,
        shape.tokens,
        shape.query_heads,
        shape.dim,
        device="cuda",
        dtype=dtype,
    )
    k = torch.randn(
        shape.batch,
        compressed_tokens,
        shape.kv_heads,
        shape.dim,
        device="cuda",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    scale = shape.dim**-0.5

    def run():
        return parallel_nsa_compression(
            q=q,
            k=k,
            v=v,
            block_size=BLOCK_SIZE,
            scale=scale,
            cu_seqlens=None,
        )

    with torch.no_grad():
        output, lse = run()
    torch.cuda.synchronize()
    assert tuple(output.shape) == (
        shape.batch,
        shape.tokens,
        shape.query_heads,
        shape.dim,
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()

    return profile(run)


def main():
    smoke = os.environ.get("NSA_BENCH_SMOKE", "0") == "1"
    shapes = SMOKE_SHAPES if smoke else COMPREHENSIVE_SHAPES

    dtype_names = tuple(
        name.strip()
        for name in os.environ.get(
            "NSA_BENCH_DTYPES",
            "float16,bfloat16",
        ).split(",")
        if name.strip()
    )
    dtypes = tuple((name, DTYPE_MAP[name]) for name in dtype_names)

    print("device:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print("mode: forward-only adapted MetaX NSA self benchmark")
    print(
        "timing:",
        f"CUDA Event, warmup={WARMUP}, "
        f"calls/sample={CALLS}, samples={SAMPLES}",
    )
    print(
        "operator,dtype,B,T,HQ,Hkv,D,min_ms,p50_ms,mean_ms"
    )

    for operator_name, benchmark in (
        ("parallel_nsa", benchmark_selected),
        ("parallel_nsa_compression", benchmark_compression),
    ):
        for dtype_name, dtype in dtypes:
            for shape in shapes:
                result = benchmark(shape, dtype)
                print(
                    f"{operator_name},{dtype_name},"
                    f"{shape.batch},{shape.tokens},"
                    f"{shape.query_heads},{shape.kv_heads},"
                    f"{shape.dim},"
                    f"{result['min_ms']:.6f},"
                    f"{result['p50_ms']:.6f},"
                    f"{result['mean_ms']:.6f}",
                    flush=True,
                )

    print("FLAGATTENTION_METAX_NSA_SELF_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
