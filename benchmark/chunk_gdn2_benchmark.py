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

"""Benchmark GDN2 against the original native Triton implementation."""

from __future__ import annotations

import argparse
import importlib
import math
import os

import torch
import triton

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from flag_attn.runtime.backend._mthreads.gdn2 import chunk_gdn2
from flag_attn.runtime.backend._mthreads.gdn2.gdn2_native.chunk_fwd import (
    chunk_gdn2_fwd,
)


MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()
FORCE_NATIVE_ENV = "FLAGGEMS_VLLM_GDN2_FORCE_NATIVE"

DEFAULT_SHAPES = [
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    (2, 2048, 16, 256, 512),
    (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]


def _native_gdn2_op(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_size=64,
    **kwargs,
):
    del kwargs
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.inference_mode():
        (
            o,
            final_state,
            _g,
            _Aqk,
            _Akk,
            _w_wy,
            _u_wy,
            _qg,
            _kg,
            _v_new,
            _h,
            _initial_state,
        ) = chunk_gdn2_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w_gate=w,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=True,
            state_v_first=state_v_first,
        )
    return o, final_state


def _public_gdn2_op(*args, **kwargs):
    force_native = os.environ.get(FORCE_NATIVE_ENV, "0") == "1"
    call_kwargs = dict(kwargs)
    if not force_native:
        # The public fused TLE path uses BT=16; the native baseline uses BT=64.
        call_kwargs["chunk_size"] = 16
    module = importlib.import_module(
        "flag_attn.runtime.backend._mthreads.gdn2.chunk_gdn2"
    )
    old = module.HAS_TLE_GDN2
    if force_native:
        module.HAS_TLE_GDN2 = False
    try:
        with torch.inference_mode():
            return chunk_gdn2(*args, **call_kwargs)
    finally:
        module.HAS_TLE_GDN2 = old


def _build_inputs(B, T, H, K, V, dtype):
    device = torch.device("musa")
    scale = K**-0.5
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)
    initial_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.01
    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_gate_in_kernel": False,
        "safe_gate": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
        "chunk_size": 64,
    }
    return (q, k, v, g, b, w), kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MThreads chunk_gdn2")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "both"),
        default="both",
        help="benchmark dtype; default: both, matching the source benchmark",
    )
    parser.add_argument(
        "--shape-index",
        type=int,
        nargs="+",
        default=None,
        help="run selected DEFAULT_SHAPES indices; default: all source cases",
    )
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument(
        "--iter",
        "--rep",
        dest="rep",
        type=int,
        default=100,
        help="number of benchmark repetitions",
    )
    return parser.parse_args()


def benchmark(args, print_output=True):
    if not MUSA_AVAILABLE:
        raise RuntimeError("chunk_gdn2 benchmark requires an available MUSA device")

    dtypes = (
        [torch.float16, torch.bfloat16]
        if args.dtype == "both"
        else [getattr(torch, args.dtype)]
    )
    if args.shape_index is None:
        shapes = DEFAULT_SHAPES
    else:
        shapes = [DEFAULT_SHAPES[i] for i in args.shape_index]

    results = []
    module = importlib.import_module(
        "flag_attn.runtime.backend._mthreads.gdn2.chunk_gdn2"
    )
    force_native = os.environ.get(FORCE_NATIVE_ENV, "0") == "1"
    backend = "native-triton" if force_native or not module.HAS_TLE_GDN2 else "tle"

    for dtype in dtypes:
        for B, T, H, K, V in shapes:
            inputs, kwargs = _build_inputs(B, T, H, K, V, dtype)

            def baseline_run():
                return _native_gdn2_op(*inputs, **kwargs)

            def optimized_run():
                return _public_gdn2_op(*inputs, **kwargs)

            baseline_ms = triton.testing.do_bench(
                baseline_run,
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
            latency_ms = triton.testing.do_bench(
                optimized_run,
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
            result = {
                "shape": (B, T, H, K, V),
                "dtype": str(dtype).removeprefix("torch."),
                "backend": backend,
                "latency_ms": latency_ms,
                "baseline_ms": baseline_ms,
                "speedup": baseline_ms / latency_ms,
            }
            results.append(result)
            if print_output:
                print(
                    f"device=musa shape={result['shape']} "
                    f"dtype={result['dtype']} backend={backend} "
                    f"latency_ms={latency_ms:.4f} "
                    f"baseline=Native-Triton baseline_ms={baseline_ms:.4f} "
                    f"speedup={result['speedup']:.3f}x"
                )
    return results


if __name__ == "__main__":
    benchmark(parse_args())
