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

"""Compare native Triton and MetaX TLE GDN2 inference latency."""

from __future__ import annotations

import argparse
import importlib
import math
from functools import partial

import torch
import triton
import triton.knobs

from flag_attn.runtime.backend._metax import chunk_gdn2
from flag_attn.runtime.backend._metax.gdn2.native.chunk_fwd import chunk_gdn2_fwd


SHAPES = (
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
)


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _make_inputs(shape: tuple[int, int, int, int, int], dtype: torch.dtype):
    batch, seq_len, heads, key_dim, value_dim = shape
    q = torch.randn(
        (batch, seq_len, heads, key_dim), device="cuda", dtype=dtype
    ).div_(math.sqrt(key_dim))
    k = torch.randn_like(q).div_(math.sqrt(key_dim))
    v = torch.randn(
        (batch, seq_len, heads, value_dim), device="cuda", dtype=dtype
    )
    g = (-torch.rand(q.shape, device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    b = torch.rand(q.shape, device="cuda", dtype=dtype)
    w = torch.rand(v.shape, device="cuda", dtype=dtype)
    initial_state = torch.randn(
        (batch, heads, key_dim, value_dim),
        device="cuda",
        dtype=torch.float32,
    ).mul_(0.01)
    return q, k, v, g, b, w, initial_state


def _native(inputs, scale: float):
    q, k, v, g, b, w, initial_state = inputs
    result = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        safe_gate=False,
        use_gate_in_kernel=False,
        disable_recompute=True,
        state_v_first=False,
    )
    return result[0], result[1]


def _tle(inputs, scale: float):
    q, k, v, g, b, w, initial_state = inputs
    return chunk_gdn2(
        q,
        k,
        v,
        g,
        b,
        w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=16,
        safe_gate=False,
        use_gate_in_kernel=False,
        state_v_first=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-compatible device.")
    triton.knobs.autotuning.adjust_block_size = False
    gdn2_module = importlib.import_module(
        "flag_attn.runtime.backend._metax.gdn2.chunk_gdn2"
    )
    if not gdn2_module.HAS_TLE_GDN2:
        raise RuntimeError("This benchmark requires FlagTree 3.6+ with TLE GDN2.")
    torch.manual_seed(0)

    for dtype in (torch.float16, torch.bfloat16):
        print(f"\n### dtype={dtype}")
        print("| Shape (B,T,H,K,V) | Native Triton (ms) | MetaX TLE (ms) | Speedup |")
        print("| :---: | ---: | ---: | ---: |")
        for shape in SHAPES:
            inputs = _make_inputs(shape, dtype)
            scale = shape[3] ** -0.5
            with torch.inference_mode():
                native_ms = _bench(
                    partial(_native, inputs, scale), args.warmup, args.rep
                )
                tle_ms = _bench(
                    partial(_tle, inputs, scale), args.warmup, args.rep
                )
            shape_text = "(" + ",".join(map(str, shape)) + ")"
            print(
                f"| {shape_text} | {native_ms:.6f} | {tle_ms:.6f} | "
                f"{native_ms / tle_ms:.3f} |"
            )
            del inputs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
