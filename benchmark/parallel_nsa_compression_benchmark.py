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

"""Benchmark the MThreads parallel NSA compression operator."""

import argparse

import torch
import triton

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from flag_attn.runtime.backend._mthreads.nsa import parallel_nsa_compression


MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()

# Same default coverage as the original FlagGems NSA benchmark.
DEFAULT_DTYPES = ("bfloat16", "float16")
DEFAULT_SHAPES = (
    (1, 16384, 4, 64, 64),
    (1, 8192, 16, 256, 64),
    (1, 16384, 16, 256, 64),
    (1, 65536, 16, 256, 64),
    (1, 16384, 32, 512, 64),
    (1, 16384, 16, 256, 128),
    (4, 8192, 16, 256, 64),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark MThreads parallel_nsa_compression")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="override the original shape table with these sequence lengths",
    )
    parser.add_argument("--num-kv-heads", type=int, default=16)
    parser.add_argument("--num-query-heads", type=int, default=256)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=DEFAULT_DTYPES,
        default=None,
        help="override the original dtype list (default: bfloat16 and float16)",
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


def _shape_configs(args):
    if args.seq_lens is not None:
        return [
            (args.batch_size, seq_len, args.num_kv_heads, args.num_query_heads, args.head_dim)
            for seq_len in args.seq_lens
        ]
    return DEFAULT_SHAPES


def _dtype_configs(args):
    return (args.dtype,) if args.dtype is not None else DEFAULT_DTYPES


def benchmark(args, print_output=True):
    if not MUSA_AVAILABLE:
        raise RuntimeError("parallel_nsa_compression benchmark requires an available MUSA device")

    device = torch.device("musa")
    BS = args.block_size
    results = []
    baseline_name = "FlagAttention self-comparison"

    for dtype_name in _dtype_configs(args):
        dtype = getattr(torch, dtype_name)
        for B, T, H, HQ, D in _shape_configs(args):
            TC = triton.cdiv(T, BS)
            q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
            k = torch.randn(B, TC, H, D, device=device, dtype=dtype)
            v = torch.randn(B, TC, H, D, device=device, dtype=dtype)

            def run():
                return parallel_nsa_compression(q, k, v, block_size=BS, scale=D**-0.5)

            # The original compression benchmark uses the same operator for
            # both benchmark slots because it has no separate TileLang/Torch
            # baseline. Preserve that self-comparison and speedup metric.
            baseline_ms = triton.testing.do_bench(
                run,
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
            latency_ms = triton.testing.do_bench(
                run,
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )
            result = {
                "shape": (B, T, H, HQ, D),
                "dtype": dtype_name,
                "latency_ms": latency_ms,
                "baseline_ms": baseline_ms,
                "speedup": baseline_ms / latency_ms,
            }
            results.append(result)
            if print_output:
                print(
                    f"device={device} shape={result['shape']} block_size={BS} "
                    f"dtype={dtype_name} latency_ms={latency_ms:.4f} "
                    f"baseline={baseline_name} baseline_ms={baseline_ms:.4f} "
                    f"speedup={baseline_ms / latency_ms:.3f}x"
                )
    return results


if __name__ == "__main__":
    benchmark(parse_args())
