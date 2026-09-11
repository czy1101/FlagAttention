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

# The TileLang kernel below follows the NSA implementation originally published
# by MooreThreads and subsequently migrated to Tile-AI:
# Original repository: https://github.com/MooreThreads/tilelang_musa
# Original commit: 4b59fea16afce49d39d7bca0eb936a14bf472dd1
# Migrated repository: https://github.com/tile-ai/tilelang-musa
# Migrated commit: 475bf79063776fc268d01bb8cabde24a92d753f8
# Path: examples/deepseek_nsa/example_tilelang_nsa_bwd.py
#
# MIT License
#
# Copyright (c) Tile-AI.
# During the period from December 1, 2024, to Mar 14, 2025, this project is
# subject to additional collaboration terms with Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""TileLang-MUSA baseline for the selected-attention path of parallel NSA."""

from __future__ import annotations

import argparse

import torch
import triton

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from flag_attn.runtime.backend._mthreads.nsa import parallel_nsa
from flag_attn.runtime.backend._mthreads.nsa.mean_pooling import mean_pooling

try:
    import tilelang
    from tilelang import language as T
except (ImportError, OSError) as error:
    tilelang = None
    T = None
    _TILELANG_IMPORT_ERROR = error
else:
    _TILELANG_IMPORT_ERROR = None

if tilelang is not None:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_parallel_nsa_fwd(
        Q,
        K,
        V,
        BlockIndices,
        dim,
        is_causal,
        scale=None,
        block_size=64,
        groups=1,
        selected_blocks=16,
    ):
        batch, seq_len, heads = T.const("batch, seq_len, heads")

        if scale is None:
            scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
        else:
            scale = scale * 1.44269504  # log2(e)

        head_kv = heads // groups
        q_shape = [batch, seq_len, heads, dim]
        kv_shape = [batch, seq_len, head_kv, dim]
        o_slc_shape = [batch, seq_len, heads, dim]
        lse_slc_shape = [batch, seq_len, heads]
        block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
        block_indices_dtype = T.int32
        dtype = T.float16
        accum_dtype = T.float32
        block_S = block_size
        block_T = min(128, tilelang.math.next_power_of_2(dim))

        NK = tilelang.cdiv(dim, block_T)
        NV = tilelang.cdiv(dim, block_T)
        assert NK == 1, "The key dimension can not be larger than 256"

        S = selected_blocks
        G = groups
        BS = block_S
        BK = BV = block_T
        num_stages = 0
        threads = 32

        Q: T.Tensor(q_shape, dtype)
        K: T.Tensor(kv_shape, dtype)
        V: T.Tensor(kv_shape, dtype)
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype)
        O_slc = T.empty(o_slc_shape, dtype)
        LSE_slc = T.empty(lse_slc_shape, accum_dtype)

        with T.Kernel(seq_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)
            # MP31 FMA cannot lower the official fragment x shared PV GEMM.
            # Preserve the FP16 cast, but materialize it in shared memory.
            acc_s_cast_shared = T.alloc_shared([G, BS], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, i_t, i_h, i] * BS
                if i_s <= i_t and i_s >= 0:
                    # [BS, BK]
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    if is_causal:
                        for k, j in T.Parallel(G, BS):
                            acc_s[k, j] = T.if_then_else(
                                i_t >= (i_s + j),
                                0,
                                -T.infinity(acc_s.dtype),
                            )
                    else:
                        T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    # Softmax
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)
                    for k in T.Parallel(G):
                        scores_scale[k] = T.exp2(scores_max_prev[k] * scale - scores_max[k] * scale)
                    for k, j in T.Parallel(G, BS):
                        acc_s[k, j] = T.exp2(acc_s[k, j] * scale - scores_max[k] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for k in T.Parallel(G):
                        logsum[k] = logsum[k] * scores_scale[k] + scores_sum[k]
                    T.copy(acc_s, acc_s_cast_shared)

                    # Rescale
                    for k, j in T.Parallel(G, BV):
                        acc_o[k, j] *= scores_scale[k]

                    # V * softmax(Q * K)
                    T.copy(
                        V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV],
                        V_shared,
                    )
                    T.gemm(
                        acc_s_cast_shared,
                        V_shared,
                        acc_o,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(
                O_shared,
                O_slc[
                    i_b,
                    i_t,
                    i_h * G : (i_h + 1) * G,
                    i_v * BV : (i_v + 1) * BV,
                ],
            )
            for i in T.Parallel(G):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(
                logsum,
                LSE_slc[i_b, i_t, i_h * G : (i_h + 1) * G],
            )

        return O_slc, LSE_slc

else:
    _tilelang_parallel_nsa_fwd = None


def is_available() -> bool:
    """Return whether TileLang-MUSA imported successfully."""

    return _tilelang_parallel_nsa_fwd is not None


def unavailable_reason() -> str:
    if _TILELANG_IMPORT_ERROR is None:
        return "TileLang-MUSA is unavailable"
    return f"TileLang-MUSA is unavailable: {_TILELANG_IMPORT_ERROR}"


def _validate_inputs(
    q,
    k,
    v,
    g_cmp,
    g_slc,
    g_swa,
    block_indices,
    block_counts,
    block_size,
    window_size,
    cu_seqlens,
):
    if not is_available():
        raise RuntimeError(unavailable_reason())
    if any(gate is not None for gate in (g_cmp, g_slc, g_swa)):
        raise NotImplementedError("TileLang NSA baseline does not support gates")
    if window_size != 0:
        raise NotImplementedError("TileLang NSA baseline does not support sliding-window attention")
    if cu_seqlens is not None:
        raise NotImplementedError("TileLang NSA baseline does not support varlen")
    if not isinstance(block_counts, int):
        raise NotImplementedError("TileLang NSA baseline requires a fixed integer block_counts")
    if block_indices is None:
        raise ValueError("block_indices must be provided")
    if q.device.type != "musa":
        raise ValueError("TileLang NSA baseline requires MUSA tensors")
    if any(tensor.device != q.device for tensor in (k, v, block_indices)):
        raise ValueError("all inputs must be on the same MUSA device")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k and v must have matching B, T and KV-head axes")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise NotImplementedError("TileLang NSA baseline requires K == V")
    if q.shape[-1] > 128:
        raise NotImplementedError("TileLang NSA baseline supports D <= 128")
    if q.shape[2] % (k.shape[2] * 16) != 0:
        raise ValueError("query/KV group size must be a multiple of 16")
    if q.dtype != torch.float16:
        raise TypeError("The official TileLang-MUSA NSA baseline supports float16")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must have the same dtype")
    if block_indices.dtype != torch.int32:
        raise TypeError("block_indices must have dtype int32")
    if block_indices.shape != (
        q.shape[0],
        q.shape[1],
        k.shape[2],
        block_counts,
    ):
        raise ValueError("block_indices shape does not match q, k and block_counts")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not all(tensor.is_contiguous() for tensor in (q, k, v, block_indices)):
        raise ValueError("TileLang NSA baseline requires contiguous inputs")


def parallel_nsa_tilelang_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    block_counts: torch.Tensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the TileLang-MUSA selected-attention benchmark baseline."""

    _validate_inputs(
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        block_indices,
        block_counts,
        block_size,
        window_size,
        cu_seqlens,
    )
    if scale is None:
        scale = q.shape[-1] ** -0.5

    # Match the current public parallel_nsa path, which performs these two
    # pooling kernels even when compression and automatic top-k are disabled.
    k_cmp = mean_pooling(k, block_size, None)
    v_cmp = mean_pooling(v, block_size, None)

    output, _ = _tilelang_parallel_nsa_fwd(
        q,
        k,
        v,
        block_indices,
        dim=q.shape[-1],
        is_causal=True,
        scale=float(scale),
        block_size=block_size,
        groups=q.shape[2] // k.shape[2],
        selected_blocks=block_counts,
    )

    # Keep the unused pooled tensors alive until selected attention completes,
    # matching the lifetime in the public FlagAttention wrapper.
    del k_cmp, v_cmp
    return output


MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()

# Same default coverage as the original FlagGems NSA benchmark.
DEFAULT_DTYPES = ("float16",)
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
    parser = argparse.ArgumentParser(description="Benchmark MThreads parallel_nsa")
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
    parser.add_argument("--selected-blocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=DEFAULT_DTYPES,
        default=None,
        help="override the original dtype list (default: float16)",
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
    baseline_group = parser.add_mutually_exclusive_group()
    baseline_group.add_argument(
        "--tilelang-baseline",
        dest="tilelang_baseline",
        action="store_true",
        help="force the official TileLang-MUSA kernel as the baseline",
    )
    baseline_group.add_argument(
        "--no-tilelang-baseline",
        dest="tilelang_baseline",
        action="store_false",
        help="use FlagAttention self-comparison instead of TileLang baseline",
    )
    # Match the original benchmark: automatically use TileLang when available,
    # otherwise fall back to a FlagAttention self-comparison.
    parser.set_defaults(tilelang_baseline=None)
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


def _make_block_indices(B, T, H, S, block_size, device):
    n_blocks = triton.cdiv(T, block_size)
    indices = torch.randint(0, max(1, n_blocks), (B, T, H, S), device=device, dtype=torch.int32)
    available = torch.div(
        torch.arange(T, device=device, dtype=torch.int32) + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).clamp_min_(1)
    indices = torch.remainder(indices, available[None, :, None, None])
    valid_counts = available.clamp_max(S)
    slots = torch.arange(S, device=device)[None, None, None, :]
    return indices.masked_fill(slots >= valid_counts[None, :, None, None], -1)


def benchmark(args, print_output=True):
    if not MUSA_AVAILABLE:
        raise RuntimeError("parallel_nsa benchmark requires an available MUSA device")
    if args.tilelang_baseline is True and not is_available():
        raise RuntimeError(unavailable_reason())

    if args.tilelang_baseline is False:
        # Match the original benchmark fallback when TileLang is unavailable:
        # use the same FlagAttention operator in both benchmark slots.
        baseline_op = parallel_nsa
        baseline_name = "FlagAttention self-comparison"
    elif is_available():
        baseline_op = parallel_nsa_tilelang_baseline
        baseline_name = "TileLang-MUSA"
    else:
        # This is the same fallback used by the original Benchmark-based
        # driver when TileLang is unavailable or cannot be imported.
        baseline_op = parallel_nsa
        baseline_name = "FlagAttention self-comparison"

    device = torch.device("musa")
    S, BS = args.selected_blocks, args.block_size
    results = []

    for dtype_name in _dtype_configs(args):
        dtype = getattr(torch, dtype_name)
        for B, T, H, HQ, D in _shape_configs(args):
            q = torch.randn(B, T, HQ, D, device=device, dtype=dtype)
            k = torch.randn(B, T, H, D, device=device, dtype=dtype)
            v = torch.randn(B, T, H, D, device=device, dtype=dtype)
            block_indices = _make_block_indices(B, T, H, S, BS, device)
            call_kwargs = {
                "block_indices": block_indices,
                "block_counts": S,
                "block_size": BS,
                "scale": D**-0.5,
            }

            baseline_ms = None
            if baseline_op is not None:
                baseline_ms = triton.testing.do_bench(
                    lambda: baseline_op(q, k, v, **call_kwargs),
                    warmup=args.warmup,
                    rep=args.rep,
                    return_mode="median",
                )
            latency_ms = triton.testing.do_bench(
                lambda: parallel_nsa(q, k, v, **call_kwargs),
                warmup=args.warmup,
                rep=args.rep,
                return_mode="median",
            )

            result = {
                "shape": (B, T, H, HQ, D),
                "dtype": dtype_name,
                "latency_ms": latency_ms,
                "baseline_ms": baseline_ms,
                "speedup": None if baseline_ms is None else baseline_ms / latency_ms,
                # Keep the old key for callers that expect the TileLang name.
                "tilelang_ms": baseline_ms,
            }
            results.append(result)
            if print_output:
                message = (
                    f"device={device} shape={result['shape']} selected_blocks={S} "
                    f"dtype={dtype_name} latency_ms={latency_ms:.4f}"
                )
                if baseline_ms is not None:
                    message += (
                        f" baseline={baseline_name} baseline_ms={baseline_ms:.4f} "
                        f"speedup={result['speedup']:.3f}x"
                    )
                print(message)
    return results


if __name__ == "__main__":
    benchmark(parse_args())
