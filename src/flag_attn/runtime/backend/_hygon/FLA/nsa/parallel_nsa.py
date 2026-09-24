# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os
import warnings

import torch
import triton
import triton.language as tl

from flag_attn.runtime.backend import _hygon as runtime

from .bwd_preprocess import parallel_attn_bwd_preprocess
from ..index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_lens,
    prepare_token_indices,
)
from .mean_pooling import mean_pooling
from .parallel_nsa_compression import parallel_nsa_compression
from ..triton_ops_helper import autotune_cache_kwargs, exp, log
from ..utils import _bitonic_merge, check_shared_mem, input_guard


_BW1000_EARLY_V_ENV = "FLAGGEMS_NSA_BW1000_EARLY_V"
_BW1000_SKIP_UNUSED_POOLING_ENV = "FLAGGEMS_NSA_BW1000_SKIP_UNUSED_POOLING"
_TOPK_PACKED_ENV = "FLAGGEMS_NSA_TOPK_PACKED"
_TOPK_MERGE_ENV = "FLAGGEMS_NSA_TOPK_MERGE"
_TOPK_HIER_ENV = "FLAGGEMS_NSA_TOPK_HIER_MERGE"
_TOPK_PAIR_ENV = "FLAGGEMS_NSA_TOPK_PAIR"


def _use_bw1000_early_v() -> bool:
    """Enable the accepted exact-shape V prefetch on Hygon BW1000."""
    value = os.environ.get(_BW1000_EARLY_V_ENV, "1").strip().lower()
    return runtime.device.vendor_name == "hygon" and value not in {
        "0",
        "false",
        "no",
        "off",
    }


def _skip_unused_bw1000_pooling() -> bool:
    """Skip K/V compression when the public call cannot consume it.

    ``k_cmp`` and ``v_cmp`` are used only by the gated compression branch.
    Keeping this behind an environment switch makes the public end-to-end
    saving directly measurable against the legacy behavior from the same
    source file.
    """
    value = os.environ.get(_BW1000_SKIP_UNUSED_POOLING_ENV, "1").strip().lower()
    return runtime.device.vendor_name == "hygon" and value not in {
        "0",
        "false",
        "no",
        "off",
    }


@triton.jit
def _nsa_fp32_to_sort_key(x):
    """Map IEEE FP32 values to monotonically sortable unsigned keys."""
    sign_bit = tl.full(x.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(x.shape, 0xFFFFFFFF, dtype=tl.uint32)
    x_bits = x.to(tl.uint32, bitcast=True)
    return x_bits ^ tl.where((x_bits & sign_bit) != 0, full_mask, sign_bit)


@triton.jit
def _nsa_index_to_sort_key(index):
    # Complementing the block id makes lower ids win exact score ties.
    return tl.full(index.shape, 0xFFFF, dtype=tl.uint32) - index.to(tl.uint32)


try:
    HAS_TLE = False
    import triton.experimental.tle.language as tle  # noqa: F401

    HAS_TLE = True
except ImportError:
    tle = None
    HAS_TLE = False

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None


# ===========================================================================
# Top-K selection kernel
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_topk(
    q,
    k,
    lse,
    scale,
    block_indices,
    packed_candidates,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    PACKED: tl.constexpr,
    CANDIDATE_MODE: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        boc = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
        boc = i_b * tl.cdiv(T, BS)

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    # the number of compression representations in total
    TC = tl.cdiv(T, BS)
    # the number of compression representations required to iterate over
    # incomplete compression blocks are not included
    NC = (i_t + 1) // BS
    ################################
    # 1. lse computation
    ################################
    if lse is not None:
        b_lse = tl.load(lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G))
    else:
        # max scores for the current block
        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        # lse = log(acc) + m
        b_acc = tl.zeros([G], dtype=tl.float32)
        for i_c in range(0, NC, BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))

            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where((o_c < NC)[None, :], b_s, float("-inf"))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BC]
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            b_mp = b_m
        if NC == 0:
            b_lse = tl.zeros([G], dtype=tl.float32)
        else:
            b_lse = b_m + log(b_acc)

    ################################
    # 2. topk selection
    ################################
    IC = i_t // BS
    if PACKED:
        # Keep only S packed score/index keys per block tile.  Unlike the
        # full BC-wide bitonic network, this uses the backend's native top-k
        # primitive and merges S candidates across tiles.  The low 16 bits
        # encode the tie-broken block id; invalid lanes use zero, below every
        # valid non-negative importance score.
        acc = tl.zeros([S], dtype=tl.uint64)
        first = True
        num_tiles = tl.cdiv(IC + 1, BC)
        tile_start = (num_tiles - 1) * BC
        for tile_no in tl.range(0, num_tiles):
            o_c = tile_start + tl.arange(0, BC)
            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K,
                (K, TC),
                (1, H * K),
                (0, tile_start),
                (BK, BC),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where(o_c < IC, b_s, float("-inf"))
            b_p = tl.where(
                (o_c == 0) | ((o_c == IC - 1) | (o_c == IC)),
                1.0,
                exp(b_s - b_lse[:, None]),
            )
            b_score = tl.sum(b_p, 0)
            score_key = _nsa_fp32_to_sort_key(b_score)
            index_key = _nsa_index_to_sort_key(o_c)
            packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
            packed = tl.where(o_c <= IC, packed, tl.zeros_like(packed))
            tile_top = tl.topk(packed, S)
            if CANDIDATE_MODE:
                # Store per-tile candidates for an exact Python/Torch merge.
                # The top score-key bit is constant for non-negative scores;
                # remove it so the packed value is representable as int64.
                # Invalid lanes are initialized to the int64 minimum.
                signed_top = tile_top & tl.full(
                    tile_top.shape, 0x7FFFFFFFFFFFFFFF, dtype=tl.uint64
                )
                p_candidates = (
                    packed_candidates
                    + ((bos + i_t) * H + i_h) * NT * S
                    + tile_start // BC * S
                )
                tl.store(p_candidates + tl.arange(0, S), signed_top.to(tl.int64))
            else:
                if first:
                    acc = tile_top
                    first = False
                else:
                    # ``tl.maximum`` is not a set union: on Hygon its lane-wise
                    # merge loses candidates once more than one tile is present.
                    merged = tl.sort(
                        tl.reshape(tl.join(acc, tile_top), (2 * S,)), descending=True
                    )
                    pick = tl.arange(0, S)[:, None]
                    src = tl.arange(0, 2 * S)[None, :]
                    acc = tl.sum(
                        tl.where(src == pick, merged[None, :], tl.zeros_like(merged)[None, :]),
                        axis=1,
                    )
            tile_start -= BC
        if not CANDIDATE_MODE:
            # Rotate the index key into the high bits, sort by block id
            # ascending, and recover the original logical block index.
            acc = (acc << 48) | (acc >> 16)
            acc = tl.sort(acc, descending=True)
            b_top = (_nsa_index_to_sort_key(acc >> 48).to(tl.int32))
            b_top = tl.where(tl.arange(0, S) <= IC, b_top, -1)
    else:
        # [BC]
        b_i = tl.full([BC], -1, dtype=tl.float32)
        o_i = tl.zeros([BC], dtype=tl.int32)
        m_i = tl.arange(0, BC) < BC // 2

        for i_c in range(0, tl.cdiv(i_t + 1, BS), BC):
            o_c = i_c + tl.arange(0, BC)

            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K, (K, TC), (1, H * K), (0, i_c), (BK, BC), (0, 1)
            )
            # [BK, BC]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [G, BC]
            b_s = tl.dot(b_q, b_k)
            b_s = tl.where(o_c < IC, b_s, float("-inf"))
            # [G, BC]
            # the 1st and the last 2 blocks are always selected
            b_p = tl.where(
                (o_c == 0) | ((o_c == IC - 1) | (o_c == IC)), 1.0, exp(b_s - b_lse[:, None])
            )
            # the importance scores of the current block
            # [BC]
            b_i, b_ip = tl.sum(b_p, 0), b_i
            # blocks with index < 0 will be skipped
            o_i, o_ip = tl.where(o_c <= IC, o_c, -1), o_i

            n_dims: tl.constexpr = tl.standard._log2(b_i.shape[0])
            for i in tl.static_range(1, n_dims):
                b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), i, 2, n_dims)

            if i_c != 0:
                b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, False, n_dims)
                b_i_new = b_ip * m_i + b_i * (1 - m_i)
                o_i_new = o_ip * m_i + o_i * (1 - m_i)
                b_i, o_i = _bitonic_merge(
                    b_i_new, o_i_new.to(tl.int32), n_dims, True, n_dims
                )
            else:
                b_i, o_i = _bitonic_merge(b_i, o_i.to(tl.int32), n_dims, True, n_dims)

        m_top = tl.arange(0, BC // S) == 0
        b_top = tl.sum(m_top[:, None] * tl.reshape(o_i, [BC // S, S]), 0)

    if not CANDIDATE_MODE:
        p_b = tl.make_block_ptr(
            block_indices + (bos + i_t) * H * S,
            (H * S,),
            (1,),
            (i_h * S,),
            (S,),
            (0,),
        )
        tl.store(p_b, b_top.to(p_b.dtype.element_ty))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_topk_pair(
    q,
    k,
    lse,
    scale,
    packed_candidates,
    cu_seqlens,
    token_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    S: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    NT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Generate packed Top-K candidates for two adjacent queries.

    Both queries use the same KV-head and therefore traverse the same K tiles.
    Loading a tile once and applying it to two independent Q rows removes the
    duplicated K-cache transaction while preserving the exact per-query score
    and tie-break rules.  The kernel is intentionally limited to fixed-length
    packed candidates; variable-length inputs retain the proven single-query
    path.
    """
    i_pair, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    i_t0 = i_pair * 2
    i_t1 = i_t0 + 1
    active1 = i_t1 < T
    bos, eos = i_b * T, i_b * T + T
    boc = i_b * tl.cdiv(T, BS)

    p_q0 = tl.make_block_ptr(
        q + (bos + i_t0) * HQ * K,
        (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0),
    )
    b_q0 = tl.load(p_q0, boundary_check=(0, 1))
    b_q0 = (b_q0 * scale).to(b_q0.dtype)
    b_lse0 = tl.load(lse + (bos + i_t0) * HQ + i_h * G + tl.arange(0, G))
    b_q1 = tl.zeros([G, BK], dtype=b_q0.dtype)
    b_lse1 = tl.zeros([G], dtype=tl.float32)
    if active1:
        p_q1 = tl.make_block_ptr(
            q + (bos + i_t1) * HQ * K,
            (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0),
        )
        b_q1 = tl.load(p_q1, boundary_check=(0, 1))
        b_q1 = (b_q1 * scale).to(b_q1.dtype)
        b_lse1 = tl.load(lse + (bos + i_t1) * HQ + i_h * G + tl.arange(0, G))

    ic0 = i_t0 // BS
    ic1 = i_t1 // BS
    # Iterate in reverse tile order, matching the original candidate layout.
    for tile_no in range(NT):
        tile_id = NT - 1 - tile_no
        tile_start = tile_id * BC
        valid1 = active1 and (tile_start <= ic1)
        valid0 = tile_start <= ic0
        if valid1 or valid0:
            o_c = tile_start + tl.arange(0, BC)
            p_k = tl.make_block_ptr(
                k + (boc * H + i_h) * K,
                (K, tl.cdiv(T, BS)),
                (1, H * K),
                (0, tile_start), (BK, BC), (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if valid1:
                b_s1 = tl.dot(b_q1, b_k)
                b_s1 = tl.where(o_c < ic1, b_s1, float("-inf"))
                b_p1 = tl.where(
                    (o_c == 0) | ((o_c == ic1 - 1) | (o_c == ic1)),
                    1.0, exp(b_s1 - b_lse1[:, None]),
                )
                b_score1 = tl.sum(b_p1, 0)
                packed1 = (
                    (_nsa_fp32_to_sort_key(b_score1).to(tl.uint64) << 16)
                    | _nsa_index_to_sort_key(o_c).to(tl.uint64)
                )
                packed1 = tl.where(o_c <= ic1, packed1, tl.zeros_like(packed1))
                top1 = tl.topk(packed1, S)
                p_c1 = packed_candidates + ((bos + i_t1) * H + i_h) * NT * S + tile_id * S
                tl.store(p_c1 + tl.arange(0, S), (top1 & tl.full(top1.shape, 0x7FFFFFFFFFFFFFFF, dtype=tl.uint64)).to(tl.int64))
            if valid0:
                b_s0 = tl.dot(b_q0, b_k)
                b_s0 = tl.where(o_c < ic0, b_s0, float("-inf"))
                b_p0 = tl.where(
                    (o_c == 0) | ((o_c == ic0 - 1) | (o_c == ic0)),
                    1.0, exp(b_s0 - b_lse0[:, None]),
                )
                b_score0 = tl.sum(b_p0, 0)
                packed0 = (
                    (_nsa_fp32_to_sort_key(b_score0).to(tl.uint64) << 16)
                    | _nsa_index_to_sort_key(o_c).to(tl.uint64)
                )
                packed0 = tl.where(o_c <= ic0, packed0, tl.zeros_like(packed0))
                top0 = tl.topk(packed0, S)
                p_c0 = packed_candidates + ((bos + i_t0) * H + i_h) * NT * S + tile_id * S
                tl.store(p_c0 + tl.arange(0, S), (top0 & tl.full(top0.shape, 0x7FFFFFFFFFFFFFFF, dtype=tl.uint64)).to(tl.int64))


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["NT", "S"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_merge_topk_fixed(
    packed_candidates,
    block_indices,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    NT: tl.constexpr,
    CSTRIDE: tl.constexpr,
):
    """Merge per-tile packed candidates without a framework ``torch.topk``.

    The first Top-K kernel already reduced each 64-block tile to ``S`` packed
    candidates.  For fixed-length inputs this small second kernel performs the
    remaining ``NT*S``-way merge on device and writes the logical block ids.
    It removes the launch/framework overhead of materialising a Torch top-k
    operation while keeping the exact packed score/index ordering.
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    # ``CSTRIDE`` is the physical row stride of the candidate workspace.  It
    # is normally ``NT * S``; keeping it explicit also lets the hierarchical
    # merge consume a reduced prefix of a ping-pong workspace without changing
    # the row layout.
    row = ((i_b * T + i_t) * H + i_h) * CSTRIDE
    offsets = tl.arange(0, NT * S)
    values = tl.load(packed_candidates + row + offsets).to(tl.uint64)
    top = tl.topk(values, S)
    index_key = top & tl.full(top.shape, 0xFFFF, dtype=tl.uint64)
    result = (tl.full(index_key.shape, 0xFFFF, dtype=tl.uint64) - index_key).to(tl.int32)
    result = tl.where(top != 0, result, -1)
    result = tl.sort(result, descending=False)
    p_out = block_indices + ((i_b * T + i_t) * H + i_h) * S
    tl.store(p_out + tl.arange(0, S), result)


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["NT", "S"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_kernel_merge_topk_groups(
    packed_in,
    packed_out,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    NT: tl.constexpr,
    CSTRIDE_IN: tl.constexpr,
    CSTRIDE_OUT: tl.constexpr,
    GROUP: tl.constexpr,
):
    """Reduce several per-tile Top-K lists into one list per group.

    The input contains ``NT`` lists of ``S`` packed score/index keys.  Each
    program merges at most ``GROUP`` lists (GROUP*S <= 64 on Hygon), writes
    the best S keys to the corresponding row of ``packed_out``.  Explicit
    input and output row strides keep the merge addressing independent of the
    storage layout while separate buffers avoid cross-program read-after-write
    hazards.
    """
    i_t, i_bh, i_group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    n_groups = (NT + GROUP - 1) // GROUP
    if i_group < n_groups:
        row_in = ((i_b * T + i_t) * H + i_h) * CSTRIDE_IN
        row_out = ((i_b * T + i_t) * H + i_h) * CSTRIDE_OUT
        offsets = tl.arange(0, GROUP * S)
        valid = tl.minimum(GROUP, NT - i_group * GROUP) * S
        values = tl.load(
            packed_in + row_in + i_group * GROUP * S + offsets,
            mask=offsets < valid,
            other=0,
        ).to(tl.uint64)
        top = tl.topk(values, S)
        tl.store(
            packed_out + row_out + i_group * S + tl.arange(0, S),
            top.to(packed_out.dtype.element_ty),
        )


# ===========================================================================
# Forward kernel
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV", "PREFETCH_V"],
    **autotune_cache_kwargs,
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PREFETCH_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(
        q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
    )
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o = tl.make_block_ptr(
        o + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0)
    )
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # Candidate path: issue the V transaction before QK so the load can
            # overlap the first matrix product.  The control keeps the proven
            # load-after-score schedule and therefore does not extend b_v's
            # live range.
            if PREFETCH_V:
                b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS] — compute QK^T scores
            b_s = tl.dot(b_q, b_k)
            # b_k registers are now free — will be reused for later loads
            b_s = tl.where(
                (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
            )

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = exp(b_mp - b_m)
            # [G, BS] — b_s registers reused for softmax output b_p
            b_p = exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)

            # [BS, BV] — load V tile AFTER score computation
            # so b_k registers are freed and peak register usage is reduced
            if not PREFETCH_V:
                b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# ===========================================================================
# TLE-optimized forward kernel (Triton Language Extensions)
# ===========================================================================

if HAS_TLE:

    @triton.heuristics(
        {
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
            "USE_BLOCK_COUNTS": lambda args: isinstance(
                args["block_counts"], torch.Tensor
            ),
        }
    )
    @triton.autotune(
        configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
        key=["BS", "BK", "BV", "PREFETCH_V"],
        **autotune_cache_kwargs,
    )
    @triton.jit
    def parallel_nsa_fwd_kernel_tle(
        q,
        k,
        v,
        o,
        lse,
        scale,
        block_indices,
        block_counts,
        cu_seqlens,
        token_indices,
        T,
        H: tl.constexpr,
        HQ: tl.constexpr,
        G: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        S: tl.constexpr,
        BS: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
        PREFETCH_V: tl.constexpr,
        IS_VARLEN: tl.constexpr,
        USE_BLOCK_COUNTS: tl.constexpr,
    ):
        i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
                token_indices + i_t * 2 + 1
            ).to(tl.int32)
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
                cu_seqlens + i_n + 1
            ).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        k += (bos * H + i_h) * K
        v += (bos * H + i_h) * V
        block_indices += (bos + i_t) * H * S + i_h * S

        if USE_BLOCK_COUNTS:
            NS = tl.load(block_counts + (bos + i_t) * H + i_h)
        else:
            NS = S

        p_q = tl.make_block_ptr(
            q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
        )
        # [G, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_o = tl.make_block_ptr(
            o + (bos + i_t) * HQ * V,
            (HQ, V),
            (V, 1),
            (i_h * G, i_v * BV),
            (G, BV),
            (1, 0),
        )
        p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
        # [G, BV]
        b_o = tl.zeros([G, BV], dtype=tl.float32)

        b_m = tl.full([G], float("-inf"), dtype=tl.float32)
        b_acc = tl.zeros([G], dtype=tl.float32)

        # Precompute indices for async K/V loads via tle.load(is_async=True).
        offs_k = tl.arange(0, BK)
        offs_s = tl.arange(0, BS)
        offs_v = tl.arange(0, BV)
        k_base = k  # already offset to (bos*H + i_h)*K
        v_base = v  # already offset to (bos*H + i_h)*V
        v_start = i_v * BV

        for i in range(NS):
            i_s = tl.load(block_indices + i).to(tl.int32) * BS
            if i_s <= i_t and i_s >= 0:
                # Async K load via regular pointer arithmetic
                k_ptrs = k_base + offs_k[:, None] + (i_s + offs_s[None, :]) * H * K
                k_mask = (offs_k[:, None] < K) & ((i_s + offs_s[None, :]) < T)
                b_k = tle.load(k_ptrs, mask=k_mask, other=0.0, is_async=True)

                if PREFETCH_V:
                    v_ptrs = (
                        v_base
                        + (i_s + offs_s[:, None]) * H * V
                        + (v_start + offs_v[None, :])
                    )
                    v_mask = ((i_s + offs_s[:, None]) < T) & (
                        (v_start + offs_v[None, :]) < V
                    )
                    b_v = tle.load(
                        v_ptrs, mask=v_mask, other=0.0, is_async=True
                    )

                # Compute QK^T scores
                b_s = tl.dot(b_q, b_k.to(b_q.dtype))
                b_s = tl.where(
                    (i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float("-inf")
                )

                # [G]
                b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
                b_r = exp(b_mp - b_m)
                # [G, BS] — b_s registers reused for softmax output b_p
                b_p = exp(b_s - b_m[:, None])
                # [G]
                b_acc = b_acc * b_r + tl.sum(b_p, 1)

                # Async V load AFTER score computation
                if not PREFETCH_V:
                    v_ptrs = (
                        v_base
                        + (i_s + offs_s[:, None]) * H * V
                        + (v_start + offs_v[None, :])
                    )
                    v_mask = ((i_s + offs_s[:, None]) < T) & (
                        (v_start + offs_v[None, :]) < V
                    )
                    b_v = tle.load(
                        v_ptrs, mask=v_mask, other=0.0, is_async=True
                    )

                # [G, BV]
                b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

                b_mp = b_m
        b_o = b_o / b_acc[:, None]
        b_m += log(b_acc)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))


# ===========================================================================
# Block mask kernel
# ===========================================================================


@triton.heuristics(
    {
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    NS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(
            block_counts + i_b * T * H + i_t * H + i_h
        )
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(
            block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


# ===========================================================================
# Backward kernel: dQ
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], torch.Tensor),
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dq,
    scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(
            token_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ * K
    do += (bos + i_t) * HQ * V
    lse += (bos + i_t) * HQ
    delta += (bos + i_t) * HQ
    dq += (i_v * all + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(do, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + i_h * G + tl.arange(0, G)
    p_delta = delta + i_h * G + tl.arange(0, G)

    # [G, BV]
    b_do = tl.load(p_do, boundary_check=(0, 1))
    # [G]
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)

    # [G, BK]
    b_dq = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v = tl.make_block_ptr(
                v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
            )
            # [BK, BS]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            # [BV, BS]
            b_v = tl.load(p_v, boundary_check=(0, 1))

            # [G, BS]
            b_s = tl.dot(b_q, b_k)
            b_p = exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p, 0)

            # [G, BV] @ [BV, BS] -> [G, BS]
            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            # [G, BS] @ [BS, BK] -> [G, BK]
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Backward kernel: dK / dV
# ===========================================================================


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BS", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def parallel_nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    dk,
    dv,
    block_mask,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    M: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    all = B * T
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(
            chunk_indices + i_s * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * all * H + bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (i_s * BS, 0),
        (BS, BK),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * H + i_h) * V,
        (T, V),
        (H * V, 1),
        (i_s * BS, i_v * BV),
        (BS, BV),
        (1, 0),
    )

    # [BS, BK]
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    # [BS, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m:
            p_q = tl.make_block_ptr(
                q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0)
            )
            # [G, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(
                do + (bos + i) * HQ * V,
                (HQ, V),
                (V, 1),
                (i_h * G, i_v * BV),
                (G, BV),
                (1, 0),
            )
            p_lse = lse + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta = delta + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            # [G, BV]
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [G]
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            # [BS, G]
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p, 0)
            # [BS, G] @ [G, BV] -> [BS, BV]
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            # [BS, BV] @ [BV, G] -> [BS, G]
            b_dp = tl.dot(b_v, tl.trans(b_do))
            # [BS, G]
            b_ds = b_p * (b_dp - b_delta[None, :])
            # [BS, G] @ [G, BK] -> [BS, BK]
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


# ===========================================================================
# Python-level wrapper functions
# ===========================================================================


def parallel_nsa_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.LongTensor:
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    # the number of selected blocks for each token
    S = block_counts if isinstance(block_counts, int) else block_counts.max().item()
    S = triton.next_power_of_2(S)
    # Keep the candidate scan tile aligned with the semantic selected block.
    # This is the validated Hygon path; wider BC tiles can trigger a
    # high-register-pressure code-generation path on long prefixes.
    BC = BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    assert BC >= 2 * S, f"BC ({BC}) must be greater than or equal to 2 * S ({S})"

    # For fixed-length prefixes whose total number of causal blocks does not
    # exceed the requested Top-K width, every visible block is selected.  Build
    # that exact index tensor directly and avoid launching a Q@K/bitonic scan.
    # This is a sequence-length class (T <= block_size * S), not a benchmark-
    # specific dispatch, and variable-length inputs continue through the
    # general kernel because each sequence has a different prefix length.
    short_prefix_enabled = os.environ.get(
        "FLAGGEMS_NSA_SHORT_PREFIX", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    if short_prefix_enabled and cu_seqlens is None and T <= BS * S:
        token_blocks = torch.arange(T, device=q.device, dtype=torch.int32) // BS
        candidate = torch.arange(S, device=q.device, dtype=torch.int32)
        indices = torch.where(
            candidate[None, :] <= token_blocks[:, None],
            candidate[None, :],
            torch.full((1, S), -1, device=q.device, dtype=torch.int32),
        )
        return indices.view(1, T, 1, S).expand(B, T, H, S).contiguous()

    block_indices = torch.zeros(B, T, H, S, dtype=torch.int32, device=q.device)
    token_indices = (
        prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
    )
    chunk_offsets = (
        prepare_chunk_offsets(cu_seqlens, BS) if cu_seqlens is not None else None
    )
    grid = (T, B * H)
    packed_topk = os.environ.get(_TOPK_PACKED_ENV, "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    candidate_tiles = triton.cdiv(triton.cdiv(T, BS), BC)
    candidate_mode = packed_topk and candidate_tiles > 1
    pair_mode = (
        candidate_mode
        and cu_seqlens is None
        and os.environ.get(_TOPK_PAIR_ENV, "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    merge_topk = (
        candidate_mode
        and cu_seqlens is None
        and os.environ.get(_TOPK_MERGE_ENV, "1").strip().lower()
        not in {"0", "false", "no", "off"}
        and candidate_tiles * S <= 64
    )
    hierarchical_merge = (
        candidate_mode
        and cu_seqlens is None
        and os.environ.get(_TOPK_MERGE_ENV, "1").strip().lower()
        not in {"0", "false", "no", "off"}
        and os.environ.get(_TOPK_HIER_ENV, "1").strip().lower()
        not in {"0", "false", "no", "off"}
        and candidate_tiles > 4
    )
    packed_candidates = None
    if candidate_mode:
        # Signed int64 stores a sortable version of the packed uint64 key;
        # the kernel removes the constant sign bit for valid scores.
        packed_candidates = torch.zeros(
            (B, T, H, candidate_tiles, S),
            dtype=torch.int64,
            device=q.device,
        )
    else:
        # The legacy kernel never dereferences this pointer.
        packed_candidates = block_indices
    # the 1st and the last 2 blocks are always selected
    if pair_mode:
        parallel_nsa_kernel_topk_pair[(triton.cdiv(T, 2), B * H)](
            q=q,
            k=k,
            lse=lse,
            scale=scale,
            packed_candidates=packed_candidates,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            S=S,
            BC=BC,
            BS=BS,
            BK=BK,
            NT=candidate_tiles,
        )
    else:
        parallel_nsa_kernel_topk[grid](
            q=q,
            k=k,
            lse=lse,
            scale=scale,
            block_indices=block_indices,
            packed_candidates=packed_candidates,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
            chunk_offsets=chunk_offsets,
            T=T,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            S=S,
            BC=BC,
            BS=BS,
            BK=BK,
            PACKED=packed_topk,
            CANDIDATE_MODE=candidate_mode,
            NT=candidate_tiles,
        )
    if candidate_mode:
        if merge_topk:
            parallel_nsa_kernel_merge_topk_fixed[(T, B * H)](
                packed_candidates=packed_candidates,
                block_indices=block_indices,
                T=T,
                H=H,
                S=S,
                NT=candidate_tiles,
                CSTRIDE=candidate_tiles * S,
            )
            return block_indices
        if hierarchical_merge:
            # Long fixed-length prefixes can produce more than 64 packed
            # candidates.  Merge four lists at a time on device, ping-ponging
            # two full-size workspaces to avoid cross-program read-after-write
            # hazards between reduction passes.
            merge_workspace = torch.empty_like(packed_candidates)
            merge_in = packed_candidates
            merge_out = merge_workspace
            merge_nt = candidate_tiles
            candidate_stride = candidate_tiles * S
            merge_in_stride = candidate_stride
            merge_out_stride = candidate_stride
            while merge_nt > 4:
                merge_groups = triton.cdiv(merge_nt, 4)
                parallel_nsa_kernel_merge_topk_groups[
                    (T, B * H, merge_groups)
                ](
                    packed_in=merge_in,
                    packed_out=merge_out,
                    T=T,
                    H=H,
                    S=S,
                    NT=merge_nt,
                    CSTRIDE_IN=merge_in_stride,
                    CSTRIDE_OUT=merge_out_stride,
                    GROUP=4,
                )
                merge_in, merge_out = merge_out, merge_in
                merge_in_stride, merge_out_stride = merge_out_stride, merge_in_stride
                merge_nt = merge_groups
            parallel_nsa_kernel_merge_topk_fixed[(T, B * H)](
                packed_candidates=merge_in,
                block_indices=block_indices,
                T=T,
                H=H,
                S=S,
                NT=merge_nt,
                CSTRIDE=merge_in_stride,
            )
            return block_indices
        candidates = packed_candidates.view(B, T, H, candidate_tiles * S)
        values = torch.topk(candidates, k=S, dim=-1, largest=True, sorted=False).values
        index_key = values & 0xFFFF
        result = (0xFFFF - index_key).to(torch.int32)
        result = torch.where(values > 0, result, torch.full_like(result, -1))
        if cu_seqlens is None:
            query_pos = torch.arange(T, device=q.device, dtype=torch.int32).view(1, T)
            query_pos = query_pos.expand(B, T)
        else:
            lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
            seq_ids = torch.repeat_interleave(
                torch.arange(lengths.numel(), device=q.device), lengths
            )
            starts = cu_seqlens[seq_ids].to(torch.int32)
            query_pos = torch.arange(T, device=q.device, dtype=torch.int32) - starts
            query_pos = query_pos.view(1, T)
        result = torch.where(
            result <= query_pos[:, :, None, None], result, torch.full_like(result, -1)
        )
        return torch.sort(result, dim=-1).values
    return block_indices


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    block_size: int,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if check_shared_mem("hopper", q.device.index):
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    # Dispatch to TLE-optimized kernel when available.
    _use_tle = HAS_TLE and os.environ.get("FLA_NSA_TLE", "1") != "0"

    # Keep the validated G=16/BV=128 tile.  v4 and v5 proved that splitting
    # either matrix dimension is harmful on BW1000.  Early V is useful only
    # for the true async TLE load; on plain Triton it merely extends b_v's live
    # range and can increase register pressure, so that backend stays control.
    PREFETCH_V = (
        _use_tle
        and _use_bw1000_early_v()
        and K == 128
        and V == 128
        and G == 16
        and BS == 64
        and S == 16
    )

    grid = (T, NV, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    if _use_tle:
        kernel = parallel_nsa_fwd_kernel_tle
    else:
        kernel = parallel_nsa_fwd_kernel

    kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
        PREFETCH_V=PREFETCH_V,
    )
    return o, lse


def parallel_nsa_block_mask(
    block_indices: torch.LongTensor,
    block_counts: torch.LongTensor | int,
    cu_seqlens: torch.LongTensor,
    block_size: int,
):
    B, T, H, S = block_indices.shape
    BS = block_size
    if cu_seqlens is not None:
        NS = triton.cdiv(prepare_lens(cu_seqlens).max().item(), BS)
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H * S)](
        block_indices=block_indices,
        block_counts=block_counts,
        block_mask=block_mask,
        T=T,
        H=H,
        S=S,
        BS=BS,
        NS=NS,
    )
    return block_mask


def parallel_nsa_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.Tensor,
    block_counts: torch.LongTensor | int,
    block_size: int = 64,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    token_indices: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    BK = max(triton.next_power_of_2(K), 16)
    BV = min(128, max(triton.next_power_of_2(v.shape[-1]), 16))
    NV = triton.cdiv(V, BV)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(
        NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device
    )
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        block_indices=block_indices,
        block_counts=block_counts,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dq = dq.sum(0)

    if cu_seqlens is not None:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BS)
        NS = len(chunk_indices)
    else:
        NS = triton.cdiv(T, BS)

    block_mask = parallel_nsa_block_mask(
        block_indices, block_counts, cu_seqlens, block_size
    )
    dk = torch.empty(
        NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device
    )
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q,
        k=k,
        v=v,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        block_mask=block_mask,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        M=block_mask.shape[-1],
        BS=BS,
        BK=BK,
        BV=BV,
    )
    dk = dk.sum(0)
    return dq, dk, dv


# ===========================================================================
# Autograd Function
# ===========================================================================


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx, q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens
    ):
        ctx.dtype = q.dtype

        token_indices = (
            prepare_token_indices(cu_seqlens) if cu_seqlens is not None else None
        )

        o, lse = parallel_nsa_fwd(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.cu_seqlens = cu_seqlens
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            lse=lse,
            do=do,
            block_indices=ctx.block_indices,
            block_counts=ctx.block_counts,
            block_size=ctx.block_size,
            scale=ctx.scale,
            cu_seqlens=ctx.cu_seqlens,
            token_indices=ctx.token_indices,
        )
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None


def parallel_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cmp: torch.Tensor | None = None,
    g_slc: torch.Tensor | None = None,
    g_swa: torch.Tensor | None = None,
    block_indices: torch.LongTensor | None = None,
    block_counts: torch.LongTensor | int = 16,
    block_size: int = 64,
    window_size: int = 0,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""
    Native Sparse Attention (NSA) — parallel implementation.

    Args:
        q (torch.Tensor):
            queries of shape ``[B, T, HQ, K]``.
        k (torch.Tensor):
            keys of shape ``[B, T, H, K]``.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H)
            must be a power of 2 and >= 16.
        v (torch.Tensor):
            values of shape ``[B, T, H, V]``.
        g_cmp (torch.Tensor):
            Gate score for compressed attention of shape ``[B, T, HQ]``. If provided,
            ``block_indices`` will be computed automatically.
        g_slc (torch.Tensor):
            Gate score for selected attention of shape ``[B, T, HQ]``.
        g_swa (torch.Tensor):
            Gate score for sliding window attention of shape ``[B, T, HQ]``.
        block_indices (torch.LongTensor):
            Pre-computed block indices of shape ``[B, T, H, S]``.
            If ``g_cmp`` is provided, this is computed automatically.
        block_counts (int or torch.LongTensor):
            Number of selected blocks for each query. If a tensor, shape ``[B, T, H]``.
            Default: 16.
        block_size (int):
            Size of each selected block. Default: 64.
        window_size (int):
            Sliding window size. 0 means no sliding attention. Default: 0.
        scale (float):
            Scale factor. If ``None``, defaults to ``K ** -0.5``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths for variable-length sequences.

    Returns:
        torch.Tensor of shape ``[B, T, HQ, V]``.
    """
    assert block_counts is not None, "block counts must be provided for selection"
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
            f"Please flatten variable-length inputs before processing.",
        )
    assert (
        q.shape[2] % (k.shape[2] * 16) == 0
    ), "Group size must be a multiple of 16 in NSA"

    o_cmp, lse_cmp = None, None
    legacy_k_cmp, legacy_v_cmp = None, None
    if g_cmp is not None:
        k_cmp = mean_pooling(k, block_size, cu_seqlens)
        v_cmp = mean_pooling(v, block_size, cu_seqlens)
        o_cmp, lse_cmp = parallel_nsa_compression(
            q=q,
            k=k_cmp,
            v=v_cmp,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
        if block_indices is not None:
            warnings.warn("`block_indices` will be ignored when `g_cmp` is provided")
        block_indices = parallel_nsa_topk(
            q=q,
            k=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )
    elif not _skip_unused_bw1000_pooling():
        # Same-source control for the public API A/B.  These tensors are
        # deliberately unused, matching the legacy implementation exactly.
        legacy_k_cmp = mean_pooling(k, block_size, cu_seqlens)
        legacy_v_cmp = mean_pooling(v, block_size, cu_seqlens)
    o = o_slc = ParallelNSAFunction.apply(
        q, k, v, block_indices, block_counts, block_size, scale, cu_seqlens
    )
    del legacy_k_cmp, legacy_v_cmp
    if g_slc is not None:
        o = o_slc * g_slc.unsqueeze(-1)
    if o_cmp is not None:
        o = torch.addcmul(o, o_cmp, g_cmp.unsqueeze(-1))
    if window_size > 0:
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError:
            raise ImportError(
                "Flash Attention is required for sliding window attention. "
                "Install it via `pip install flash-attn --no-build-isolation`"
            )
        if cu_seqlens is not None:
            max_seqlen = q.shape[1]
            o_swa = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(window_size - 1, 0),
            ).unsqueeze(0)
        else:
            o_swa = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(window_size - 1, 0),
            )
        o = torch.addcmul(o, o_swa, g_swa.unsqueeze(-1))
    return o
