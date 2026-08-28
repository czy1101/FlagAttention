# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""Triton kernels for MiniMax M3 lightning-indexer block scoring + top-k.

Index queries score each 128-token block of index keys (max over the block),
then the top-k blocks (plus forced init/local blocks) are selected per query
token. Adapted to vLLM's paged KV cache: the KV page size is forced to equal the
sparse block size (128), so one sparse block maps to exactly one page.

Index-K cache layout (vLLM): ``(num_blocks, 128, idx_head_dim)`` (single head).

Only the paths MiniMax M3 uses are implemented: score_type="max", index value
disabled (score-only indexer), single shared index head. The selected block ids
feed the block-sparse attention kernels in ``sparse_attn``.
"""

import torch
import triton
import triton.language as tl
from triton.errors import TritonError

from .utils import current_platform, has_triton_tle, round_up

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        _HAS_TLE = True
    except ImportError:
        tle = None
        _HAS_TLE = False
else:
    tle = None
    _HAS_TLE = False


# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128
_NPU_PREFILL_MAX_PROGRAMS = 32768


# ---------------------------------------------------------------------------
# Bitonic top-k helpers (layout-agnostic).
# ---------------------------------------------------------------------------
@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)
    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _select_topk_pair_to_ptr(
    score,
    index,
    active,
    score_ptr,
    index_ptr,
    score_stride,
    index_stride,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
):
    """Select score/index pairs without sub-vector permutations.

    Ascend's vector core requires aligned UB accesses for vectorized
    permutation instructions.  The usual hypercube bitonic implementation
    creates short, non-contiguous subvectors (for example ``[2, 2, ...]``),
    which can lower to an unaligned vector access.  This reduction-based
    selector keeps every reduction on the original contiguous lane vector and
    writes each result through a scalar pointer offset.
    """
    for rank in tl.static_range(0, block_size_t):
        if rank < topk:
            current = tl.where(active, score, -1e30)
            max_score = tl.max(current, axis=0)
            candidate = active & (score == max_score)
            candidate_index = tl.where(candidate, -index, -1e30)
            selected_index = -tl.max(candidate_index, axis=0)
            selected = candidate & (index == selected_index)
            found = tl.max(selected.to(tl.float32), axis=0) > 0.0
            output_score = tl.where(found, max_score, -1e30)
            output_index = tl.where(found, selected_index, 0.0).to(tl.int32)
            active = active & ~selected
        else:
            output_score = -1e30
            output_index = 0
        tl.store(score_ptr + rank * score_stride, output_score)
        tl.store(index_ptr + rank * index_stride, output_index)


@triton.jit
def _select_topk_index_to_ptr(
    score,
    index,
    active,
    index_ptr,
    index_stride,
    topk: tl.constexpr,
    block_size_t: tl.constexpr,
):
    for rank in tl.static_range(0, block_size_t):
        if rank < topk:
            current = tl.where(active, score, -1e30)
            max_score = tl.max(current, axis=0)
            candidate = active & (score == max_score)
            candidate_index = tl.where(candidate, -index, -1e30)
            selected_index = -tl.max(candidate_index, axis=0)
            selected = candidate & (index == selected_index)
            found = tl.max(selected.to(tl.float32), axis=0) > 0.0
            output_index = tl.where(
                found,
                selected_index - 1.0,
                -1.0,
            ).to(tl.int32)
            active = active & ~selected
            tl.store(index_ptr + rank * index_stride, output_index)


# ---------------------------------------------------------------------------
# Index block-score kernel (paged). score[h, token, block] = max over the
# 128-token block of (idx_q . index_k), causal-masked. BLOCK_SIZE_K == 128 so
# each K-tile is exactly one page (BLOCKS_PER_K_BLOCK == 1).
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_b = pid_bh // num_idx_heads
    pid_h = pid_bh % num_idx_heads

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if BLOCK_SIZE_Q * pid_q >= q_len:
        return

    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n + pid_h * stride_q_h,
        shape=(q_len, head_dim),
        strides=(stride_q_n, stride_q_d),
        offsets=(pid_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")
    q_start = prefix_len + pid_q * BLOCK_SIZE_Q

    off_q = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q + prefix_len
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    # Block table row for this request.
    bt_row = block_table_ptr + pid_b * stride_bt_b
    # Causal window: only blocks up to the last query token's position.
    hi = min(seq_len, prefix_len + (pid_q + 1) * BLOCK_SIZE_Q)
    for i in tl.range(0, hi, BLOCK_SIZE_K):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        # index-K for this page: [BLOCK_SIZE_D, BLOCK_SIZE_K] (transposed)
        # we don't need masked load for K, because KV cache ensures
        # allocation is multiple of BLOCK_SIZE_K.
        # for tokens beyond seqlen, they will be masked in qk later.
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k)
        # apply causal mask as needed
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(off_q[:, None] >= pos[None, :], qk, float("-inf"))
        # one sparse block per K-tile -> max over the 128 positions
        score = tl.max(qk, axis=1)  # [BLOCK_SIZE_Q]
        s_ptrs = (
            score_ptr
            + pid_h * stride_s_h
            + (seq_start + pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q))
            * stride_s_n
            + blk * stride_s_k
        )
        q_store_mask = (pid_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < q_len
        tl.store(s_ptrs, score, mask=q_store_mask)


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel_npu(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens,  # [batch+1] query start offsets
    seq_lens,  # [batch] total K length
    prefix_lens,  # [batch] context length before this chunk's queries
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
):
    """NPU Prefill score kernel using the official MSA row tiling.

    The official kernel treats ``(token, head)`` as one flattened M
    dimension.  Keeping those rows in one program makes a K page reusable
    across index heads.  The M tile is kept at 64 here because a 128x128
    Triton tile exceeds the 910B UB once the compiler's local buffers are
    included.
    The public score layout remains head-major, so only the address arithmetic
    differs from the legacy per-head kernel.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    rows = q_len * num_idx_heads
    row_start = pid_m * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < rows
    if row_start >= rows:
        return

    token_offsets = row_offsets // num_idx_heads
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + seq_start * stride_q_n,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start, 0),
        block_shape=(BLOCK_SIZE_M, head_dim),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0,), padding_option="zero")

    # Only pages through the last valid token in this M tile are needed.
    last_row = tl.minimum(row_start + BLOCK_SIZE_M - 1, rows - 1)
    last_token = last_row // num_idx_heads
    hi = tl.minimum(seq_len, prefix_len + last_token + 1)
    q_start = prefix_len + row_start // num_idx_heads
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    q_positions = prefix_len + token_offsets

    for i in tl.range(0, hi, BLOCK_SIZE_K, num_stages=2):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk = tl.dot(q, k)
        if q_start < i + BLOCK_SIZE_K:
            qk = tl.where(
                row_mask[:, None] & (q_positions[:, None] >= pos[None, :]),
                qk,
                float("-inf"),
            )
        score = tl.max(qk, axis=1)
        s_ptrs = (
            score_ptr
            + (row_offsets - token_offsets * num_idx_heads) * stride_s_h
            + (seq_start + token_offsets) * stride_s_n
            + blk * stride_s_k
        )
        tl.store(s_ptrs, score, mask=row_mask)


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _index_block_score_kernel_npu_m128_split(
    q_ptr,
    ik_cache_ptr,
    score_ptr,
    block_table_ptr,
    cu_seqlens,
    seq_lens,
    prefix_lens,
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """NPU M128 schedule with two UB-sized M64 matrix operations.

    A single 128x128 Triton dot exceeds the 910B UB.  Splitting M into two
    64-row dots preserves the official 128-row task granularity while keeping
    the K page loaded only once for both halves.
    """
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    seq_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - seq_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    rows = q_len * num_idx_heads
    row_start = pid_m * BLOCK_SIZE_M
    if row_start >= rows:
        return

    half_m: tl.constexpr = BLOCK_SIZE_M // 2
    row0 = row_start + tl.arange(0, half_m)
    row1 = row0 + half_m
    mask0 = row0 < rows
    mask1 = row1 < rows
    token0 = row0 // num_idx_heads
    token1 = row1 // num_idx_heads
    qbase = q_ptr + seq_start * stride_q_n
    q0_ptrs = tl.make_block_ptr(
        base=qbase,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start, 0),
        block_shape=(half_m, head_dim),
        order=(1, 0),
    )
    q1_ptrs = tl.make_block_ptr(
        base=qbase,
        shape=(rows, head_dim),
        strides=(stride_q_h, stride_q_d),
        offsets=(row_start + half_m, 0),
        block_shape=(half_m, head_dim),
        order=(1, 0),
    )
    q0 = tl.load(q0_ptrs, boundary_check=(0,), padding_option="zero")
    q1 = tl.load(q1_ptrs, boundary_check=(0,), padding_option="zero")
    qpos0 = prefix_len + token0
    qpos1 = prefix_len + token1
    qstart = prefix_len + row_start // num_idx_heads
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    hi = tl.minimum(
        seq_len,
        prefix_len + (row_start + BLOCK_SIZE_M - 1) // num_idx_heads + 1,
    )
    for i in tl.range(0, hi, BLOCK_SIZE_K, num_stages=1):
        blk = i // BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = i + off_k
        k = tl.load(
            ik_cache_ptr
            + page * stride_ik_blk
            + off_k[None, :] * stride_ik_pos
            + off_d[:, None] * stride_ik_d,
        )
        qk0 = tl.dot(q0, k)
        if qstart < i + BLOCK_SIZE_K:
            qk0 = tl.where(
                mask0[:, None] & (qpos0[:, None] >= pos[None, :]),
                qk0,
                float("-inf"),
            )
        score0 = tl.max(qk0, axis=1)
        flat_base = seq_start * num_idx_heads
        s0 = score_ptr + blk * stride_s_k + flat_base + row0
        tl.store(s0, score0, mask=mask0)
        if row_start + half_m < rows:
            qk1 = tl.dot(q1, k)
            if qstart < i + BLOCK_SIZE_K:
                qk1 = tl.where(
                    mask1[:, None] & (qpos1[:, None] >= pos[None, :]),
                    qk1,
                    float("-inf"),
                )
            score1 = tl.max(qk1, axis=1)
            s1 = score_ptr + blk * stride_s_k + flat_base + row1
            tl.store(s1, score1, mask=mask1)


# ---------------------------------------------------------------------------
# Top-k selection over per-token block scores (layout-agnostic). block_size_q
# is 1 for M3, so top-k is computed per query token.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["BLOCK_SIZE_T"],
)
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel_fallback(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
        + off_k * stride_s_k
    )
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        if MASK_INIT:
            score = tl.where(causal_mask & init_mask, score - 1e29, score)
        else:
            score = tl.where(causal_mask & init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(causal_mask & local_mask, score - 1e28, score)
        else:
            score = tl.where(causal_mask & local_mask, 1e29, score)
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    store_mask = off_t < topk
    valid_mask = off_t < valid_blocks
    topk_idx = tl.where(store_mask & valid_mask, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=store_mask)


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(
    do_not_specialize=["query_start"],
    do_not_specialize_on_alignment=["prefix_lens"],
)
def _topk_index_kernel_fallback_npu(
    s_ptr,
    ti_ptr,
    sample_interval: tl.constexpr,
    block_size: tl.constexpr,
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    score_capacity,
    query_start,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    pid_q = tl.program_id(0) + query_start
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return

    valid_blocks = (
        prefix_len + pid_q * sample_interval + block_size
    ) // block_size
    score_capacity = tl.maximum(score_capacity, 1)
    score_row = (
        s_ptr
        + (seq_start + pid_q * sample_interval) * stride_s_n
        + pid_h * stride_s_h
    )
    off = tl.arange(0, BLOCK_SIZE_K)
    score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    index = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    local_start = tl.maximum(0, valid_blocks - local_blocks)
    for lane in tl.static_range(0, BLOCK_SIZE_K):
        lane_valid = lane < valid_blocks
        safe_lane = tl.minimum(lane, score_capacity - 1)
        lane_score = tl.load(
            score_row + safe_lane * stride_s_k,
            mask=lane_valid,
            other=-1e30,
        ).to(tl.float32)
        lane_score = tl.where(lane_score != lane_score, -1e30, lane_score)
        init_mask = lane < init_blocks
        local_mask = lane >= local_start
        if MASK_INIT:
            lane_score = tl.where(
                lane_valid & init_mask,
                lane_score - 1e29,
                lane_score,
            )
        else:
            lane_score = tl.where(
                lane_valid & init_mask,
                1e30,
                lane_score,
            )
        if MASK_LOCAL:
            lane_score = tl.where(
                lane_valid & local_mask,
                lane_score - 1e28,
                lane_score,
            )
        else:
            lane_score = tl.where(
                lane_valid & local_mask,
                1e29,
                lane_score,
            )
        score = tl.where(off == lane, lane_score, score)
        index = tl.where(
            off == lane,
            tl.where(lane_valid, lane + 1.0, 0.0),
            index,
        )

    active = (index > 0.0) & (index <= valid_blocks)
    output_base = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
    )
    _select_topk_index_to_ptr(
        score,
        index,
        active,
        output_base,
        stride_ti_t,
        topk,
        BLOCK_SIZE_T,
    )


# ---------------------------------------------------------------------------
# Streaming Top-K adapted to MSA prefill semantics.
#
# The reference implementation is ``FlagTree/python/tutorials/tle/03-topk.py``
# and operates on a uniform 2-D [M, N] tensor.  This adapted kernel keeps the
# MSA row mapping, per-query causal length, forced init/local blocks, NaN
# handling, output layout, and -1 padding.  Only the row-local selection engine
# is replaced by the packed-key ``tl.topk`` + ``tl.bitonic_merge`` algorithm.
#
# This path intentionally remains separate from ``_topk_index_kernel_fallback``
# so unsupported shapes preserve the previous behavior.  The streaming
# implementation uses only standard Triton APIs; it does not depend on TLE.
# ---------------------------------------------------------------------------
@triton.jit
def _streaming_topk_fpval_to_key(x_bits):
    sign_bit = tl.full(x_bits.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(x_bits.shape, 0xFFFFFFFF, dtype=tl.uint32)
    mask = tl.where((x_bits & sign_bit) != 0, full_mask, sign_bit)
    return x_bits ^ mask


@triton.jit
def _streaming_topk_index_to_key(index):
    max_u16 = tl.full(index.shape, 0xFFFF, dtype=tl.uint32)
    return max_u16 - index.to(tl.uint32)


@triton.jit
def _streaming_topk_key_to_index(index_key):
    max_u16 = tl.full(index_key.shape, 0xFFFF, dtype=tl.uint32)
    return (max_u16 - index_key.to(tl.uint32)).to(tl.int32)


_MAX_STREAMING_TILE_SIZE = 2048
_RADIX_BITS = 4
_RADIX_MIN_PADDED_BLOCKS = 1024
_RADIX_MAX_TOPK = 64
_RADIX_MIN_BLOCKS_PER_TOPK = 16
_FAILED_RADIX_CONFIGS: set[tuple[int, int]] = set()


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
def _topk_index_kernel_streaming(
    s_ptr,  # [num_heads, total_q, max_block]
    ti_ptr,  # [num_heads, total_q, topk]
    sample_interval: tl.constexpr,  # block_size_q (1 for M3)
    block_size: tl.constexpr,  # sparse block size (128)
    cu_seqlens,
    cu_seqblocks_q,
    prefix_lens,
    topk: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_T == topk)
    tl.static_assert(BLOCK_SIZE_K >= BLOCK_SIZE_T)

    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    seq_start = tl.load(cu_seqlens + pid_b)
    block_start = tl.load(cu_seqblocks_q + pid_b)
    block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q >= block_num:
        return

    valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
    off_t = tl.arange(0, BLOCK_SIZE_T)
    ti_ptrs = (
        ti_ptr
        + (block_start + pid_q) * stride_ti_n
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )

    # When N <= K every visible block must be selected.  The normal streaming
    # path also returns these ids in ascending order, so this is exactly the
    # same output without loading or sorting scores.
    if valid_blocks <= topk:
        topk_idx = tl.where(off_t < valid_blocks, off_t, -1)
        tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty))
        return

    off_k = tl.arange(0, BLOCK_SIZE_K)

    # Start from the last (possibly partial) tile.  All preceding tiles are
    # full, matching the order used by the source streaming kernel.
    num_tiles = tl.cdiv(valid_blocks, BLOCK_SIZE_K)
    tile_start = (num_tiles - 1) * BLOCK_SIZE_K
    block_idx = tile_start + off_k
    causal_mask = block_idx < valid_blocks
    local_mask = block_idx >= max(0, valid_blocks - local_blocks)
    init_mask = block_idx < init_blocks

    score_row = (
        s_ptr + (seq_start + pid_q * sample_interval) * stride_s_n + pid_h * stride_s_h
    )
    score = tl.load(
        score_row + block_idx * stride_s_k,
        mask=causal_mask,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    if MASK_INIT:
        score = tl.where(causal_mask & init_mask, score - 1e29, score)
    else:
        score = tl.where(causal_mask & init_mask, 1e30, score)
    if MASK_LOCAL:
        score = tl.where(causal_mask & local_mask, score - 1e28, score)
    else:
        score = tl.where(causal_mask & local_mask, 1e29, score)

    score_key = _streaming_topk_fpval_to_key(score.to(tl.uint32, bitcast=True))
    index_key = _streaming_topk_index_to_key(block_idx)
    packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
    # A finite sentinel such as -1e30 is not below every valid FP32 value.
    # Force lanes outside this row's causal range below the key for -inf so a
    # partial tile can never return an out-of-range logical block id.
    packed = tl.where(causal_mask, packed, tl.zeros_like(packed))
    acc = tl.topk(packed, BLOCK_SIZE_T)

    for _ in tl.range(0, num_tiles - 1):
        acc = tl.bitonic_merge(acc)
        tile_start -= BLOCK_SIZE_K
        block_idx = tile_start + off_k
        local_mask = block_idx >= max(0, valid_blocks - local_blocks)
        init_mask = block_idx < init_blocks

        score = tl.load(score_row + block_idx * stride_s_k).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        if MASK_INIT:
            score = tl.where(init_mask, score - 1e29, score)
        else:
            score = tl.where(init_mask, 1e30, score)
        if MASK_LOCAL:
            score = tl.where(local_mask, score - 1e28, score)
        else:
            score = tl.where(local_mask, 1e29, score)

        score_key = _streaming_topk_fpval_to_key(score.to(tl.uint32, bitcast=True))
        index_key = _streaming_topk_index_to_key(block_idx)
        packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
        acc = tl.maximum(acc, tl.topk(packed, BLOCK_SIZE_T))

    # Match the source streaming implementation's output convention: rotate
    # the index key into the high bits, then sort selected blocks by ascending
    # logical block id.
    acc = (acc << 48) | (acc >> 16)
    acc = tl.sort(acc, descending=True)
    selected_index_key = (acc >> 48).to(tl.uint32)
    topk_idx = _streaming_topk_key_to_index(selected_index_key)

    valid_result = (off_t < valid_blocks) & (topk_idx < valid_blocks)
    topk_idx = tl.where(valid_result, topk_idx, -1)
    tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty))


# ---------------------------------------------------------------------------
# TLE shared-memory Radix Select for wide Prefill rows.
#
# This adapts ``FlagTree/python/tutorials/tle/03-topk.py`` to MSA metadata.
# Unlike the tutorial's atomic output compaction, the final pass scans block ids
# in ascending order and uses prefix sums.  Equal scores therefore use the same
# smaller-block-id tie-break as the packed-key streaming path.
# ---------------------------------------------------------------------------
@triton.jit
def _load_prefill_topk_score(
    score_row,
    block_idx,
    valid_blocks,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_k,
    MASK_INIT: tl.constexpr,
    MASK_LOCAL: tl.constexpr,
):
    valid = block_idx < valid_blocks
    score = tl.load(
        score_row + block_idx * stride_s_k,
        mask=valid,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    init_mask = block_idx < init_blocks
    local_mask = block_idx >= max(0, valid_blocks - local_blocks)
    if MASK_INIT:
        score = tl.where(valid & init_mask, score - 1e29, score)
    else:
        score = tl.where(valid & init_mask, 1e30, score)
    if MASK_LOCAL:
        score = tl.where(valid & local_mask, score - 1e28, score)
    else:
        score = tl.where(valid & local_mask, 1e29, score)
    return score, valid


if _HAS_TLE:

    @triton.jit(do_not_specialize_on_alignment=["prefix_lens"])
    def _topk_index_kernel_radix_tle(
        s_ptr,  # [num_heads, total_q, max_block]
        ti_ptr,  # [num_heads, total_q, topk]
        sample_interval: tl.constexpr,
        block_size: tl.constexpr,
        cu_seqlens,
        cu_seqblocks_q,
        prefix_lens,
        topk: tl.constexpr,
        init_blocks: tl.constexpr,
        local_blocks: tl.constexpr,
        stride_s_h,
        stride_s_n,
        stride_s_k,
        stride_ti_h,
        stride_ti_n,
        stride_ti_t,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_T: tl.constexpr,
        RADIX_BITS: tl.constexpr,
        MASK_INIT: tl.constexpr,
        MASK_LOCAL: tl.constexpr,
    ):
        tl.static_assert(RADIX_BITS == 4)
        tl.static_assert(BLOCK_SIZE_T >= topk)
        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1

        pid_q = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_h = tl.program_id(2)
        seq_start = tl.load(cu_seqlens + pid_b)
        block_start = tl.load(cu_seqblocks_q + pid_b)
        block_num = tl.load(cu_seqblocks_q + pid_b + 1) - block_start
        prefix_len = tl.load(prefix_lens + pid_b)
        if pid_q >= block_num:
            return

        valid_blocks = (prefix_len + pid_q * sample_interval + block_size) // block_size
        off_t = tl.arange(0, BLOCK_SIZE_T)
        output_row = ti_ptr + (block_start + pid_q) * stride_ti_n + pid_h * stride_ti_h

        # Per-row Identity path.  It is particularly important for early
        # Prefill queries, whose causal range has not yet grown beyond K.
        if valid_blocks <= topk:
            topk_idx = tl.where(off_t < valid_blocks, off_t, -1)
            tl.store(
                output_row + off_t * stride_ti_t,
                topk_idx.to(ti_ptr.dtype.element_ty),
                mask=off_t < topk,
            )
            return

        score_row = (
            s_ptr
            + (seq_start + pid_q * sample_interval) * stride_s_n
            + pid_h * stride_s_h
        )
        lane = tl.arange(0, BLOCK_SIZE_K)
        bins = tl.arange(0, RADIX_SIZE)
        one = tl.full([BLOCK_SIZE_K], 1, tl.int32)
        n_tiles = tl.cdiv(valid_blocks, BLOCK_SIZE_K)

        # The histogram is CTA-local, exactly as in the TLE tutorial.
        smem_counts = tle.gpu.alloc(
            [RADIX_SIZE],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        smem_count_ptrs = tle.gpu.local_ptr(smem_counts, (bins,))

        desired = tl.full((), 0, dtype=tl.uint32)
        desired_mask = tl.full((), 0, dtype=tl.uint32)
        k_to_find = tl.full((), topk, dtype=tl.int32)
        radix_mask_u32 = tl.full((), RADIX_MASK, dtype=tl.uint32)

        # Eight 4-bit MSD passes locate the exact FP32 key at the K boundary.
        for digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
            tl.store(
                smem_count_ptrs,
                tl.zeros([RADIX_SIZE], dtype=tl.int32),
            )
            tl.debug_barrier()
            for tile in tl.range(0, n_tiles):
                block_idx = tile * BLOCK_SIZE_K + lane
                score, valid = _load_prefill_topk_score(
                    score_row,
                    block_idx,
                    valid_blocks,
                    init_blocks,
                    local_blocks,
                    stride_s_k,
                    MASK_INIT,
                    MASK_LOCAL,
                )
                score_key = _streaming_topk_fpval_to_key(
                    score.to(tl.uint32, bitcast=True)
                )
                matches = (score_key & desired_mask) == desired
                digit = ((score_key >> digit_pos) & RADIX_MASK).to(tl.int32)
                count_ptrs = tle.gpu.local_ptr(smem_counts, (digit,))
                tl.atomic_add(
                    count_ptrs,
                    one,
                    mask=valid & matches,
                    sem="relaxed",
                    scope="cta",
                )
            tl.debug_barrier()

            counts = tl.load(smem_count_ptrs)
            cumsum_desc = tl.cumsum(counts, axis=0, reverse=True)
            selected_mask = cumsum_desc >= k_to_find
            selected = tl.max(tl.where(selected_mask, bins, 0), axis=0).to(tl.int32)
            counts_gt = tl.max(tl.where(bins == selected + 1, cumsum_desc, 0), axis=0)
            selected_u32 = selected.to(tl.uint32)
            desired = desired | (selected_u32 << digit_pos)
            desired_mask = desired_mask | (radix_mask_u32 << digit_pos)
            k_to_find = k_to_find - counts_gt

        # At this point ``desired`` is the exact threshold key and
        # ``k_to_find`` is how many threshold-equal elements are still needed.
        # Scan once in ascending block-id order, emitting all greater keys and
        # the first equal keys.  This combines the tutorial's two collection
        # passes and makes ties deterministic.
        written = tl.full((), 0, dtype=tl.int32)
        equal_seen = tl.full((), 0, dtype=tl.int32)
        for tile in tl.range(0, n_tiles):
            block_idx = tile * BLOCK_SIZE_K + lane
            score, valid = _load_prefill_topk_score(
                score_row,
                block_idx,
                valid_blocks,
                init_blocks,
                local_blocks,
                stride_s_k,
                MASK_INIT,
                MASK_LOCAL,
            )
            score_key = _streaming_topk_fpval_to_key(score.to(tl.uint32, bitcast=True))
            take_gt = valid & (score_key > desired)
            take_equal = valid & (score_key == desired)
            equal_rank = tl.cumsum(take_equal.to(tl.int32), axis=0)
            take_equal = take_equal & (equal_seen + equal_rank <= k_to_find)
            take = take_gt | take_equal
            take_rank = tl.cumsum(take.to(tl.int32), axis=0)
            out_pos = written + take_rank - 1
            tl.store(
                output_row + out_pos * stride_ti_t,
                block_idx.to(ti_ptr.dtype.element_ty),
                mask=take,
            )
            written = written + tl.sum(take.to(tl.int32), axis=0)
            equal_seen = equal_seen + tl.sum(
                (valid & (score_key == desired)).to(tl.int32), axis=0
            )

else:
    _topk_index_kernel_radix_tle = None


# ---------------------------------------------------------------------------
# Decode index-score kernel (split-K over seq blocks). Decode batches are
# flattened request-major, with a runtime query length used to map each query
# token back to its request metadata. Chunk counts depend only on shape
# constants so the grid is fixed within a cuda graph. The score scale is omitted
# because decode only consumes block ordering.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["num_kv_chunks", "decode_query_len"])
def _decode_index_score_kernel(
    q_ptr,  # idx_q: [total_q, num_idx_heads, head_dim]
    ik_cache_ptr,  # index-K cache: [num_blocks, 128, head_dim]
    score_ptr,  # [num_idx_heads, total_q, max_block]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    num_idx_heads: tl.constexpr,
    head_dim: tl.constexpr,
    init_blocks,
    local_blocks,
    decode_query_len,
    stride_q_n,
    stride_q_h,
    stride_q_d,
    stride_ik_blk,
    stride_ik_pos,
    stride_ik_d,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_Q: tl.constexpr,
    num_kv_chunks,
    HEADS_PER_PROGRAM: tl.constexpr,
    IS_NPU: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    BLOCK_SIZE_HQ: tl.constexpr = HEADS_PER_PROGRAM * BLOCK_SIZE_Q
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    hq_offsets = tl.arange(0, BLOCK_SIZE_HQ)
    if IS_NPU:
        h_offsets = pid_h * HEADS_PER_PROGRAM + hq_offsets % HEADS_PER_PROGRAM
        q_offsets = hq_offsets // HEADS_PER_PROGRAM
    else:
        h_offsets = pid_h * HEADS_PER_PROGRAM + hq_offsets // BLOCK_SIZE_Q
        q_offsets = hq_offsets % BLOCK_SIZE_Q
    q_mask = (q_offsets < decode_query_len) & (h_offsets < num_idx_heads)
    q_ids = pid_r * decode_query_len + q_offsets

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + pid_r)
    query_pos = seq_len - decode_query_len + q_offsets
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks_q = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    kv_len_max = tl.max(tl.where(q_mask, kv_len, 0), axis=0)
    num_blocks = (kv_len_max + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K

    # block-aligned fixed-count split: grid independent of seq_len (cuda graph).
    chunk_size_blocks = (num_blocks + num_kv_chunks - 1) // num_kv_chunks
    chunk_start_block = pid_c * chunk_size_blocks
    chunk_end_block = tl.minimum(chunk_start_block + chunk_size_blocks, num_blocks)
    if chunk_start_block >= chunk_end_block:
        return
    off_k = tl.arange(0, BLOCK_SIZE_K)  # positions within a 128-block
    off_d = tl.arange(0, head_dim)
    bt_row = block_table_ptr + pid_r * stride_bt_b
    # Force-select init (1e30) and local (1e29, higher priority) blocks.
    local_start = tl.maximum(0, num_blocks_q - local_blocks)
    if IS_NPU:
        q = tl.load(
            q_ptr
            + q_ids[:, None] * stride_q_n
            + h_offsets[:, None] * stride_q_h
            + off_d[None, :] * stride_q_d,
            mask=q_mask[:, None],
            other=0.0,
        )
    else:
        q = tl.load(
            q_ptr
            + q_ids[None, :] * stride_q_n
            + h_offsets[None, :] * stride_q_h
            + off_d[:, None] * stride_q_d,
            mask=q_mask[None, :],
            other=0.0,
        )
    for blk in tl.range(chunk_start_block, chunk_end_block):
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_k
        if IS_NPU:
            pos_mask = pos[None, :] < kv_len[:, None]
            k = tl.load(
                ik_cache_ptr
                + page * stride_ik_blk
                + off_k[:, None] * stride_ik_pos
                + off_d[None, :] * stride_ik_d,
            )
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            qk = tl.where(pos_mask & q_mask[:, None], qk, float("-inf"))
            score = tl.max(qk, axis=1)
        else:
            pos_mask = pos[:, None] < kv_len[None, :]
            # we don't need masked load for K, because KV cache ensures
            # allocation is multiple of BLOCK_SIZE_K.
            # for tokens beyond seqlen, they will be masked in qk later.
            k = tl.load(
                ik_cache_ptr
                + page * stride_ik_blk
                + off_k[:, None] * stride_ik_pos
                + off_d * stride_ik_d,
            )
            # fp32 accumulation is required for the fp8 (e4m3) index cache: q/k are
            # loaded in their stored dtype (bf16 or e4m3) and the MMA accumulates in
            # fp32 so the per-block max score is exact for the fp8 indexer too.
            kq = tl.dot(k, q, out_dtype=tl.float32)
            kq = tl.where(pos_mask & q_mask[None, :], kq, float("-inf"))
            score = tl.max(kq, axis=0)
        is_visible_block = blk < num_blocks_q
        is_init = (blk < init_blocks) & is_visible_block
        is_local = (blk >= local_start) & is_visible_block
        score = tl.where(is_local, 1e29, tl.where(is_init, 1e30, score))
        tl.store(
            score_ptr + h_offsets * stride_s_h + q_ids * stride_s_n + blk * stride_s_k,
            score,
            mask=q_mask,
        )


# ---------------------------------------------------------------------------
# Decode top-k: identity for N <= K, otherwise per-chunk partial top-k plus a
# merge. Forced init/local blocks are already encoded in the scores.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len"])
def _decode_topk_identity_kernel(
    ti_final_ptr,  # [num_idx_heads, total_q, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,
    topk: tl.constexpr,
    decode_query_len,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Emit every visible block when the padded maximum cannot exceed K."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off_t = tl.arange(0, BLOCK_SIZE_T)
    topk_idx = tl.where(off_t < num_blocks, off_t, -1)
    tl.store(
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t,
        topk_idx.to(ti_final_ptr.dtype.element_ty),
        mask=off_t < topk,
    )


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE_K": 64}, num_warps=2, num_stages=2),
    ],
    key=["topk"],
)
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_kernel(
    s_ptr,  # score: [num_idx_heads, total_q, max_block]
    ts_partial_ptr,  # partial scores out: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx out (1-indexed global, 0=invalid): same shape
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    chunk_blocks,  # how many score-blocks each chunk owns
    decode_query_len,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # Slice this chunk owns within [0, num_blocks).
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)
    chunk_actual = tl.maximum(chunk_end - chunk_start, 0)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)

    s_ptrs = (
        s_ptr
        + pid_b * stride_s_b
        + pid_h * stride_s_h
        + (chunk_start + off_k) * stride_s_k
    )

    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2

    # Streaming top-K within this chunk. tl.range(0, 0) is a no-op so empty
    # chunks (chunk_actual == 0) skip the body and store sentinel -1e30 / 0.
    for i in tl.range(0, chunk_actual, BLOCK_SIZE_K):
        mask = off_k < chunk_actual - i
        score = tl.load(s_ptrs, mask=mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (
            tl.where(mask, chunk_start + i + off_k + 1, 0),  # 1-indexed global
            topk_idx,
        )
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Extract first BLOCK_SIZE_T entries (top-K of this chunk after the sort).
    topk_mask_extract = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    final_score = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_score, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    final_idx = tl.sum(
        topk_mask_extract[:, None]
        * tl.reshape(topk_idx, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    # Always write all BLOCK_SIZE_T slots — invalid slots carry -1e30 / 0
    # sentinels and lose to real scores in the merge stage.
    ts_ptrs = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
        + off_t * stride_ts_t
    )
    ti_ptrs = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
        + off_t * stride_ti_t
    )
    tl.store(ts_ptrs, final_score)
    tl.store(ti_ptrs, final_idx)


@triton.heuristics({"BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"])})
@triton.jit(do_not_specialize=["chunk_blocks", "decode_query_len"])
def _topk_index_partial_kernel_npu(
    s_ptr,
    ts_partial_ptr,
    ti_partial_ptr,
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    chunk_blocks,
    decode_query_len,
    stride_s_h,
    stride_s_b,
    stride_s_k,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    tl.static_assert(topk < BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size
    chunk_start = pid_chunk * chunk_blocks
    chunk_end = tl.minimum(chunk_start + chunk_blocks, num_blocks)

    off_k = tl.arange(0, BLOCK_SIZE_K)
    score_row = s_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    aligned_start = (chunk_start // 8) * 8
    block_idx = aligned_start + off_k
    valid = (block_idx >= chunk_start) & (block_idx < chunk_end)
    score_capacity = tl.maximum(stride_s_b // stride_s_k, 1)
    safe_block_idx = tl.minimum(block_idx, score_capacity - 1)
    score = tl.load(
        score_row + safe_block_idx * stride_s_k,
        mask=valid,
        other=-1e30,
    ).to(tl.float32)
    score = tl.where(score != score, -1e30, score)
    index = tl.where(valid, block_idx.to(tl.float32) + 1.0, 0.0)

    ts_base = (
        ts_partial_ptr
        + pid_chunk * stride_ts_c
        + pid_b * stride_ts_b
        + pid_h * stride_ts_h
    )
    ti_base = (
        ti_partial_ptr
        + pid_chunk * stride_ti_c
        + pid_b * stride_ti_b
        + pid_h * stride_ti_h
    )
    _select_topk_pair_to_ptr(
        score,
        index,
        valid,
        ts_base,
        ti_base,
        stride_ts_t,
        stride_ti_t,
        topk,
        BLOCK_SIZE_T,
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def _topk_index_merge_kernel(
    ts_partial_ptr,  # partial scores: [NUM_TOPK_CHUNKS, num_idx_heads, total_q, T]
    ti_partial_ptr,  # partial idx (1-indexed global, 0=invalid): same shape
    ti_final_ptr,  # final idx (0-indexed, -1=invalid): [num_idx_heads, total_q, topk]
    seq_lens,  # [num_reqs]
    block_size: tl.constexpr,  # sparse block size (128)
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b = tl.program_id(0)  # flattened query-token id
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len

    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    # Load NUM_TOPK_CHUNKS * BLOCK_SIZE_T candidates, padded to BLOCK_SIZE_K.
    # Candidate at flat position p comes from chunk = p // BLOCK_SIZE_T,
    # in_chunk = p % BLOCK_SIZE_T.
    off = tl.arange(0, BLOCK_SIZE_K)
    chunk_idx = off // BLOCK_SIZE_T
    in_chunk_idx = off % BLOCK_SIZE_T
    valid = chunk_idx < num_topk_chunks

    score_offset = (
        chunk_idx * stride_ts_c
        + pid_h * stride_ts_h
        + pid_b * stride_ts_b
        + in_chunk_idx * stride_ts_t
    )
    idx_offset = (
        chunk_idx * stride_ti_c
        + pid_h * stride_ti_h
        + pid_b * stride_ti_b
        + in_chunk_idx * stride_ti_t
    )

    score = tl.load(ts_partial_ptr + score_offset, mask=valid, other=-1e30).to(
        tl.float32
    )
    score = tl.where(score != score, -1e30, score)
    idx = tl.load(ti_partial_ptr + idx_offset, mask=valid, other=0).to(tl.int32)

    # Full bitonic descending sort of BLOCK_SIZE_K items.
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for j in tl.static_range(1, n_dims):
        score, idx = _bitonic_merge(score, idx.to(tl.int32), j, 2, n_dims)
    score, idx = _bitonic_merge(score, idx.to(tl.int32), n_dims, True, n_dims)

    # Extract first BLOCK_SIZE_T positions — these are the global top-K.
    extract_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx_final = tl.sum(
        extract_mask[:, None]
        * tl.reshape(idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )

    off_t = tl.arange(0, BLOCK_SIZE_T)
    tif_ptrs = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
        + off_t * stride_tif_t
    )
    store_mask = off_t < topk
    topk_idx_final = tl.where(off_t < tl.minimum(topk, num_blocks), topk_idx_final, -1)
    tl.store(
        tif_ptrs, topk_idx_final.to(ti_final_ptr.dtype.element_ty), mask=store_mask
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["topk"]),
        "BLOCK_SIZE_K": lambda args: triton.next_power_of_2(
            args["num_topk_chunks"] * triton.next_power_of_2(args["topk"])
        ),
    }
)
@triton.jit(do_not_specialize=["num_topk_chunks", "decode_query_len"])
def _topk_index_merge_kernel_npu(
    ts_partial_ptr,
    ti_partial_ptr,
    ti_final_ptr,
    seq_lens,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    decode_query_len,
    stride_ts_c,
    stride_ts_h,
    stride_ts_b,
    stride_ts_t,
    stride_ti_c,
    stride_ti_h,
    stride_ti_b,
    stride_ti_t,
    stride_tif_h,
    stride_tif_b,
    stride_tif_t,
    num_topk_chunks,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    kv_len = tl.maximum(query_pos + 1, 0)
    num_blocks = (kv_len + block_size - 1) // block_size

    off = tl.arange(0, BLOCK_SIZE_K)
    score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    index = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)
    for lane in tl.static_range(0, BLOCK_SIZE_K):
        chunk_idx = lane // BLOCK_SIZE_T
        in_chunk_idx = lane % BLOCK_SIZE_T
        lane_valid = chunk_idx < num_topk_chunks
        score_value = tl.load(
            ts_partial_ptr
            + chunk_idx * stride_ts_c
            + pid_h * stride_ts_h
            + pid_b * stride_ts_b
            + in_chunk_idx * stride_ts_t,
            mask=lane_valid,
            other=-1e30,
        ).to(tl.float32)
        index_value = tl.load(
            ti_partial_ptr
            + chunk_idx * stride_ti_c
            + pid_h * stride_ti_h
            + pid_b * stride_ti_b
            + in_chunk_idx * stride_ti_t,
            mask=lane_valid,
            other=0,
        ).to(tl.float32)
        score = tl.where(off == lane, score_value, score)
        index = tl.where(off == lane, index_value, index)
    score = tl.where(score != score, -1e30, score)

    active = (index > 0.0) & (index <= num_blocks)
    tif_base = (
        ti_final_ptr
        + pid_h * stride_tif_h
        + pid_b * stride_tif_b
    )
    _select_topk_index_to_ptr(
        score,
        index,
        active,
        tif_base,
        stride_tif_t,
        topk,
        BLOCK_SIZE_T,
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.  The NPU
    # flattened kernel writes one page across adjacent M rows.  Store that
    # buffer as [block, flattened-M] and expose a logical [head, token, block]
    # view so the existing top-k kernels need no layout-specific changes.
    score_block_stride = round_up(max_block, 16)
    use_npu_flat_score = (
        idx_q.device.type == "npu" and num_idx_heads > 1 and idx_q.is_contiguous()
    )
    if use_npu_flat_score:
        score_storage = torch.empty(
            (score_block_stride, total_q * num_idx_heads),
            dtype=torch.float32,
            device=idx_q.device,
        )
        score = torch.as_strided(
            score_storage,
            (num_idx_heads, total_q, score_block_stride),
            (1, num_idx_heads, total_q * num_idx_heads),
        )
    else:
        score = torch.empty(
            (num_idx_heads, total_q, score_block_stride),
            dtype=torch.float32,
            device=idx_q.device,
        )
    if use_npu_flat_score:
        # Flatten token and head like MSA.  Keep the official 128-row task
        # granularity, but split its matrix operation into two M64 dots in the
        # kernel so the complete M128xK128 dot never has to reside in UB.
        block_size_m = 128
        grid_score = (
            triton.cdiv(max_query_len * num_idx_heads, block_size_m),
            batch,
        )
        try:
            _index_block_score_kernel_npu_m128_split[grid_score](
                idx_q,
                index_kv_cache,
                score_storage,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                num_idx_heads,
                head_dim,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                index_kv_cache.stride(0),
                index_kv_cache.stride(1),
                index_kv_cache.stride(2),
                score_storage.stride(0),
                block_table.stride(0),
                BLOCK_SIZE_M=block_size_m,
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                num_warps=8,
                num_stages=1,
            )
        except TritonError:
            # Keep the previously validated M64 path as a compile-time
            # fallback for CANN/Triton versions that reject the split tile.
            fallback_m = 64
            fallback_grid = (
                triton.cdiv(max_query_len * num_idx_heads, fallback_m),
                batch,
            )
            _index_block_score_kernel_npu[fallback_grid](
                idx_q,
                index_kv_cache,
                score,
                block_table,
                cu_seqlens_q,
                seq_lens,
                prefix_lens,
                num_idx_heads,
                head_dim,
                idx_q.stride(0),
                idx_q.stride(1),
                idx_q.stride(2),
                index_kv_cache.stride(0),
                index_kv_cache.stride(1),
                index_kv_cache.stride(2),
                score.stride(0),
                score.stride(1),
                score.stride(2),
                block_table.stride(0),
                BLOCK_SIZE_M=fallback_m,
                BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
                num_warps=4,
                num_stages=1,
            )
    else:
        block_size_q = 64
        grid_score = (triton.cdiv(max_query_len, block_size_q), batch * num_idx_heads)
        _index_block_score_kernel[grid_score](
            idx_q,
            index_kv_cache,
            score,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_idx_heads,
            head_dim,
            idx_q.stride(0),
            idx_q.stride(1),
            idx_q.stride(2),
            index_kv_cache.stride(0),
            index_kv_cache.stride(1),
            index_kv_cache.stride(2),
            score.stride(0),
            score.stride(1),
            score.stride(2),
            block_table.stride(0),
            BLOCK_SIZE_Q=block_size_q,
            BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        )
    return score


def _supports_prefill_selector_input(score: torch.Tensor, topk: int) -> bool:
    """Common correctness boundary for the optimized Prefill selectors."""
    if score.device.type not in {"cuda", "npu"}:
        return False
    if score.dtype != torch.float32 or score.ndim != 3:
        return False
    if topk <= 0 or score.shape[2] <= 0:
        return False
    return True


def _can_use_radix_prefill_topk(
    score: torch.Tensor,
    topk: int,
    max_query_len: int,
) -> bool:
    """Use Radix for a short Prefill chunk over a wide existing context."""
    if score.device.type != "cuda":
        return False
    if not _HAS_TLE or _topk_index_kernel_radix_tle is None:
        return False
    if not _supports_prefill_selector_input(score, topk):
        return False
    max_score_blocks = score.shape[2]
    block_size_k, _ = _radix_prefill_launch_config(max_score_blocks)
    return (
        # A chunk no longer than one sparse page keeps per-row valid_blocks
        # nearly constant.  Full Prefill has N growing from 1 to max_block and
        # is better served by the single-pass Streaming selector.
        0 < max_query_len <= SPARSE_BLOCK_SIZE
        and max_score_blocks >= _RADIX_MIN_PADDED_BLOCKS
        and topk <= _RADIX_MAX_TOPK
        and max_score_blocks >= _RADIX_MIN_BLOCKS_PER_TOPK * topk
        and (block_size_k, topk) not in _FAILED_RADIX_CONFIGS
    )


def _can_use_streaming_prefill_topk(score: torch.Tensor, topk: int) -> bool:
    """Return whether packed-key Streaming supports this call exactly."""
    if not _supports_prefill_selector_input(score, topk):
        return False
    if topk > _MAX_STREAMING_TILE_SIZE:
        return False
    # The streaming selector packs the logical block id into 16 bits and uses
    # an exact (not padded) compile-time K in ``tl.topk``.
    if score.shape[2] > 0xFFFF:
        return False
    if (topk & (topk - 1)) != 0:
        return False
    return True


def _select_prefill_topk_path(
    score: torch.Tensor,
    topk: int,
    max_query_len: int,
) -> str:
    """Choose an algorithm; correctness support and performance policy differ."""
    if _can_use_radix_prefill_topk(score, topk, max_query_len):
        return "radix"
    if _can_use_streaming_prefill_topk(score, topk):
        return "streaming"
    return "fallback"


def _topk_tile_num_warps(block_size: int) -> int:
    if block_size <= 64:
        return 2
    if block_size <= 128:
        return 4
    return 8


def _streaming_prefill_launch_config(
    max_score_blocks: int,
    topk: int,
) -> tuple[int, int]:
    """Choose Streaming resources from MSA's padded max-block dimension."""
    # MSA rows have different valid_blocks, but they share this padded upper
    # bound.  Cap the normal tile at 1024 as in TLE, then enlarge only when K
    # itself needs more lanes.  No synthetic kernel argument is required.
    block_size = max(
        64,
        topk,
        triton.next_power_of_2(min(max_score_blocks, 1024)),
    )
    return block_size, _topk_tile_num_warps(block_size)


def _radix_prefill_launch_config(max_score_blocks: int) -> tuple[int, int]:
    """Choose Radix resources from MSA's padded max-block dimension."""
    block_size = max(
        32,
        triton.next_power_of_2(min(max_score_blocks, 1024)),
    )
    return block_size, _topk_tile_num_warps(block_size)


@torch.no_grad()
def minimax_m3_index_topk(
    score: torch.Tensor,  # [num_idx_heads, total_q, max_block]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select index top-k from a precomputed score tensor.

    Dispatches short Prefill chunks over wide contexts to TLE Radix Select,
    full/medium Prefill to packed-key Streaming Top-K, and unsupported inputs to
    the original Bitonic fallback.  Optimized kernels contain a per-row N <= K
    identity path.

    When ``out`` is provided (a ``[num_idx_heads, >=total_q, topk]`` buffer), the
    result is written into ``out[:, :total_q, :]`` instead of a fresh tensor --
    used to keep the top-k output at a stable address for cudagraph capture.
    """
    num_idx_heads = score.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    total_q = score.shape[1]
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=score.device,
        )
    # block_size_q == 1 -> query blocks coincide with query tokens.
    grid_topk = (max_query_len, batch, num_idx_heads)
    kernel_args = (
        score,
        topk_idx,
        1,  # sample_interval (block_size_q)
        SPARSE_BLOCK_SIZE,
        cu_seqlens_q,
        cu_seqlens_q,  # cu_seqblocks_q == cu_seqlens_q when block_size_q == 1
        prefix_lens,
        topk,
        init_blocks,
        local_blocks,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
    )
    path = _select_prefill_topk_path(score, topk, max_query_len)
    if score.device.type == "npu":
        path = "fallback"
    if path == "radix":
        assert _topk_index_kernel_radix_tle is not None
        block_size_k, num_warps = _radix_prefill_launch_config(score.shape[2])
        try:
            _topk_index_kernel_radix_tle[grid_topk](
                *kernel_args,
                BLOCK_SIZE_K=block_size_k,
                BLOCK_SIZE_T=triton.next_power_of_2(topk),
                RADIX_BITS=_RADIX_BITS,
                MASK_INIT=False,
                MASK_LOCAL=False,
                num_warps=num_warps,
                num_stages=1,
            )
            return topk_idx
        except TritonError:
            # Some CUDA/Triton combinations expose the TLE module but cannot
            # lower this shared-memory kernel.  Remember the failed compile
            # configuration so later calls do not repeatedly pay that cost.
            _FAILED_RADIX_CONFIGS.add((block_size_k, topk))
            path = (
                "streaming"
                if _can_use_streaming_prefill_topk(score, topk)
                else "fallback"
            )

    if path == "streaming":
        block_size_k, num_warps = _streaming_prefill_launch_config(score.shape[2], topk)
        _topk_index_kernel_streaming[grid_topk](
            *kernel_args,
            BLOCK_SIZE_K=block_size_k,
            MASK_INIT=False,
            MASK_LOCAL=False,
            num_warps=num_warps,
            num_stages=2,
        )
    elif path == "fallback" and score.device.type == "npu":
        fallback_block_size = max(
            64,
            triton.next_power_of_2(score.shape[2]),
            triton.next_power_of_2(topk + 1),
        )
        fixed_grid_size = batch * num_idx_heads
        max_query_programs = max(
            1,
            _NPU_PREFILL_MAX_PROGRAMS // fixed_grid_size,
        )
        for query_start in range(0, max_query_len, max_query_programs):
            query_programs = min(
                max_query_programs,
                max_query_len - query_start,
            )
            npu_grid_topk = (query_programs, batch, num_idx_heads)
            _topk_index_kernel_fallback_npu[npu_grid_topk](
                *kernel_args,
                score.shape[2],
                query_start,
                BLOCK_SIZE_K=fallback_block_size,
                MASK_INIT=False,
                MASK_LOCAL=False,
                num_warps=_topk_tile_num_warps(fallback_block_size),
                num_stages=1,
            )
    else:
        _topk_index_kernel_fallback[grid_topk](
            *kernel_args,
            MASK_INIT=False,
            MASK_LOCAL=False,
        )
    return topk_idx


@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score (split-K, cudagraph-safe); no top-k.

    Returns score [num_kv_heads, total_q, >=max_block] (fp32; init/local blocks
    forced to 1e30/1e29). When ``score_out`` is given the scores are written into
    it (read/written by strides, so a transposed view of a unified buffer is
    accepted) instead of a fresh tensor -- used to share a unified score buffer
    with the prefill side and run a single top-k over both.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    # NPU decode uses one program for a bounded Q/head tile.  Keeping the
    # product below 64 columns fits the 910B UB while allowing qlen=1 to
    # process all index heads together.
    score_kwargs = pdl_kwargs.copy()
    if idx_q.device.type == "npu":
        score_kwargs.update({"num_stages": 1})
    elif num_idx_heads > 1 and max_decode_query_len > 1:
        score_kwargs.update({"num_warps": 4, "num_stages": 2})

    if score_out is not None:
        score = score_out
    else:
        # Keep score strides 16-divisible to avoid Triton recompiles.
        score_block_stride = round_up(max_block, 16)
        score = torch.empty(
            (num_idx_heads, total_q, score_block_stride),
            dtype=torch.float32,
            device=idx_q.device,
        )
    # split-K over seq blocks; chunk count depends only on shape constants so
    # the grid is fixed within a cuda graph.
    TARGET_GRID = 512
    MAX_NUM_KV_CHUNKS = 256
    # Use the configured max decode length to avoid Triton recompiles when
    # switching between qlen=1 and spec-decode verification batches.
    BLOCK_SIZE_Q = triton.next_power_of_2(max_decode_query_len)
    if idx_q.device.type == "npu":
        block_size_per_chunk = 32
        target_chunks = max(1, triton.cdiv(max_block, block_size_per_chunk))
        num_kv_chunks = min(
            MAX_NUM_KV_CHUNKS,
            1 << (target_chunks - 1).bit_length(),
        )
        max_query_head_columns = 64
        heads_per_program = min(
            num_idx_heads,
            max(1, max_query_head_columns // BLOCK_SIZE_Q),
        )
        score_kwargs["num_warps"] = 4 if heads_per_program * BLOCK_SIZE_Q > 16 else 2
    else:
        score_ctas_per_chunk = seq_lens.shape[0]
        target = max(
            1,
            min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, score_ctas_per_chunk)),
        )
        num_kv_chunks = 1 << (target.bit_length() - 1)
        heads_per_program = num_idx_heads
    head_programs = triton.cdiv(num_idx_heads, heads_per_program)
    grid_score = (seq_lens.shape[0], num_kv_chunks, head_programs)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        num_kv_chunks=num_kv_chunks,
        HEADS_PER_PROGRAM=heads_per_program,
        IS_NPU=idx_q.device.type == "npu",
        USE_PDL=use_pdl,
        **score_kwargs,
    )
    return score


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + dispatched top-k (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    When ``score_out`` ([num_kv_heads, total_q, >=max_block]) is given, the block
    scores are written into it (read back by the top-k) instead of a fresh
    tensor -- used to share a unified score buffer with the prefill side. Reads
    via strides, so a transposed view of a block-major buffer is accepted.
    """
    total_q, num_idx_heads, _ = idx_q.shape
    assert (
        num_idx_heads == num_kv_heads
    ), "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})

    block_size_t = triton.next_power_of_2(topk)
    if max_block <= topk:
        # Every visible block is selected, so scores cannot affect the result.
        # Preserve the documented score_out mutation when the caller supplies
        # that buffer; otherwise skip both score generation and Top-K sorting.
        if out is not None:
            topk_idx = out[:, :total_q, :]
        else:
            topk_idx = torch.empty(
                (num_idx_heads, total_q, topk),
                dtype=torch.int32,
                device=idx_q.device,
            )
        if score_out is not None:
            minimax_m3_index_decode_score(
                idx_q,
                index_kv_cache,
                block_table,
                seq_lens,
                max_seq_len,
                init_blocks,
                local_blocks,
                num_kv_heads,
                decode_query_len,
                max_decode_query_len,
                score_out=score_out,
            )
        identity_use_pdl = use_pdl and score_out is not None
        identity_pdl_kwargs = pdl_kwargs if identity_use_pdl else {}
        _decode_topk_identity_kernel[(batch, num_idx_heads)](
            topk_idx,
            seq_lens,
            SPARSE_BLOCK_SIZE,
            topk,
            decode_query_len,
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            BLOCK_SIZE_T=block_size_t,
            USE_PDL=identity_use_pdl,
            **identity_pdl_kwargs,
        )
        return topk_idx

    score = minimax_m3_index_decode_score(
        idx_q,
        index_kv_cache,
        block_table,
        seq_lens,
        max_seq_len,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_query_len,
        max_decode_query_len,
        score_out=score_out,
    )

    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )

    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * num_idx_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.float32,
        device=idx_q.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.int32,
        device=idx_q.device,
    )
    partial_args = (
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
    )
    partial_grid = (batch, num_idx_heads, num_topk_chunks)
    if idx_q.device.type == "npu":
        partial_block_size = max(
            64,
            triton.next_power_of_2(chunk_blocks + 8),
            triton.next_power_of_2(topk + 1),
        )
        _topk_index_partial_kernel_npu[partial_grid](
            *partial_args,
            BLOCK_SIZE_K=partial_block_size,
            num_warps=_topk_tile_num_warps(partial_block_size),
            num_stages=1,
        )
    else:
        _topk_index_partial_kernel[partial_grid](
            *partial_args,
            USE_PDL=use_pdl,
            **pdl_kwargs,
        )
    merge_args = (
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
    )
    merge_grid = (batch, num_idx_heads)
    if idx_q.device.type == "npu":
        _topk_index_merge_kernel_npu[merge_grid](
            *merge_args,
            num_topk_chunks=num_topk_chunks,
            num_warps=_topk_tile_num_warps(
                triton.next_power_of_2(num_topk_chunks * block_size_t)
            ),
            num_stages=1,
        )
    else:
        _topk_index_merge_kernel[merge_grid](
            *merge_args,
            num_topk_chunks=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_kwargs,
        )
    return topk_idx
