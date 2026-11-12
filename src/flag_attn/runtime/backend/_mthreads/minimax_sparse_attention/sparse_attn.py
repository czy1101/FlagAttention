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

"""Triton kernels for MiniMax M3 block-sparse GQA attention.

The main heads attend only to the blocks selected by the lightning indexer (see
``index_topk``). Adapted to vLLM's paged KV cache: the KV page size is forced to
equal the sparse block size (128), so one selected block maps to exactly one
page.

Main K/V cache layout (vLLM):
  ``(num_blocks, num_kv_heads, 128, 2 * head_dim)``
  K=[..., :head_dim] V=[..., head_dim:]

Only the paths MiniMax M3 uses are implemented: no attention sink, base-2
(exp2/log2) softmax. The decode kernels use split-K (flash-decoding) over the
selected blocks with a separate merge step, since one query token per request
leaves the prefill kernels (which parallelize over the query dim) idle.
"""

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .utils import current_platform as current_platform  # Benchmark compatibility.

# Restrict the verified LLC options to the inspected FlagTree release.
try:
    _PAIR_LLC_OPTIONS_SUPPORTED = version("flagtree") == "0.6.1+mthreads3.6" and triton.__version__ == "3.6.0"
except PackageNotFoundError:
    _PAIR_LLC_OPTIONS_SUPPORTED = False


# One sparse block == one KV page.
SPARSE_BLOCK_SIZE = 128
# Optional decode tuning controls. A value of 0 for the chunk override keeps
# the original shape-based split-K policy.
_DECODE_MIN_BLOCK_H = int(os.getenv("FLAG_ATTN_DECODE_MIN_BLOCK_H", "16"))
_DECODE_NUM_TOPK_CHUNKS = int(os.getenv("FLAG_ATTN_DECODE_NUM_TOPK_CHUNKS", "0"))
_DECODE_NUM_WARPS = int(os.getenv("FLAG_ATTN_DECODE_NUM_WARPS", "8"))
_DECODE_BF16_MIN_BLOCK_H = int(os.getenv("FLAG_ATTN_DECODE_MIN_BLOCK_H", "8"))
_DECODE_BF16_NUM_WARPS = int(os.getenv("FLAG_ATTN_DECODE_NUM_WARPS", "4"))

# Final MUSA Prefill direct-kernel configuration for the current MTT S5000
# workload. H=8 lowers the per-CTA accumulator/register footprint and exposes
# more head-level parallelism. The launch parameters below are the selected
# values after the num_warps and num_stages sweeps.
_PREFILL_BLOCK_H = 8
_PREFILL_NUM_WARPS = 4
_PREFILL_NUM_STAGES = 3

if _DECODE_NUM_WARPS not in (1, 2, 4, 8):
    raise ValueError("FLAG_ATTN_DECODE_NUM_WARPS must be one of 1, 2, 4, or 8")

if _PREFILL_BLOCK_H not in (4, 8, 16):
    raise ValueError("_PREFILL_BLOCK_H must be one of 4, 8, or 16")


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)


# ---------------------------------------------------------------------------
# GQA block-sparse attention (paged). Main heads attend only to the selected
# blocks. BLOCK_SIZE_K == 128 so each selected block is one page.
# ---------------------------------------------------------------------------
# since prefill metadata is sliced from mixed batch metadata, seq_lens and prefix_lens
# might lose pointer alignment, which trigger Triton recompiles. we don't actually
# need pointer alignment for those tensors anyway because we do scalar load.
@triton.jit
def _gqa_sparse_bulk_logits_kernel(
    q_ptr,
    kv_desc,
    routes_ptr,
    logits_ptr,
    ranks_ptr,
    table_ptr,
    cu_ptr,
    length_ptr,
    prefix_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    total_q,
    pages: tl.constexpr,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_tb,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    QT: tl.constexpr,
):
    tile, head_request, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    request, kv_head = head_request // num_kv_heads, head_request % num_kv_heads
    start = tl.load(cu_ptr + request)
    qlen = tl.load(cu_ptr + request + 1) - start
    if tile * QT >= qlen:
        return
    prefix = tl.load(prefix_ptr + request)
    length = tl.load(length_ptr + request)
    qs = tile * QT + tl.arange(0, QT)
    ranks = tl.arange(0, BLOCK_T)
    counts = tl.minimum(topk, tl.cdiv(prefix + qs + 1, 128))
    routes = tl.load(
        routes_ptr + kv_head * stride_th + (start + qs[:, None]) * stride_tn + ranks[None, :] * stride_tk,
        mask=(qs[:, None] < qlen) & (ranks[None, :] < counts[:, None]) & (ranks[None, :] < topk),
        other=-1,
    )
    matches = routes == block
    selected = tl.sum(matches.to(tl.int32), axis=1) > 0
    rank = tl.sum(tl.where(matches, ranks[None, :], 0), axis=1)
    tl.store(ranks_ptr + (kv_head * total_q + start + qs) * pages + block, tl.where(selected, rank, -1), mask=qs < qlen)
    if tl.sum(selected.to(tl.int32), axis=0) == 0:
        return
    h = tl.arange(0, QT * 8)
    query = tile * QT + h // 8
    d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr
        + (start + query[:, None]) * stride_qn
        + (kv_head * 8 + h[:, None] % 8) * stride_qh
        + d[None, :] * stride_qd,
        mask=(query[:, None] < qlen) & (d[None, :] < head_dim),
        other=0.0,
    )
    page = tl.load(table_ptr + request * stride_tb + block).to(tl.int32)
    k = kv_desc.load([(page * num_kv_heads + kv_head) * 128, 0])
    products = tl.trans(tl.dot(k, tl.trans(q))) * (sm_scale * 1.4426950409)
    n = tl.arange(0, 128)
    position = block * 128 + n
    logits = tl.where(
        (position[None, :] <= prefix + query[:, None]) & (position[None, :] < length), products, float("-inf")
    )
    rank_h = tl.reshape(tl.broadcast_to(rank[:, None], (QT, 8)), (QT * 8,))
    selected_h = tl.reshape(tl.broadcast_to(selected[:, None], (QT, 8)), (QT * 8,))
    offsets = (((kv_head * total_q + start + query[:, None]) * 8 + h[:, None] % 8) * topk + rank_h[:, None]) * 128 + n[
        None, :
    ]
    tl.store(logits_ptr + offsets, logits, mask=selected_h[:, None] & (query[:, None] < qlen))


@triton.jit
def _gqa_sparse_bulk_softmax_kernel(
    logits_ptr,
    probability_ptr,
    cu_ptr,
    prefix_ptr,
    total_q,
    num_kv_heads: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    query, head_request = tl.program_id(0), tl.program_id(1)
    request = head_request // (num_kv_heads * 8)
    head = head_request % (num_kv_heads * 8)
    kv_head, local_head = head // 8, head % 8
    start = tl.load(cu_ptr + request)
    qlen = tl.load(cu_ptr + request + 1) - start
    if query >= qlen:
        return
    prefix = tl.load(prefix_ptr + request)
    count = tl.minimum(topk, tl.cdiv(prefix + query + 1, 128))
    x = tl.arange(0, BLOCK_T * 128)
    offset = ((kv_head * total_q + start + query) * 8 + local_head) * topk * 128
    logits = tl.load(logits_ptr + offset + x, mask=(x // 128 < count) & (x // 128 < topk), other=float("-inf"))
    maximum = tl.max(logits, axis=0)
    prob = tl.exp2(logits - maximum)
    probability = prob / tl.sum(prob, axis=0)
    tl.store(probability_ptr + offset + x, probability.to(probability_ptr.dtype.element_ty), mask=x < topk * 128)


@triton.jit
def _gqa_sparse_bulk_pv_kernel(
    kv_desc,
    ranks_ptr,
    probability_ptr,
    output_ptr,
    table_ptr,
    cu_ptr,
    length_ptr,
    prefix_ptr,
    total_q,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    topk: tl.constexpr,
    pages: tl.constexpr,
    stride_on,
    stride_oh,
    stride_od,
    stride_tb,
    BLOCK_D: tl.constexpr,
    QT: tl.constexpr,
    D_TILES: tl.constexpr,
):
    tile, head_output_tile, request = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kv_head = head_output_tile // D_TILES
    output_start = (head_output_tile % D_TILES) * BLOCK_D
    start = tl.load(cu_ptr + request)
    qlen = tl.load(cu_ptr + request + 1) - start
    if tile * QT >= qlen:
        return
    prefix = tl.load(prefix_ptr + request)
    length = tl.load(length_ptr + request)
    qs = tile * QT + tl.arange(0, QT)
    h = tl.arange(0, QT * 8)
    query = tile * QT + h // 8
    n = tl.arange(0, 128)
    d = output_start + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D, QT * 8), tl.float32)
    hi = tl.minimum(length, prefix + tl.minimum((tile + 1) * QT, qlen))
    for block in range(tl.cdiv(hi, 128)):
        rank = tl.load(ranks_ptr + (kv_head * total_q + start + qs) * pages + block, mask=qs < qlen, other=-1)
        selected = rank >= 0
        if tl.sum(selected.to(tl.int32), axis=0) > 0:
            rank_h = tl.reshape(tl.broadcast_to(rank[:, None], (QT, 8)), (QT * 8,))
            selected_h = tl.reshape(tl.broadcast_to(selected[:, None], (QT, 8)), (QT * 8,))
            offsets = (
                ((kv_head * total_q + start + query[:, None]) * 8 + h[:, None] % 8) * topk + rank_h[:, None]
            ) * 128 + n[None, :]
            prob = tl.load(probability_ptr + offsets, mask=selected_h[:, None] & (query[:, None] < qlen), other=0.0)
            page = tl.load(table_ptr + request * stride_tb + block).to(tl.int32)
            if (block + 1) * 128 <= length:
                v_full = kv_desc.load([(page * num_kv_heads + kv_head) * 128, head_dim + output_start])
                acc += tl.dot(tl.trans(v_full), tl.trans(prob))
            else:
                v_tail = kv_desc.load([(page * num_kv_heads + kv_head) * 128, head_dim + output_start])
                v_tail = tl.where(block * 128 + n[:, None] < length, v_tail, 0.0)
                acc += tl.dot(tl.trans(v_tail), tl.trans(prob))
    result = tl.trans(acc)
    tl.store(
        output_ptr
        + (start + query[:, None]) * stride_on
        + (kv_head * 8 + h[:, None] % 8) * stride_oh
        + d[None, :] * stride_od,
        result.to(output_ptr.dtype.element_ty),
        mask=(query[:, None] < qlen) & (d[None, :] < head_dim),
    )


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_direct(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads: tl.constexpr, 128, 2*head_dim]
    kv_cache_desc,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    num_q_loop: tl.constexpr,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    NUM_HEAD_TILES: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_TME: tl.constexpr,
):
    NATIVE_QK: tl.constexpr = (
        not USE_FP8
        and q_ptr.dtype.element_ty == tl.bfloat16
        and USE_TME
        and BLOCK_SIZE_Q == 1
        and BLOCK_SIZE_QH == 8
        and head_dim >= 64
        and head_dim == BLOCK_SIZE_D
    )
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_hg = tl.program_id(1)
    pid_kh = pid_hg // NUM_HEAD_TILES
    pid_ht = pid_hg % NUM_HEAD_TILES
    head_start = pid_ht * BLOCK_SIZE_H
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    if USE_FP8 and KV_SCALE_MODE == 1:
        # Scalar scales are invariant across every selected page. Fold K into
        # the logits scale and delay V until the normalized output.
        qk_scale = sm_scale_log2e * tl.load(k_scale_ptr)
        v_scale_scalar = tl.load(v_scale_ptr)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        valid_blocks = (q_abs + BLOCK_SIZE_K) // BLOCK_SIZE_K
        real_topk = tl.minimum(max_topk, valid_blocks)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, head_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        # Keep the direct kernel's PV accumulator in [QH, D] order.  The
        # transposed [D, QH] form spills heavily for the FP8 QH=32 tile.
        if USE_FP8:
            acc_o = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)
        else:
            acc_o = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        if USE_FP8:
            # Keep QK in FP8 Tensor Core form.  Q is per-head dynamically
            # scaled once, then its scale is restored on the FP32 logits.
            # This keeps the existing K cache in FP8 through tl.dot instead
            # of materializing a BF16 K tile for every selected page.
            qk_q_scale = tl.maximum(tl.max(tl.abs(q), axis=1) * (1.0 / 448.0), 1.0e-8)
            q_qk = (q / qk_q_scale[:, None]).to(tl.float8e4nv)
        else:
            q_qk = q
        if NATIVE_QK:
            # Fill native matrix columns from Q while retaining H8 softmax/PV.
            q_native = tl.reshape(tl.broadcast_to(q[None, :, :], (2, BLOCK_SIZE_QH, BLOCK_SIZE_D)), (16, BLOCK_SIZE_D))
        for _ in range(real_topk):
            blk = tl.load(t_ptr_j).to(tl.int32)
            t_ptr_j = t_ptr_j + stride_tk
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)

            pos = c + off_n
            pos_mask = pos < seq_len

            k_base_ptr = kv_cache_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
            if USE_TME:
                k = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), 0])
            else:
                k_ptrs = tl.make_block_ptr(
                    base=k_base_ptr,
                    shape=(BLOCK_SIZE_K, head_dim),
                    strides=(stride_kv_pos, stride_kv_d),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                k = tl.load(
                    k_ptrs,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr + pid_kh * stride_ks_h + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
            if NATIVE_QK:
                products = tl.dot(k, tl.trans(q_native))
                active, unused = tl.split(tl.permute(tl.reshape(products, (BLOCK_SIZE_K, 2, 8)), (0, 2, 1)))
                qk_dot = tl.trans(active)
            else:
                qk_dot = tl.dot(q_qk, tl.trans(k))
            if USE_FP8:
                qk_dot *= qk_q_scale[:, None]

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))

            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            if USE_FP8:
                if KV_SCALE_MODE == 1:
                    qk += qk_dot * qk_scale
                elif KV_SCALE_MODE == 2:
                    # Q @ (K * scale[token]) is equivalent to scaling the
                    # smaller [QH, K] logits instead of the [K, D] K tile.
                    qk += qk_dot * (sm_scale_log2e * k_scale[None, :])
                else:
                    qk += qk_dot * sm_scale_log2e
            else:
                qk += qk_dot * sm_scale_log2e

            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - m_ij)
            if USE_FP8:
                acc_o = acc_o * alpha[:, None]
            else:
                acc_o = acc_o * alpha[None, :]
            l_i = l_i * alpha + l_ij

            v_base_ptr = k_base_ptr + head_dim * stride_kv_d
            if USE_TME:
                v = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), head_dim])
            else:
                v_ptrs = tl.make_block_ptr(
                    base=v_base_ptr,
                    shape=(BLOCK_SIZE_K, head_dim),
                    strides=(stride_kv_pos, stride_kv_d),
                    offsets=(0, 0),
                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_D),
                    order=(1, 0),
                )
                v = tl.load(
                    v_ptrs,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )

            if USE_FP8:
                if KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr + pid_kh * stride_vs_h + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )

            if USE_FP8 and BLOCK_SIZE_H < 32:
                # The 8/16-head tiles lower efficiently as FP8 PV MMAs.
                # Fold a per-token V scale into P before quantizing P
                # row-wise, and keep V in its FP8 cache format.
                pv_p = p
                if KV_SCALE_MODE == 2:
                    pv_p *= v_scale[None, :]
                pv_p_scale = tl.maximum(tl.max(tl.abs(pv_p), axis=1) * (1.0 / 448.0), 1.0e-8)
                p_dot = (pv_p / pv_p_scale[:, None]).to(tl.float8e4nv)
                acc_o += tl.dot(p_dot, v) * pv_p_scale[:, None]
            else:
                # QH=32 uses the native PV accumulator layout above, but its
                # FP8 PV lowering spills heavily; retain BF16 operands there.
                if USE_FP8:
                    v = v.to(q.dtype)
                p_dot = p.to(v.dtype)
                if USE_FP8 and KV_SCALE_MODE == 2:
                    p_dot = (p * v_scale[None, :]).to(v.dtype)
                if USE_FP8:
                    acc_o += tl.dot(p_dot, v)
                else:
                    acc_o += tl.dot(tl.trans(v), tl.trans(p_dot))
            m_i = m_ij

        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        if USE_FP8 and KV_SCALE_MODE == 1:
            inv_l *= v_scale_scalar
        if USE_FP8:
            acc_o = acc_o * inv_l[:, None]
        else:
            acc_o = tl.trans(acc_o * inv_l[None, :])
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, head_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@triton.jit(do_not_specialize_on_alignment=["seq_lens", "prefix_lens"])
def _gqa_sparse_fwd_pair_union(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # [total_q, num_heads, head_dim]
    block_table_ptr,  # [num_reqs, max_blocks]
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    num_q_loop: tl.constexpr,
    sm_scale,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_bt_b,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    NUM_HEAD_TILES: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    pid_q = tl.program_id(0)
    pid_hg = tl.program_id(1)
    pid_kh = pid_hg // NUM_HEAD_TILES
    pid_ht = pid_hg % NUM_HEAD_TILES
    head_start = pid_ht * BLOCK_SIZE_H
    pid_b = tl.program_id(2)
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    if pid_q * num_q_loop * BLOCK_SIZE_Q >= q_block_len:
        return
    real_q_loop = min(num_q_loop, tl.cdiv(q_block_len, BLOCK_SIZE_Q) - pid_q * num_q_loop)
    bt_row = block_table_ptr + pid_b * stride_bt_b
    off_n = tl.arange(0, BLOCK_SIZE_K)
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        t_ptr_j = t_ptr + (q_block_start + pid_q_j * BLOCK_SIZE_Q) * stride_tn + pid_kh * stride_th
        # Valid block count from seq position (no sentinel): block_size_q == 1.
        q_abs = prefix_len + pid_q_j * BLOCK_SIZE_Q
        route_q = pid_q_j * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        route_t = tl.arange(0, BLOCK_SIZE_T)
        route_count = tl.minimum(max_topk, (prefix_len + route_q + BLOCK_SIZE_K) // BLOCK_SIZE_K)
        routes = tl.load(
            t_ptr_j + tl.arange(0, BLOCK_SIZE_Q)[:, None] * stride_tn + route_t[None, :] * stride_tk,
            mask=(route_q[:, None] < q_len) & (route_t[None, :] < route_count[:, None]) & (route_t[None, :] < max_topk),
            other=2147483647,
        ).to(tl.int32)
        routes = tl.where(routes >= 0, routes, 2147483647)
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_kh * gqa_group_size * stride_qh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, head_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        off_q = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)

        l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        # Keep the BF16 pair accumulator in the transposed PV layout.
        acc_o = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_QH), dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_D)
        q_qk = q
        remaining = (tl.min(tl.min(routes, axis=1), axis=0) < 2147483647).to(tl.int32)
        while remaining > 0:
            blk = tl.min(tl.min(routes, axis=1), axis=0)
            selected_q = tl.sum((routes == blk).to(tl.int32), axis=1) > 0
            routes = tl.where(routes == blk, 2147483647, routes)
            c = blk * BLOCK_SIZE_K
            page = tl.load(bt_row + blk).to(tl.int64)

            pos = c + off_n
            pos_mask = pos < seq_len
            k = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), 0])
            if q_ptr.dtype.element_ty != tl.bfloat16 or head_dim < 64:
                qk_dot = tl.dot(q_qk, tl.trans(k))
            else:
                # Match QK's native column layout to the transposed PV path.
                qk_dot = tl.trans(tl.dot(k, tl.trans(q_qk)))

            is_full_causal = (c + BLOCK_SIZE_K) <= q_abs
            is_full_seq = (c + BLOCK_SIZE_K) <= seq_len

            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            if not is_full_causal:
                qk += tl.where(off_q[:, None, :] >= c, 0, float("-inf"))

            qk += tl.where(selected_q[:, None, None], 0.0, float("-inf"))
            qk = tl.reshape(qk, BLOCK_SIZE_QH, BLOCK_SIZE_K)
            qk += qk_dot * sm_scale_log2e

            if not is_full_seq:
                qk += tl.where(pos_mask[None, :], 0, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            safe_m = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            p = tl.exp2(qk - safe_m[:, None])
            l_ij = tl.sum(p, axis=1)

            alpha = tl.exp2(m_i - safe_m)
            acc_o = acc_o * alpha[None, :]
            l_i = l_i * alpha + l_ij
            v = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), head_dim])

            p_dot = p.to(v.dtype)
            acc_o += tl.dot(tl.trans(v), tl.trans(p_dot))
            m_i = m_ij
            remaining = (tl.min(tl.min(routes, axis=1), axis=0) < 2147483647).to(tl.int32)

        inv_l = tl.where(l_i > 0, 1.0 / l_i, 0.0)
        acc_o = tl.trans(acc_o * inv_l[None, :])
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_kh * gqa_group_size * stride_oh,
            shape=(q_len, gqa_group_size, head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, head_start, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


# ---------------------------------------------------------------------------
# Decode kernels (split-K). Decode batches are flattened request-major, with a
# runtime query length used to map each query token back to its request metadata.
# This parallelizes over the selected top-k blocks, producing partials that the
# merge kernel combines (flash-decoding). All chunk counts depend only on shape
# constants so the grid is fixed within a device graph. Base-2 (exp2/log2)
# softmax matches the prefill kernel.
# ---------------------------------------------------------------------------
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_output_slice_kernel(
    q_ptr,
    kv_desc,
    v_desc,
    routes_ptr,
    out_ptr,
    table_ptr,
    lengths_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_tb,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    query, kv_head, d_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    request = query // decode_query_len
    query_offset = query - request * decode_query_len
    length = tl.load(lengths_ptr + request)
    kv_length = tl.maximum(length - decode_query_len + query_offset + 1, 0)
    count = tl.minimum(max_topk, tl.cdiv(kv_length, 128))
    d = tl.arange(0, BLOCK_D)
    h = tl.arange(0, 16)
    q = tl.load(
        q_ptr + query * stride_qn + (kv_head * 8 + h[:, None]) * stride_qh + d[None, :] * stride_qd,
        mask=(h[:, None] < 8) & (d[None, :] < head_dim),
        other=0.0,
    )
    n = tl.arange(0, 128)
    maximum = tl.full((8,), float("-inf"), tl.float32)
    lse = tl.full((8,), float("-inf"), tl.float32)
    acc = tl.zeros((BLOCK_V, 16), tl.float32)
    for rank in range(count):
        block = tl.load(routes_ptr + kv_head * stride_th + query * stride_tn + rank * stride_tk).to(tl.int32)
        page = tl.load(table_ptr + request * stride_tb + block).to(tl.int32)
        row = (page * num_kv_heads + kv_head) * 128
        k = kv_desc.load([row, 0])
        products = tl.dot(k, tl.trans(q))
        products, unused = tl.split(tl.permute(tl.reshape(products, (128, 2, 8)), (0, 2, 1)))
        valid = block * 128 + n < kv_length
        logits = tl.where(valid[None, :], tl.trans(products) * (sm_scale * 1.4426950409), float("-inf"))
        next_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        prob = tl.exp2(logits - next_maximum[:, None])
        alpha = tl.exp2(maximum - next_maximum)
        prob_sum = tl.sum(prob, axis=1)
        alpha_native = tl.reshape(tl.broadcast_to(alpha[None, :], (2, 8)), (16,))
        acc *= alpha_native[None, :]
        # Consume each descriptor load inside its defining full/tail branch.
        if (block + 1) * 128 <= kv_length:
            v = v_desc.load([row, head_dim + d_tile * BLOCK_V])
            probability = tl.trans(prob.to(v.dtype))
            probability_native = tl.reshape(tl.permute(tl.join(probability, probability), (0, 2, 1)), (128, 16))
            acc = tl.dot(tl.trans(v), probability_native, acc=acc)
        else:
            v = v_desc.load([row, head_dim + d_tile * BLOCK_V])
            v = tl.where(valid[:, None], v, 0.0)
            probability = tl.trans(prob.to(v.dtype))
            probability_native = tl.reshape(tl.permute(tl.join(probability, probability), (0, 2, 1)), (128, 16))
            acc = tl.dot(tl.trans(v), probability_native, acc=acc)
        lse = next_maximum + tl.log2(tl.exp2(lse - next_maximum) + prob_sum)
        maximum = next_maximum
    scale = tl.where(lse > float("-inf"), tl.exp2(maximum - lse), 0.0)
    scale_native = tl.reshape(tl.broadcast_to(scale[None, :], (2, 8)), (16,))
    result = tl.trans(acc * scale_native[None, :])
    weight = tl.exp2(lse - lse)
    weight_native = tl.reshape(tl.broadcast_to(weight[None, :], (2, 8)), (16,))
    result *= (weight_native / weight_native)[:, None]
    oh = tl.arange(0, 16)
    od = d_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    tl.store(
        out_ptr + query * stride_on + (kv_head * 8 + oh[:, None]) * stride_oh + od[None, :] * stride_od,
        result.to(out_ptr.dtype.element_ty),
        mask=(oh[:, None] < 8) & (od[None, :] < head_dim),
    )


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            _DECODE_MIN_BLOCK_H if args["USE_FP8"] else _DECODE_BF16_MIN_BLOCK_H,
            triton.next_power_of_2(args["gqa_group_size"]),
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    kv_cache_desc,
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    num_kv_heads: tl.constexpr,
    gqa_group_size: tl.constexpr,
    head_dim: tl.constexpr,
    max_topk: tl.constexpr,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_TME: tl.constexpr,
):
    NATIVE_QK: tl.constexpr = not USE_FP8 and BLOCK_SIZE_H == 8 and head_dim >= 64 and head_dim == BLOCK_SIZE_D
    TRANSPOSE_PV: tl.constexpr = USE_TME and not USE_FP8
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    if TRANSPOSE_PV:
        acc_o = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_H), dtype=tl.float32)
    else:
        acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    if NATIVE_QK:
        qh = tl.arange(0, 16)
        q_native = tl.load(
            q_ptr + pid_b * stride_qn + (pid_h + qh[:, None]) * stride_qh + off_d[None, :] * stride_qd,
            mask=(qh[:, None] < gqa_group_size) & d_mask[None, :],
            other=0.0,
        )

    cur_idx_ptr = idx_base + chunk_start_topk * stride_tk
    for _ in tl.range(chunk_start_topk, chunk_end_topk):
        blk = tl.load(cur_idx_ptr).to(tl.int32)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        c = blk * BLOCK_SIZE_K
        page = tl.load(bt_row + blk).to(tl.int64)
        pos = c + off_n
        pos_mask = pos < kv_len
        if USE_TME:
            k_page = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), 0])
            if NATIVE_QK:
                k = k_page
            else:
                k = tl.trans(k_page)
                k = tl.where(pos_mask[None, :], k, 0.0)
        else:
            k = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + pid_kh * stride_kv_h
                + off_n[None, :] * stride_kv_pos
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
        if NATIVE_QK and not USE_TME:
            k = tl.trans(k)
        if USE_FP8:
            k = k.to(q.dtype)
            if KV_SCALE_MODE == 1:
                k = (k * tl.load(k_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                k_scale = tl.load(
                    k_scale_ptr + pid_kh * stride_ks_h + (page * BLOCK_SIZE_K + off_n) * stride_ks_t,
                    mask=pos_mask,
                    other=1.0,
                )
                k = (k * k_scale[None, :]).to(q.dtype)
        if NATIVE_QK:
            products = tl.dot(k, tl.trans(q_native))
            products, unused = tl.split(tl.permute(tl.reshape(products, (BLOCK_SIZE_K, 2, 8)), (0, 2, 1)))
            qk = tl.where(pos_mask[None, :], tl.trans(products) * sm_scale_log2e, float("-inf"))
        else:
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        alpha = tl.exp2(m_i - m_ij)
        if TRANSPOSE_PV:
            acc_o = acc_o * alpha[None, :]
        else:
            acc_o = acc_o * alpha[:, None]
        if USE_TME:
            v = kv_cache_desc.load([((page * num_kv_heads + pid_kh) * BLOCK_SIZE_K).to(tl.int32), head_dim])
            v = tl.where(pos_mask[:, None], v, 0.0)
        else:
            v = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk
                + pid_kh * stride_kv_h
                + off_n[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
        if USE_FP8:
            v = v.to(q.dtype)
            if KV_SCALE_MODE == 1:
                v = (v * tl.load(v_scale_ptr)).to(q.dtype)
            elif KV_SCALE_MODE == 2:
                v_scale = tl.load(
                    v_scale_ptr + pid_kh * stride_vs_h + (page * BLOCK_SIZE_K + off_n) * stride_vs_t,
                    mask=pos_mask,
                    other=1.0,
                )
                v = (v * v_scale[:, None]).to(q.dtype)
        if TRANSPOSE_PV:
            acc_o += tl.dot(tl.trans(v), tl.trans(p.to(v.dtype)))
        else:
            acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), tl.zeros_like(lse_i))
    if TRANSPOSE_PV:
        acc_o = tl.trans(acc_o * scale[None, :])
    else:
        acc_o = acc_o * scale[:, None]
    if NUM_TOPK_CHUNKS == 1:
        weight = tl.exp2(lse_i - lse_i)
        acc_o *= (weight / weight)[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    if NUM_TOPK_CHUNKS != 1:
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
            shape=(gqa_group_size,),
            strides=(stride_l_h,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_H,),
            order=(0,),
        )
        tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
_KV_SCALE_NONE = 0
_KV_SCALE_SCALAR = 1
_KV_SCALE_PER_TOKEN_HEAD = 2


def _kv_scale_args(
    output: torch.Tensor,
    num_kv_heads: int,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    if k_scale is None and v_scale is None:
        return output, output, 0, 0, 0, 0, _KV_SCALE_NONE
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale must be both provided or both None")
    if k_scale.device != output.device or v_scale.device != output.device:
        raise ValueError("k_scale and v_scale must be on the same dvice as output")
    if k_scale.numel() == 1 and v_scale.numel() == 1:
        return k_scale, v_scale, 0, 0, 0, 0, _KV_SCALE_SCALAR
    if k_scale.dim() == 2 and v_scale.dim() == 2:
        if k_scale.shape[0] != num_kv_heads or v_scale.shape[0] != num_kv_heads:
            raise ValueError("per-token/head KV scales must have shape " f"[{num_kv_heads}, max_kv_tokens]")
        if k_scale.shape != v_scale.shape:
            raise ValueError("k_scale and v_scale must have matching shapes")
        return (
            k_scale,
            v_scale,
            k_scale.stride(0),
            k_scale.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            _KV_SCALE_PER_TOKEN_HEAD,
        )
    raise ValueError(
        "MiniMax-M3 sparse attention supports scalar KV scales or "
        "[num_kv_heads, max_kv_tokens] per-token/head scales"
    )


@torch.no_grad()
def minimax_m3_sparse_attn(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention over the selected blocks. block_size_q == 1."""
    total_q, num_heads, head_dim = q.shape
    batch = cu_seqlens_q.shape[0] - 1
    topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )
    block_size_h = min(_PREFILL_BLOCK_H, triton.next_power_of_2(gqa_group_size))
    num_head_tiles = triton.cdiv(gqa_group_size, block_size_h)

    use_tme = (
        not use_fp8
        and head_dim == triton.next_power_of_2(head_dim)
        and head_dim >= 32
        and kv_cache.is_contiguous()
        and kv_cache.data_ptr() % 16 == 0
        and kv_cache.numel() // (2 * head_dim) < 2**31
    )
    if use_tme:
        props = torch.musa.get_device_properties(q.device)
        use_tme = (props.major, props.minor) >= (3, 1)
    direct_desc = (
        TensorDescriptor.from_tensor(kv_cache.view(-1, 2 * head_dim), [SPARSE_BLOCK_SIZE, head_dim])
        if use_tme
        else None
    )
    if (
        use_tme
        and q.dtype == torch.bfloat16
        and gqa_group_size == 8
        and head_dim >= 64
        and block_table.shape[1] <= 2 * topk
        and total_q * num_kv_heads >= 256
        and topk <= 32
        and num_kv_heads * total_q * (8 * topk * SPARSE_BLOCK_SIZE * 6 + block_table.shape[1] * 4) <= 2 * 1024**3
    ):
        sparse_logits = torch.empty(
            (num_kv_heads, total_q, 8, topk, SPARSE_BLOCK_SIZE), dtype=torch.float32, device=q.device
        )
        sparse_prob = torch.empty_like(sparse_logits, dtype=q.dtype)
        sparse_ranks = torch.empty((num_kv_heads, total_q, block_table.shape[1]), dtype=torch.int32, device=q.device)
        bulk_options = {}
        if _PAIR_LLC_OPTIONS_SUPPORTED:
            bulk_options[
                "llc_options"
            ] = "-mtgpu-opt-level=1 -mtgpu-combine-instr-with-burst=1 -mtgpu-combine-fop-instr=1 -misched=mtgpu-max-ilp"
        query_tile = max(2, min(16, 2048 // head_dim))
        score_query_tile = 8
        output_tile = 64
        output_tiles = head_dim // output_tile
        value_desc = TensorDescriptor.from_tensor(kv_cache.view(-1, 2 * head_dim), [SPARSE_BLOCK_SIZE, output_tile])
        _gqa_sparse_bulk_logits_kernel[
            (triton.cdiv(max_query_len, score_query_tile), batch * num_kv_heads, block_table.shape[1])
        ](
            q,
            direct_desc,
            topk_idx,
            sparse_logits,
            sparse_ranks,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            num_kv_heads,
            head_dim,
            topk,
            total_q,
            block_table.shape[1],
            sm_scale,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            block_table.stride(0),
            BLOCK_D=head_dim,
            BLOCK_T=triton.next_power_of_2(topk),
            QT=score_query_tile,
            num_warps=8,
            num_stages=1,
            **bulk_options,
        )
        _gqa_sparse_bulk_softmax_kernel[(max_query_len, batch * num_kv_heads * 8)](
            sparse_logits,
            sparse_prob,
            cu_seqlens_q,
            prefix_lens,
            total_q,
            num_kv_heads,
            topk,
            BLOCK_T=triton.next_power_of_2(topk),
            num_warps=4,
        )
        _gqa_sparse_bulk_pv_kernel[(triton.cdiv(max_query_len, query_tile), num_kv_heads * output_tiles, batch)](
            value_desc,
            sparse_ranks,
            sparse_prob,
            output,
            block_table,
            cu_seqlens_q,
            seq_lens,
            prefix_lens,
            total_q,
            num_kv_heads,
            head_dim,
            topk,
            block_table.shape[1],
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_D=output_tile,
            QT=query_tile,
            D_TILES=output_tiles,
            num_warps=8,
            num_stages=1,
            **bulk_options,
        )
        return
    use_pair = not use_fp8 and use_tme and block_size_h == 8 and max_query_len >= 2
    if use_pair:
        # Pair packing halves the query grid. Keep at least two CTAs
        # per processor available before choosing the larger matrix tile.
        pair_ctas = triton.cdiv(max_query_len, 2) * num_kv_heads * num_head_tiles * batch
        use_pair = pair_ctas >= 2 * props.multi_processor_count
    query_tile = 2 if use_pair else 1
    direct_grid = (triton.cdiv(max_query_len, query_tile), num_kv_heads * num_head_tiles, batch)
    direct_kernel = _gqa_sparse_fwd_pair_union if use_pair else _gqa_sparse_fwd_direct
    pair_options = (
        dict(BLOCK_SIZE_T=triton.next_power_of_2(topk))
        if use_pair
        else dict(USE_FP8=use_fp8, KV_SCALE_MODE=kv_scale_mode, USE_TME=use_tme)
    )
    if use_pair and _PAIR_LLC_OPTIONS_SUPPORTED:
        pair_options["llc_options"] = (
            "-mtgpu-opt-level=1 -mtgpu-combine-instr-with-burst=1 " "-mtgpu-combine-fop-instr=1 -misched=mtgpu-max-ilp"
        )
    direct_kernel[direct_grid](
        q,
        kv_cache,
        direct_desc,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        output,
        block_table,
        cu_seqlens_q,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        topk,
        1,  # num_q_loop
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=query_tile,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=triton.next_power_of_2(head_dim),
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_QH=query_tile * block_size_h,
        NUM_HEAD_TILES=num_head_tiles,
        **pair_options,
        num_warps=_PREFILL_NUM_WARPS,
        num_stages=1 if not use_fp8 else _PREFILL_NUM_STAGES,
    )
    return


@lru_cache(maxsize=16)
def _native_decode_device_properties(device):
    # Device topology is constant for this process; no tensor data is cached.
    return torch.musa.get_device_properties(device)


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )
    # split-K over the selected blocks; chunk count is shape-constant (device graph).
    TARGET_GRID = 256
    target = max(1, min(max_topk, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = 1 << (target.bit_length() - 1)
    if _DECODE_NUM_TOPK_CHUNKS > 0:
        override = _DECODE_NUM_TOPK_CHUNKS
        if override > max_topk:
            raise ValueError(f"Decode top-k chunks ({override}) cannot exceed topk ({max_topk})")
        if override & (override - 1):
            raise ValueError("Decode top-k chunks must be a power of two")
        if max_topk % override != 0:
            raise ValueError(f"topk ({max_topk}) must be divisible by chunks ({override})")
        num_topk_chunks = override
    # The native D64 slices repeat QK. Require four complete output-CTA
    # waves, and test this cheap crossover before validating cache descriptors.
    if (
        num_topk_chunks > 1
        and _DECODE_NUM_TOPK_CHUNKS == 0
        and q.device.type == "musa"
        and q.dtype == torch.bfloat16
        and not use_fp8
        and gqa_group_size == 8
        and head_dim >= 128
    ):
        native_props = _native_decode_device_properties(q.device)
        native_ctas = total_q * num_kv_heads * (head_dim // 64)
        if (
            native_ctas >= 4 * native_props.multi_processor_count
            and (native_props.major, native_props.minor) >= (3, 1)
            and head_dim == triton.next_power_of_2(head_dim)
            and "FLAG_ATTN_DECODE_NUM_WARPS" not in os.environ
            and "FLAG_ATTN_DECODE_MIN_BLOCK_H" not in os.environ
            and kv_cache.is_contiguous()
            and kv_cache.data_ptr() % 16 == 0
            and kv_cache.numel() // (2 * head_dim) < 2**31
        ):
            num_topk_chunks = 1
    if num_topk_chunks == 1:
        o_partial = output.unsqueeze(0)
        lse_partial = output
    else:
        o_partial = torch.empty(num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device)
        lse_partial = torch.empty(num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device)
    use_tme = (
        q.device.type == "musa"
        and not use_fp8
        and num_topk_chunks == 1
        and max_topk > 1
        and head_dim == triton.next_power_of_2(head_dim)
        and head_dim >= 32
        and kv_cache.is_contiguous()
        and kv_cache.data_ptr() % 16 == 0
        and kv_cache.numel() // (2 * head_dim) < 2**31
    )
    if use_tme:
        props = torch.musa.get_device_properties(q.device)
        use_tme = (props.major, props.minor) >= (3, 1)
    decode_desc = (
        TensorDescriptor.from_tensor(kv_cache.view(-1, 2 * head_dim), [SPARSE_BLOCK_SIZE, head_dim])
        if use_tme
        else None
    )
    decode_options = {}
    if use_tme and gqa_group_size <= 8 and _PAIR_LLC_OPTIONS_SUPPORTED:
        decode_options[
            "llc_options"
        ] = "-mtgpu-opt-level=1 -mtgpu-combine-instr-with-burst=1 -mtgpu-combine-fop-instr=1 -misched=mtgpu-max-ilp"
    if (
        use_tme
        and q.dtype == torch.bfloat16
        and gqa_group_size == 8
        and head_dim >= 128
        and "FLAG_ATTN_DECODE_NUM_WARPS" not in os.environ
        and "FLAG_ATTN_DECODE_MIN_BLOCK_H" not in os.environ
    ):
        v_desc = TensorDescriptor.from_tensor(kv_cache.view(-1, 2 * head_dim), [SPARSE_BLOCK_SIZE, 64])
        _gqa_sparse_decode_output_slice_kernel[(total_q, num_kv_heads, head_dim // 64)](
            q,
            decode_desc,
            v_desc,
            topk_idx,
            output,
            block_table,
            seq_lens,
            num_kv_heads,
            head_dim,
            max_topk,
            sm_scale,
            decode_query_len,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            topk_idx.stride(0),
            topk_idx.stride(1),
            topk_idx.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            block_table.stride(0),
            BLOCK_D=head_dim,
            BLOCK_V=64,
            num_warps=4,
            num_stages=1,
            **decode_options,
        )
        return
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _gqa_sparse_decode_kernel[grid](
        q,
        kv_cache,
        decode_desc,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        num_kv_heads,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        USE_TME=use_tme,
        # Tune only the main decode-attention kernel. Keep the merge kernel on
        # its existing default so this switch isolates the main-kernel effect.
        num_warps=_DECODE_NUM_WARPS
        if use_fp8
        else (8 if num_topk_chunks > 1 and "FLAG_ATTN_DECODE_NUM_WARPS" not in os.environ else _DECODE_BF16_NUM_WARPS),
        num_stages=3 if use_fp8 else 1,
        **decode_options,
    )
    if num_topk_chunks == 1:
        return
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
    )
