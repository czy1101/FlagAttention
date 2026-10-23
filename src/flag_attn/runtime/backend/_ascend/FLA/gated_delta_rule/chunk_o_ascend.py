# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend-only ``chunk_fwd_o`` for GDN v1.

Math identical to ``flag_attn.FLA.chunk_o``; the generic module is untouched.
Differences, all established by measurement on 910B:

1. **No autotune.**  ``check_shared_mem()`` is False on Ascend, so the generic
   kernel searched 36 configs per new ``(H, K, V, BT)`` key (7-8.5 s per-process
   cold start).  The tile is fixed here and measured (``BV``=min(V,128),
   4 warps: 0.248 ms vs 0.479 ms at BV=64 for T=1024/H=8/K=V=128, BT=64).
2. **K is a single tile** (``BK = K``): the generic loop over ``cdiv(K, 64)``
   trips twice for K=128 and carries a ``BT x BV`` fp32 cube accumulator
   across the loop, which is miscompiled on 910B (see ``kkt.py``).
3. **Decay applied through 2D factor tiles.**  The generic form
   ``dot(q, k^T) * exp(g_i - g_j)`` / ``o * exp(g_i)`` does elementwise work on
   cube outputs (layout conversion) and the obvious fix
   ``q * exp(g)[:, None]`` is even worse: on this part a 1D->2D broadcast
   multiply of a ``[64, 128]`` tile costs ~6.4 us per program while a 2D x 2D
   multiply costs ~0.1 us (measured 0.82 ms vs 0.05 ms for 128 programs).
   The decay factors are therefore *gathered as 2D tiles*

       b_g2 = tl.load(g + base + o_t[:, None] * H + 0 * arange(K)[None, :], ...)

   and folded into the dot operands:

       A[i, j] = (q_i exp(g_i)) . (k_j exp(-g_j))
       o       = (q exp(g)) @ h + (masked A) @ v

   ``exp(-g_j)`` is clamped to ``e^80`` like ``kkt.py``; with the causal mask
   the kept entries satisfy ``exp(g_i - g_j) <= 1``, so the clamp only touches
   contributions below fp32 resolution.
4. **No ``o_t < T`` term in the causal mask.**  Every load is
   ``boundary_check``-guarded and pads with 0, so out-of-range rows/cols
   already contribute nothing.  The generic kernel's extra broadcast
   ``o_t < T`` mask is what pushes the BT=64 UB footprint over 192KB (measured
   ub overflow: 3358720 bits required vs 1572864 available).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit(do_not_specialize=["T"])
def _chunk_fwd_o_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    p_q = tl.make_block_ptr(q, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0))
    p_k = tl.make_block_ptr(k, (K, T), (1, Hg * K), (0, i_t * BT), (K, BT), (0, 1))
    p_h = tl.make_block_ptr(h, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0))
    p_v = tl.make_block_ptr(
        v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_h = tl.load(p_h, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    if USE_G:
        o_t = i_t * BT + tl.arange(0, BT)
        o_k = tl.arange(0, K)
        m_t = o_t < T
        gb = g + bos * H + i_h
        b_g2 = tl.load(
            gb + o_t[:, None] * H + 0 * o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32)
        b_gr = tl.load(
            gb + o_t[None, :] * H + 0 * o_k[:, None],
            mask=m_t[None, :],
            other=0.0,
        ).to(tl.float32)
        b_qe = b_q * tl.exp(b_g2).to(b_q.dtype)
        b_kt = tl.load(p_k, boundary_check=(0, 1))
        b_kr = b_kt * tl.exp(tl.minimum(-b_gr, 80.0)).to(b_kt.dtype)
        b_A = tl.dot(b_qe, b_kr)
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        b_qe = b_q
        b_kt = tl.load(p_k, boundary_check=(0, 1))
        b_A = tl.dot(b_q, b_kt)

    # Only the causal mask: boundary-checked loads already zero the padding.
    b_A = tl.where(o_t[:, None] >= o_t[None, :], b_A, 0)
    b_o = tl.dot(b_qe, b_h) + tl.dot(b_A.to(b_v.dtype), b_v)
    b_o = b_o * scale
    p_o = tl.make_block_ptr(
        o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def _default_bv(V: int) -> int:
    # BV=128 measured best at BT=64 (0.247 ms vs 0.479 ms at BV=64 on
    # T=1024/H=8/K=V=128): fewer, larger programs beat more, smaller ones.
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_O_BV", "0")) or min(V, 128)


def _default_warps() -> int:
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_O_WARPS", "4"))


def _default_stages() -> int:
    return int(os.environ.get("FLAG_ATTN_ASCEND_GDN_O_STAGES", "2"))


def chunk_fwd_o_ascend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = K**-0.5
    BV = min(V, int(bv)) if bv else _default_bv(V)
    o = torch.empty_like(v)
    grid = (
        triton.cdiv(V, BV),
        NT,
        H if cu_seqlens is not None else B * H,
    )
    _chunk_fwd_o_kernel[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        USE_G=g is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps or _default_warps(),
        num_stages=num_stages or _default_stages(),
    )
    return o


__all__ = ["chunk_fwd_o_ascend"]
