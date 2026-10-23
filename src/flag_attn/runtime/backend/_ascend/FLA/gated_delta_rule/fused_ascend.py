# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused Ascend stages for the GDN v1 fast path.

The pipeline is three Triton launches:

1. ``fold`` (grid ``(chunks, B*H)``) -- a **vector-only** kernel: chunk-local
   cumsum (fp32), then all per-token decay/beta folds written out as bf16
   tensors: ``q_e = q*exp(g)``, ``k_e = k*beta*exp(g)``, ``k_r = k*exp(-g)``,
   ``v_beta = v*beta``, ``k_s = k*exp(G-g)`` (``G`` = chunk-total log decay, so
   the state recurrence becomes ``h <- exp(G) h + k_s @ v_new`` with no
   per-token decay in the sequential loop).  No ``tl.dot`` in this kernel.
2. ``prep_w_u`` (grid ``(chunks, B*H)``) -- a **cube-only** kernel: KKT
   ``L = k_e @ k_r^T``, Newton inverse, ``A = tril(q_e @ k_r^T)``,
   ``w = A_inv @ k_e`` and ``u = A_inv @ v_beta`` from block-pointer loads.
   ``A_inv`` never hits GM.
3. ``fwd_h_o`` (grid ``(V/BV, N*H)``) -- state scan fused with the output
   stage; ``h`` and ``v_new`` never hit GM.

Why the fold/prep split, and why fold is two kernels
----------------------------------------------------
Measured on T=1024/H=8 with the old fused prep (which gathered the decay
factors as 2D fp32 tiles *inside* the kernel and fed them into ``tl.dot``)::

    prep kernel (all dots + all gathers)      0.386 ms
    ... without the Newton chain              0.279 ms   -> Newton = 0.107
    ... with trivial factor tiles             0.092 ms   -> gathers = 0.294
    cube-only prep on pre-folded operands     0.080 ms   (no Newton, no A)

i.e. the dominant cost was *not* the Newton chain (0.107 ms) but the two 2D
fp32 gathers (0.294 ms): a gather result produced by the vector pipeline has to
be re-laid-out before it can feed a cube dot, and on 910B that conversion costs
~1.1 us per program per tile.  Feeding the same folds into *stores* (vector
layout) is cheap, so all folds moved into a vector kernel and the prep kernel
became cube-only: **prep device 0.386 -> 0.150 ms**.

That vector fold kernel then became the bottleneck (1.287 ms device at
T=1024/H=8), because on triton-ascend 3.5.1 several vector patterns *serialize*
across the grid (~5 us/program) instead of running in parallel.  The minimal
repro in ``test_script/probe_fold_parallel.py`` (see ``FOLD_PARALLELISM.md``)
identifies them: a bf16 2D gather, two or more 2D gathers, and -- the one that
bit the broadcast form -- **arithmetic on a 1D vector (e.g. ``exp``) whose
result is then broadcast into a 2D tile**.  Block-ptr 2D loads/stores, 2D
``cumsum``, multiple stores and plain 1D-load + ``[:, None]`` broadcast
multiplies all parallelise fine.

The fold is therefore two kernels:

* ``factor``: 1D only -- ``g_cum``, ``exp(g_cum)``, ``exp(-g_cum)``, fp32
  ``beta`` and ``beta*exp(g_cum)``, each written to a 1D fp32 buffer;
* ``fold``: plain 1D block loads of those buffers + ``[:, None]`` broadcast
  multiplies + 4 stores.  No 1D arithmetic is ever broadcast, no gathers.

Measured fold device time: 1.287 -> **0.102 (factor) + 0.076 (fold) = 0.18 ms**
at T=1024/H=8, which makes the 4-launch fused path the best one at every size
(``chunk.py`` now always uses it when eligible).

Fast-path invariants (measured on 910B, see ``wy_fast_ascend.py``): fp32
buffers for anything gathered as a 2D tile, gathered factors never feed a dot,
K in a single tile, no autotune.

Two-pass state scan (``_affine_kernel`` + ``_fwd_h_o_2pass_kernel``, enabled for
``NT >= 32`` / ``FLAG_ATTN_ASCEND_GDN_TWOPASS=1``)
    The delta-rule chunk recurrence is affine *with a matrix transition*:
    ``S' = a S + kr (u - w S) = (a I - a kr w) S + a kr u``.  Pass 1
    (chunk-parallel) precomputes ``qe2 = qe - A w``, ``Au = A u``,
    ``Cu = kr u`` and ``Mw = kr w``; pass 2 then needs only two dots per chunk
    (``o = qe2 S + Au``, ``S' = a (S + Cu - Mw S)``) instead of four, halving
    the serial chain.  Measured at T=2048/H=8: fwd_h_o device **0.256 -> 0.170
    ms**, e2e 0.813 -> ~0.80 ms.  Note that dropping the ``Mw S`` term (i.e.
    using only ``C_c = sum_j exp(G-g_j) k_j u_j^T`` and scalar prefix factors)
    is *not* exact; a fully parallel prefix scan would have to compose the
    K x K matrices (~31 combines/head, ~1 GMAC) and is more expensive than the
    sequential scan at these sizes.  Because the affine pass costs 0.109 ms of
    device plus ~0.10 ms of host, the two-pass path is enabled only once the
    sequential chain is long (NT >= 32); below that it is a net loss.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


# ---------------------------------------------------------------------------
# kernel 1: chunk cumsum + all per-token folds (vector-only, no dots)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _factor_kernel(
    g,
    beta,
    g_cum,
    e1,
    r1,
    b1,
    be1,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """1D-only per-token factors (all parallel-safe, see FOLD_PARALLELISM.md).

    Writes the chunk-local cumsum ``g_cum`` plus ``exp(g_cum)``,
    ``exp(-g_cum)`` and the fp32 copy of ``beta`` as 1D fp32 buffers.  Doing the
    exponentials here (1D) is what allows the fold kernel to use plain
    broadcast multiplies instead of ``exp`` on a 1D vector (which serializes).
    """
    i_t0, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos0 = i_b * T
    for j in tl.static_range(0, NCHUNK):
        i_t = i_t0 * NCHUNK + j
        bos = bos0
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        base = bos * H + i_h
        b_g = tl.load(
            tl.make_block_ptr(g + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_gc = tl.cumsum(b_g, axis=0)
        tl.store(
            tl.make_block_ptr(g_cum + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_gc, boundary_check=(0,),
        )
        b_e = tl.exp(b_gc)
        tl.store(
            tl.make_block_ptr(e1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_e, boundary_check=(0,),
        )
        tl.store(
            tl.make_block_ptr(r1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            tl.exp(tl.minimum(-b_gc, 80.0)), boundary_check=(0,),
        )
        b_beta = tl.load(
            tl.make_block_ptr(beta + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        tl.store(
            tl.make_block_ptr(b1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_beta, boundary_check=(0,),
        )
        tl.store(
            tl.make_block_ptr(be1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            b_beta * b_e, boundary_check=(0,),
        )


@triton.jit(do_not_specialize=["T"])
def _fold_kernel(
    q,
    k,
    v,
    e1,
    r1,
    b1,
    be1,
    qe,
    ke,
    kr,
    vb,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """Vector-only folds: plain 1D block loads + ``[:, None]`` broadcast muls.

    No 2D gather, no ``exp`` on a 1D vector, no 2D cumsum — all three would
    serialize this kernel on triton-ascend 3.5.1 (see FOLD_PARALLELISM.md).
    """
    i_t0, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    rep = H // Hg
    i_hg = i_h // rep
    bos0 = i_b * T
    for j in tl.static_range(0, NCHUNK):
        i_t = i_t0 * NCHUNK + j
        bos = bos0
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
            )
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        rep = H // Hg
        i_hg = i_h // rep
        base = bos * H + i_h
        base_qk = (bos * Hg + i_hg) * K

        b_e = tl.load(
            tl.make_block_ptr(e1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_r = tl.load(
            tl.make_block_ptr(r1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_beta = tl.load(
            tl.make_block_ptr(b1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)
        b_be = tl.load(
            tl.make_block_ptr(be1 + base, (T,), (H,), (i_t * BT,), (BT,), (0,)),
            boundary_check=(0,),
        ).to(tl.float32)

        b_k = tl.load(
            tl.make_block_ptr(
                k + base_qk, (T, K), (Hg * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                ke + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (b_k.to(tl.float32) * b_be[:, None]).to(ke.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                kr + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (b_k.to(tl.float32) * b_r[:, None]).to(kr.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
            ),
            (
                tl.load(
                    tl.make_block_ptr(
                        q + base_qk,
                        (T, K),
                        (Hg * K, 1),
                        (i_t * BT, 0),
                        (BT, K),
                        (1, 0),
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * b_e[:, None]
            ).to(qe.dtype.element_ty),
            boundary_check=(0, 1),
        )
        tl.store(
            tl.make_block_ptr(
                vb + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
            ),
            (
                tl.load(
                    tl.make_block_ptr(
                        v + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V),
                        (1, 0)
                    ),
                    boundary_check=(0, 1),
                ).to(tl.float32)
                * b_beta[:, None]
            ).to(vb.dtype.element_ty),
            boundary_check=(0, 1),
        )


@triton.jit(do_not_specialize=["T"])
def _fold_gather_kernel(
    q,
    k,
    v,
    g,
    beta,
    g_cum,
    qe,
    ke,
    kr,
    vb,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Single-kernel fold for *small* workloads (one chunk-head tile per program).

    This is the task-7 kernel: it does 2D gathers from the bf16 inputs (which
    serialize this backend's vector kernels, see FOLD_PARALLELISM.md), but with
    only ``chunks*H <= 64`` programs the serialized cost is small and one launch
    beats the two-launch ``factor`` + ``fold`` pair on host time.
    """
    i_t, i_bh, i_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    rep = H // Hg
    i_hg = i_h // rep
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = i_k * BK + tl.arange(0, BK)
    m_t = o_t < T
    base = bos * H + i_h
    base_qk = (bos * Hg + i_hg) * K

    p_g = tl.make_block_ptr(g + base, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_g1 = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    p_gc = tl.make_block_ptr(g_cum + base, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_gc, tl.cumsum(b_g1, axis=0), boundary_check=(0,))

    b_gc = tl.cumsum(
        tl.load(
            g + base + o_t[:, None] * H + 0 * o_k[None, :],
            mask=m_t[:, None],
            other=0.0,
        ).to(tl.float32),
        axis=0,
    )
    b_e = tl.exp(b_gc).to(tl.bfloat16)
    b_beta = tl.load(
        beta + base + o_t[:, None] * H + 0 * o_k[None, :],
        mask=m_t[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    b_k = tl.load(
        tl.make_block_ptr(
            k + base_qk,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            ke + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        b_k * (b_beta * b_e),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            kr + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        b_k * tl.exp(tl.minimum(-b_gc, 80.0)).to(tl.bfloat16),
        boundary_check=(0, 1),
    )
    tl.store(
        tl.make_block_ptr(
            qe + base * K,
            (T, K),
            (H * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        tl.load(
            tl.make_block_ptr(
                q + base_qk,
                (T, K),
                (Hg * K, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            ),
            boundary_check=(0, 1),
        )
        * b_e,
        boundary_check=(0, 1),
    )
    # The fast path requires K == V, so this K slice is the same V slice.
    tl.store(
        tl.make_block_ptr(
            vb + base * V,
            (T, V),
            (H * V, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        ),
        tl.load(
            tl.make_block_ptr(
                v + base * V,
                (T, V),
                (H * V, 1),
                (i_t * BT, i_k * BK),
                (BT, BK),
                (1, 0),
            ),
            boundary_check=(0, 1),
        )
        * b_beta,
        boundary_check=(0, 1),
    )


# ---------------------------------------------------------------------------
# kernel 2: cube-only prep (KKT + Newton inverse + w/u + output A)
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _prep_wu_kernel(
    qe,
    ke,
    kr,
    vb,
    w,
    u,
    a_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    N_NEWTON: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_b * T
    rep = H // Hg
    i_hg = i_h // rep

    o_t = i_t * BT + tl.arange(0, BT)
    base = bos * H + i_h
    # q/k are shared inside a GQA group but the decay is per head, so the
    # pre-folded q_e/k_e/k_r/k_s tensors are indexed per head; only ``w`` and
    # ``u`` are per H as well.
    base_qk = (bos * Hg + i_hg) * K

    p_ke = tl.make_block_ptr(
        ke + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base * K, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_ke = tl.load(p_ke, boundary_check=(0, 1))
    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    b_L = tl.dot(b_ke, b_kr)
    b_L = tl.where(o_t[:, None] > o_t[None, :], b_L, 0.0)

    # A_inv = (I + L)^-1 by Newton doubling; L is strictly lower triangular
    # (nilpotent, L^BT = 0), so log2(BT) - 1 steps are exact.
    o_r = tl.arange(0, BT)
    m_I = (o_r[:, None] == o_r[None, :]).to(tl.float32)
    b_M = b_L + m_I
    b_Ai = m_I - b_L
    for _ in tl.static_range(0, N_NEWTON):
        b_Y = tl.dot(b_M, b_Ai)
        b_Ai = tl.dot(b_Ai, 2.0 * m_I - b_Y)

    # A = tril(q_e @ k_r^T), the decayed causal q/k product chunk_o needs.
    p_qe = tl.make_block_ptr(
        qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    b_qe = tl.load(p_qe, boundary_check=(0, 1))
    b_A = tl.dot(b_qe, b_kr)
    b_A = tl.where(o_t[:, None] >= o_t[None, :], b_A, 0)
    p_A = tl.make_block_ptr(
        a_out + base * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    tl.store(p_A, b_A.to(p_A.dtype.element_ty), boundary_check=(0, 1))

    p_w = tl.make_block_ptr(
        w + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    b_w = tl.dot(b_Ai.to(b_ke.dtype), b_ke)
    tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))

    p_vb = tl.make_block_ptr(
        vb + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
    )
    p_u = tl.make_block_ptr(
        u + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, BV), (1, 0)
    )
    b_vb = tl.load(p_vb, boundary_check=(0, 1))
    b_u = tl.dot(b_Ai.to(b_vb.dtype), b_vb)
    tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# kernel 3: state scan + output stage, state tile stays in registers
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _fwd_h_o_kernel(
    qe,
    kr,
    a_in,
    w,
    u,
    g,
    o,
    h0,
    ht,
    scale,
    cu_seqlens,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_n * T
    NT = tl.cdiv(T, BT)

    b_h = tl.zeros([K, BV], dtype=tl.float32)

    base_h = (bos * H + i_h) * K
    base_a = (bos * H + i_h) * BT
    base_v = (bos * H + i_h) * V
    gb = g + bos * H + i_h
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
    if STORE_FINAL_STATE:
        ht += i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        p_qe = tl.make_block_ptr(
            qe + base_h, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        b_qe = tl.load(p_qe, boundary_check=(0, 1))
        p_A = tl.make_block_ptr(
            a_in + base_a, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
        )
        b_A = tl.load(p_A, boundary_check=(0, 1))
        p_kr = tl.make_block_ptr(
            kr + base_h, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
        )
        b_kr = tl.load(p_kr, boundary_check=(0, 1))
        p_w = tl.make_block_ptr(
            w + base_h, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_u = tl.make_block_ptr(
            u + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.load(p_u, boundary_check=(0, 1)).to(tl.float32)
        b_v = b_v - tl.dot(b_w, b_h.to(b_w.dtype))

        b_o = tl.dot(b_qe, b_h.to(b_qe.dtype)) + tl.dot(
            b_A, b_v.to(a_in.dtype.element_ty)
        )
        p_o = tl.make_block_ptr(
            o + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        tl.store(p_o, (b_o * scale).to(p_o.dtype.element_ty), boundary_check=(0, 1))

        # h <- exp(G) h + k_s^T v_new with k_s = k_r exp(G); equivalently
        # h <- exp(G) (h + k_r^T v_new), i.e. one scalar multiply per chunk.
        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.exp(tl.load(gb + last_idx * H).to(tl.float32))
        b_h = b_h + tl.dot(b_kr, b_v.to(kr.dtype.element_ty))
        if USE_G:
            b_h = b_h * b_g_last

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# exact two-pass state scan
#
# Per chunk the delta-rule recurrence is affine *with a matrix transition*:
#     v_new = u - w S
#     S'    = a S + kr (u - w S) = (a I - a kr w) S + a kr u
# so a chunk-parallel pass can precompute
#     qe2 = qe - A w        (o = qe2 S + A u)
#     Au  = A u             (o = qe2 S + Au)
#     Cu  = kr u            (S' = a (S + Cu - Mw S))
#     Mw  = kr w
# and the sequential part then needs only two dots per chunk (S' needs one:
# Mw S; the output needs one: qe2 S) with no u/w/A/kr loads at all.
#
# NOTE: the "C_c = sum_j exp(G-g_j) k_j u_j^T, prefix-combine with a_c only"
# recurrence would *not* be exact: the -a (kr w) S term is what Mw @ S below
# evaluates, and dropping it changes the results by O(1) per chunk.
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["T"])
def _affine_kernel(
    qe,
    a_in,
    kr,
    w,
    u,
    qe2,
    au,
    cu,
    mw,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    bos = i_b * T
    base = bos * H + i_h
    p_A = tl.make_block_ptr(
        a_in + base * BT, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0)
    )
    p_w = tl.make_block_ptr(
        w + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_u = tl.make_block_ptr(
        u + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
    )
    p_qe = tl.make_block_ptr(
        qe + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base * K, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_w = tl.load(p_w, boundary_check=(0, 1))
    b_u = tl.load(p_u, boundary_check=(0, 1))
    b_qe = tl.load(p_qe, boundary_check=(0, 1))

    b_qw = tl.dot(b_A, b_w.to(b_A.dtype))
    p_qe2 = tl.make_block_ptr(
        qe2 + base * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    tl.store(
        p_qe2,
        (b_qe.to(tl.float32) - b_qw).to(p_qe2.dtype.element_ty),
        boundary_check=(0, 1),
    )
    p_au = tl.make_block_ptr(
        au + base * V, (T, V), (H * V, 1), (i_t * BT, 0), (BT, V), (1, 0)
    )
    tl.store(p_au, tl.dot(b_A, b_u).to(p_au.dtype.element_ty),
             boundary_check=(0, 1))

    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    cbase_v = ((i_b * NT + i_t) * H + i_h) * K * V
    p_cu = tl.make_block_ptr(cu + cbase_v, (K, V), (V, 1), (0, 0), (K, V), (1, 0))
    tl.store(p_cu, tl.dot(b_kr, b_u).to(p_cu.dtype.element_ty))
    cbase_m = ((i_b * NT + i_t) * H + i_h) * K * K
    p_mw = tl.make_block_ptr(mw + cbase_m, (K, K), (K, 1), (0, 0), (K, K), (1, 0))
    tl.store(p_mw, tl.dot(b_kr, b_w).to(p_mw.dtype.element_ty))


@triton.jit(do_not_specialize=["T"])
def _fwd_h_o_2pass_kernel(
    qe2,
    au,
    cu,
    mw,
    g,
    o,
    h0,
    ht,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    bos = i_n * T
    b_h = tl.zeros([K, BV], dtype=tl.float32)
    base_v = (bos * H + i_h) * V
    base_k = (bos * H + i_h) * K
    gb = g + bos * H + i_h
    if USE_INITIAL_STATE:
        h0 += i_nh * K * V
    if STORE_FINAL_STATE:
        ht += i_nh * K * V
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(
            h0, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        cb_v = ((i_n * NT + i_t) * H + i_h) * K * V
        cb_m = ((i_n * NT + i_t) * H + i_h) * K * K
        p_qe2 = tl.make_block_ptr(
            qe2 + base_k, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
        )
        p_au = tl.make_block_ptr(
            au + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_qe2 = tl.load(p_qe2, boundary_check=(0, 1))
        b_au = tl.load(p_au, boundary_check=(0, 1)).to(tl.float32)
        p_o = tl.make_block_ptr(
            o + base_v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_o = tl.dot(b_qe2, b_h.to(b_qe2.dtype)) + b_au
        tl.store(p_o, (b_o * scale).to(p_o.dtype.element_ty), boundary_check=(0, 1))

        b_cu = tl.load(
            tl.make_block_ptr(
                cu + cb_v, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
            )
        )
        b_mw = tl.load(
            tl.make_block_ptr(mw + cb_m, (K, K), (K, 1), (0, 0), (K, K), (1, 0))
        )
        b_corr = tl.dot(b_mw, b_h.to(b_mw.dtype))
        last_idx = min((i_t + 1) * BT, T) - 1
        b_a = tl.exp(tl.load(gb + last_idx * H).to(tl.float32))
        b_h = (b_h + b_cu.to(tl.float32) - b_corr) * b_a

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht, (K, V), (V, 1), (0, i_v * BV), (K, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# ---------------------------------------------------------------------------
# host wrappers
# ---------------------------------------------------------------------------


# Workload size below which the single-kernel (gather) fold wins on host time.
_SMALL_FOLD_PROGRAMS = 64

# Chunk-batched fold/factor kernels (vector-only loops, no dots): each program
# handles NCHUNK consecutive chunks.  FLAG_ATTN_ASCEND_GDN_NCHUNK overrides the
# automatic choice; 1 disables batching.
_NCHUNK_ENV = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_NCHUNK", "0"))
# Automatic batching: measured net effect of NCHUNK>1 is at most -6% (T2048) and
# is neutral/negative elsewhere, so it is OFF unless requested explicitly.
_NCHUNK_AUTO = 1


def _fold_nchunk(NC: int, is_varlen: bool) -> int:
    if is_varlen:
        return 1
    if _NCHUNK_ENV > 0:
        return _NCHUNK_ENV if NC % _NCHUNK_ENV == 0 else 1
    return _NCHUNK_AUTO

# Fixed kernel configs, read once (not per call).
_FOLD_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_F_WARPS", "4"))
_FOLD_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_F_STAGES", "2"))
_PREP_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_WARPS", "4"))
_PREP_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_P_STAGES", "2"))
_HO_WARPS = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_WARPS", "4"))
_HO_STAGES = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_STAGES", "2"))
_HO_BV = int(os.environ.get("FLAG_ATTN_ASCEND_GDN_HO_BV", "0"))

# The fast path is host-bound, so its per-call intermediate tensors are cached
# (keyed by shape/dtype/device) instead of being allocated and freed every
# call.  ``o`` and the final state are always freshly allocated (they are
# returned to the caller); only never-returned scratch is pooled.  The cache is
# bounded, and can be disabled with FLAG_ATTN_ASCEND_GDN_SCRATCH=0.
_SCRATCH_LIMIT = 8
_scratch: "OrderedDict[tuple, dict]" = OrderedDict()


_OFF_VALUES = {"0", "false", "off", "no"}
_SCRATCH_ENABLED = (
    os.environ.get("FLAG_ATTN_ASCEND_GDN_SCRATCH", "1") not in _OFF_VALUES
)


def _scratch_enabled() -> bool:
    return _SCRATCH_ENABLED


def _fast_scratch(B, T, H, K, V, BT, dtype, device) -> dict:
    key = (B, T, H, K, V, BT, dtype, device.type, device.index)
    ent = _scratch.get(key)
    if ent is not None:
        _scratch.move_to_end(key)
        return ent
    if len(_scratch) >= _SCRATCH_LIMIT:
        # Defensive: the evicted buffers may still be referenced by kernels that
        # are queued but not executed yet.  A sync here is cheap (only happens
        # when a new shape is first seen) and keeps eviction safe.
        torch.npu.synchronize()
        _scratch.popitem(last=False)
    fp = torch.float32
    ent = {
        "g_cum": torch.empty(B, T, H, dtype=fp, device=device),
        "e1": torch.empty(B, T, H, dtype=fp, device=device),
        "r1": torch.empty(B, T, H, dtype=fp, device=device),
        "b1": torch.empty(B, T, H, dtype=fp, device=device),
        "be1": torch.empty(B, T, H, dtype=fp, device=device),
        "qe": torch.empty(B, T, H, K, dtype=dtype, device=device),
        "ke": torch.empty(B, T, H, K, dtype=dtype, device=device),
        "vb": torch.empty(B, T, H, V, dtype=dtype, device=device),
        "kr": torch.empty(B, T, H, K, dtype=dtype, device=device),
        "w": torch.empty(B, T, H, K, dtype=dtype, device=device),
        "u": torch.empty(B, T, H, V, dtype=dtype, device=device),
        "a": torch.empty(B, T, H, BT, dtype=dtype, device=device),
    }
    _scratch[key] = ent
    return ent


def clear_scratch_cache() -> None:
    """Drop the scratch pool (e.g. before releasing the device)."""
    _scratch.clear()


def _fold_config() -> tuple[int, int]:
    return _FOLD_WARPS, _FOLD_STAGES


def _prep_wu_config() -> tuple[int, int]:
    return _PREP_WARPS, _PREP_STAGES


def _fwd_h_o_bv(V: int) -> int:
    return _HO_BV or min(V, 64)


def _fwd_h_o_config() -> tuple[int, int]:
    return _HO_WARPS, _HO_STAGES


def fold_ascend(
    g: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """Return ``(g_cum, q_e, k_e, k_r, v_beta)`` (all bf16 but g_cum).

    Fixed-length inputs take the torch/CANN elementwise fold; varlen inputs use
    the two Triton vector kernels (``_factor_kernel`` + ``_fold_kernel``).
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    V = v.shape[-1]
    BT = int(chunk_size)
    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NC = triton.cdiv(T, BT) if not is_varlen else len(chunk_indices)
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, k.dtype, k.device)
        g_cum = buf["g_cum"]
        qe, ke, kr, vb = buf["qe"], buf["ke"], buf["kr"], buf["vb"]
    else:
        g_cum = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        qe = q.new_empty(B, T, H, K)
        ke = q.new_empty(B, T, H, K)
        kr = q.new_empty(B, T, H, K)
        vb = v.new_empty(B, T, H, V)
    warps, stages = _fold_config()
    if k.shape[-1] == v.shape[-1] and NC * H <= _SMALL_FOLD_PROGRAMS:
        # Small workload: one launch (serialized 2D gathers are cheap when the
        # grid is small, and host time dominates here).
        BK = 64 if K % 64 == 0 else K
        _fold_gather_kernel[(NC, H if is_varlen else B * H, K // BK)](
            q, k, v, g, beta, g_cum, qe, ke, kr, vb,
            cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, BK, is_varlen,
            num_warps=warps, num_stages=stages,
        )
        return g_cum, qe, ke, kr, vb
    if _scratch_enabled():
        e1, r1, b1, be1 = buf["e1"], buf["r1"], buf["b1"], buf["be1"]
    else:
        e1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        r1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        b1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
        be1 = torch.empty(B, T, H, dtype=torch.float32, device=g.device)
    nchunk = _fold_nchunk(NC, is_varlen)
    _factor_kernel[(NC // nchunk, H if is_varlen else B * H)](
        g, beta, g_cum, e1, r1, b1, be1,
        cu_seqlens, chunk_indices, T, H, BT, is_varlen, nchunk,
        num_warps=warps, num_stages=stages,
    )
    _fold_kernel[(NC // nchunk, H if is_varlen else B * H)](
        q, k, v, e1, r1, b1, be1, qe, ke, kr, vb,
        cu_seqlens, chunk_indices, T, H, Hg, K, V, BT, is_varlen, nchunk,
        num_warps=warps, num_stages=stages,
    )
    return g_cum, qe, ke, kr, vb


def prep_w_u_ascend(
    qe: torch.Tensor,
    ke: torch.Tensor,
    kr: torch.Tensor,
    vb: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(w, u, A)``: KKT/Newton inverse applied to the folded operands."""
    B, T, H, K = ke.shape
    V = vb.shape[-1]
    BT = int(chunk_size)
    BV = min(V, 128)  # single V tile in the kernel (V <= 128 on the fast path)
    is_varlen = cu_seqlens is not None
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if is_varlen else None
    NC = triton.cdiv(T, BT) if not is_varlen else len(chunk_indices)
    if _scratch_enabled():
        buf = _fast_scratch(B, T, H, K, V, BT, ke.dtype, ke.device)
        w, u, a_masked = buf["w"], buf["u"], buf["a"]
    else:
        w = ke.new_empty(B, T, H, K)
        u = vb.new_empty(B, T, H, V)
        a_masked = ke.new_empty(B, T, H, BT)
    warps, stages = _prep_wu_config()
    _prep_wu_kernel[(NC, H if is_varlen else B * H)](
        qe, ke, kr, vb, w, u, a_masked,
        cu_seqlens, chunk_indices, T, H, H, K, V, BT, BV,
        max(1, BT.bit_length() - 2), is_varlen,
        num_warps=warps, num_stages=stages,
    )
    return w, u, a_masked


def fwd_h_o_ascend(
    qe: torch.Tensor,
    a_in: torch.Tensor,
    kr: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None,
    chunk_size: int,
    bv: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``(o, final_state)``: state scan fused with the output stage."""
    B, T, H, _ = u.shape
    V = u.shape[-1]
    Hg, K = qe.shape[-2], qe.shape[-1]
    BT = int(chunk_size)
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BV = min(V, int(bv) if bv else _fwd_h_o_bv(V))
    o = torch.empty_like(u)
    final_state = (
        u.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )
    warps, stages = _fwd_h_o_config()
    _fwd_h_o_kernel[(triton.cdiv(V, BV), N * H)](
        qe, kr, a_in, w, u, g, o, initial_state, final_state, scale,
        cu_seqlens, T, H, Hg, K, V, BT, BV,
        g is not None,
        initial_state is not None,
        final_state is not None,
        cu_seqlens is not None,
        num_warps=num_warps or warps,
        num_stages=num_stages or stages,
    )
    return o, final_state


def affine_precompute_ascend(
    qe: torch.Tensor,
    a_in: torch.Tensor,
    kr: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-pass scan pass 1: chunk-parallel ``(qe2, Au, Cu, Mw)`` buffers."""
    B, T, H, K = qe.shape
    V = u.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    # Not the default path: allocate fresh (these buffers are large: Cu/Mw are
    # K x V / K x K per chunk, so they are not pooled in the scratch cache).
    qe2 = qe.new_empty(B, T, H, K)
    au = u.new_empty(B, T, H, V)
    cu = qe.new_empty(B, NT, H, K, V)
    mw = qe.new_empty(B, NT, H, K, K)
    warps, stages = _prep_wu_config()
    _affine_kernel[(NT, B * H)](
        qe, a_in, kr, w, u, qe2, au, cu, mw,
        T, H, K, V, BT, NT,
        num_warps=warps, num_stages=stages,
    )
    return qe2, au, cu, mw


def fwd_h_o_2pass_ascend(
    qe2: torch.Tensor,
    au: torch.Tensor,
    cu: torch.Tensor,
    mw: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    bv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Two-pass scan pass 2: chunk-parallel output + 2-dot state advance."""
    B, T, H, _ = au.shape
    V = au.shape[-1]
    K = qe2.shape[-1]
    BT = int(chunk_size)
    NT = triton.cdiv(T, BT)
    BV = min(V, int(bv) if bv else _fwd_h_o_bv(V))
    o = torch.empty_like(au)
    final_state = (
        au.new_empty(B, H, K, V, dtype=torch.float32)
        if output_final_state
        else None
    )
    warps, stages = _fwd_h_o_config()
    _fwd_h_o_2pass_kernel[(triton.cdiv(V, BV), B * H)](
        qe2, au, cu, mw, g, o, initial_state, final_state, scale,
        T, H, K, V, BT, BV, NT,
        initial_state is not None,
        final_state is not None,
        num_warps=warps, num_stages=stages,
    )
    return o, final_state


__all__ = [
    "fold_ascend",
    "prep_w_u_ascend",
    "fwd_h_o_ascend",
    "affine_precompute_ascend",
    "fwd_h_o_2pass_ascend",
    "clear_scratch_cache",
]
