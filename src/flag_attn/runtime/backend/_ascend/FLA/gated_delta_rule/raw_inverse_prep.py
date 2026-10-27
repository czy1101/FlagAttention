"""Fixed-length TLE-Raw inverse preparation for eligible GDN workloads.

This module is deliberately separate from the existing prep implementation.
It materializes a dense contiguous FP32 64x64 chunk-local KKT matrix, applies
the Raw inverse to its two diagonal 32x32 blocks, stitches the full 64x64
inverse, then computes the existing two-pass affine outputs. The dense KKT
write avoids the severe serialization seen with dynamic compact-block
addressing on CANN 9.1. Dispatch is restricted to the measured large-workload
class and can be disabled with ``FLAG_ATTN_ASCEND_GDN_RAW_INVERSE=0``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .raw_inverse_ascend import (
    launch_inverse64_rawdiag_outer_batch2 as launch_inverse64_rawdiag_outer,
)


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_kernel(ke, kr, L_FULL, T,
                       H: tl.constexpr, K: tl.constexpr,
                       NT: tl.constexpr, BT: tl.constexpr):
    """Write dense FP32 `[BT,BT]` strictly-lower `ke @ kr.T` per chunk."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = (i_b * T) * H * K + i_h * K
    p_ke = tl.make_block_ptr(
        ke + base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_l = tl.dot(tl.load(p_ke, boundary_check=(0, 1)),
                 tl.load(p_kr, boundary_check=(0, 1)))
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    lower = r[:, None] > c[None, :]
    b_l = tl.where(lower, b_l, 0.0).to(tl.float32)

    tile = (i_b * NT + i_t) * H + i_h
    # Keep the KKT tile in its native contiguous 64x64 layout. Dynamic
    # three-block offset selection serialized this producer on CANN 9.1.
    tl.store(L_FULL + tile * (BT * BT) + r[:, None] * BT + c[None, :], b_l)


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_one_job_batched(ke, kr, L_FULL, T,
                                H: tl.constexpr, K: tl.constexpr,
                                NT: tl.constexpr, BT: tl.constexpr,
                                i_t, i_bh):
    i_b, i_h = i_bh // H, i_bh % H
    base = (i_b * T) * H * K + i_h * K
    p_ke = tl.make_block_ptr(
        ke + base, (T, K), (H * K, 1), (i_t * BT, 0), (BT, K), (1, 0)
    )
    p_kr = tl.make_block_ptr(
        kr + base, (K, T), (1, H * K), (0, i_t * BT), (K, BT), (0, 1)
    )
    b_l = tl.dot(
        tl.load(p_ke, boundary_check=(0, 1), padding_option="zero"),
        tl.load(p_kr, boundary_check=(0, 1), padding_option="zero"),
    )
    r = tl.arange(0, BT)
    c = tl.arange(0, BT)
    valid_rows = (i_t * BT + r < T)[:, None]
    valid_cols = (i_t * BT + c < T)[None, :]
    lower = r[:, None] > c[None, :]
    b_l = tl.where(lower & valid_rows & valid_cols, b_l, 0.0).to(tl.float32)
    tile = (i_b * NT + i_t) * H + i_h
    tl.store(L_FULL + tile * (BT * BT) + r[:, None] * BT + c[None, :], b_l)


@triton.jit(do_not_specialize=["T"])
def _kkt_blocks_batched_kernel(ke, kr, L_FULL, T,
                               H: tl.constexpr, K: tl.constexpr,
                               NT: tl.constexpr, BT: tl.constexpr,
                               B: tl.constexpr, JOBS: tl.constexpr,
                               BATCH: tl.constexpr):
    first = tl.program_id(0) * BATCH
    for j in tl.static_range(BATCH):
        job = first + j
        if job < JOBS:
            i_t = job // (B * H)
            i_bh = job % (B * H)
            _kkt_blocks_one_job_batched(
                ke, kr, L_FULL, T, H, K, NT, BT, i_t, i_bh
            )


def launch_kkt_blocks_batched(ke: torch.Tensor, kr: torch.Tensor,
                              L_full: torch.Tensor, T: int, H: int, K: int,
                              NT: int, BT: int, batch_size: int = 4) -> None:
    """Produce dense KKT tiles with 2 or 4 independent jobs per CTA."""
    if batch_size not in (2, 4):
        raise ValueError("KKT batch_size must be 2 or 4")
    batch = ke.shape[0]
    jobs = batch * NT * H
    _kkt_blocks_batched_kernel[(triton.cdiv(jobs, batch_size),)](
        ke, kr, L_full, T, H, K, NT, BT, batch, jobs, batch_size,
        num_warps=4, num_stages=2,
    )


@triton.jit
def _stitch_inverse_kernel(L_FULL, DIAG, INVERSE):
    tile = tl.program_id(0)
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    a = tl.load(DIAG + tile * 2112 + r[:, None] * 32 + c[None, :])
    d = tl.load(DIAG + tile * 2112 + 1056 + r[:, None] * 32 + c[None, :])
    cb = tl.load(L_FULL + tile * 4096 + (r[:, None] + 32) * 64 + c[None, :])
    lo = -tl.dot(tl.dot(d, cb), a)
    base = INVERSE + tile * 4096 + r[:, None] * 64 + c[None, :]
    tl.store(base, a)
    tl.store(base + 32 * 64, lo)
    tl.store(base + 32, tl.full((32, 32), 0.0, tl.float32))
    tl.store(base + 32 * 64 + 32, d)


@triton.jit
def _stitch_inverse_one_tile(L_FULL, DIAG, INVERSE, tile):
    r = tl.arange(0, 32)
    c = tl.arange(0, 32)
    a = tl.load(DIAG + tile * 2112 + r[:, None] * 32 + c[None, :])
    d = tl.load(DIAG + tile * 2112 + 1056 + r[:, None] * 32 + c[None, :])
    cb = tl.load(L_FULL + tile * 4096 + (r[:, None] + 32) * 64 + c[None, :])
    lo = -tl.dot(tl.dot(d, cb), a)
    base = INVERSE + tile * 4096 + r[:, None] * 64 + c[None, :]
    tl.store(base, a)
    tl.store(base + 32 * 64, lo)
    tl.store(base + 32, tl.full((32, 32), 0.0, tl.float32))
    tl.store(base + 32 * 64 + 32, d)


@triton.jit
def _stitch_inverse_batched_kernel(L_FULL, DIAG, INVERSE,
                                   JOBS: tl.constexpr,
                                   BATCH: tl.constexpr):
    first = tl.program_id(0) * BATCH
    for j in tl.static_range(BATCH):
        tile = first + j
        if tile < JOBS:
            _stitch_inverse_one_tile(L_FULL, DIAG, INVERSE, tile)


def launch_stitch_inverse_batched(L_full: torch.Tensor,
                                  diag: torch.Tensor,
                                  inverse: torch.Tensor,
                                  batch_size: int = 4) -> None:
    """Stitch a small static batch of independent inverse tiles per CTA."""
    if batch_size < 1 or batch_size > 4:
        raise ValueError("stitch batch_size must be in [1, 4]")
    jobs = L_full.shape[0]
    _stitch_inverse_batched_kernel[(triton.cdiv(jobs, batch_size),)](
        L_full, diag, inverse, jobs, batch_size, num_warps=4, num_stages=1,
    )


def launch_stitch_inverse_batch4(L_full: torch.Tensor,
                                 diag: torch.Tensor,
                                 inverse: torch.Tensor) -> None:
    """Default four-tile stitch launch for the eligible Raw prep path."""
    launch_stitch_inverse_batched(L_full, diag, inverse, batch_size=4)


@triton.jit(do_not_specialize=["T"])
def _prep_affine_from_inverse_kernel(
    qe, ke, kr, vb, INVERSE,
    qe2, au, cu, mw, a_out,
    T,
    H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    NT: tl.constexpr, BT: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    base = i_b * T * H + i_h
    base_k = base * K
    base_v = base * V
    tile = (i_b * NT + i_t) * H + i_h
    r = tl.arange(0, BT)
    p_ai = tl.make_block_ptr(INVERSE + tile * BT * BT,
                             (BT, BT), (BT, 1), (0, 0), (BT, BT), (1, 0))
    b_ai = tl.load(p_ai)

    p_ke = tl.make_block_ptr(ke + base_k, (T, K), (H * K, 1),
                             (i_t * BT, 0), (BT, K), (1, 0))
    p_kr = tl.make_block_ptr(kr + base_k, (K, T), (1, H * K),
                             (0, i_t * BT), (K, BT), (0, 1))
    b_ke = tl.load(p_ke, boundary_check=(0, 1))
    b_kr = tl.load(p_kr, boundary_check=(0, 1))
    b_w = tl.dot(b_ai.to(b_ke.dtype), b_ke)
    b_u = tl.dot(b_ai.to(b_ke.dtype), tl.load(tl.make_block_ptr(
        vb + base_v, (T, V), (H * V, 1), (i_t * BT, 0),
        (BT, V), (1, 0)), boundary_check=(0, 1)))
    b_w = b_w.to(a_out.dtype.element_ty)
    b_u = b_u.to(a_out.dtype.element_ty)

    p_qe = tl.make_block_ptr(qe + base_k, (T, K), (H * K, 1),
                             (i_t * BT, 0), (BT, K), (1, 0))
    b_qe = tl.load(p_qe, boundary_check=(0, 1))
    o_t = i_t * BT + r
    b_a = tl.dot(b_qe, b_kr)
    b_a = tl.where(o_t[:, None] >= o_t[None, :], b_a, 0.0)
    b_a = b_a.to(a_out.dtype.element_ty)
    tl.store(tl.make_block_ptr(a_out + base * BT, (T, BT), (H * BT, 1),
                              (i_t * BT, 0), (BT, BT), (1, 0)),
             b_a, boundary_check=(0, 1))

    b_qw = tl.dot(b_a, b_w.to(b_a.dtype))
    tl.store(tl.make_block_ptr(qe2 + base_k, (T, K), (H * K, 1),
                              (i_t * BT, 0), (BT, K), (1, 0)),
             (b_qe.to(tl.float32) - b_qw).to(qe2.dtype.element_ty),
             boundary_check=(0, 1))
    tl.store(tl.make_block_ptr(au + base_v, (T, V), (H * V, 1),
                              (i_t * BT, 0), (BT, V), (1, 0)),
             tl.dot(b_a, b_u).to(au.dtype.element_ty), boundary_check=(0, 1))

    cbase_v = ((i_b * NT + i_t) * H + i_h) * K * V
    p_cu = tl.make_block_ptr(cu + cbase_v, (K, V), (V, 1),
                             (0, 0), (K, V), (1, 0))
    tl.store(p_cu, tl.dot(b_kr, b_u.to(b_kr.dtype)).to(p_cu.dtype.element_ty))
    cbase_m = ((i_b * NT + i_t) * H + i_h) * K * K
    p_mw = tl.make_block_ptr(mw + cbase_m, (K, K), (K, 1),
                             (0, 0), (K, K), (1, 0))
    tl.store(p_mw, tl.dot(b_kr, b_w.to(b_kr.dtype)).to(p_mw.dtype.element_ty))


def raw_inverse_prep_affine_ascend(qe, ke, kr, vb, chunk_size, output_dtype=None):
    """Replace affine prep with dense KKT, Raw inverse, stitch, and affine prep."""
    B, T, H, K = ke.shape
    V = vb.shape[-1]
    BT = int(chunk_size)
    if BT != 64 or V != K or K > 128:
        raise ValueError("Raw prep supports only fixed BT=64 and K=V<=128")
    NT = triton.cdiv(T, BT)
    output_dtype = output_dtype or ke.dtype
    L_full = torch.empty((B, NT, H, BT, BT),
                         dtype=torch.float32, device=ke.device)
    diag = torch.empty((B, NT, H, 2, 33, 32),
                       dtype=torch.float32, device=ke.device)
    inverse = torch.empty((B, NT, H, BT, BT),
                          dtype=torch.float32, device=ke.device)
    qe2 = qe.new_empty((B, T, H, K))
    au = torch.empty((B, T, H, V), dtype=output_dtype, device=vb.device)
    cu = qe.new_empty((B, NT, H, K, V))
    mw = qe.new_empty((B, NT, H, K, K))
    a_out = torch.empty((B, T, H, BT), dtype=output_dtype, device=ke.device)
    launch_kkt_blocks_batched(ke, kr, L_full, T, H, K, NT, BT,
                              batch_size=4)
    L_flat = L_full.view(-1, BT, BT)
    diag_flat = diag.view(-1, 2, 33, 32)
    launch_inverse64_rawdiag_outer(L_flat, diag_flat)
    launch_stitch_inverse_batch4(L_flat, diag_flat, inverse)
    _prep_affine_from_inverse_kernel[(NT, B * H)](
        qe, ke, kr, vb, inverse, qe2, au, cu, mw, a_out,
        T, H, K, V, NT, BT,
        num_warps=4, num_stages=2,
    )
    return qe2, au, cu, mw, a_out


__all__ = ["raw_inverse_prep_affine_ascend"]
