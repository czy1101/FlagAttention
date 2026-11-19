# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend Triton kernels for paged DiffKV decode attention.

Design notes (backend lessons):

* one physical page per loop iteration, so both loads stay affine (the page
  base is a scalar from ``block_table``); a multi-page tile would need a
  gather, and gathers serialise on this backend;
* the inner tile is exactly one page, so the page-table lookup stays scalar;
* boundaries are an ``-inf`` score mask, never a masked load;
* fp32 accumulation, dot in the input dtype;
* KV split across programs, partials combined by a separate reduce kernel.
"""

import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950216)
_NEG = tl.constexpr(-1.0e30)


@triton.jit
def _diffkv_decode_kernel(
    q,
    k_cache,
    v_cache,
    block_table,
    lens,
    o_part,
    m_part,
    l_part,
    scale,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    DQK: tl.constexpr,
    DPAD: tl.constexpr,
    DV: tl.constexpr,
    BS: tl.constexpr,
    SPLITS: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_kv = tl.program_id(1)
    i_b = tl.program_id(2)

    length = tl.load(lens + i_b).to(tl.int32)
    start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    total = tl.maximum(length - start, 0)

    # tokens per split, rounded up to whole pages
    chunk = ((total + SPLITS - 1) // SPLITS + BS - 1) // BS * BS
    t_lo = start + i_s * chunk
    t_hi = tl.minimum(t_lo + chunk, length)

    rg = tl.arange(0, GROUP)
    rd = tl.arange(0, DPAD)
    rt = tl.arange(0, BS)
    rv = tl.arange(0, DV)

    h = i_kv * GROUP + rg
    msk_d = rd < DQK
    b_q = tl.load(q + (i_b * HQ + h)[:, None] * DQK + rd[None, :],
                  mask=msk_d[None, :], other=0.0)
    b_q = (b_q.to(tl.float32) * scale).to(b_q.dtype)

    b_m = tl.full([GROUP], _NEG, tl.float32)
    b_l = tl.zeros([GROUP], tl.float32)
    b_o = tl.zeros([GROUP, DV], tl.float32)

    n_pages = (t_hi + BS - 1) // BS
    p0 = t_lo // BS
    for i_p in tl.range(p0, n_pages):
        pid = tl.load(block_table + i_b * BT_STRIDE + i_p).to(tl.int64)
        # key page loaded transposed: [DPAD, BS]
        b_k = tl.load(k_cache + pid * (BS * HKV * DQK) + i_kv * DQK
                      + rt[None, :] * (HKV * DQK) + rd[:, None],
                      mask=msk_d[:, None], other=0.0)
        b_s = tl.dot(b_q, b_k)                       # [GROUP, BS]
        pos = i_p * BS + rt
        valid = (pos >= t_lo) & (pos < t_hi)
        b_s = tl.where(valid[None, :], b_s, _NEG)

        m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.exp2((b_m - m_new) * _LOG2E)
        b_p = tl.exp2((b_s - m_new[:, None]) * _LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)
        b_l = b_l * alpha + tl.sum(b_p, 1)

        b_v = tl.load(v_cache + pid * (BS * HKV * DV) + i_kv * DV
                      + rt[:, None] * (HKV * DV) + rv[None, :])
        b_o = b_o * alpha[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = m_new

    row = (i_b * HQ + h) * SPLITS + i_s
    tl.store(o_part + row[:, None] * DV + rv[None, :], b_o)
    tl.store(m_part + row, b_m)
    tl.store(l_part + row, b_l)


@triton.jit
def _diffkv_reduce_kernel(
    o_part,
    m_part,
    l_part,
    out,
    DV: tl.constexpr,
    SPLITS: tl.constexpr,
    SPAD: tl.constexpr,
):
    i_row = tl.program_id(0)
    rs = tl.arange(0, SPAD)
    rv = tl.arange(0, DV)
    msk_s = rs < SPLITS

    base = i_row * SPLITS
    b_m = tl.load(m_part + base + rs, mask=msk_s, other=_NEG)
    b_l = tl.load(l_part + base + rs, mask=msk_s, other=0.0)

    b_M = tl.max(b_m, 0)
    b_w = tl.exp2((b_m - b_M) * _LOG2E)
    denom = tl.sum(b_w * b_l, 0)

    b_o = tl.load(o_part + (base + rs)[:, None] * DV + rv[None, :],
                  mask=msk_s[:, None], other=0.0)
    acc = tl.sum(b_w[:, None] * b_o, 0)
    acc = acc / tl.where(denom > 0.0, denom, 1.0)
    tl.store(out + i_row * DV + rv, acc.to(out.dtype.element_ty))


@triton.jit
def _diffkv_depage_kernel(
    k_cache,
    v_cache,
    block_table,
    lens,
    k_dense,
    v_dense,
    HKV: tl.constexpr,
    DQK: tl.constexpr,
    KFLAT: tl.constexpr,     # next_pow2(HKV * DQK)
    DV: tl.constexpr,
    VFLAT: tl.constexpr,     # next_pow2(HKV * DV)
    BS: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    LSTRIDE: tl.constexpr,   # elements per token in the dense buffer
    PPG: tl.constexpr,       # pages per program
):
    """De-page the cache: one program copies PPG contiguous pages."""
    i_pg = tl.program_id(0)
    i_b = tl.program_id(1)

    length = tl.load(lens + i_b).to(tl.int32)
    n_pages = (length + BS - 1) // BS

    rt = tl.arange(0, BS)
    rk = tl.arange(0, KFLAT)
    rv = tl.arange(0, VFLAT)
    msk_k = rk < HKV * DQK

    for j in tl.static_range(PPG):
        p = i_pg * PPG + j
        if p < n_pages:
            pid = tl.load(block_table + i_b * BT_STRIDE + p).to(tl.int64)
            b_k = tl.load(k_cache + pid * (BS * HKV * DQK) + rt[:, None] * (HKV * DQK) + rk[None, :],
                          mask=msk_k[None, :], other=0.0)
            tl.store(k_dense + ((i_b * LSTRIDE + p * BS) + rt)[:, None] * (HKV * DQK) + rk[None, :],
                     b_k, mask=msk_k[None, :])
            b_v = tl.load(v_cache + pid * (BS * HKV * DV) + rt[:, None] * (HKV * DV) + rv[None, :])
            tl.store(v_dense + ((i_b * LSTRIDE + p * BS) + rt)[:, None] * (HKV * DV) + rv[None, :], b_v)


@triton.jit
def _diffkv_decode_dense_kernel(
    q,
    k_dense,
    v_dense,
    lens,
    o_part,
    m_part,
    l_part,
    scale,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GROUP: tl.constexpr,
    DQK: tl.constexpr,
    DPAD: tl.constexpr,
    DV: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    LSTRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
):
    """Split-KV decode over a de-paged, contiguous KV buffer."""
    i_s = tl.program_id(0)
    i_kv = tl.program_id(1)
    i_b = tl.program_id(2)

    length = tl.load(lens + i_b).to(tl.int32)
    start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    total = tl.maximum(length - start, 0)
    chunk = (total + SPLITS - 1) // SPLITS
    t_lo = start + i_s * chunk
    t_hi = tl.minimum(t_lo + chunk, length)

    rg = tl.arange(0, GROUP)
    rd = tl.arange(0, DPAD)
    rt = tl.arange(0, BN)
    rv = tl.arange(0, DV)

    h = i_kv * GROUP + rg
    msk_d = rd < DQK
    b_q = tl.load(q + (i_b * HQ + h)[:, None] * DQK + rd[None, :],
                  mask=msk_d[None, :], other=0.0)
    b_q = (b_q.to(tl.float32) * scale).to(b_q.dtype)

    b_m = tl.full([GROUP], _NEG, tl.float32)
    b_l = tl.zeros([GROUP], tl.float32)
    b_o = tl.zeros([GROUP, DV], tl.float32)

    k_base = k_dense + i_b * LSTRIDE * HKV * DQK + i_kv * DQK
    v_base = v_dense + i_b * LSTRIDE * HKV * DV + i_kv * DV
    n_blocks = tl.cdiv(t_hi - t_lo, BN)
    for i_t in tl.range(0, n_blocks):
        t0 = t_lo + i_t * BN
        b_k = tl.load(k_base + (t0 + rt)[None, :] * (HKV * DQK) + rd[:, None],
                      mask=msk_d[:, None], other=0.0)          # [DPAD, BN]
        b_s = tl.dot(b_q, b_k)                                 # [GROUP, BN]
        pos = t0 + rt
        valid = (pos >= t_lo) & (pos < t_hi)
        b_s = tl.where(valid[None, :], b_s, _NEG)

        m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.exp2((b_m - m_new) * _LOG2E)
        b_p = tl.exp2((b_s - m_new[:, None]) * _LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)
        b_l = b_l * alpha + tl.sum(b_p, 1)

        b_v = tl.load(v_base + (t0 + rt)[:, None] * (HKV * DV) + rv[None, :])
        b_o = b_o * alpha[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = m_new

    row = (i_b * HQ + h) * SPLITS + i_s
    tl.store(o_part + row[:, None] * DV + rv[None, :], b_o)
    tl.store(m_part + row, b_m)
    tl.store(l_part + row, b_l)
