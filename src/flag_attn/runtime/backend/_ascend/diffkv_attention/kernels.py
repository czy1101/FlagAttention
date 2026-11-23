# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Triton kernels for paged DiffKV decode attention."""

import triton
import triton.language as tl

_LOG2E = tl.constexpr(1.4426950216)
_NEG = tl.constexpr(-1.0e30)


@triton.jit
def _diffkv_decode_window_kernel(
    q,
    k_cache,
    v_cache,
    lens,
    block_table,
    out,
    scale,
    HQ: tl.constexpr,
    GROUP: tl.constexpr,
    BS: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    WINDOW: tl.constexpr,
    DQK: tl.constexpr,
    DPAD: tl.constexpr,
    DV: tl.constexpr,
    BN: tl.constexpr,
):
    """Gather the active window into one tile and normalize it in place."""
    i_b = tl.program_id(0)
    i_kv = tl.program_id(1)
    length = tl.load(lens + i_b)
    start = tl.maximum(0, length - WINDOW)
    pos = start + tl.arange(0, BN)
    valid = pos < length
    page = tl.load(
        block_table + i_b.to(tl.int64) * BT_STRIDE + pos // BS,
        mask=valid,
        other=0,
    ).to(tl.int64)
    token = pos % BS
    h = i_kv * GROUP + tl.arange(0, GROUP)
    rd = tl.arange(0, DPAD)
    rv = tl.arange(0, DV)
    b_q = tl.load(
        q + (i_b * HQ + h[:, None]) * DQK + rd[None, :],
        mask=rd[None, :] < DQK,
        other=0.0,
    )
    k_base = page * K_STRIDES[0] + token * K_STRIDES[1] + i_kv * K_STRIDES[2]
    b_k = tl.load(
        k_cache + k_base[:, None] + rd[None, :],
        mask=valid[:, None] & (rd[None, :] < DQK),
        other=0.0,
    )
    b_s = tl.where(valid[None, :], tl.dot(b_q, tl.trans(b_k)) * scale, _NEG)
    b_m = tl.max(b_s, 1)
    b_p = tl.where(valid[None, :], tl.exp(b_s - b_m[:, None]), 0.0)
    b_l = tl.sum(b_p, 1)
    v_base = page * V_STRIDES[0] + token * V_STRIDES[1] + i_kv * V_STRIDES[2]
    b_v = tl.load(v_cache + v_base[:, None] + rv[None, :], mask=valid[:, None], other=0.0)
    b_o = tl.dot(b_p.to(b_v.dtype), b_v) / tl.where(b_l > 0.0, b_l, 1.0)[:, None]
    tl.store(out + (i_b * HQ + h[:, None]) * DV + rv[None, :], b_o)


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
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    SPLITS: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    FINAL: tl.constexpr,
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

    b_m = tl.full([GROUP], _NEG, tl.float32)
    b_l = tl.zeros([GROUP], tl.float32)
    b_o = tl.zeros([GROUP, DV], tl.float32)

    n_pages = tl.where(t_hi > t_lo, tl.cdiv(t_hi, BS), 0)
    p0 = t_lo // BS
    for i_p in tl.range(p0, n_pages):
        pid = tl.load(block_table + i_b.to(tl.int64) * BT_STRIDE + i_p).to(tl.int64)
        # key page loaded transposed: [DPAD, BS]
        b_k = tl.load(k_cache + pid * K_STRIDES[0] + i_kv * K_STRIDES[2]
                      + rt[None, :] * K_STRIDES[1] + rd[:, None],
                      mask=msk_d[:, None], other=0.0)
        b_s = tl.dot(b_q, b_k) * scale                       # [GROUP, BS]
        pos = i_p * BS + rt
        valid = (pos >= t_lo) & (pos < t_hi)
        b_s = tl.where(valid[None, :], b_s, _NEG)

        m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.exp2((b_m - m_new) * _LOG2E)
        b_p = tl.exp2((b_s - m_new[:, None]) * _LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)
        b_l = b_l * alpha + tl.sum(b_p, 1)

        b_v = tl.load(v_cache + pid * V_STRIDES[0] + i_kv * V_STRIDES[2]
                      + rt[:, None] * V_STRIDES[1] + rv[None, :],
                      mask=valid[:, None], other=0.0)
        b_o = b_o * alpha[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = m_new

    if FINAL:
        b_o = b_o / tl.where(b_l > 0.0, b_l, 1.0)[:, None]
        tl.store(o_part + (i_b * HQ + h)[:, None] * DV + rv[None, :],
                 b_o.to(o_part.dtype.element_ty))
    else:
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
    DV: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    BS: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    LSTRIDE: tl.constexpr,
    PPG: tl.constexpr,
    WINDOW: tl.constexpr,
    BN: tl.constexpr,
    NPAGES_MAX: tl.constexpr,
):
    i_pg = tl.program_id(0)
    i_b = tl.program_id(1)
    if HEAD_MAJOR:
        i_kv = tl.program_id(2)
        K_WIDTH: tl.constexpr = DQK
        V_WIDTH: tl.constexpr = DV
        dense_base = (i_b * HKV + i_kv).to(tl.int64) * LSTRIDE
    else:
        i_kv = 0
        K_WIDTH: tl.constexpr = HKV * DQK
        V_WIDTH: tl.constexpr = HKV * DV
        dense_base = i_b.to(tl.int64) * LSTRIDE

    length = tl.load(lens + i_b)
    start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    p_start = start // BS
    n_pages = tl.cdiv(length, BS) - p_start
    # Dense decode reads a whole tile at the tail, before masking its scores.
    n_zero = tl.minimum(n_pages + tl.cdiv(BN, BS), NPAGES_MAX)
    rt = tl.arange(0, BS)
    rk = tl.arange(0, triton.next_power_of_2(K_WIDTH))
    rv = tl.arange(0, triton.next_power_of_2(V_WIDTH))

    for j in tl.static_range(PPG):
        p = i_pg * PPG + j
        tokens = dense_base + p * BS + rt
        dst_k = k_dense + tokens[:, None] * K_WIDTH + rk[None, :]
        dst_v = v_dense + tokens[:, None] * V_WIDTH + rv[None, :]
        if p < n_pages:
            logical_page = p_start + p
            pid = tl.load(block_table + i_b.to(tl.int64) * BT_STRIDE + logical_page).to(tl.int64)
            src_k = k_cache + pid * K_STRIDES[0] + i_kv * K_STRIDES[2]
            src_v = v_cache + pid * V_STRIDES[0] + i_kv * V_STRIDES[2]
            b_k = tl.load(src_k + rt[:, None] * K_STRIDES[1] + rk[None, :],
                          mask=rk[None, :] < K_WIDTH, other=0.0)
            b_v = tl.load(src_v + rt[:, None] * V_STRIDES[1] + rv[None, :],
                          mask=rv[None, :] < V_WIDTH, other=0.0)
            tl.store(dst_k, b_k, mask=rk[None, :] < K_WIDTH)
            tl.store(dst_v, b_v, mask=rv[None, :] < V_WIDTH)
        elif p < n_zero:
            tl.store(dst_k, 0.0, mask=rk[None, :] < K_WIDTH)
            tl.store(dst_v, 0.0, mask=rv[None, :] < V_WIDTH)


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
    BS_PAGE: tl.constexpr,
    HEAD_MAJOR: tl.constexpr,
    SPLITS: tl.constexpr,
    LSTRIDE: tl.constexpr,
    WINDOW: tl.constexpr,
    FINAL: tl.constexpr,
):
    """Split-KV decode over a de-paged, contiguous KV buffer.

    ``FINAL`` means the program owns the whole KV range, so it normalises and
    writes the output itself instead of leaving partials for a reduce kernel
    (saves one launch, which on this backend costs more than the work).
    """
    i_s = tl.program_id(0)
    i_kv = tl.program_id(1)
    i_b = tl.program_id(2)

    length = tl.load(lens + i_b).to(tl.int32)
    start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    # The de-page pass fills whole pages from the window start, so the token
    # base is page-aligned to guarantee every address it touches was written
    t_base = (start // BS_PAGE) * BS_PAGE
    start -= t_base
    length -= t_base
    chunk = tl.cdiv(length, SPLITS)
    t_lo = i_s * chunk
    t_hi = tl.minimum(t_lo + chunk, length)

    rg = tl.arange(0, GROUP)
    rd = tl.arange(0, DPAD)
    rt = tl.arange(0, BN)
    rv = tl.arange(0, DV)

    h = i_kv * GROUP + rg
    msk_d = rd < DQK
    b_q = tl.load(q + (i_b * HQ + h)[:, None] * DQK + rd[None, :],
                  mask=msk_d[None, :], other=0.0)

    b_m = tl.full([GROUP], _NEG, tl.float32)
    b_l = tl.zeros([GROUP], tl.float32)
    b_o = tl.zeros([GROUP, DV], tl.float32)

    if HEAD_MAJOR:
        dense_base = (i_b * HKV + i_kv).to(tl.int64) * LSTRIDE
        K_TOKEN: tl.constexpr = DQK
        V_TOKEN: tl.constexpr = DV
        k_base = k_dense + dense_base * DQK
        v_base = v_dense + dense_base * DV
    else:
        dense_base = i_b.to(tl.int64) * LSTRIDE
        K_TOKEN: tl.constexpr = HKV * DQK
        V_TOKEN: tl.constexpr = HKV * DV
        k_base = k_dense + dense_base * K_TOKEN + i_kv * DQK
        v_base = v_dense + dense_base * V_TOKEN + i_kv * DV
    n_blocks = tl.cdiv(t_hi - t_lo, BN)
    for i_t in tl.range(0, n_blocks):
        t0 = t_lo + i_t * BN
        pos = t0 + rt
        b_k = tl.load(k_base + pos[None, :] * K_TOKEN + rd[:, None],
                      mask=msk_d[:, None], other=0.0)          # [DPAD, BN]
        b_s = tl.dot(b_q, b_k) * scale
        valid = (pos >= start) & (pos < t_hi)
        b_s = tl.where(valid[None, :], b_s, _NEG)

        m_new = tl.maximum(b_m, tl.max(b_s, 1))
        alpha = tl.exp2((b_m - m_new) * _LOG2E)
        b_p = tl.exp2((b_s - m_new[:, None]) * _LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)
        b_l = b_l * alpha + tl.sum(b_p, 1)

        b_v = tl.load(v_base + pos[:, None] * V_TOKEN + rv[None, :])
        b_v = tl.where(valid[:, None], b_v, 0.0)
        b_o = b_o * alpha[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_m = m_new

    if FINAL:
        b_o = b_o / tl.where(b_l > 0.0, b_l, 1.0)[:, None]
        tl.store(o_part + (i_b * HQ + h)[:, None] * DV + rv[None, :],
                 b_o.to(o_part.dtype.element_ty))
    else:
        row = (i_b * HQ + h) * SPLITS + i_s
        tl.store(o_part + row[:, None] * DV + rv[None, :], b_o)
        tl.store(m_part + row, b_m)
        tl.store(l_part + row, b_l)
