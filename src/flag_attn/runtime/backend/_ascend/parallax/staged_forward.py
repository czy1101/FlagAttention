"""Fixed-length, full-causal Parallax forward staged through GM intermediates."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _score_tiles(
    q,
    r,
    k,
    score_ptr,
    rk_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int32)
    pid_n = tl.program_id(1).to(tl.int32)
    bh = tl.program_id(2).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    row_start = pid_m * BM
    col_start = pid_n * BN
    if col_start > row_start + BM - 1:
        return

    rows = row_start + tl.arange(0, BM)
    cols = col_start + tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    col_mask = cols < T
    feature_mask = d < D
    q_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    r_offsets = q_offsets
    kv_offsets = ((batch * T + cols[:, None]) * H + hk) * D + d[None, :]
    bq = tl.load(q + q_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    br = tl.load(r + r_offsets, row_mask[:, None] & feature_mask[None, :], other=0.0)
    bk = tl.load(k + kv_offsets, col_mask[:, None] & feature_mask[None, :], other=0.0)

    qk = tl.dot(bq, tl.trans(bk), out_dtype=tl.float32) * SCALE_LOG2
    rk = tl.dot(br, tl.trans(bk), out_dtype=tl.float32)
    if col_start >= row_start:
        causal = rows[:, None] >= cols[None, :]
        qk = tl.where(causal, qk, -float("inf"))

    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    tile_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(score_ptr + score_offsets, qk, tile_mask)
    tl.store(rk_ptr + score_offsets, rk, tile_mask)


@triton.jit
def _softmax_correction(
    score_ptr,
    rk_ptr,
    prob_ptr,
    weighted_rk_ptr,
    d1_ptr,
    bart_ptr,
    m_ptr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    BR: tl.constexpr,
):
    row_block = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    rows = row_block * BR + tl.arange(0, BR)
    cols = tl.arange(0, T)
    score_offsets = ((bh * T + rows[:, None]) * T + cols[None, :])
    row_valid = rows < T
    causal = cols[None, :] <= rows[:, None]
    valid = row_valid[:, None] & causal
    scores = tl.load(score_ptr + score_offsets, valid, other=-float("inf")).to(tl.float32)
    rk = tl.load(rk_ptr + score_offsets, valid, other=0.0).to(tl.float32)
    row_max = tl.max(scores, axis=1, keep_dims=True)
    p = tl.exp2(scores - row_max)
    denom = tl.sum(p, axis=1, keep_dims=True)
    weighted_rk = p * rk
    bart = tl.sum(weighted_rk, axis=1, keep_dims=True) / denom
    scalar_offsets = (batch * T + rows) * HQ + hq
    dense_output = row_valid[:, None] & (cols[None, :] < T)

    normalized_p = p / denom
    corrected_weight = (p * (1.0 + bart) - weighted_rk) / denom
    tl.store(prob_ptr + score_offsets, normalized_p.to(tl.bfloat16), dense_output)
    tl.store(weighted_rk_ptr + score_offsets, corrected_weight.to(tl.bfloat16), dense_output)
    tl.store(d1_ptr + scalar_offsets, tl.sum(p, axis=1), row_valid)
    tl.store(bart_ptr + scalar_offsets, tl.sum(bart, axis=1), row_valid)
    tl.store(m_ptr + scalar_offsets, tl.sum(row_max, axis=1), row_valid)


@triton.jit
def _pv_tiles(
    prob_ptr,
    corrected_ptr,
    v,
    o,
    barv,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int32)
    bh = tl.program_id(1).to(tl.int32)
    batch = bh // HQ
    hq = bh % HQ
    hk = hq // G
    rows = pid_m * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    d = tl.arange(0, BD)
    row_mask = rows < T
    accum = tl.zeros((BM, BD), dtype=tl.float32)
    correction_accum = tl.zeros((BM, BD), dtype=tl.float32)

    for col_start in range(0, T, BN):
        key_rows = col_start + cols
        key_mask = key_rows < T
        weight_offsets = ((bh * T + rows[:, None]) * T + key_rows[None, :])
        value_offsets = ((batch * T + key_rows[:, None]) * H + hk) * D + d[None, :]
        weight_mask = row_mask[:, None] & key_mask[None, :]
        value_mask = key_mask[:, None] & (d[None, :] < D)
        prob = tl.load(prob_ptr + weight_offsets, weight_mask, other=0.0)
        corrected = tl.load(corrected_ptr + weight_offsets, weight_mask, other=0.0)
        values = tl.load(v + value_offsets, value_mask, other=0.0)
        accum = tl.dot(prob, values, out_dtype=tl.float32, acc=accum)
        correction_accum = tl.dot(corrected, values, out_dtype=tl.float32, acc=correction_accum)

    output_offsets = ((batch * T + rows[:, None]) * HQ + hq) * D + d[None, :]
    output_mask = row_mask[:, None] & (d[None, :] < D)
    tl.store(barv + output_offsets, accum, output_mask)
    tl.store(o + output_offsets, correction_accum, output_mask)


def staged_parallax_forward(q, r, k, v, scale):
    """Staged fixed-length D64 full-causal path, returning the backward state."""
    batch, time, hq, dim = q.shape
    h = k.shape[2]
    group_size = hq // h
    score_shape = (batch, hq, time, time)
    score = torch.empty(score_shape, device=q.device, dtype=q.dtype)
    rk = torch.empty_like(score)
    prob = torch.empty(score_shape, device=q.device, dtype=torch.bfloat16)
    weighted_rk = torch.empty_like(prob)
    output = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((batch, time, hq), device=q.device, dtype=torch.float32)
    bart = torch.empty_like(d1)
    m = torch.empty_like(d1)

    bd = triton.next_power_of_2(dim)
    bm_score = 128
    bn_score = 128
    _score_tiles[(triton.cdiv(time, bm_score), triton.cdiv(time, bn_score), batch * hq)](
        q,
        r,
        k,
        score,
        rk,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm_score,
        BN=bn_score,
        BD=bd,
        SCALE_LOG2=scale * 1.4426950216,
        num_warps=8,
        num_stages=1,
    )
    br = 8
    _softmax_correction[(triton.cdiv(time, br), batch * hq)](
        score,
        rk,
        prob,
        weighted_rk,
        d1,
        bart,
        m,
        time,
        HQ=hq,
        BR=br,
        num_warps=4,
        num_stages=1,
    )
    bm_pv = 128
    bn_pv = 64
    bd_pv = triton.next_power_of_2(dim)
    _pv_tiles[(triton.cdiv(time, bm_pv), batch * hq)](
        prob,
        weighted_rk,
        v,
        output,
        barv,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BM=bm_pv,
        BN=bn_pv,
        BD=bd_pv,
        num_warps=8,
        num_stages=1,
    )
    return output, barv, d1, bart, m
