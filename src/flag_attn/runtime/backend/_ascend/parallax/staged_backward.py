"""Large-GEMM staged backward for fixed full-causal D64 Parallax attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weights_kernel(q, r, k, v, o, barv, grad_o, d1, bart, m,
                    gl_ptr, gu_ptr, w_ptr, scale: tl.constexpr,
                    T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
                    G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    """Compute row-local deltas once, then stream causal key tiles."""
    mi = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    hq = bh % HQ
    hk = hq // G
    row0 = mi * BM
    rows = row0 + tl.arange(0, BM)
    ds = tl.arange(0, BD)
    rm = rows < T
    dm = ds < D
    qoff = ((b * T + rows[:, None]) * HQ + hq) * D + ds[None, :]
    qv = tl.load(q + qoff, rm[:, None] & dm[None, :], 0.0)
    rv = tl.load(r + qoff, rm[:, None] & dm[None, :], 0.0)
    go = tl.load(grad_o + qoff, rm[:, None] & dm[None, :], 0.0)
    ov = tl.load(o + qoff, rm[:, None] & dm[None, :], 0.0).to(tl.float32)
    bv = tl.load(barv + qoff, rm[:, None] & dm[None, :], 0.0).to(tl.float32)
    dt = tl.sum(go.to(tl.float32) * ov, axis=1)[:, None]
    db = tl.sum(go.to(tl.float32) * bv, axis=1)[:, None]
    scalar = (b * T + rows) * HQ + hq
    norm = tl.load(d1 + scalar, rm, 1.0)[:, None]
    maxv = tl.load(m + scalar, rm, 0.0)[:, None]
    bt = tl.load(bart + scalar, rm, 0.0)[:, None]
    for ni in range(tl.cdiv(row0 + BM, BN)):
        col0 = ni * BN
        cols = col0 + tl.arange(0, BN)
        cm = cols < T
        koff = ((b * T + cols[:, None]) * H + hk) * D + ds[None, :]
        kv = tl.load(k + koff, cm[:, None] & dm[None, :], 0.0)
        vv = tl.load(v + koff, cm[:, None] & dm[None, :], 0.0)
        allowed = rm[:, None] & cm[None, :] & (rows[:, None] >= cols[None, :])
        score = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
        p = tl.where(allowed, tl.math.exp2(score - maxv) / norm, 0.0)
        rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
        a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
        delta = a - db
        correction = bt - rk
        gl = p * (a - dt + correction * delta) * scale
        gu = -p * delta
        weights = p * (1.0 + correction)
        off = (bh * T + rows[:, None]) * T + cols[None, :]
        tl.store(gl_ptr + off, gl.to(tl.bfloat16), allowed)
        tl.store(gu_ptr + off, gu.to(tl.bfloat16), allowed)
        tl.store(w_ptr + off, weights.to(tl.bfloat16), allowed)


@triton.jit
def _dqr_gemm(gl_ptr, gu_ptr, k, grad_q, grad_r,
              T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
              G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    mi = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    hq = bh % HQ
    hk = hq // G
    rows = mi * BM + tl.arange(0, BM)
    ns = tl.arange(0, BN)
    ds = tl.arange(0, BD)
    rm = rows < T
    dm = ds < D
    aq = tl.zeros((BM, BD), tl.float32)
    ar = tl.zeros((BM, BD), tl.float32)
    for n0 in range(0, T, BN):
        cols = n0 + ns
        valid = rm[:, None] & (cols[None, :] < T) & (cols[None, :] <= rows[:, None])
        off = (bh * T + rows[:, None]) * T + cols[None, :]
        gl = tl.load(gl_ptr + off, valid, 0.0)
        gu = tl.load(gu_ptr + off, valid, 0.0)
        koff = ((b * T + cols[:, None]) * H + hk) * D + ds[None, :]
        kv = tl.load(k + koff, (cols[:, None] < T) & dm[None, :], 0.0)
        aq = tl.dot(gl, kv, out_dtype=tl.float32, acc=aq)
        ar = tl.dot(gu, kv, out_dtype=tl.float32, acc=ar)
    qoff = ((b * T + rows[:, None]) * HQ + hq) * D + ds[None, :]
    tl.store(grad_q + qoff, aq, rm[:, None] & dm[None, :])
    tl.store(grad_r + qoff, ar, rm[:, None] & dm[None, :])


@triton.jit
def _dkv_gemm(gl_ptr, gu_ptr, w_ptr, q, r, grad_o, grad_k, grad_v,
              T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
              G: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
              BM: tl.constexpr, BK: tl.constexpr):
    ni = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    hq = bh % HQ
    hk = hq // G
    cols = ni * BM + tl.arange(0, BM)
    qs0 = tl.arange(0, BK)
    ds = tl.arange(0, BD)
    cm = cols < T
    dm = ds < D
    ak = tl.zeros((BM, BD), tl.float32)
    av = tl.zeros((BM, BD), tl.float32)
    for q0 in range(0, T, BK):
        rows = q0 + qs0
        valid = (rows[:, None] < T) & cm[None, :] & (rows[:, None] >= cols[None, :])
        off = (bh * T + rows[:, None]) * T + cols[None, :]
        gl = tl.load(gl_ptr + off, valid, 0.0)
        gu = tl.load(gu_ptr + off, valid, 0.0)
        ww = tl.load(w_ptr + off, valid, 0.0)
        qoff = ((b * T + rows[:, None]) * HQ + hq) * D + ds[None, :]
        qv = tl.load(q + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
        rv = tl.load(r + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
        go = tl.load(grad_o + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
        ak = tl.dot(tl.trans(gl), qv, out_dtype=tl.float32, acc=ak)
        ak = tl.dot(tl.trans(gu), rv, out_dtype=tl.float32, acc=ak)
        av = tl.dot(tl.trans(ww), go, out_dtype=tl.float32, acc=av)
    koff = ((b * T + cols[:, None]) * HQ + hq) * D + ds[None, :]
    tl.store(grad_k + koff, ak, cm[:, None] & dm[None, :])
    tl.store(grad_v + koff, av, cm[:, None] & dm[None, :])


@triton.jit
def _dkv_grouped_gemm(gl_ptr, gu_ptr, w_ptr, q, r, grad_o, grad_k, grad_v,
                      T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
                      G: tl.constexpr, PARTS: tl.constexpr, GP: tl.constexpr,
                      D: tl.constexpr, BD: tl.constexpr,
                      BM: tl.constexpr, BK: tl.constexpr):
    ni = tl.program_id(0)
    bp = tl.program_id(1)
    b = bp // (H * PARTS)
    hp = bp % (H * PARTS)
    hk = hp // PARTS
    part = hp % PARTS
    cols = ni * BM + tl.arange(0, BM)
    qs0 = tl.arange(0, BK)
    ds = tl.arange(0, BD)
    cm = cols < T
    dm = ds < D
    ak = tl.zeros((BM, BD), tl.float32)
    av = tl.zeros((BM, BD), tl.float32)
    for gi in range(GP):
        hq = hk * G + part * GP + gi
        bh = b * HQ + hq
        for q0 in range(0, T, BK):
            rows = q0 + qs0
            valid = (rows[:, None] < T) & cm[None, :] & (rows[:, None] >= cols[None, :])
            off = (bh * T + rows[:, None]) * T + cols[None, :]
            gl = tl.load(gl_ptr + off, valid, 0.0)
            gu = tl.load(gu_ptr + off, valid, 0.0)
            ww = tl.load(w_ptr + off, valid, 0.0)
            qoff = ((b * T + rows[:, None]) * HQ + hq) * D + ds[None, :]
            qv = tl.load(q + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
            rv = tl.load(r + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
            go = tl.load(grad_o + qoff, (rows[:, None] < T) & dm[None, :], 0.0)
            ak = tl.dot(tl.trans(gl), qv, out_dtype=tl.float32, acc=ak)
            ak = tl.dot(tl.trans(gu), rv, out_dtype=tl.float32, acc=ak)
            av = tl.dot(tl.trans(ww), go, out_dtype=tl.float32, acc=av)
    outoff = ((b * T + cols[:, None]) * (H * PARTS) + hk * PARTS + part) * D + ds[None, :]
    tl.store(grad_k + outoff, ak, cm[:, None] & dm[None, :])
    tl.store(grad_v + outoff, av, cm[:, None] & dm[None, :])


def staged_parallax_backward(q, r, k, v, o, barv, d1, bart, m, grad_o, scale):
    """Fixed-length full-causal D64 staged derivative path."""
    bsz, time, hq, dim = q.shape
    h = k.shape[2]
    group = hq // h
    matrix_shape = (bsz, hq, time, time)
    gl = torch.empty(matrix_shape, device=q.device, dtype=torch.bfloat16)
    gu = torch.empty_like(gl)
    weights = torch.empty_like(gl)
    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    heads_per_part = 16 if group == 32 else 8 if group > 1 else 1
    parts = group // heads_per_part if group > 1 else hq
    scratch_heads = h * parts if group > 1 else hq
    grad_k_buf = torch.empty((bsz, time, scratch_heads, dim), device=q.device, dtype=q.dtype)
    grad_v_buf = torch.empty_like(grad_k_buf)
    bd = triton.next_power_of_2(dim)
    weight_bm, weight_bn = 32, 256
    _weights_kernel[(triton.cdiv(time, weight_bm), bsz * hq)](
        q, r, k, v, o, barv, grad_o, d1, bart, m, gl, gu, weights,
        scale, time, hq, h, group, dim, bd, weight_bm, weight_bn, num_warps=8, num_stages=1)
    bm = bn = 128
    if torch.npu.is_current_stream_capturing():
        _dqr_gemm[(triton.cdiv(time, bm), bsz * hq)](
            gl, gu, k, grad_q, grad_r, time, hq, h, group, dim, bd, bm, bn,
            num_warps=8, num_stages=1)
        if group == 1:
            _dkv_gemm[(triton.cdiv(time, bm), bsz * hq)](
                gl, gu, weights, q, r, grad_o, grad_k_buf, grad_v_buf,
                time, hq, h, group, dim, bd, bm, bn, num_warps=8, num_stages=1)
        else:
            _dkv_grouped_gemm[(triton.cdiv(time, bm), bsz * h * parts)](
                gl, gu, weights, q, r, grad_o, grad_k_buf, grad_v_buf,
                time, hq, h, group, parts, heads_per_part, dim, bd, bm, bn,
                num_warps=8, num_stages=1)
    else:
        current_stream = torch.npu.current_stream(q.device)
        dqr_stream = torch.npu.Stream(device=q.device)
        dkv_stream = torch.npu.Stream(device=q.device)
        dqr_stream.wait_stream(current_stream)
        dkv_stream.wait_stream(current_stream)
        with torch.npu.stream(dqr_stream):
            _dqr_gemm[(triton.cdiv(time, bm), bsz * hq)](
                gl, gu, k, grad_q, grad_r, time, hq, h, group, dim, bd, bm, bn,
                num_warps=8, num_stages=1)
        with torch.npu.stream(dkv_stream):
            if group == 1:
                _dkv_gemm[(triton.cdiv(time, bm), bsz * hq)](
                    gl, gu, weights, q, r, grad_o, grad_k_buf, grad_v_buf,
                    time, hq, h, group, dim, bd, bm, bn, num_warps=8, num_stages=1)
            else:
                _dkv_grouped_gemm[(triton.cdiv(time, bm), bsz * h * parts)](
                    gl, gu, weights, q, r, grad_o, grad_k_buf, grad_v_buf,
                    time, hq, h, group, parts, heads_per_part, dim, bd, bm, bn,
                    num_warps=8, num_stages=1)
        current_stream.wait_stream(dqr_stream)
        current_stream.wait_stream(dkv_stream)
    if group > 1:
        from .backward import _reduce_gqa_kernel
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        n = bsz * time * h * dim
        reduce_block = 128
        _reduce_gqa_kernel[(triton.cdiv(n, reduce_block),)](
            grad_k_buf, grad_v_buf, grad_k, grad_v, n, h * parts, h, parts, dim, reduce_block,
            num_warps=4,
        )
        return grad_q, grad_r, grad_k, grad_v
    return grad_q, grad_r, grad_k_buf, grad_v_buf
