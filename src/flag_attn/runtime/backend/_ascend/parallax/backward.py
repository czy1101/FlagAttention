"""Ascend Triton backward kernels for Parallax attention."""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _preprocess_kernel(o, barv, grad_o, delta_t, delta_b, T: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BD)
    x = row * HQ + h
    offsets = x * D + d
    mask = d < D
    go = tl.load(grad_o + offsets, mask, 0).to(tl.float32)
    out = tl.load(o + offsets, mask, 0).to(tl.float32)
    bv = tl.load(barv + offsets, mask, 0).to(tl.float32)
    tl.store(delta_t + x, tl.sum(go * out, axis=0))
    tl.store(delta_b + x, tl.sum(go * bv, axis=0))


@triton.jit
def _dqr_kernel(q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_q, grad_r,
                scale, T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                D: tl.constexpr, BD: tl.constexpr, WINDOW: tl.constexpr,
                BT: tl.constexpr, BS: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b, hq = bh // HQ, bh % HQ
    h = hq // G
    bos_q = b * T * HQ * D + hq * D
    bos_k = b * T * H * D + h * D
    rows = tile * BT + tl.arange(0, BT)
    ds = tl.arange(0, BD)
    rowmask = rows < T
    qoff = bos_q + rows[:, None] * HQ * D + ds[None, :]
    qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
    dt = tl.load(delta_t + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    db = tl.load(delta_b + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    norm = tl.load(d1 + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    maxv = tl.load(m + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    bt = tl.load(bart + b * T * HQ + rows * HQ + hq, rowmask, 0)[:, None]
    gq = tl.zeros((BT, BD), tl.float32)
    gr = tl.zeros((BT, BD), tl.float32)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    cols = tl.arange(0, BS)
    for block in range(tl.cdiv(T, BS)):
        ks = block * BS + cols
        kmask = (ks < T)[:, None] & (ds[None, :] < D)
        koff = bos_k + ks[:, None] * H * D + ds[None, :]
        kv = tl.load(k + koff, kmask, 0)
        vv = tl.load(v + koff, kmask, 0)
        allowed = (ks[None, :] <= rows[:, None]) & (ks[None, :] < T)
        if WINDOW >= 0:
            allowed = allowed & (ks[None, :] >= rows[:, None] - WINDOW + 1)
        scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
        scores = tl.where(allowed & rowmask[:, None], scores, float("-inf"))
        p = tl.where(allowed & rowmask[:, None], tl.math.exp2(scores - maxv) / safe_norm, 0.0)
        a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
        a = tl.where(allowed & rowmask[:, None], a, 0.0)
        delta = a - db
        gr += tl.dot((-p * delta).to(kv.dtype), kv, out_dtype=tl.float32)
        rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
        rk = tl.where(allowed & rowmask[:, None], rk, 0.0)
        gl = p * (a - dt + (bt - rk) * delta)
        gl = tl.where(allowed & rowmask[:, None], gl, 0.0)
        gq += tl.dot(gl.to(kv.dtype), kv, out_dtype=tl.float32)
    gq *= scale
    tl.store(grad_q + qoff, gq, rowmask[:, None] & (ds[None, :] < D))
    tl.store(grad_r + qoff, gr, rowmask[:, None] & (ds[None, :] < D))


@triton.jit
def _dkv_kernel(q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o, grad_k, grad_v,
                scale, T: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                D: tl.constexpr, BD: tl.constexpr, WINDOW: tl.constexpr,
                BT: tl.constexpr, BS: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    b, hq = bh // HQ, bh % HQ
    h = hq // G
    bos_q = b * T * HQ * D + hq * D
    bos_k = b * T * H * D + h * D
    cols = tile * BS + tl.arange(0, BS)
    ds = tl.arange(0, BD)
    colmask = cols < T
    koff = bos_k + cols[:, None] * H * D + ds[None, :]
    kv = tl.load(k + koff, colmask[:, None] & (ds[None, :] < D), 0)
    vv = tl.load(v + koff, colmask[:, None] & (ds[None, :] < D), 0)
    gk = tl.zeros((BS, BD), tl.float32)
    gv = tl.zeros((BS, BD), tl.float32)
    rows = tl.arange(0, BT)
    for block in range(tl.cdiv(T, BT)):
        qs = block * BT + rows
        rowmask = qs < T
        qoff = bos_q + qs[:, None] * HQ * D + ds[None, :]
        qv = tl.load(q + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
        rv = tl.load(r + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
        go = tl.load(grad_o + qoff, rowmask[:, None] & (ds[None, :] < D), 0)
        dt = tl.load(delta_t + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
        db = tl.load(delta_b + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
        norm = tl.load(d1 + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
        maxv = tl.load(m + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
        bt = tl.load(bart + b * T * HQ + qs * HQ + hq, rowmask, 0)[:, None]
        allowed = (qs[:, None] >= cols[None, :]) & rowmask[:, None] & colmask[None, :]
        if WINDOW >= 0:
            allowed = allowed & (cols[None, :] >= qs[:, None] - WINDOW + 1)
        scores = tl.dot(qv, tl.trans(kv), out_dtype=tl.float32) * (scale * 1.4426950408889634)
        scores = tl.where(allowed, scores, float("-inf"))
        safe_norm = tl.where(norm > 0.0, norm, 1.0)
        p = tl.where(allowed, tl.math.exp2(scores - maxv) / safe_norm, 0.0)
        rk = tl.dot(rv, tl.trans(kv), out_dtype=tl.float32)
        a = tl.dot(go, tl.trans(vv), out_dtype=tl.float32)
        rk = tl.where(allowed, rk, 0.0)
        a = tl.where(allowed, a, 0.0)
        delta = a - db
        correction = bt - rk
        gl = p * (a - dt + correction * delta) * scale
        gu = -p * delta
        weights = p * (1.0 + correction)
        gl = tl.where(allowed, gl, 0.0)
        gu = tl.where(allowed, gu, 0.0)
        weights = tl.where(allowed, weights, 0.0)
        gk += tl.dot(tl.trans(gl).to(qv.dtype), qv, out_dtype=tl.float32)
        gk += tl.dot(tl.trans(gu).to(rv.dtype), rv, out_dtype=tl.float32)
        gv += tl.dot(tl.trans(weights).to(go.dtype), go, out_dtype=tl.float32)
    # Per-query-head scratch has HQ in its token/head stride.
    outoff = (b * T * HQ + cols[:, None] * HQ + hq) * D + ds[None, :]
    tl.store(grad_k + outoff, gk, colmask[:, None] & (ds[None, :] < D))
    tl.store(grad_v + outoff, gv, colmask[:, None] & (ds[None, :] < D))


@triton.jit
def _reduce_gqa_kernel(gk_buf, gv_buf, gk, gv, N: tl.constexpr, HQ: tl.constexpr,
                       H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hdim = x % D
    rowhead = x // D
    h = rowhead % H
    bhrow = rowhead // H
    src = (bhrow * HQ + h * G) * D + hdim
    ak = tl.zeros((BLOCK,), tl.float32)
    av = tl.zeros((BLOCK,), tl.float32)
    for g in range(G):
        ak += tl.load(gk_buf + src + g * D, x < N, 0).to(tl.float32)
        av += tl.load(gv_buf + src + g * D, x < N, 0).to(tl.float32)
    tl.store(gk + x, ak, x < N)
    tl.store(gv + x, av, x < N)


def parallax_backward(q, r, k, v, o, barv, d1, bart, m, grad_o, scale,
                      cu_seqlens=None, window_size_left=-1):
    """Compute gradients for Q, R, K and V using saved Parallax forward state."""
    if q.ndim != 4 or r.shape != q.shape or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("q/r/k/v must have compatible [B, T, H, D] shapes")
    bsz, total_t, hq, dim = q.shape
    hkv = k.shape[2]
    if k.shape[:2] != (bsz, total_t) or k.shape[-1] != dim or hq % hkv:
        raise ValueError("q and k/v shapes must agree and HQ must be divisible by H")
    if q.device.type != "npu":
        raise ValueError(f"Ascend Parallax backward requires NPU tensors, got {q.device}")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"unsupported dtype {q.dtype}; expected float16 or bfloat16")
    if grad_o.shape != q.shape:
        raise ValueError("grad_o must have the same shape as q")

    G = hq // hkv
    bd = triton.next_power_of_2(dim)
    bt = bs = 32
    delta_t = torch.empty((bsz, total_t, hq), device=q.device, dtype=torch.float32)
    delta_b = torch.empty_like(delta_t)
    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    grad_k_buf = torch.empty((bsz, total_t, hq, dim), device=q.device, dtype=torch.float32)
    grad_v_buf = torch.empty_like(grad_k_buf)

    if cu_seqlens is None:
        segments = [(0, total_t, bi) for bi in range(bsz)]
    else:
        if bsz != 1:
            raise ValueError("packed varlen backward requires B == 1")
        offsets = cu_seqlens.detach().cpu().tolist()
        segments = [(int(start), int(end - start), 0) for start, end in zip(offsets[:-1], offsets[1:])]

    _preprocess_kernel[(bsz * total_t, hq)](
        o, barv, grad_o, delta_t, delta_b, bsz * total_t, hq, dim, bd,
        num_warps=4,
    )
    for bos, t, bi in segments:
        if t == 0:
            continue
        # Kernels index from each batch/sequence's absolute token offset. Pass views
        # based at the segment, and use local row/head statistics similarly.
        _dqr_kernel[(triton.cdiv(t, bt), hq)](
            q[bi, bos:], r[bi, bos:], k[bi, bos:], v[bi, bos:],
            d1[bi, bos:], bart[bi, bos:], m[bi, bos:],
            delta_t[bi, bos:], delta_b[bi, bos:], grad_o[bi, bos:],
            grad_q[bi, bos:], grad_r[bi, bos:], scale, t, hq, hkv, G, dim, bd,
            window_size_left, bt, bs, num_warps=4,
        )
        _dkv_kernel[(triton.cdiv(t, bs), hq)](
            q[bi, bos:], r[bi, bos:], k[bi, bos:], v[bi, bos:],
            d1[bi, bos:], bart[bi, bos:], m[bi, bos:],
            delta_t[bi, bos:], delta_b[bi, bos:], grad_o[bi, bos:],
            grad_k_buf[bi, bos:], grad_v_buf[bi, bos:], scale, t, hq, hkv, G, dim, bd,
            window_size_left, bt, bs, num_warps=4,
        )

    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    if G == 1:
        # DKV already holds one contribution per query head in token-major storage.
        grad_k.copy_(grad_k_buf)
        grad_v.copy_(grad_v_buf)
    else:
        n = bsz * total_t * hkv * dim
        _reduce_gqa_kernel[(triton.cdiv(n, 128),)](
            grad_k_buf, grad_v_buf, grad_k, grad_v, n, hq, hkv, G, dim, 128,
            num_warps=4,
        )
    return grad_q, grad_r, grad_k, grad_v
