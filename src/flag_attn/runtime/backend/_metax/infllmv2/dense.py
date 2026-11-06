# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl


@triton.jit
def _dense_kernel(
    Q,
    K,
    V,
    CQ,
    CK,
    O,
    LSE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    STORE_LSE: tl.constexpr,
    QK_TRANS: tl.constexpr,
    PV_TRANS: tl.constexpr,
):
    tile, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    rm = tile * BM + tl.arange(0, BM)
    if tile * BM >= qe - qs:
        return
    rn, rd = tl.arange(0, BN), tl.arange(0, BD)
    hk = h // (HQ // HK)
    qoff = (qs + rm[:, None]) * HQ * D + h * D + rd[None, :]
    q = tl.load(Q + qoff, (rm[:, None] < qe - qs) & (rd[None, :] < D), 0)
    absolute = ke - ks - (qe - qs) + rm
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BD, BM) if PV_TRANS else (BM, BD), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    end = tl.minimum(ke - ks, ke - ks - (qe - qs) + (tile + 1) * BM)
    for start in range(0, end, BN):
        pos = start + rn
        off = (ks + pos[:, None]) * HK * D + hk * D + rd[None, :]
        mask = (pos[:, None] < ke - ks) & (rd[None, :] < D)
        k = tl.load(K + off, mask, 0)
        score = (tl.trans(tl.dot(k, tl.trans(q))) if QK_TRANS else tl.dot(q, tl.trans(k))) * (SCALE * log2e)
        score = tl.where((pos[None, :] < ke - ks) & (pos[None, :] <= absolute[:, None]), score, -float("inf"))
        mn = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mn)
        p = tl.exp2(score - mn[:, None])
        v = tl.load(V + off, mask, 0)
        if PV_TRANS:
            acc = tl.dot(tl.trans(v), tl.trans(p.to(v.dtype)), acc * alpha[None, :])
        else:
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        li = li * alpha + tl.sum(p, 1)
        mi = mn
    if PV_TRANS:
        acc = tl.trans(acc)
    tl.store(O + qoff, acc / li[:, None], (rm[:, None] < qe - qs) & (rd[None, :] < D))
    if STORE_LSE:
        tl.store(LSE + (qs + rm) * HQ + h, (mi + tl.log2(li)) / log2e, rm < qe - qs)


def _forward(
    q, k, v, cq, ck, max_q, scale, store_lse, *, bm=None, bn=None, warps=None, qk_transpose=None, pv_transpose=None
):
    tq, hq, d = q.shape

    native = q.dtype in (torch.float16, torch.bfloat16)
    padded128 = native and 64 < d < 128
    large = native and 128 < d <= 256
    bm = (16 if large else 32) if bm is None else bm
    bn = (16 if large else 32 if d >= 128 else 64) if bn is None else bn
    warps = (4 if padded128 else 2) if warps is None else warps
    qk_transpose = (padded128 or large) if qk_transpose is None else qk_transpose
    pv_transpose = (padded128 or large) if pv_transpose is None else pv_transpose
    out = torch.empty_like(q)
    lse = torch.empty((tq, hq), device=q.device, dtype=torch.float32) if store_lse else out
    _dense_kernel[(triton.cdiv(max_q, bm), hq, cq.numel() - 1)](
        q,
        k,
        v,
        cq,
        ck,
        out,
        lse,
        hq,
        k.shape[1],
        d,
        d**-0.5 if scale is None else scale,
        bm,
        bn,
        max(16, triton.next_power_of_2(d)),
        store_lse,
        qk_transpose,
        pv_transpose,
        num_warps=warps,
        num_stages=1,
    )
    return out, lse


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cq, ck, max_q, scale, max_k):
        out, lse = _forward(q, k, v, cq, ck, max_q, scale, True)
        ctx.save_for_backward(q, k, v, cq, ck, out, lse)
        ctx.max_q, ctx.scale = max_q, scale
        ctx.max_k = max_k
        return out

    @staticmethod
    def backward(ctx, dout):
        from .backward import sparse_backward

        q, k, v, cq, ck, out, lse = ctx.saved_tensors
        max_k = ctx.max_k
        if max_k is None:
            max_k = k.shape[0] if ck.numel() == 2 else int((ck[1:] - ck[:-1]).max().item())
        blocks = triton.cdiv(max_k, 64)

        ids = torch.arange(blocks, device=q.device, dtype=torch.int32)
        selected = ids.view(1, 1, blocks).expand(k.shape[1], q.shape[0], blocks).contiguous()
        dq, dk, dv = sparse_backward(
            q, k, v, out, dout.contiguous(), lse, selected, cq, ck, ctx.max_q, 64, ctx.scale, True, max_k=max_k
        )
        return dq, dk, dv, None, None, None, None, None


def dense_attention(q, k, v, cq, ck, max_q, scale=None, max_k=None):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _DenseAttention.apply(q, k, v, cq, ck, max_q, scale, max_k)
    return _forward(q, k, v, cq, ck, max_q, scale, False)[0]
