# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl
from flag_attn.infllmv2.backward import _build_reverse_index


@triton.jit
def _dq_c550(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    Selected,
    CQ,
    CK,
    DQ,
    Delta,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    TQ: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    HEAD_SPLITS: tl.constexpr,
):
    qi, hgroup, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = hgroup // HEAD_SPLITS
    head_part = hgroup % HEAD_SPLITS
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    if qi >= qe - qs:
        return
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    g, d, n = head_part * BM + tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, BN)
    head = hk * G + g
    qo = (qs + qi) * HQ * D + head[:, None] * D + d[None, :]
    mask = (g[:, None] < G) & (d[None, :] < D)
    q = tl.load(Q + qo, mask, 0)
    do = tl.load(DO + qo, mask, 0)
    o = tl.load(O + qo, mask, 0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(Delta + (qs + qi) * HQ + head, delta, g < G)
    lse = tl.load(LSE + (qs + qi) * HQ + head, g < G, -float("inf"))
    dq = tl.zeros((BM, BD), tl.float32)
    limit = ke - ks - (qe - qs) + qi + 1
    steps: tl.constexpr = triton.cdiv(BS, BN)
    ranks = TOPK
    if CAUSAL:
        ranks = TOPK
    for iteration in tl.range(0, ranks * steps):
        block = tl.load(Selected + (hk * TQ + qs + qi) * TOPK + iteration // steps)
        if block >= 0:
            pos = block * BS + (iteration % steps) * BN + n
            valid = (pos < ke - ks) & ((iteration % steps) * BN + n < BS)
            if CAUSAL:
                valid = valid & (pos < limit)
            kv = (ks + pos[:, None]) * HK * D + hk * D + d[None, :]
            k = tl.load(K + kv, valid[:, None] & (d[None, :] < D), 0)
            logits = tl.dot(q, tl.trans(k)) * SCALE
            p = tl.where(
                (g[:, None] < G) & valid[None, :] & (lse[:, None] != -float("inf")), tl.exp(logits - lse[:, None]), 0.0
            )
            v = tl.load(V + kv, valid[:, None] & (d[None, :] < D), 0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * SCALE
            dq = tl.dot(ds.to(k.dtype), k, dq)
    tl.store(DQ + qo, dq, mask)


@triton.jit
def _dkdv_c550(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    CQ,
    CK,
    Counts,
    Queries,
    DK,
    DV,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    GP: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    NB: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    QSPLIT: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
    HEAD_SPLITS: tl.constexpr,
):
    owner, part, subgroup = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sub = subgroup % triton.cdiv(BS, BN)
    head_part = subgroup // triton.cdiv(BS, BN)
    block = owner % NB
    hk = owner // NB % HK
    batch = owner // (NB * HK)
    count = tl.load(Counts + owner)
    start = part * QSPLIT
    if not DIRECT_STORE:
        if start >= count:
            return
    end = tl.minimum(count, start + QSPLIT)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    m, n, d = tl.arange(0, BM), tl.arange(0, BN), tl.arange(0, BD)
    head = hk * G + head_part * GP + m % GP
    pos = block * BS + sub * BN + n
    kv = (ks + pos[:, None]) * HK * D + hk * D + d[None, :]
    kv_mask = (pos[:, None] < ke - ks) & (sub * BN + n[:, None] < BS) & (d[None, :] < D)
    k = tl.load(K + kv, kv_mask, 0)
    v = tl.load(V + kv, kv_mask, 0)
    dk = tl.zeros((BN, BD), tl.float32)
    dv = tl.zeros((BN, BD), tl.float32)
    for offset in tl.range(start, end, BM // GP):
        slot = offset + m // GP
        qi = tl.load(Queries + owner * MAX_Q + slot, slot < end, 0)
        valid_m = (slot < end) & (head_part * GP + m % GP < G)
        qo = (qs + qi[:, None]) * HQ * D + head[:, None] * D + d[None, :]
        q = tl.load(Q + qo, valid_m[:, None] & (d[None, :] < D), 0)
        do = tl.load(DO + qo, valid_m[:, None] & (d[None, :] < D), 0)
        lse = tl.load(LSE + (qs + qi) * HQ + head, valid_m, -float("inf"))
        delta = tl.load(Delta + (qs + qi) * HQ + head, valid_m, 0.0)
        valid = (
            valid_m[:, None] & (pos[None, :] < ke - ks) & (sub * BN + n[None, :] < BS) & (lse[:, None] != -float("inf"))
        )
        if CAUSAL:
            valid = valid & (pos[None, :] < ke - ks - (qe - qs) + qi[:, None] + 1)
        logits = tl.dot(q, tl.trans(k)) * SCALE
        p = tl.where(valid, tl.exp(logits - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None]) * SCALE
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)
        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)
    if DIRECT_STORE:
        tl.store(DK + kv, dk, kv_mask)
        tl.store(DV + kv, dv, kv_mask)
    else:
        tl.atomic_add(DK + kv, dk, kv_mask, sem="relaxed")
        tl.atomic_add(DV + kv, dv, kv_mask, sem="relaxed")


def sparse_backward(
    q,
    k,
    v,
    out,
    dout,
    lse,
    selected,
    cq,
    ck,
    max_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    *,
    max_k=None,
    bn=None,
    warps=None,
    query_split=64,
):
    tq, hq, d = q.shape
    hk = k.shape[1]
    group = hq // hk
    bd = triton.next_power_of_2(d)
    full_group = triton.next_power_of_2(group)

    gp = min(full_group, max(16, 4096 // bd))
    head_splits = triton.cdiv(group, gp)
    if max_k is None:
        max_k = k.shape[0] if ck.numel() == 2 else int((ck[1:] - ck[:-1]).max().item())
    nb = triton.cdiv(max_k, block_size)
    owners = (cq.numel() - 1) * hk * nb
    counts = torch.zeros(owners, device=q.device, dtype=torch.int32)
    queries = torch.empty((owners, max_q), device=q.device, dtype=torch.int32)
    topk = selected.shape[-1]
    _build_reverse_index[(triton.cdiv(max_q * topk, 256), hk, cq.numel() - 1)](
        selected, cq, counts, queries, tq, topk, hk, max_q, nb, 256, num_warps=4
    )
    dq = torch.empty_like(q)
    delta = torch.empty((tq, hq), device=q.device, dtype=torch.float32)
    direct = max_q <= query_split and head_splits == 1
    if direct:
        dk, dv = torch.empty_like(k), torch.empty_like(v)
    else:
        accum = torch.zeros((2, *k.shape), device=k.device, dtype=torch.float32)
        dk, dv = accum[0], accum[1]
    scale = d**-0.5 if softmax_scale is None else softmax_scale
    bn = (16 if bd > 128 else 32 if bd >= 128 else 64) if bn is None else bn
    if warps is None:
        warps = 2 if bd == 128 and d >= 128 and group <= 16 else 4
    _dq_c550[(max_q, hk * head_splits, cq.numel() - 1)](
        q,
        k,
        v,
        out,
        dout,
        lse,
        selected,
        cq,
        ck,
        dq,
        delta,
        hq,
        hk,
        group,
        d,
        tq,
        topk,
        block_size,
        scale,
        causal,
        max(16, gp),
        bn,
        bd,
        head_splits,
        num_warps=warps,
        num_stages=1,
    )
    _dkdv_c550[(owners, triton.cdiv(max_q, query_split), triton.cdiv(block_size, bn) * head_splits)](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        cq,
        ck,
        counts,
        queries,
        dk,
        dv,
        hq,
        hk,
        group,
        gp,
        d,
        max_q,
        nb,
        block_size,
        bd,
        max(16, gp),
        bn,
        query_split,
        scale,
        causal,
        direct,
        head_splits,
        num_warps=warps,
        num_stages=1,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype)
