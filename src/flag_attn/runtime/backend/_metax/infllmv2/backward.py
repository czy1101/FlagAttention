# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl

from flag_attn.infllmv2.backward import _build_reverse_index


_MAX_HEAD_TILE_ELEMENTS = 4096


@triton.jit
def _dq_kernel(
    Q,
    K,
    V,
    O,
    DO,
    LSE,
    Selected,
    CuQ,
    CuK,
    DQ,
    Delta,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    D: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_SPLITS: tl.constexpr,
):
    qi, hgroup, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = hgroup // HEAD_SPLITS
    head_part = hgroup % HEAD_SPLITS
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    if qi >= qe - qs:
        return
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    g, d, n = head_part * BLOCK_M + tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_N)
    head = hk * GQA_GROUP_SIZE + g
    qo = (qs + qi) * HQ * D + head[:, None] * D + d[None, :]
    mask = (g[:, None] < GQA_GROUP_SIZE) & (d[None, :] < D)
    q = tl.load(Q + qo, mask, 0)
    do = tl.load(DO + qo, mask, 0)
    o = tl.load(O + qo, mask, 0)
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), 1)
    tl.store(Delta + (qs + qi) * HQ + head, delta, g < GQA_GROUP_SIZE)
    lse = tl.load(LSE + (qs + qi) * HQ + head, g < GQA_GROUP_SIZE, -float("inf"))
    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    limit = ke - ks - (qe - qs) + qi + 1
    steps: tl.constexpr = triton.cdiv(BLOCK_SIZE, BLOCK_N)
    ranks = TOPK
    for iteration in tl.range(0, ranks * steps):
        block = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + iteration // steps)
        if block >= 0:
            pos = block * BLOCK_SIZE + (iteration % steps) * BLOCK_N + n
            valid = (pos < ke - ks) & ((iteration % steps) * BLOCK_N + n < BLOCK_SIZE)
            if CAUSAL:
                valid = valid & (pos < limit)
            kv = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
            k = tl.load(K + kv, valid[:, None] & (d[None, :] < D), 0)
            logits = tl.dot(q, tl.trans(k)) * SCALE
            p = tl.where(
                (g[:, None] < GQA_GROUP_SIZE) & valid[None, :] & (lse[:, None] != -float("inf")),
                tl.exp(logits - lse[:, None]),
                0.0,
            )
            v = tl.load(V + kv, valid[:, None] & (d[None, :] < D), 0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - delta[:, None]) * SCALE
            dq = tl.dot(ds.to(k.dtype), k, dq)
    tl.store(DQ + qo, dq, mask)


@triton.jit
def _dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    CuQ,
    CuK,
    Counts,
    Queries,
    DK,
    DV,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    GROUP_TILE: tl.constexpr,
    D: tl.constexpr,
    MAX_Q: tl.constexpr,
    NUM_KV_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUERY_SPLIT: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
    HEAD_SPLITS: tl.constexpr,
):
    owner, part, subgroup = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sub = subgroup % triton.cdiv(BLOCK_SIZE, BLOCK_N)
    head_part = subgroup // triton.cdiv(BLOCK_SIZE, BLOCK_N)
    block = owner % NUM_KV_BLOCKS
    hk = owner // NUM_KV_BLOCKS % HKV
    batch = owner // (NUM_KV_BLOCKS * HKV)
    count = tl.load(Counts + owner)
    start = part * QUERY_SPLIT
    if not DIRECT_STORE:
        if start >= count:
            return
    end = tl.minimum(count, start + QUERY_SPLIT)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    m, n, d = tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_N), tl.arange(0, BLOCK_D)
    head = hk * GQA_GROUP_SIZE + head_part * GROUP_TILE + m % GROUP_TILE
    pos = block * BLOCK_SIZE + sub * BLOCK_N + n
    kv = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
    kv_mask = (pos[:, None] < ke - ks) & (sub * BLOCK_N + n[:, None] < BLOCK_SIZE) & (d[None, :] < D)
    k = tl.load(K + kv, kv_mask, 0)
    v = tl.load(V + kv, kv_mask, 0)
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    for offset in tl.range(start, end, BLOCK_M // GROUP_TILE):
        slot = offset + m // GROUP_TILE
        qi = tl.load(Queries + owner * MAX_Q + slot, slot < end, 0)
        valid_m = (slot < end) & (head_part * GROUP_TILE + m % GROUP_TILE < GQA_GROUP_SIZE)
        qo = (qs + qi[:, None]) * HQ * D + head[:, None] * D + d[None, :]
        q = tl.load(Q + qo, valid_m[:, None] & (d[None, :] < D), 0)
        do = tl.load(DO + qo, valid_m[:, None] & (d[None, :] < D), 0)
        lse = tl.load(LSE + (qs + qi) * HQ + head, valid_m, -float("inf"))
        delta = tl.load(Delta + (qs + qi) * HQ + head, valid_m, 0.0)
        valid = (
            valid_m[:, None]
            & (pos[None, :] < ke - ks)
            & (sub * BLOCK_N + n[None, :] < BLOCK_SIZE)
            & (lse[:, None] != -float("inf"))
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    selected: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    *,
    max_seqlen_k: int | None = None,
    bn: int | None = None,
    warps: int | None = None,
    query_split: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    bd = triton.next_power_of_2(head_dim)
    full_group = triton.next_power_of_2(gqa_group_size)

    gp = min(full_group, max(16, _MAX_HEAD_TILE_ELEMENTS // bd))
    head_splits = triton.cdiv(gqa_group_size, gp)
    if max_seqlen_k is None:
        max_seqlen_k = (
            k.shape[0]
            if cu_seqlens_k.numel() == 2
            else int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
        )
    nb = triton.cdiv(max_seqlen_k, block_size)
    owners = (cu_seqlens_q.numel() - 1) * num_kv_heads * nb
    counts = torch.zeros(owners, device=q.device, dtype=torch.int32)
    queries = torch.empty((owners, max_seqlen_q), device=q.device, dtype=torch.int32)
    topk = selected.shape[-1]
    _build_reverse_index[(triton.cdiv(max_seqlen_q * topk, 256), num_kv_heads, cu_seqlens_q.numel() - 1)](
        selected,
        cu_seqlens_q,
        counts,
        queries,
        TOTAL_Q=total_q,
        TOPK=topk,
        HKV=num_kv_heads,
        MAX_Q=max_seqlen_q,
        MAX_BLOCKS=nb,
        TILE=256,
        num_warps=4,
    )
    dq = torch.empty_like(q)
    delta = torch.empty((total_q, num_q_heads), device=q.device, dtype=torch.float32)
    direct = max_seqlen_q <= query_split and head_splits == 1
    if direct:
        dk, dv = torch.empty_like(k), torch.empty_like(v)
    else:
        accum = torch.zeros((2, *k.shape), device=k.device, dtype=torch.float32)
        dk, dv = accum[0], accum[1]
    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    if bn is None:
        if bd > 128:
            bn = 16
        elif bd >= 128:
            bn = 32
        else:
            bn = 64
    if warps is None:
        warps = 2 if bd == 128 and head_dim >= 128 and gqa_group_size <= 16 else 4
    _dq_kernel[(max_seqlen_q, num_kv_heads * head_splits, cu_seqlens_q.numel() - 1)](
        q,
        k,
        v,
        out,
        dout,
        lse,
        selected,
        cu_seqlens_q,
        cu_seqlens_k,
        dq,
        delta,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        D=head_dim,
        TOTAL_Q=total_q,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        SCALE=scale,
        CAUSAL=causal,
        BLOCK_M=max(16, gp),
        BLOCK_N=bn,
        BLOCK_D=bd,
        HEAD_SPLITS=head_splits,
        num_warps=warps,
        num_stages=1,
    )
    _dkdv_kernel[(owners, triton.cdiv(max_seqlen_q, query_split), triton.cdiv(block_size, bn) * head_splits)](
        q,
        k,
        v,
        dout,
        lse,
        delta,
        cu_seqlens_q,
        cu_seqlens_k,
        counts,
        queries,
        dk,
        dv,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        GROUP_TILE=gp,
        D=head_dim,
        MAX_Q=max_seqlen_q,
        NUM_KV_BLOCKS=nb,
        BLOCK_SIZE=block_size,
        BLOCK_D=bd,
        BLOCK_M=max(16, gp),
        BLOCK_N=bn,
        QUERY_SPLIT=query_split,
        SCALE=scale,
        CAUSAL=causal,
        DIRECT_STORE=direct,
        HEAD_SPLITS=head_splits,
        num_warps=warps,
        num_stages=1,
    )
    return dq, dk.to(k.dtype), dv.to(v.dtype)
