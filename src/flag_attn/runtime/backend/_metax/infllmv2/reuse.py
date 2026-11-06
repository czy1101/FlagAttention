# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl
from .attention import _sparse_c550_kernel


_C550_REUSE_CONFIG = {
    64: dict(
        bq=4,
        bn=64,
        warps=2,
        pv_transpose=True,
        qk_transpose=True,
        residual_preload=False,
        residual_bn=32,
        residual_flatten=True,
        residual_warps=1,
        sorted_ids=False,
        omit_empty=True,
    ),
    128: dict(
        bq=4,
        bn=16,
        warps=4,
        pv_transpose=True,
        qk_transpose=True,
        residual_preload=False,
        residual_bn=16,
        residual_flatten=False,
        residual_warps=1,
        sorted_ids=True,
        omit_empty=True,
    ),
}


def reuse_config(group, d, dtype, block_size, topk, local_blocks, max_q, total_q):
    if local_blocks is None or dtype not in (torch.float16, torch.bfloat16):
        return None
    init, local = local_blocks
    if not (max_q >= 1024 and total_q >= 1024 and 1 <= group <= 64 and 16 <= d <= 256):
        return None
    if not (
        block_size in (16, 32, 64, 128)
        and topk in (16, 32, 64, 128)
        and init >= 0
        and local >= 8
        and init + local + 1 <= min(64, topk)
    ):
        return None
    bg = triton.next_power_of_2(group)
    bd = triton.next_power_of_2(d)
    if group == 16 and d in _C550_REUSE_CONFIG and block_size == 64 and topk == 64 and local_blocks == (1, 32):
        return dict(_C550_REUSE_CONFIG[d])
    cfg = dict(
        bq=min(32, 64 // bg),
        bn=min(block_size, 64 if bd <= 64 else 16),
        warps=2 if bd <= 64 else 4,
        qk_transpose=True,
        pv_transpose=True,
        residual_bn=min(block_size, 32 if bd <= 64 else 16),
        residual_warps=1 if group <= 16 and bd <= 128 else 2,
        residual_flatten=bd <= 64,
        sorted_ids=bd > 64,
        omit_empty=True,
        residual_pv_transpose=bd == 64,
    )
    if bg == 1 and bd == 128:
        cfg.update(bq=32, bn=min(block_size, 64), warps=4)
    elif bg == 2 and bd == 128:
        cfg.update(bq=16, bn=min(block_size, 32), warps=2)
    if bd == 256:
        cfg.update(residual_warps=4 if bg <= 16 else 8, residual_qk_transpose=True, residual_pv_transpose=True)
    if bg == 64 and bd >= 128:
        cfg.update(residual_warps=4 if bd == 128 else 8)
    return cfg


@triton.jit
def _or_mask(a, b):
    return a | b


@triton.jit
def _local_reuse_c550(
    Q,
    K,
    V,
    Selected,
    CQ,
    CK,
    PA,
    PM,
    PL,
    O,
    LSE,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    G: tl.constexpr,
    BG: tl.constexpr,
    D: tl.constexpr,
    TQ: tl.constexpr,
    TOPK: tl.constexpr,
    BS: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    SCALE: tl.constexpr,
    BQ: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
    QK_TRANS: tl.constexpr,
    PV_TRANS: tl.constexpr,
    MERGE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    OMIT_EMPTY: tl.constexpr,
):
    tile, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    if tile * BQ >= qe - qs:
        return
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    m, d, n = tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, BN)
    qi = tile * BQ + m // BG
    head = hk * G + m % BG
    valid_m = (qi < qe - qs) & (m % BG < G)
    absolute = ke - ks - (qe - qs) + qi

    query = tile * BQ + tl.arange(0, BQ)
    selected_ranks = tl.arange(0, triton.next_power_of_2(TOPK))
    position = ke - ks - (qe - qs) + query
    ids = tl.load(
        Selected + (hk * TQ + qs + query[:, None]) * TOPK + selected_ranks[None, :],
        (query[:, None] < qe - qs) & (selected_ranks[None, :] < TOPK),
        -1,
    )
    local_start = position // BS - LOCAL
    in_local = (ids >= 0) & ((ids < INIT) | ((ids >= local_start[:, None]) & (ids <= position[:, None] // BS)))
    in_local = in_local & (selected_ranks[None, :] < tl.minimum(TOPK, tl.cdiv(position[:, None] + 1, BS)))
    id_bit = tl.where(ids < INIT, ids, INIT + ids - local_start[:, None])
    id_bit = tl.minimum(63, tl.maximum(0, id_bit)).to(tl.uint64)
    bits = tl.where(in_local, tl.full((), 1, tl.uint64) << id_bit, tl.full((), 0, tl.uint64))
    masks = tl.reduce(bits, 1, _or_mask)
    membership = tl.reshape(tl.broadcast_to(masks[:, None], (BQ, BG)), (BM,))
    qo = (qs + qi[:, None]) * HQ * D + head[:, None] * D + d[None, :]
    q = tl.load(Q + qo, valid_m[:, None] & (d[None, :] < D), 0)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BD, BM) if PV_TRANS else (BM, BD), tl.float32)
    first = ke - ks - (qe - qs) + tile * BQ
    last = ke - ks - (qe - qs) + tl.minimum((tile + 1) * BQ, qe - qs) - 1
    prefix = tl.maximum(0, tl.minimum(INIT, tl.cdiv(last + 1, BS)))
    local_first = tl.maximum(INIT, first // BS - LOCAL)
    local_last = last // BS + 1
    count = prefix + tl.maximum(0, local_last - local_first)
    steps: tl.constexpr = triton.cdiv(BS, BN)
    for iteration in tl.range(0, count * steps):
        rank = iteration // steps
        block = tl.where(rank < prefix, rank, local_first + rank - prefix)
        pos = block * BS + iteration % steps * BN + n
        valid_n = (pos < ke - ks) & (iteration % steps * BN + n < BS)
        ko = (ks + pos[:, None]) * HK * D + hk * D + d[None, :]
        k = tl.load(K + ko, valid_n[:, None] & (d[None, :] < D), 0)
        score = tl.trans(tl.dot(k, tl.trans(q))) if QK_TRANS else tl.dot(q, tl.trans(k))
        score = score * (SCALE * 1.4426950408889634)
        bit = tl.where(block < INIT, block, INIT + block - (absolute // BS - LOCAL))
        in_window = (block < INIT) | ((block >= absolute // BS - LOCAL) & (block <= absolute // BS))
        shift = tl.minimum(63, tl.maximum(0, bit)).to(tl.uint64)
        local = in_window & (((membership >> shift) & 1) != 0)
        valid = valid_m[:, None] & valid_n[None, :] & (pos[None, :] <= absolute[:, None]) & local[:, None]
        score = tl.where(valid, score, -float("inf"))
        mn = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.where(mi != -float("inf"), tl.exp2(mi - mn), 0.0)
        p = tl.where(valid, tl.exp2(score - mn[:, None]), 0.0)
        v = tl.load(V + ko, valid_n[:, None] & (d[None, :] < D), 0)
        if PV_TRANS:
            acc = tl.dot(tl.trans(v), tl.trans(p.to(v.dtype)), acc * alpha[None, :])
        else:
            acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None])
        li = li * alpha + tl.sum(p, 1)
        mi = mn
    if PV_TRANS:
        acc = tl.trans(acc)
    so = (qs + qi) * HQ + head
    if MERGE:
        remote_valid = valid_m
        if OMIT_EMPTY:
            remote_valid = remote_valid & (absolute + 1 > (INIT + LOCAL + 1) * BS)
        remote_m = tl.load(PM + so, remote_valid, -float("inf"))
        remote_l = tl.load(PL + so, remote_valid, 0.0)
        maximum = tl.maximum(mi, remote_m)
        local_alpha = tl.where(mi != -float("inf"), tl.exp2(mi - maximum), 0.0)
        remote_alpha = tl.where(remote_m != -float("inf"), tl.exp2(remote_m - maximum), 0.0)
        den = li * local_alpha + remote_l * remote_alpha
        remote_acc = tl.load(PA + qo, remote_valid[:, None] & (d[None, :] < D), 0.0)
        value = acc * local_alpha[:, None] + remote_acc * remote_alpha[:, None]
        result = tl.where(den[:, None] > 0.0, value / den[:, None], 0.0)
        tl.store(O + qo, result, valid_m[:, None] & (d[None, :] < D))
        if STORE_LSE:
            tl.store(
                LSE + so, tl.where(den > 0.0, (maximum + tl.log2(den)) * 0.6931471805599453, -float("inf")), valid_m
            )
    else:
        tl.store(PA + qo, acc, valid_m[:, None] & (d[None, :] < D))
        tl.store(PM + so, mi, valid_m)
        tl.store(PL + so, li, valid_m)


@triton.jit
def _merge_local_remote_c550(
    PA, PM, PL, O, LSE, NH: tl.constexpr, D: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr, STORE_LSE: tl.constexpr
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    d = tl.arange(0, BD)
    m0 = tl.load(PM + h, h < NH, -float("inf"))
    m1 = tl.load(PM + NH + h, h < NH, -float("inf"))
    l0 = tl.load(PL + h, h < NH, 0.0)
    l1 = tl.load(PL + NH + h, h < NH, 0.0)
    maximum = tl.maximum(m0, m1)
    a0 = tl.where(m0 != -float("inf"), tl.exp2(m0 - maximum), 0.0)
    a1 = tl.where(m1 != -float("inf"), tl.exp2(m1 - maximum), 0.0)
    den = l0 * a0 + l1 * a1
    off = h[:, None] * D + d[None, :]
    valid = (h[:, None] < NH) & (d[None, :] < D)
    x = tl.load(PA + off, valid, 0.0)
    y = tl.load(PA + NH * D + off, valid, 0.0)
    out = tl.where(den[:, None] > 0.0, (x * a0[:, None] + y * a1[:, None]) / den[:, None], 0.0)
    tl.store(O + off, out, valid)
    if STORE_LSE:
        tl.store(LSE + h, tl.where(den > 0.0, (maximum + tl.log2(den)) * 0.6931471805599453, -float("inf")), h < NH)


def sparse_reuse(
    q,
    k,
    v,
    selected,
    cq,
    ck,
    max_q,
    block_size=64,
    scale=None,
    store_lse=False,
    init_blocks=1,
    local_blocks=32,
    *,
    bq=4,
    bn=32,
    warps=4,
    qk_transpose=True,
    pv_transpose=False,
    residual_preload=False,
    merge_heads=8,
    fused_merge=True,
    residual_bn=None,
    residual_flatten=True,
    residual_warps=1,
    sorted_ids=False,
    omit_empty=False,
    residual_qk_transpose=True,
    residual_pv_transpose=None,
):
    tq, hq, d = q.shape
    hk = k.shape[1]
    group = hq // hk
    assert group > 0 and block_size > 0
    assert selected.shape[-1] > 0 and 0 <= init_blocks and 0 <= local_blocks
    assert init_blocks + local_blocks + 1 <= 64
    assert not omit_empty or fused_merge
    padded_group = triton.next_power_of_2(group)
    assert bq * padded_group >= 16
    if residual_pv_transpose is None:
        residual_pv_transpose = d == 64
    out = torch.empty_like(q)
    lse = torch.empty((tq, hq), device=q.device, dtype=torch.float32) if store_lse else out
    parts = 1 if fused_merge else 2
    pa = torch.empty((parts, tq, hq, d), device=q.device, dtype=torch.float32)
    pm = torch.empty((parts, tq, hq), device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    factor = d**-0.5 if scale is None else scale

    _sparse_c550_kernel[(max_q, hk, cq.numel() - 1)](
        q,
        k,
        v,
        selected,
        cq,
        ck,
        out,
        lse,
        pa,
        pm,
        pl,
        hq,
        hk,
        group,
        d,
        tq,
        selected.shape[-1],
        block_size,
        factor,
        True,
        max(16, padded_group),
        (16 if d >= 128 else 32) if residual_bn is None else residual_bn,
        triton.next_power_of_2(d),
        1,
        False,
        True,
        True,
        True,
        residual_flatten,
        residual_qk_transpose,
        residual_pv_transpose,
        SKIP_LOCAL=True,
        INIT=init_blocks,
        LOCAL=local_blocks,
        PARTIAL=True,
        PART_OFFSET=0 if fused_merge else 1,
        PRELOAD=residual_preload,
        SORTED_IDS=sorted_ids,
        OMIT_EMPTY=omit_empty,
        num_warps=residual_warps,
        num_stages=1,
    )
    _local_reuse_c550[(triton.cdiv(max_q, bq), hk, cq.numel() - 1)](
        q,
        k,
        v,
        selected,
        cq,
        ck,
        pa,
        pm,
        pl,
        out,
        lse,
        hq,
        hk,
        group,
        padded_group,
        d,
        tq,
        selected.shape[-1],
        block_size,
        init_blocks,
        local_blocks,
        factor,
        bq,
        bq * padded_group,
        bn,
        triton.next_power_of_2(d),
        qk_transpose,
        pv_transpose,
        fused_merge,
        store_lse,
        omit_empty,
        num_warps=warps,
        num_stages=1,
    )
    if not fused_merge:
        _merge_local_remote_c550[(triton.cdiv(tq * hq, merge_heads),)](
            pa,
            pm,
            pl,
            out,
            lse,
            tq * hq,
            d,
            merge_heads,
            triton.next_power_of_2(d),
            store_lse,
            num_warps=1,
            num_stages=1,
        )
    return out, lse
