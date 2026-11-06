# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl

from flag_attn.infllmv2.forward import _check_cuda_contiguous


@triton.jit
def _sparse_c550_kernel(
    Q,
    K,
    V,
    Selected,
    CQ,
    CK,
    O,
    LSE,
    PA,
    PM,
    PL,
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
    SPLITS: tl.constexpr,
    STORE_LSE: tl.constexpr,
    FUSE_ACC: tl.constexpr,
    EXP2: tl.constexpr,
    TRANSPOSED_K: tl.constexpr,
    FLATTEN: tl.constexpr,
    QK_TRANS: tl.constexpr,
    PV_TRANS: tl.constexpr,
    SKIP_LOCAL: tl.constexpr = False,
    INIT: tl.constexpr = 0,
    LOCAL: tl.constexpr = 0,
    PARTIAL: tl.constexpr = False,
    PART_OFFSET: tl.constexpr = 0,
    PRELOAD: tl.constexpr = False,
    SORTED_IDS: tl.constexpr = False,
    OMIT_EMPTY: tl.constexpr = False,
):
    qi, hk, z = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, split = z // SPLITS, z % SPLITS
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    if qi >= qe - qs:
        return
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    limit = ke - ks - (qe - qs) + qi + 1
    if SKIP_LOCAL and OMIT_EMPTY:
        if limit <= (INIT + LOCAL + 1) * BS:
            return
    gm, nd, nn = tl.arange(0, BM), tl.arange(0, BD), tl.arange(0, BN)
    heads = hk * G + gm
    qo = (qs + qi) * HQ * D + heads[:, None] * D + nd[None, :]
    q = tl.load(Q + qo, (gm[:, None] < G) & (nd[None, :] < D), 0)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BD, BM) if PV_TRANS else (BM, BD), tl.float32)
    ranks = TOPK
    if CAUSAL:
        ranks = tl.minimum(ranks, tl.cdiv(limit, BS))
        if SKIP_LOCAL:
            ranks = tl.where(limit <= (INIT + LOCAL + 1) * BS, 0, ranks)
    per_split: tl.constexpr = triton.cdiv(TOPK, SPLITS)
    first, last = split * per_split, tl.minimum(ranks, (split + 1) * per_split)
    if SKIP_LOCAL and SORTED_IDS:
        rr = tl.arange(0, triton.next_power_of_2(TOPK))
        selected_ids = tl.load(Selected + (hk * TQ + qs + qi) * TOPK + rr, rr < ranks, -1)
        valid_ids = (rr < ranks) & (selected_ids >= 0)
        first = tl.sum((valid_ids & (selected_ids < INIT)).to(tl.int32), 0)
        last = tl.sum((valid_ids & (selected_ids < tl.maximum(INIT, (limit - 1) // BS - LOCAL))).to(tl.int32), 0)
    log2e: tl.constexpr = 1.4426950408889634
    steps: tl.constexpr = triton.cdiv(BS, BN)
    if PRELOAD:
        rr = tl.arange(0, triton.next_power_of_2(TOPK))
        resident_ids = tl.load(Selected + (hk * TQ + qs + qi) * TOPK + rr, rr < TOPK, -1)
    for iteration in tl.range(first * (steps if FLATTEN else 1), last * (steps if FLATTEN else 1)):
        rank = iteration // steps if FLATTEN else iteration
        if PRELOAD:
            block = tl.sum(tl.where(rr == rank, resident_ids, 0), 0)
        else:
            block = tl.load(Selected + (hk * TQ + qs + qi) * TOPK + rank)
        qblock = (limit - 1) // BS
        common = (block >= 0) & ((block < INIT) | ((block >= qblock - LOCAL) & (block <= qblock)))
        if not SKIP_LOCAL or SORTED_IDS or not common:
            for sub_static in tl.static_range(1 if FLATTEN else steps):
                sub = iteration % steps if FLATTEN else sub_static
                pos = tl.maximum(block, 0) * BS + sub * BN + nn
                valid = (block >= 0) & (sub * BN + nn < BS) & (pos < ke - ks)
                if CAUSAL:
                    valid = valid & (pos < limit)
                if QK_TRANS:
                    ko = (ks + pos[:, None]) * HK * D + hk * D + nd[None, :]
                    k = tl.load(K + ko, (nd[None, :] < D) & valid[:, None], 0)
                    logits = tl.trans(tl.dot(k, tl.trans(q)))
                elif TRANSPOSED_K:
                    ko = (ks + pos[None, :]) * HK * D + hk * D + nd[:, None]
                    k = tl.load(K + ko, (nd[:, None] < D) & valid[None, :], 0)
                    logits = tl.dot(q, k)
                else:
                    ko = (ks + pos[:, None]) * HK * D + hk * D + nd[None, :]
                    k = tl.trans(tl.load(K + ko, (nd[None, :] < D) & valid[:, None], 0))
                    logits = tl.dot(q, k)
                logits = logits * (SCALE * log2e if EXP2 else SCALE)
                logits = tl.where((gm[:, None] < G) & valid[None, :], logits, -float("inf"))
                mn = tl.maximum(mi, tl.max(logits, 1))
                alpha = tl.where(mi != -float("inf"), tl.exp2(mi - mn) if EXP2 else tl.exp(mi - mn), 0.0)
                p = tl.where(
                    (gm[:, None] < G) & valid[None, :],
                    tl.exp2(logits - mn[:, None]) if EXP2 else tl.exp(logits - mn[:, None]),
                    0.0,
                )
                vo = (ks + pos[:, None]) * HK * D + hk * D + nd[None, :]
                v = tl.load(V + vo, valid[:, None] & (nd[None, :] < D), 0)
                if PV_TRANS:
                    acc = acc * alpha[None, :]
                    acc = tl.dot(tl.trans(v), tl.trans(p.to(v.dtype)), acc)
                elif FUSE_ACC:
                    acc = acc * alpha[:, None]
                    acc = tl.dot(p.to(v.dtype), v, acc)
                else:
                    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                li = li * alpha + tl.sum(p, 1)
                mi = mn
    if PV_TRANS:
        acc = tl.trans(acc)
    if SPLITS == 1 and not PARTIAL:
        result = tl.where(li[:, None] > 0.0, acc / li[:, None], 0.0)
        tl.store(O + qo, result, (gm[:, None] < G) & (nd[None, :] < D))
        if STORE_LSE:
            lse = tl.where(li > 0.0, (mi + tl.log2(li)) / log2e if EXP2 else mi + tl.log(li), -float("inf"))
            tl.store(LSE + (qs + qi) * HQ + heads, lse, gm < G)
    else:
        po = (split + PART_OFFSET) * TQ * HQ * D + qo
        so = (split + PART_OFFSET) * TQ * HQ + (qs + qi) * HQ + heads
        tl.store(PA + po, acc, (gm[:, None] < G) & (nd[None, :] < D))
        tl.store(PM + so, mi if EXP2 else mi * log2e, gm < G)
        tl.store(PL + so, li, gm < G)


@triton.jit
def _merge_c550_kernel(
    PA,
    PM,
    PL,
    O,
    LSE,
    HQ: tl.constexpr,
    D: tl.constexpr,
    TQ: tl.constexpr,
    SPLITS: tl.constexpr,
    BD: tl.constexpr,
    NS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    ss, dd = tl.arange(0, NS), tl.arange(0, BD)
    so = ss * TQ * HQ + token * HQ + head
    m = tl.load(PM + so, ss < SPLITS, -float("inf"))
    l = tl.load(PL + so, ss < SPLITS, 0.0)
    maximum = tl.max(m, 0)
    alpha = tl.where(m != -float("inf"), tl.exp2(m - maximum), 0.0)
    denom = tl.sum(l * alpha, 0)
    partial = tl.load(PA + so[:, None] * D + dd[None, :], (ss[:, None] < SPLITS) & (dd[None, :] < D), 0.0)
    value = tl.sum(partial * alpha[:, None], 0)
    tl.store(O + (token * HQ + head) * D + dd, tl.where(denom > 0.0, value / denom, 0.0), dd < D)
    if STORE_LSE:
        tl.store(
            LSE + token * HQ + head,
            tl.where(denom > 0.0, (maximum + tl.log2(denom)) * 0.6931471805599453, -float("inf")),
        )


def sparse_forward(
    q,
    k,
    v,
    selected,
    cq,
    ck,
    max_q,
    block_size=64,
    scale=None,
    causal=True,
    store_lse=False,
    *,
    bn=None,
    warps=None,
    splits=None,
    fuse_acc=True,
    exp2=True,
    transposed_k=True,
    flatten=True,
    num_stages=1,
    pipeline="basic",
    scenario="",
    pipeline_load_num=-1,
    qk_transpose=None,
    pv_transpose=None,
    preload=False,
):
    tq, hq, d = q.shape
    hk = k.shape[1]
    group = hq // hk

    native_gqa16 = group == 16 and d in (64, 128)
    default_qk = native_gqa16
    default_pv = native_gqa16 and d == 64
    if max_q > 32 and q.dtype in (torch.float16, torch.bfloat16):
        if group <= 16 and 16 <= d <= 256:
            default_qk = group > 1 or d > 64
            default_pv = (group > 1 and d <= 64) or d > 128
        elif 32 < group <= 64 and 64 < d <= 128:
            default_qk = True
            default_pv = False
    qk_transpose = default_qk if qk_transpose is None else qk_transpose
    pv_transpose = default_pv if pv_transpose is None else pv_transpose
    topk = selected.shape[-1]
    bn = (16 if d >= 128 else 32) if bn is None else bn
    warps = (1 if d <= 128 and group <= 16 else 2 if group <= 32 else 4) if warps is None else warps
    if splits is None:
        splits = (
            min(
                topk,
                triton.next_power_of_2(max(1, triton.cdiv(k.shape[0], block_size))),
                triton.next_power_of_2(max(1, triton.cdiv(104, max(1, tq * hk)))),
            )
            if max_q <= 32
            else 1
        )
    out = torch.empty_like(q)
    lse = torch.empty((tq, hq), dtype=torch.float32, device=q.device) if store_lse else out
    if splits > 1:
        pa = torch.empty((splits, tq, hq, d), dtype=torch.float32, device=q.device)
        pm = torch.empty((splits, tq, hq), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
    else:
        pa = pm = pl = out
    _sparse_c550_kernel[(max_q, hk, (cq.numel() - 1) * splits)](
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
        topk,
        block_size,
        d**-0.5 if scale is None else scale,
        causal,
        max(16, triton.next_power_of_2(group)),
        bn,
        triton.next_power_of_2(d),
        splits,
        store_lse,
        fuse_acc,
        exp2,
        transposed_k,
        flatten,
        qk_transpose,
        pv_transpose,
        PRELOAD=preload,
        num_warps=warps,
        num_stages=num_stages,
        pipeline=pipeline,
        scenario=scenario,
        pipeline_load_num=pipeline_load_num,
    )
    if splits > 1:
        _merge_c550_kernel[(tq, hq)](
            pa,
            pm,
            pl,
            out,
            lse,
            hq,
            d,
            tq,
            splits,
            triton.next_power_of_2(d),
            triton.next_power_of_2(splits),
            store_lse,
            num_warps=4,
            num_stages=1,
        )
    return out, lse


def _sparse_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    store_lse: bool = True,
    _c550_local_blocks: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_cuda_contiguous(q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    if selected_blocks.ndim != 3 or selected_blocks.shape[:2] != (hkv, total_q):
        raise ValueError(
            "selected_blocks must have [Hkv, total_q_tokens, topk] layout; "
            "all query heads in one GQA group share the per-token Hkv selection"
        )
    from .reuse import sparse_reuse, reuse_config

    reuse = (
        reuse_config(
            hq // hkv, d, q.dtype, block_size, selected_blocks.shape[-1], _c550_local_blocks, max_seqlen_q, total_q
        )
        if causal
        else None
    )
    if reuse is not None:
        return sparse_reuse(
            q,
            k,
            v,
            selected_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            block_size,
            softmax_scale,
            store_lse,
            *_c550_local_blocks,
            **reuse,
        )
    return sparse_forward(
        q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, block_size, softmax_scale, causal, store_lse
    )


class _SparseAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_blocks: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        block_size: int,
        softmax_scale: float | None,
        causal: bool,
        max_seqlen_k: int | None,
        c550_local_blocks: tuple[int, int] | None,
    ) -> torch.Tensor:
        out, lse = _sparse_attention_forward(
            q,
            k,
            v,
            selected_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            block_size,
            softmax_scale,
            causal,
            _c550_local_blocks=c550_local_blocks,
        )
        ctx.save_for_backward(q, k, v, selected_blocks, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, selected, out, lse, cu_q, cu_k = ctx.saved_tensors
        dout = dout.contiguous()
        from .backward import sparse_backward

        dq, dk, dv = sparse_backward(
            q,
            k,
            v,
            out,
            dout,
            lse,
            selected,
            cu_q,
            cu_k,
            ctx.max_seqlen_q,
            ctx.block_size,
            ctx.softmax_scale,
            ctx.causal,
            max_k=ctx.max_seqlen_k,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_blocks: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    *,
    _allow_tle: bool = True,
    _max_seqlen_k: int | None = None,
    _c550_local_blocks: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Differentiable sparse attention over selected blocks of continuous packed K/V."""
    if not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, k, v)):
        return _sparse_attention_forward(
            q,
            k,
            v,
            selected_blocks,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            block_size,
            softmax_scale,
            causal,
            store_lse=False,
            _c550_local_blocks=_c550_local_blocks,
        )[0]
    return _SparseAttentionFunction.apply(
        q,
        k,
        v,
        selected_blocks,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        block_size,
        softmax_scale,
        causal,
        _max_seqlen_k,
        _c550_local_blocks,
    )
