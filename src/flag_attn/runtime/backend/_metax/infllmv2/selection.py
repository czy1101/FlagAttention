# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.


import torch
import triton
import triton.language as tl

from flag_attn.infllmv2.forward import (
    _check_cuda_contiguous,
    _compress_k_kernel,
    _topk_float_to_ordered_key,
    _topk_index_to_key,
    _topk_key_to_index,
)


@triton.jit
def _stage1_kernel(
    q_ptr,
    k1_ptr,
    k2_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k1_ptr,
    cu_k2_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    max_k1,
    scale,
    k1_stride: tl.constexpr,
    k2_stride: tl.constexpr,
    causal: tl.constexpr,
    tune_key: tl.constexpr,
    block_q: tl.constexpr,
    block_g: tl.constexpr,
    block_d: tl.constexpr,
    block_n: tl.constexpr,
    selection_topk: tl.constexpr = 0,
    selection_block_size: tl.constexpr = 64,
    selection_local_blocks: tl.constexpr = 0,
):
    q_local_start = tl.program_id(0) * block_q
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if causal and selection_topk > 0:
        kv_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
        unused = tl.maximum(0, selection_topk * selection_block_size - (kv_len - (q_end - q_start)))
        q_local_start += unused // block_q * block_q
    if q_start + q_local_start >= q_end:
        return
    k1_start = tl.load(cu_k1_ptr + batch)
    k1_end = tl.load(cu_k1_ptr + batch + 1)
    k2_start = tl.load(cu_k2_ptr + batch)
    k2_end = tl.load(cu_k2_ptr + batch + 1)
    k1_len = k1_end - k1_start
    k2_len = k2_end - k2_start
    q_len = q_end - q_start
    offs_q = tl.arange(0, block_q)
    q_local = q_local_start + offs_q
    q_index = q_start + q_local
    q_mask = q_index < q_end

    offs_g = tl.arange(0, block_g)
    offs_d = tl.arange(0, block_d)
    q_ptrs = q_ptr + q_index[:, None, None] * hq * d + (hk * group + offs_g[None, :, None]) * d + offs_d[None, None, :]
    q = tl.load(
        q_ptrs,
        mask=q_mask[:, None, None] & (offs_g[None, :, None] < group) & (offs_d[None, None, :] < d),
        other=0.0,
    )
    q = tl.reshape(q, [block_q * block_g, block_d])

    m = tl.full((block_q, block_g), float("-inf"), tl.float32)
    l = tl.zeros((block_q, block_g), tl.float32)
    # Align in COMPRESSED coordinates. Integer divisions here match positive
    # sequence lengths with C++ truncation toward zero, including Lq < stride.
    coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
    coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
    coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
    offs_n = tl.arange(0, block_n)
    coarse_end = k2_len
    if causal and selection_topk > 0:
        coarse_end = tl.minimum(k2_len, tl.max(coarse_right, 0))
    for start_n in tl.range(0, coarse_end, block_n):
        pos = start_n + offs_n
        k_ptrs = k2_ptr + (k2_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k2_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid = q_mask[:, None] & (pos[None, :] < k2_len)
        if causal:
            valid = valid & (pos[None, :] < coarse_right[:, None])
        logits = tl.where(
            valid[:, None, :] & (offs_g[None, :, None] < group),
            logits,
            float("-inf"),
        )
        tile_m = tl.max(logits, axis=2)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(
            valid[:, None, :] & (offs_g[None, :, None] < group),
            tl.exp(logits - new_m[:, :, None]),
            0.0,
        )
        l = l * alpha + tl.sum(p, axis=2)
        m = new_m
    lse = m + tl.log(l)

    fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
    fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
    fine_end = k1_len
    if causal and selection_topk > 0:
        position = kv_len - q_len + tl.minimum(q_local_start + block_q, q_len) - 1
        first_local = position // selection_block_size - selection_local_blocks
        fine_end = tl.minimum(k1_len, tl.maximum(0, first_local * (selection_block_size // k1_stride)))
    for start_n in tl.range(0, fine_end, block_n):
        pos = start_n + offs_n
        k_ptrs = k1_ptr + (k1_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k1_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid_n = q_mask[:, None] & (pos[None, :] < k1_len)
        if causal:
            valid_n = valid_n & (pos[None, :] < fine_right[:, None])
        valid_g = offs_g < group
        probs = tl.where(
            valid_g[None, :, None] & valid_n[:, None, :] & (l[:, :, None] > 0.0),
            tl.exp(logits - lse[:, :, None]),
            0.0,
        )
        score = tl.sum(probs, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
        out_ptrs = out_ptr + hk * total_q * max_k1 + q_index[:, None] * max_k1 + pos[None, :]
        tl.store(out_ptrs, score, mask=q_mask[:, None] & (pos[None, :] < k1_len))


def stage1(
    q: torch.Tensor,
    k1: torch.Tensor,
    k2: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    cu_seqlens_k2: torch.Tensor,
    max_seqlen_q: int,
    k1_stride: int = 16,
    k2_stride: int = 64,
    softmax_scale: float | None = None,
    causal: bool = True,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    _allow_tle: bool = True,
    _max_k1: int | None = None,
    _selection_topk: int = 0,
    _selection_block_size: int = 64,
    _selection_local_blocks: int = 0,
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item()) if _max_k1 is None else _max_k1
    # Stage1 probabilities are already rounded to the Q dtype in the kernel.
    # Storing them as FP32 only doubles traffic and capacity without retaining
    # additional information.

    allocate = torch.zeros if _max_k1 is None else torch.empty
    out = allocate((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    if max_k1 == 0:
        return out
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    tune_key = triton.next_power_of_2(max_seqlen_q)
    block_q = 4 if max_seqlen_q >= 1024 and group == 16 and d in (64, 128) else 1
    _stage1_kernel[(triton.cdiv(max_seqlen_q, block_q), hkv, cu_seqlens_q.numel() - 1)](
        q,
        k1,
        k2,
        out,
        cu_seqlens_q,
        cu_seqlens_k1,
        cu_seqlens_k2,
        cu_seqlens_k,
        hq=hq,
        hkv=hkv,
        group=group,
        d=d,
        total_q=total_q,
        max_k1=max_k1,
        scale=scale,
        k1_stride=k1_stride,
        k2_stride=k2_stride,
        causal=causal,
        tune_key=tune_key,
        block_q=block_q,
        block_g=max(16, triton.next_power_of_2(group)),
        block_d=triton.next_power_of_2(d),
        block_n=64,
        selection_topk=_selection_topk,
        selection_block_size=_selection_block_size,
        selection_local_blocks=_selection_local_blocks,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit
def _select_blocks_streaming_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k_ptr,
    total_q,
    max_blocks,
    hkv: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block_b: tl.constexpr,
):
    """Packed-key exact TopK with a streaming tile merge."""
    tl.static_assert(block_b >= topk)
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return

    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    q_position = k_len - q_len + q_local
    valid_blocks = tl.minimum(
        max_blocks,
        tl.minimum(
            (k_len + block_size - 1) // block_size,
            (q_position + block_size) // block_size,
        ),
    )
    q_index = q_start + q_local
    offs_t = tl.arange(0, topk)
    out_ptrs = out_ptr + hk * total_q * topk + q_index * topk + offs_t

    # Causal rows with N <= K select every visible block.  Their canonical
    # output is already the ascending identity, so scores need not be loaded.
    if valid_blocks <= topk:
        selected = tl.where(offs_t < valid_blocks, offs_t, -1)
        tl.store(out_ptrs, selected)
        return

    offs_b = tl.arange(0, block_b)
    num_tiles = tl.cdiv(valid_blocks, block_b)
    tile_start = (num_tiles - 1) * block_b
    block_idx = tile_start + offs_b
    valid = block_idx < valid_blocks
    score_row = score_ptr + hk * total_q * max_blocks + q_index * max_blocks
    score = tl.load(score_row + block_idx, mask=valid, other=float("-inf")).to(tl.float32)
    score = tl.where(score == score, score, float("-inf"))
    score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
    index_key = _topk_index_to_key(block_idx)
    packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
    packed = tl.where(valid, packed, tl.zeros_like(packed))

    selected_keys = tl.max(packed, axis=0, keep_dims=True) if topk == 1 else tl.topk(packed, topk)

    # Merge one sorted TopK list per score tile.  Packing score in the high
    # bits and the reversed block id in the low bits gives deterministic,
    # stable ties: equal scores prefer the smaller logical block id.
    for _ in tl.range(0, num_tiles - 1):
        if topk > 1:
            selected_keys = tl.bitonic_merge(selected_keys)
        tile_start -= block_b
        block_idx = tile_start + offs_b
        score = tl.load(score_row + block_idx).to(tl.float32)
        score = tl.where(score == score, score, float("-inf"))
        score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
        index_key = _topk_index_to_key(block_idx)
        packed = (score_key.to(tl.uint64) << 16) | index_key.to(tl.uint64)
        tile_keys = tl.max(packed, axis=0, keep_dims=True) if topk == 1 else tl.topk(packed, topk)
        selected_keys = tl.maximum(selected_keys, tile_keys)

    # Rotate the 16-bit index key above the score key and sort once more so
    # Stage2 receives the same ascending block-id order as the baseline.
    selected_keys = (selected_keys << 48) | (selected_keys >> 16)
    if topk > 1:
        selected_keys = tl.sort(selected_keys, descending=True)
    selected = _topk_key_to_index((selected_keys >> 48).to(tl.uint32))
    selected = tl.where(selected < valid_blocks, selected, -1)
    tl.store(out_ptrs, selected)


def select_blocks(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    _allow_tle: bool = True,
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(block_score, cu_seqlens_q, cu_seqlens_k)
    hkv, total_q, max_blocks = block_score.shape
    if max_blocks > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk <= 0 or topk > 2048 or (topk & (topk - 1)) != 0:
        raise ValueError("TopK must be a power of two no greater than 2048")
    out = torch.full((hkv, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if max_blocks == 0:
        return out
    max_q = int(torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item())
    grid = (max_q, hkv, cu_seqlens_q.numel() - 1)
    block_b = max(64, topk, triton.next_power_of_2(min(max_blocks, 1024)))
    num_warps = 2 if block_b <= 64 else (4 if block_b <= 128 else 8)
    _select_blocks_streaming_kernel[grid](
        block_score,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        max_blocks,
        hkv=hkv,
        block_size=block_size,
        topk=topk,
        block_b=block_b,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@triton.jit
def _compress_dual_c550(
    K,
    CK,
    KFine,
    KCoarse,
    C1,
    C2,
    B: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    N1: tl.constexpr,
    K1: tl.constexpr,
    S1: tl.constexpr,
    K2: tl.constexpr,
    S2: tl.constexpr,
    BB: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BD: tl.constexpr,
):
    chunk, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bi = tl.arange(0, BB)
    starts = tl.load(CK + bi, bi < B, 0)
    ends = tl.load(CK + bi + 1, bi < B, 0)
    counts1 = tl.where((bi < B) & (ends - starts >= K1), (ends - starts - K1) // S1 + 1, 0)
    counts2 = tl.where((bi < B) & (ends - starts >= K2), (ends - starts - K2) // S2 + 1, 0)
    base1 = tl.sum(tl.where(bi < batch, counts1, 0), 0)
    base2 = tl.sum(tl.where(bi < batch, counts2, 0), 0)
    length1 = tl.sum(tl.where(bi == batch, counts1, 0), 0)
    length2 = tl.sum(tl.where(bi == batch, counts2, 0), 0)
    if chunk == 0 and hk == 0:
        tl.store(C1 + batch, base1)
        tl.store(C2 + batch, base2)
        if batch == B - 1:
            tl.store(C1 + B, base1 + length1)
            tl.store(C2 + B, base2 + length2)
    start = tl.load(CK + batch)
    end = tl.load(CK + batch + 1)
    d = tl.arange(0, BD)
    if chunk < N1:
        if chunk < length1:
            w = tl.arange(0, W1)
            token = start + chunk * S1 + w
            values = tl.load(
                K + token[:, None] * HK * D + hk * D + d[None, :],
                (w[:, None] < K1) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K1
            tl.store(KFine + (base1 + chunk) * HK * D + hk * D + d, mean, d < D)
    else:
        coarse = chunk - N1
        if coarse < length2:
            w = tl.arange(0, W2)
            token = start + coarse * S2 + w
            values = tl.load(
                K + token[:, None] * HK * D + hk * D + d[None, :],
                (w[:, None] < K2) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K2
            tl.store(KCoarse + (base2 + coarse) * HK * D + hk * D + d, mean, d < D)


@triton.jit
def _compressed_metadata(
    CK,
    C1,
    C2,
    B: tl.constexpr,
    K1: tl.constexpr,
    S1: tl.constexpr,
    K2: tl.constexpr,
    S2: tl.constexpr,
    BB: tl.constexpr,
):
    i = tl.arange(0, BB)
    a = tl.load(CK + i, i < B, 0)
    b = tl.load(CK + i + 1, i < B, 0)
    n1 = tl.where((i < B) & (b - a >= K1), (b - a - K1) // S1 + 1, 0)
    n2 = tl.where((i < B) & (b - a >= K2), (b - a - K2) // S2 + 1, 0)
    tl.store(C1 + i + 1, tl.cumsum(n1), i < B)
    tl.store(C2 + i + 1, tl.cumsum(n2), i < B)
    tl.store(C1, 0)
    tl.store(C2, 0)


@triton.jit
def _pooled_keys(
    Score,
    blocks,
    valid,
    forced,
    fine_len,
    qindex,
    hk,
    TQ: tl.constexpr,
    N1: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
):
    value = tl.full(blocks.shape, -float("inf"), tl.float32)
    for j in tl.static_range(WINDOW):
        fine = blocks * STRIDE - PAD + j
        x = tl.load(
            Score + hk * TQ * N1 + qindex * N1 + fine,
            valid & (~forced) & (fine >= 0) & (fine < fine_len),
            -float("inf"),
        )
        value = tl.maximum(value, x)
    value = tl.where(forced, float("inf"), value)
    value = tl.where(value == value, value, -float("inf"))
    score_key = _topk_float_to_ordered_key(value.to(tl.uint32, bitcast=True))
    key = (score_key.to(tl.uint64) << 16) | _topk_index_to_key(blocks).to(tl.uint64)
    return tl.where(valid, key, tl.zeros_like(key))


@triton.jit
def _pool_select_c550(
    Score,
    CQ,
    CK,
    C1,
    Selected,
    TQ: tl.constexpr,
    N1: tl.constexpr,
    NB: tl.constexpr,
    BS: tl.constexpr,
    TOPK: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    BB: tl.constexpr,
    IDENTITY: tl.constexpr,
):
    qi, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    if qi >= qe - qs:
        return
    kl = tl.load(CK + batch + 1) - tl.load(CK + batch)
    position = kl - (qe - qs) + qi
    count = tl.minimum(NB, tl.minimum(tl.cdiv(kl, BS), tl.cdiv(position + 1, BS)))
    ranks = tl.arange(0, TOPK)
    dest = Selected + ((hk * TQ + qs + qi) * TOPK) + ranks
    if IDENTITY:
        tl.store(dest, tl.where(ranks < count, ranks, -1))
    else:
        if count <= TOPK:
            tl.store(dest, tl.where(ranks < count, ranks, -1))
            return
        fine_len = tl.load(C1 + batch + 1) - tl.load(C1 + batch)
        tiles = tl.cdiv(count, BB)
        start = (tiles - 1) * BB
        offsets = tl.arange(0, BB)
        blocks = start + offsets
        qblock = position // BS
        forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
        keys = _pooled_keys(Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TQ, N1, STRIDE, PAD, WINDOW)
        best = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
        for _ in tl.range(0, tiles - 1):
            if TOPK > 1:
                best = tl.bitonic_merge(best)
            start -= BB
            blocks = start + offsets
            forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
            keys = _pooled_keys(
                Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TQ, N1, STRIDE, PAD, WINDOW
            )
            top = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
            best = tl.maximum(best, top)
        best = (best << 48) | (best >> 16)
        if TOPK > 1:
            best = tl.sort(best, descending=True)
        selected = _topk_key_to_index((best >> 48).to(tl.uint32))
        tl.store(dest, tl.where(selected < count, selected, -1))


def select(q, k, cq, ck, max_q, max_k, config, scale):
    batch = cq.numel() - 1
    tq, hq, d = q.shape
    hk = k.shape[1]
    bs, topk = config.block_size, config.topk
    nb = triton.cdiv(max_k, bs)
    if nb > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk <= 0 or topk > 2048 or topk & (topk - 1):
        raise ValueError("TopK must be a power of two no greater than 2048")
    selected = torch.empty((hk, tq, topk), device=q.device, dtype=torch.int32)
    n1 = max(0, (max_k - config.k1_kernel_size) // config.k1_stride + 1)
    if nb <= topk:
        c1 = ck
        score = q
    else:
        n2 = max(0, (max_k - config.k2_kernel_size) // config.k2_stride + 1)
        c1, c2 = torch.empty_like(ck), torch.empty_like(ck)
        if batch > 32:
            _compressed_metadata[(1,)](
                ck,
                c1,
                c2,
                batch,
                config.k1_kernel_size,
                config.k1_stride,
                config.k2_kernel_size,
                config.k2_stride,
                triton.next_power_of_2(batch),
                num_warps=4,
            )
        compressed = []
        for n, window, stride, cu in [
            (n1, config.k1_kernel_size, config.k1_stride, c1),
            (n2, config.k2_kernel_size, config.k2_stride, c2),
        ]:
            capacity = min(batch * n, k.shape[0] // stride + batch)
            output = torch.empty((capacity, hk, d), device=k.device, dtype=k.dtype)
            if n and batch > 32:
                _compress_k_kernel[(n, hk, batch)](
                    k,
                    output,
                    ck,
                    cu,
                    hk,
                    d,
                    window,
                    stride,
                    triton.next_power_of_2(window),
                    triton.next_power_of_2(d),
                    num_warps=4,
                )
            compressed.append(output)
        if batch <= 32:
            _compress_dual_c550[(max(1, n1 + n2), hk, batch)](
                k,
                ck,
                *compressed,
                c1,
                c2,
                batch,
                hk,
                d,
                n1,
                config.k1_kernel_size,
                config.k1_stride,
                config.k2_kernel_size,
                config.k2_stride,
                triton.next_power_of_2(batch),
                triton.next_power_of_2(config.k1_kernel_size),
                triton.next_power_of_2(config.k2_kernel_size),
                triton.next_power_of_2(d),
                num_warps=4,
                num_stages=1,
            )
        score = stage1(
            q,
            *compressed,
            cq,
            c1,
            c2,
            max_q,
            config.k1_stride,
            config.k2_stride,
            scale,
            config.causal,
            ck,
            _max_k1=n1,
            _selection_topk=topk,
            _selection_block_size=bs,
            _selection_local_blocks=config.local_blocks,
        )
    bb = max(64, topk, triton.next_power_of_2(min(nb, 1024)))
    _pool_select_c550[(max_q, hk, batch)](
        score,
        cq,
        ck,
        c1,
        selected,
        tq,
        n1,
        nb,
        bs,
        topk,
        bs // config.k1_stride,
        config.k1_kernel_size // config.k1_stride - 1,
        config.k1_kernel_size // config.k1_stride + bs // config.k1_stride - 1,
        config.init_blocks,
        config.local_blocks,
        bb,
        nb <= topk,
        num_warps=2 if bb <= 64 else 4 if bb <= 128 else 8,
        num_stages=1,
    )
    return selected
