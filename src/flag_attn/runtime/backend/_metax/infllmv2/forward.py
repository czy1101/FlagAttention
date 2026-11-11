# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

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
from flag_attn.infllmv2.operator import InfLLMV2Config


_C550_MULTIPROCESSOR_COUNT = 104


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
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    _check_cuda_contiguous(q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k)
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k1.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError("number of query heads must be divisible by KV heads")
    gqa_group_size = num_q_heads // num_kv_heads
    max_k1 = int(torch.max(cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).item()) if _max_k1 is None else _max_k1
    # Stage1 probabilities are already rounded to the Q dtype in the kernel.
    # Storing them as FP32 only doubles traffic and capacity without retaining
    # additional information.

    allocate = torch.zeros if _max_k1 is None else torch.empty
    out = allocate((num_kv_heads, total_q, max_k1), dtype=q.dtype, device=q.device)
    if max_k1 == 0:
        return out
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    tune_key = triton.next_power_of_2(max_seqlen_q)
    block_q = 4 if max_seqlen_q >= 1024 and gqa_group_size == 16 and head_dim in (64, 128) else 1
    _stage1_kernel[(triton.cdiv(max_seqlen_q, block_q), num_kv_heads, cu_seqlens_q.numel() - 1)](
        q,
        k1,
        k2,
        out,
        cu_seqlens_q,
        cu_seqlens_k1,
        cu_seqlens_k2,
        cu_seqlens_k,
        hq=num_q_heads,
        hkv=num_kv_heads,
        group=gqa_group_size,
        d=head_dim,
        total_q=total_q,
        max_k1=max_k1,
        scale=scale,
        k1_stride=k1_stride,
        k2_stride=k2_stride,
        causal=causal,
        tune_key=tune_key,
        block_q=block_q,
        block_g=max(16, triton.next_power_of_2(gqa_group_size)),
        block_d=triton.next_power_of_2(head_dim),
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
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    _check_cuda_contiguous(block_score, cu_seqlens_q, cu_seqlens_k)
    num_kv_heads, total_q, max_blocks = block_score.shape
    if max_blocks > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk <= 0 or topk > 2048 or (topk & (topk - 1)) != 0:
        raise ValueError("TopK must be a power of two no greater than 2048")
    out = torch.full((num_kv_heads, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if max_blocks == 0:
        return out
    max_seqlen_q = int(torch.max(cu_seqlens_q[1:] - cu_seqlens_q[:-1]).item())
    grid = (max_seqlen_q, num_kv_heads, cu_seqlens_q.numel() - 1)
    block_b = max(64, topk, triton.next_power_of_2(min(max_blocks, 1024)))
    if block_b <= 64:
        num_warps = 2
    elif block_b <= 128:
        num_warps = 4
    else:
        num_warps = 8
    _select_blocks_streaming_kernel[grid](
        block_score,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        total_q,
        max_blocks,
        hkv=num_kv_heads,
        block_size=block_size,
        topk=topk,
        block_b=block_b,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


@triton.jit
def _compress_dual_kernel(
    K,
    CuK,
    KFine,
    KCoarse,
    C1,
    C2,
    B: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    N1: tl.constexpr,
    K1: tl.constexpr,
    S1: tl.constexpr,
    K2: tl.constexpr,
    S2: tl.constexpr,
    BB: tl.constexpr,
    W1: tl.constexpr,
    W2: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    bi = tl.arange(0, BB)
    starts = tl.load(CuK + bi, bi < B, 0)
    ends = tl.load(CuK + bi + 1, bi < B, 0)
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
    start = tl.load(CuK + batch)
    end = tl.load(CuK + batch + 1)
    d = tl.arange(0, BLOCK_D)
    if chunk < N1:
        if chunk < length1:
            w = tl.arange(0, W1)
            token = start + chunk * S1 + w
            values = tl.load(
                K + token[:, None] * HKV * D + hk * D + d[None, :],
                (w[:, None] < K1) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K1
            tl.store(KFine + (base1 + chunk) * HKV * D + hk * D + d, mean, d < D)
    else:
        coarse = chunk - N1
        if coarse < length2:
            w = tl.arange(0, W2)
            token = start + coarse * S2 + w
            values = tl.load(
                K + token[:, None] * HKV * D + hk * D + d[None, :],
                (w[:, None] < K2) & (token[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            mean = tl.sum(values, 0) / K2
            tl.store(KCoarse + (base2 + coarse) * HKV * D + hk * D + d, mean, d < D)


@triton.jit
def _compressed_metadata_kernel(
    CuK,
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
    a = tl.load(CuK + i, i < B, 0)
    b = tl.load(CuK + i + 1, i < B, 0)
    n1 = tl.where((i < B) & (b - a >= K1), (b - a - K1) // S1 + 1, 0)
    n2 = tl.where((i < B) & (b - a >= K2), (b - a - K2) // S2 + 1, 0)
    tl.store(C1 + i + 1, tl.cumsum(n1), i < B)
    tl.store(C2 + i + 1, tl.cumsum(n2), i < B)
    tl.store(C1, 0)
    tl.store(C2, 0)


@triton.jit
def _pooled_keys_kernel(
    Score,
    blocks,
    valid,
    forced,
    fine_len,
    qindex,
    hk,
    TOTAL_Q: tl.constexpr,
    N1: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
):
    value = tl.full(blocks.shape, -float("inf"), tl.float32)
    for j in tl.static_range(WINDOW):
        fine = blocks * STRIDE - PAD + j
        x = tl.load(
            Score + hk * TOTAL_Q * N1 + qindex * N1 + fine,
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
def _pool_select_kernel(
    Score,
    CuQ,
    CuK,
    C1,
    Selected,
    TOTAL_Q: tl.constexpr,
    N1: tl.constexpr,
    NUM_KV_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
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
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    if qi >= qe - qs:
        return
    kl = tl.load(CuK + batch + 1) - tl.load(CuK + batch)
    position = kl - (qe - qs) + qi
    count = tl.minimum(NUM_KV_BLOCKS, tl.minimum(tl.cdiv(kl, BLOCK_SIZE), tl.cdiv(position + 1, BLOCK_SIZE)))
    ranks = tl.arange(0, TOPK)
    dest = Selected + ((hk * TOTAL_Q + qs + qi) * TOPK) + ranks
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
        qblock = position // BLOCK_SIZE
        forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
        keys = _pooled_keys_kernel(
            Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TOTAL_Q, N1, STRIDE, PAD, WINDOW
        )
        best = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
        for _ in tl.range(0, tiles - 1):
            if TOPK > 1:
                best = tl.bitonic_merge(best)
            start -= BB
            blocks = start + offsets
            forced = (blocks < INIT) | ((qblock >= blocks) & (qblock <= blocks + LOCAL))
            keys = _pooled_keys_kernel(
                Score, blocks, blocks < count, forced, fine_len, qs + qi, hk, TOTAL_Q, N1, STRIDE, PAD, WINDOW
            )
            top = tl.max(keys, 0, keep_dims=True) if TOPK == 1 else tl.topk(keys, TOPK)
            best = tl.maximum(best, top)
        best = (best << 48) | (best >> 16)
        if TOPK > 1:
            best = tl.sort(best, descending=True)
        selected = _topk_key_to_index((best >> 48).to(tl.uint32))
        tl.store(dest, tl.where(selected < count, selected, -1))


def _select_attention_blocks(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    scale: float | None,
) -> torch.Tensor:
    batch = cu_seqlens_q.numel() - 1
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    bs, topk = config.block_size, config.topk
    nb = triton.cdiv(max_seqlen_k, bs)
    if nb > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk <= 0 or topk > 2048 or topk & (topk - 1):
        raise ValueError("TopK must be a power of two no greater than 2048")
    selected = torch.empty((num_kv_heads, total_q, topk), device=q.device, dtype=torch.int32)
    n1 = max(0, (max_seqlen_k - config.k1_kernel_size) // config.k1_stride + 1)
    if nb <= topk:
        c1 = cu_seqlens_k
        score = q
    else:
        n2 = max(0, (max_seqlen_k - config.k2_kernel_size) // config.k2_stride + 1)
        c1, c2 = torch.empty_like(cu_seqlens_k), torch.empty_like(cu_seqlens_k)
        if batch > 32:
            _compressed_metadata_kernel[(1,)](
                cu_seqlens_k,
                c1,
                c2,
                B=batch,
                K1=config.k1_kernel_size,
                S1=config.k1_stride,
                K2=config.k2_kernel_size,
                S2=config.k2_stride,
                BB=triton.next_power_of_2(batch),
                num_warps=4,
            )
        compressed = []
        for n, window, stride, cu in [
            (n1, config.k1_kernel_size, config.k1_stride, c1),
            (n2, config.k2_kernel_size, config.k2_stride, c2),
        ]:
            capacity = min(batch * n, k.shape[0] // stride + batch)
            output = torch.empty((capacity, num_kv_heads, head_dim), device=k.device, dtype=k.dtype)
            if n and batch > 32:
                _compress_k_kernel[(n, num_kv_heads, batch)](
                    k,
                    output,
                    cu_seqlens_k,
                    cu,
                    hkv=num_kv_heads,
                    d=head_dim,
                    kernel_size=window,
                    stride=stride,
                    block_window=triton.next_power_of_2(window),
                    block_d=triton.next_power_of_2(head_dim),
                    num_warps=4,
                )
            compressed.append(output)
        if batch <= 32:
            _compress_dual_kernel[(max(1, n1 + n2), num_kv_heads, batch)](
                k,
                cu_seqlens_k,
                compressed[0],
                compressed[1],
                c1,
                c2,
                B=batch,
                HKV=num_kv_heads,
                D=head_dim,
                N1=n1,
                K1=config.k1_kernel_size,
                S1=config.k1_stride,
                K2=config.k2_kernel_size,
                S2=config.k2_stride,
                BB=triton.next_power_of_2(batch),
                W1=triton.next_power_of_2(config.k1_kernel_size),
                W2=triton.next_power_of_2(config.k2_kernel_size),
                BLOCK_D=triton.next_power_of_2(head_dim),
                num_warps=4,
                num_stages=1,
            )
        score = stage1(
            q,
            *compressed,
            cu_seqlens_q,
            c1,
            c2,
            max_seqlen_q,
            config.k1_stride,
            config.k2_stride,
            scale,
            config.causal,
            cu_seqlens_k,
            _max_k1=n1,
            _selection_topk=topk,
            _selection_block_size=bs,
            _selection_local_blocks=config.local_blocks,
        )
    bb = max(64, topk, triton.next_power_of_2(min(nb, 1024)))
    _pool_select_kernel[(max_seqlen_q, num_kv_heads, batch)](
        score,
        cu_seqlens_q,
        cu_seqlens_k,
        c1,
        selected,
        TOTAL_Q=total_q,
        N1=n1,
        NUM_KV_BLOCKS=nb,
        BLOCK_SIZE=bs,
        TOPK=topk,
        STRIDE=bs // config.k1_stride,
        PAD=config.k1_kernel_size // config.k1_stride - 1,
        WINDOW=config.k1_kernel_size // config.k1_stride + bs // config.k1_stride - 1,
        INIT=config.init_blocks,
        LOCAL=config.local_blocks,
        BB=bb,
        IDENTITY=nb <= topk,
        num_warps=2 if bb <= 64 else 4 if bb <= 128 else 8,
        num_stages=1,
    )
    return selected


@triton.jit
def _dense_attention_kernel(
    Q,
    K,
    V,
    CuQ,
    CuK,
    O,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_LSE: tl.constexpr,
    QK_TRANS: tl.constexpr,
    PV_TRANS: tl.constexpr,
):
    tile, h, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    rm = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    if tile * BLOCK_M >= qe - qs:
        return
    rn, rd = tl.arange(0, BLOCK_N), tl.arange(0, BLOCK_D)
    hk = h // (HQ // HKV)
    qoff = (qs + rm[:, None]) * HQ * D + h * D + rd[None, :]
    q = tl.load(Q + qoff, (rm[:, None] < qe - qs) & (rd[None, :] < D), 0)
    absolute = ke - ks - (qe - qs) + rm
    mi = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    li = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_D, BLOCK_M) if PV_TRANS else (BLOCK_M, BLOCK_D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    end = tl.minimum(ke - ks, ke - ks - (qe - qs) + (tile + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        pos = start + rn
        off = (ks + pos[:, None]) * HKV * D + hk * D + rd[None, :]
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


def _dense_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float | None,
    store_lse: bool,
    *,
    bm: int | None = None,
    bn: int | None = None,
    warps: int | None = None,
    qk_transpose: bool | None = None,
    pv_transpose: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_q_heads, head_dim = q.shape

    native = q.dtype in (torch.float16, torch.bfloat16)
    padded128 = native and 64 < head_dim < 128
    large = native and 128 < head_dim <= 256
    if bm is None:
        if large:
            bm = 16
        else:
            bm = 32
    if bn is None:
        if large:
            bn = 16
        elif head_dim >= 128:
            bn = 32
        else:
            bn = 64
    if warps is None:
        if padded128:
            warps = 4
        else:
            warps = 2
    if qk_transpose is None:
        qk_transpose = padded128 or large
    if pv_transpose is None:
        pv_transpose = padded128 or large
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_q_heads), device=q.device, dtype=torch.float32) if store_lse else out
    _dense_attention_kernel[(triton.cdiv(max_seqlen_q, bm), num_q_heads, cu_seqlens_q.numel() - 1)](
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse,
        HQ=num_q_heads,
        HKV=k.shape[1],
        D=head_dim,
        SCALE=head_dim**-0.5 if scale is None else scale,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
        STORE_LSE=store_lse,
        QK_TRANS=qk_transpose,
        PV_TRANS=pv_transpose,
        num_warps=warps,
        num_stages=1,
    )
    return out, lse


class _DenseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, max_seqlen_k):
        out, lse = _dense_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, True)
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k, out, lse)
        ctx.max_seqlen_q, ctx.scale = max_seqlen_q, scale
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, dout):
        from .backward import sparse_backward

        q, k, v, cu_seqlens_q, cu_seqlens_k, out, lse = ctx.saved_tensors
        max_seqlen_k = ctx.max_seqlen_k
        if max_seqlen_k is None:
            max_seqlen_k = (
                k.shape[0]
                if cu_seqlens_k.numel() == 2
                else int((cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())
            )
        blocks = triton.cdiv(max_seqlen_k, 64)

        ids = torch.arange(blocks, device=q.device, dtype=torch.int32)
        selected = ids.view(1, 1, blocks).expand(k.shape[1], q.shape[0], blocks).contiguous()
        dq, dk, dv = sparse_backward(
            q, k, v, out, dout.contiguous(), lse, selected, cu_seqlens_q, cu_seqlens_k,
            ctx.max_seqlen_q, 64, ctx.scale, True, max_seqlen_k=max_seqlen_k
        )
        return dq, dk, dv, None, None, None, None, None


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    scale: float | None = None,
    max_seqlen_k: int | None = None,
) -> torch.Tensor:
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _DenseAttention.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, max_seqlen_k)
    return _dense_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, scale, False)[0]


@triton.jit
def _sparse_attention_kernel(
    Q,
    K,
    V,
    Selected,
    CuQ,
    CuK,
    O,
    LSE,
    PA,
    PM,
    PL,
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
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    if qi >= qe - qs:
        return
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    limit = ke - ks - (qe - qs) + qi + 1
    if SKIP_LOCAL and OMIT_EMPTY:
        if limit <= (INIT + LOCAL + 1) * BLOCK_SIZE:
            return
    gm, nd, nn = tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_N)
    heads = hk * GQA_GROUP_SIZE + gm
    qo = (qs + qi) * HQ * D + heads[:, None] * D + nd[None, :]
    q = tl.load(Q + qo, (gm[:, None] < GQA_GROUP_SIZE) & (nd[None, :] < D), 0)
    mi = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    li = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_D, BLOCK_M) if PV_TRANS else (BLOCK_M, BLOCK_D), tl.float32)
    ranks = TOPK
    if CAUSAL:
        ranks = tl.minimum(ranks, tl.cdiv(limit, BLOCK_SIZE))
        if SKIP_LOCAL:
            ranks = tl.where(limit <= (INIT + LOCAL + 1) * BLOCK_SIZE, 0, ranks)
    per_split: tl.constexpr = triton.cdiv(TOPK, SPLITS)
    first, last = split * per_split, tl.minimum(ranks, (split + 1) * per_split)
    if SKIP_LOCAL and SORTED_IDS:
        rr = tl.arange(0, triton.next_power_of_2(TOPK))
        selected_ids = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + rr, rr < ranks, -1)
        valid_ids = (rr < ranks) & (selected_ids >= 0)
        first = tl.sum((valid_ids & (selected_ids < INIT)).to(tl.int32), 0)
        last = tl.sum(
            (valid_ids & (selected_ids < tl.maximum(INIT, (limit - 1) // BLOCK_SIZE - LOCAL))).to(tl.int32), 0
        )
    log2e: tl.constexpr = 1.4426950408889634
    steps: tl.constexpr = triton.cdiv(BLOCK_SIZE, BLOCK_N)
    if PRELOAD:
        rr = tl.arange(0, triton.next_power_of_2(TOPK))
        resident_ids = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + rr, rr < TOPK, -1)
    for iteration in tl.range(first * (steps if FLATTEN else 1), last * (steps if FLATTEN else 1)):
        rank = iteration // steps if FLATTEN else iteration
        if PRELOAD:
            block = tl.sum(tl.where(rr == rank, resident_ids, 0), 0)
        else:
            block = tl.load(Selected + (hk * TOTAL_Q + qs + qi) * TOPK + rank)
        qblock = (limit - 1) // BLOCK_SIZE
        common = (block >= 0) & ((block < INIT) | ((block >= qblock - LOCAL) & (block <= qblock)))
        if not SKIP_LOCAL or SORTED_IDS or not common:
            for sub_static in tl.static_range(1 if FLATTEN else steps):
                sub = iteration % steps if FLATTEN else sub_static
                pos = tl.maximum(block, 0) * BLOCK_SIZE + sub * BLOCK_N + nn
                valid = (block >= 0) & (sub * BLOCK_N + nn < BLOCK_SIZE) & (pos < ke - ks)
                if CAUSAL:
                    valid = valid & (pos < limit)
                if QK_TRANS:
                    ko = (ks + pos[:, None]) * HKV * D + hk * D + nd[None, :]
                    k = tl.load(K + ko, (nd[None, :] < D) & valid[:, None], 0)
                    logits = tl.trans(tl.dot(k, tl.trans(q)))
                elif TRANSPOSED_K:
                    ko = (ks + pos[None, :]) * HKV * D + hk * D + nd[:, None]
                    k = tl.load(K + ko, (nd[:, None] < D) & valid[None, :], 0)
                    logits = tl.dot(q, k)
                else:
                    ko = (ks + pos[:, None]) * HKV * D + hk * D + nd[None, :]
                    k = tl.trans(tl.load(K + ko, (nd[None, :] < D) & valid[:, None], 0))
                    logits = tl.dot(q, k)
                logits = logits * (SCALE * log2e if EXP2 else SCALE)
                logits = tl.where((gm[:, None] < GQA_GROUP_SIZE) & valid[None, :], logits, -float("inf"))
                mn = tl.maximum(mi, tl.max(logits, 1))
                alpha = tl.where(mi != -float("inf"), tl.exp2(mi - mn) if EXP2 else tl.exp(mi - mn), 0.0)
                p = tl.where(
                    (gm[:, None] < GQA_GROUP_SIZE) & valid[None, :],
                    tl.exp2(logits - mn[:, None]) if EXP2 else tl.exp(logits - mn[:, None]),
                    0.0,
                )
                vo = (ks + pos[:, None]) * HKV * D + hk * D + nd[None, :]
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
        tl.store(O + qo, result, (gm[:, None] < GQA_GROUP_SIZE) & (nd[None, :] < D))
        if STORE_LSE:
            lse = tl.where(li > 0.0, (mi + tl.log2(li)) / log2e if EXP2 else mi + tl.log(li), -float("inf"))
            tl.store(LSE + (qs + qi) * HQ + heads, lse, gm < GQA_GROUP_SIZE)
    else:
        po = (split + PART_OFFSET) * TOTAL_Q * HQ * D + qo
        so = (split + PART_OFFSET) * TOTAL_Q * HQ + (qs + qi) * HQ + heads
        tl.store(PA + po, acc, (gm[:, None] < GQA_GROUP_SIZE) & (nd[None, :] < D))
        tl.store(PM + so, mi if EXP2 else mi * log2e, gm < GQA_GROUP_SIZE)
        tl.store(PL + so, li, gm < GQA_GROUP_SIZE)


@triton.jit
def _merge_attention_kernel(
    PA,
    PM,
    PL,
    O,
    LSE,
    HQ: tl.constexpr,
    D: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NS: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    ss, dd = tl.arange(0, NS), tl.arange(0, BLOCK_D)
    so = ss * TOTAL_Q * HQ + token * HQ + head
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


def _sparse_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    scale: float | None = None,
    causal: bool = True,
    store_lse: bool = False,
    *,
    bn: int | None = None,
    warps: int | None = None,
    splits: int | None = None,
    fuse_acc: bool = True,
    exp2: bool = True,
    transposed_k: bool = True,
    flatten: bool = True,
    num_stages: int = 1,
    pipeline: str = 'basic',
    scenario: str = '',
    pipeline_load_num: int = -1,
    qk_transpose: bool | None = None,
    pv_transpose: bool | None = None,
    preload: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads

    native_gqa16 = gqa_group_size == 16 and head_dim in (64, 128)
    default_qk = native_gqa16
    default_pv = native_gqa16 and head_dim == 64
    if max_seqlen_q > 32 and q.dtype in (torch.float16, torch.bfloat16):
        if gqa_group_size <= 16 and 16 <= head_dim <= 256:
            default_qk = gqa_group_size > 1 or head_dim > 64
            default_pv = (gqa_group_size > 1 and head_dim <= 64) or head_dim > 128
        elif 32 < gqa_group_size <= 64 and 64 < head_dim <= 128:
            default_qk = True
            default_pv = False
    if qk_transpose is None:
        qk_transpose = default_qk
    if pv_transpose is None:
        pv_transpose = default_pv
    topk = selected.shape[-1]
    if bn is None:
        if head_dim >= 128:
            bn = 16
        else:
            bn = 32
    if warps is None:
        if head_dim <= 128 and gqa_group_size <= 16:
            warps = 1
        elif gqa_group_size <= 32:
            warps = 2
        else:
            warps = 4
    if splits is None:
        if max_seqlen_q <= 32:
            splits = min(
                topk,
                triton.next_power_of_2(max(1, triton.cdiv(k.shape[0], block_size))),
                triton.next_power_of_2(
                    max(1, triton.cdiv(_C550_MULTIPROCESSOR_COUNT, max(1, total_q * num_kv_heads)))
                ),
            )
        else:
            splits = 1
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device) if store_lse else out
    if splits > 1:
        pa = torch.empty((splits, total_q, num_q_heads, head_dim), dtype=torch.float32, device=q.device)
        pm = torch.empty((splits, total_q, num_q_heads), dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
    else:
        pa = pm = pl = out
    _sparse_attention_kernel[(max_seqlen_q, num_kv_heads, (cu_seqlens_q.numel() - 1) * splits)](
        q,
        k,
        v,
        selected,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse,
        pa,
        pm,
        pl,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        D=head_dim,
        TOTAL_Q=total_q,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        SCALE=head_dim**-0.5 if scale is None else scale,
        CAUSAL=causal,
        BLOCK_M=max(16, triton.next_power_of_2(gqa_group_size)),
        BLOCK_N=bn,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SPLITS=splits,
        STORE_LSE=store_lse,
        FUSE_ACC=fuse_acc,
        EXP2=exp2,
        TRANSPOSED_K=transposed_k,
        FLATTEN=flatten,
        QK_TRANS=qk_transpose,
        PV_TRANS=pv_transpose,
        PRELOAD=preload,
        num_warps=warps,
        num_stages=num_stages,
        pipeline=pipeline,
        scenario=scenario,
        pipeline_load_num=pipeline_load_num,
    )
    if splits > 1:
        _merge_attention_kernel[(total_q, num_q_heads)](
            pa,
            pm,
            pl,
            out,
            lse,
            HQ=num_q_heads,
            D=head_dim,
            TOTAL_Q=total_q,
            SPLITS=splits,
            BLOCK_D=triton.next_power_of_2(head_dim),
            NS=triton.next_power_of_2(splits),
            STORE_LSE=store_lse,
            num_warps=4,
            num_stages=1,
        )
    return out, lse


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


def _get_local_attention_config(
    gqa_group_size: int,
    head_dim: int,
    dtype: torch.dtype,
    block_size: int,
    topk: int,
    local_blocks: tuple[int, int] | None,
    max_seqlen_q: int,
    total_q: int,
) -> dict[str, int | bool] | None:
    if local_blocks is None or dtype not in (torch.float16, torch.bfloat16):
        return None
    init, local = local_blocks
    if not (max_seqlen_q >= 1024 and total_q >= 1024 and 1 <= gqa_group_size <= 64 and 16 <= head_dim <= 256):
        return None
    if not (
        block_size in (16, 32, 64, 128)
        and topk in (16, 32, 64, 128)
        and init >= 0
        and local >= 8
        and init + local + 1 <= min(64, topk)
    ):
        return None
    bg = triton.next_power_of_2(gqa_group_size)
    bd = triton.next_power_of_2(head_dim)
    if (
        gqa_group_size == 16
        and head_dim in _C550_REUSE_CONFIG
        and block_size == 64
        and topk == 64
        and local_blocks == (1, 32)
    ):
        return dict(_C550_REUSE_CONFIG[head_dim])
    cfg = dict(
        bq=min(32, 64 // bg),
        bn=min(block_size, 64 if bd <= 64 else 16),
        warps=2 if bd <= 64 else 4,
        qk_transpose=True,
        pv_transpose=True,
        residual_bn=min(block_size, 32 if bd <= 64 else 16),
        residual_warps=1 if gqa_group_size <= 16 and bd <= 128 else 2,
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
def _local_attention_kernel(
    Q,
    K,
    V,
    Selected,
    CuQ,
    CuK,
    PA,
    PM,
    PL,
    O,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    BLOCK_G: tl.constexpr,
    D: tl.constexpr,
    TOTAL_Q: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INIT: tl.constexpr,
    LOCAL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QK_TRANS: tl.constexpr,
    PV_TRANS: tl.constexpr,
    MERGE: tl.constexpr,
    STORE_LSE: tl.constexpr,
    OMIT_EMPTY: tl.constexpr,
):
    tile, hk, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CuQ + batch), tl.load(CuQ + batch + 1)
    if tile * BLOCK_Q >= qe - qs:
        return
    ks, ke = tl.load(CuK + batch), tl.load(CuK + batch + 1)
    m, d, n = tl.arange(0, BLOCK_M), tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_N)
    qi = tile * BLOCK_Q + m // BLOCK_G
    head = hk * GQA_GROUP_SIZE + m % BLOCK_G
    valid_m = (qi < qe - qs) & (m % BLOCK_G < GQA_GROUP_SIZE)
    absolute = ke - ks - (qe - qs) + qi

    query = tile * BLOCK_Q + tl.arange(0, BLOCK_Q)
    selected_ranks = tl.arange(0, triton.next_power_of_2(TOPK))
    position = ke - ks - (qe - qs) + query
    ids = tl.load(
        Selected + (hk * TOTAL_Q + qs + query[:, None]) * TOPK + selected_ranks[None, :],
        (query[:, None] < qe - qs) & (selected_ranks[None, :] < TOPK),
        -1,
    )
    local_start = position // BLOCK_SIZE - LOCAL
    in_local = (ids >= 0) & ((ids < INIT) | ((ids >= local_start[:, None]) & (ids <= position[:, None] // BLOCK_SIZE)))
    in_local = in_local & (selected_ranks[None, :] < tl.minimum(TOPK, tl.cdiv(position[:, None] + 1, BLOCK_SIZE)))
    id_bit = tl.where(ids < INIT, ids, INIT + ids - local_start[:, None])
    id_bit = tl.minimum(63, tl.maximum(0, id_bit)).to(tl.uint64)
    bits = tl.where(in_local, tl.full((), 1, tl.uint64) << id_bit, tl.full((), 0, tl.uint64))
    masks = tl.reduce(bits, 1, _or_mask)
    membership = tl.reshape(tl.broadcast_to(masks[:, None], (BLOCK_Q, BLOCK_G)), (BLOCK_M,))
    qo = (qs + qi[:, None]) * HQ * D + head[:, None] * D + d[None, :]
    q = tl.load(Q + qo, valid_m[:, None] & (d[None, :] < D), 0)
    mi = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    li = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_D, BLOCK_M) if PV_TRANS else (BLOCK_M, BLOCK_D), tl.float32)
    first = ke - ks - (qe - qs) + tile * BLOCK_Q
    last = ke - ks - (qe - qs) + tl.minimum((tile + 1) * BLOCK_Q, qe - qs) - 1
    prefix = tl.maximum(0, tl.minimum(INIT, tl.cdiv(last + 1, BLOCK_SIZE)))
    local_first = tl.maximum(INIT, first // BLOCK_SIZE - LOCAL)
    local_last = last // BLOCK_SIZE + 1
    count = prefix + tl.maximum(0, local_last - local_first)
    steps: tl.constexpr = triton.cdiv(BLOCK_SIZE, BLOCK_N)
    for iteration in tl.range(0, count * steps):
        rank = iteration // steps
        block = tl.where(rank < prefix, rank, local_first + rank - prefix)
        pos = block * BLOCK_SIZE + iteration % steps * BLOCK_N + n
        valid_n = (pos < ke - ks) & (iteration % steps * BLOCK_N + n < BLOCK_SIZE)
        ko = (ks + pos[:, None]) * HKV * D + hk * D + d[None, :]
        k = tl.load(K + ko, valid_n[:, None] & (d[None, :] < D), 0)
        score = tl.trans(tl.dot(k, tl.trans(q))) if QK_TRANS else tl.dot(q, tl.trans(k))
        score = score * (SCALE * 1.4426950408889634)
        bit = tl.where(block < INIT, block, INIT + block - (absolute // BLOCK_SIZE - LOCAL))
        in_window = (block < INIT) | ((block >= absolute // BLOCK_SIZE - LOCAL) & (block <= absolute // BLOCK_SIZE))
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
            remote_valid = remote_valid & (absolute + 1 > (INIT + LOCAL + 1) * BLOCK_SIZE)
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
def _merge_local_remote_kernel(
    PA,
    PM,
    PL,
    O,
    LSE,
    NH: tl.constexpr,
    D: tl.constexpr,
    BH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    STORE_LSE: tl.constexpr,
):
    h = tl.program_id(0) * BH + tl.arange(0, BH)
    d = tl.arange(0, BLOCK_D)
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


def _sparse_forward_with_local_reuse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    block_size: int = 64,
    scale: float | None = None,
    store_lse: bool = False,
    init_blocks: int = 1,
    local_blocks: int = 32,
    *,
    bq: int = 4,
    bn: int = 32,
    warps: int = 4,
    qk_transpose: bool = True,
    pv_transpose: bool = False,
    residual_preload: bool = False,
    merge_heads: int = 8,
    fused_merge: bool = True,
    residual_bn: int | None = None,
    residual_flatten: bool = True,
    residual_warps: int = 1,
    sorted_ids: bool = False,
    omit_empty: bool = False,
    residual_qk_transpose: bool = True,
    residual_pv_transpose: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    gqa_group_size = num_q_heads // num_kv_heads
    assert gqa_group_size > 0 and block_size > 0
    assert selected.shape[-1] > 0 and 0 <= init_blocks and 0 <= local_blocks
    assert init_blocks + local_blocks + 1 <= 64
    assert not omit_empty or fused_merge
    padded_group = triton.next_power_of_2(gqa_group_size)
    assert bq * padded_group >= 16
    if residual_pv_transpose is None:
        residual_pv_transpose = head_dim == 64
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_q_heads), device=q.device, dtype=torch.float32) if store_lse else out
    parts = 1 if fused_merge else 2
    pa = torch.empty((parts, total_q, num_q_heads, head_dim), device=q.device, dtype=torch.float32)
    pm = torch.empty((parts, total_q, num_q_heads), device=q.device, dtype=torch.float32)
    pl = torch.empty_like(pm)
    factor = head_dim**-0.5 if scale is None else scale

    _sparse_attention_kernel[(max_seqlen_q, num_kv_heads, cu_seqlens_q.numel() - 1)](
        q,
        k,
        v,
        selected,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse,
        pa,
        pm,
        pl,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        D=head_dim,
        TOTAL_Q=total_q,
        TOPK=selected.shape[-1],
        BLOCK_SIZE=block_size,
        SCALE=factor,
        CAUSAL=True,
        BLOCK_M=max(16, padded_group),
        BLOCK_N=(16 if head_dim >= 128 else 32) if residual_bn is None else residual_bn,
        BLOCK_D=triton.next_power_of_2(head_dim),
        SPLITS=1,
        STORE_LSE=False,
        FUSE_ACC=True,
        EXP2=True,
        TRANSPOSED_K=True,
        FLATTEN=residual_flatten,
        QK_TRANS=residual_qk_transpose,
        PV_TRANS=residual_pv_transpose,
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
    _local_attention_kernel[(triton.cdiv(max_seqlen_q, bq), num_kv_heads, cu_seqlens_q.numel() - 1)](
        q,
        k,
        v,
        selected,
        cu_seqlens_q,
        cu_seqlens_k,
        pa,
        pm,
        pl,
        out,
        lse,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GQA_GROUP_SIZE=gqa_group_size,
        BLOCK_G=padded_group,
        D=head_dim,
        TOTAL_Q=total_q,
        TOPK=selected.shape[-1],
        BLOCK_SIZE=block_size,
        INIT=init_blocks,
        LOCAL=local_blocks,
        SCALE=factor,
        BLOCK_Q=bq,
        BLOCK_M=bq * padded_group,
        BLOCK_N=bn,
        BLOCK_D=triton.next_power_of_2(head_dim),
        QK_TRANS=qk_transpose,
        PV_TRANS=pv_transpose,
        MERGE=fused_merge,
        STORE_LSE=store_lse,
        OMIT_EMPTY=omit_empty,
        num_warps=warps,
        num_stages=1,
    )
    if not fused_merge:
        _merge_local_remote_kernel[(triton.cdiv(total_q * num_q_heads, merge_heads),)](
            pa,
            pm,
            pl,
            out,
            lse,
            NH=total_q * num_q_heads,
            D=head_dim,
            BH=merge_heads,
            BLOCK_D=triton.next_power_of_2(head_dim),
            STORE_LSE=store_lse,
            num_warps=1,
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
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError("number of query heads must be divisible by KV heads")
    if selected_blocks.ndim != 3 or selected_blocks.shape[:2] != (num_kv_heads, total_q):
        raise ValueError(
            "selected_blocks must have [Hkv, total_q_tokens, topk] layout; "
            "all query heads in one GQA group share the per-token Hkv selection"
        )
    reuse = (
        _get_local_attention_config(
            num_q_heads // num_kv_heads, head_dim, q.dtype, block_size,
            selected_blocks.shape[-1], _c550_local_blocks, max_seqlen_q, total_q
        )
        if causal
        else None
    )
    if reuse is not None:
        return _sparse_forward_with_local_reuse(
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
    return _sparse_forward(
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
    def backward(
        ctx,
        dout: torch.Tensor,
    ):
        q, k, v, selected, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
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
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.block_size,
            ctx.softmax_scale,
            ctx.causal,
            max_seqlen_k=ctx.max_seqlen_k,
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
