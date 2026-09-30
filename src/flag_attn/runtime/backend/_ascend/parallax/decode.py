# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dense KV-cache Parallax decode for Ascend Triton."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


_LOG2E = tl.constexpr(1.4426950408889634)
_BLOCK_N = 128
_MAX_SPLITS = 32
_SPLIT_THRESHOLD = 512
_AICORE_TARGET = 32


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=["Skv"])
def _decode_single_kernel(
    q,
    r,
    k,
    v,
    out,
    scale,
    seqused_k,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    USE_SEQLEN: tl.constexpr,
):
    """Online reduction for one batch/query-head pair."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    kv_hi = Skv
    if USE_SEQLEN:
        kv_hi = tl.minimum(tl.maximum(tl.load(seqused_k + i_b), 0), Skv)
    kv_lo = 0
    if WINDOW_SIZE_LEFT >= 0:
        kv_lo = tl.maximum(kv_hi - WINDOW_SIZE_LEFT, 0)

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    q_offset = i_bh * K + offs_d
    b_q = tl.load(q + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    b_r = tl.load(r + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    scale_log2 = scale * _LOG2E

    m = tl.full((1,), -float("inf"), tl.float32)
    d1 = tl.zeros((1,), tl.float32)
    d2 = tl.zeros((1,), tl.float32)
    o1 = tl.zeros((BK,), tl.float32)
    o2 = tl.zeros((BK,), tl.float32)

    first_block = kv_lo // BN
    last_block = tl.cdiv(kv_hi, BN)
    for block_id in range(first_block, last_block):
        cols = block_id * BN + tl.arange(0, BN)
        mask_n = (cols >= kv_lo) & (cols < kv_hi)
        kv_offsets = ((i_b * Skv + cols[:, None]) * H + i_h) * K + offs_d[None, :]
        mask_kv = mask_n[:, None] & mask_d[None, :]
        b_k = tl.load(k + kv_offsets, mask=mask_kv, other=0.0)

        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        b_v = tl.load(v + kv_offsets, mask=mask_kv, other=0.0).to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + q_offset, result, mask=mask_d)


@triton.jit(do_not_specialize=["Skv"])
def _decode_split_kernel(
    q,
    r,
    k,
    v,
    partial_m,
    partial_d1,
    partial_d2,
    partial_o1,
    partial_o2,
    scale,
    Skv,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Compute the online-softmax state for one KV partition."""
    i_bh = tl.program_id(0).to(tl.int64)
    i_split = tl.program_id(1)
    i_b = i_bh // HQ
    i_hq = i_bh % HQ
    i_h = i_hq // G

    num_blocks = tl.cdiv(Skv, BN)
    first_block = (i_split * num_blocks) // NUM_SPLITS
    last_block = ((i_split + 1) * num_blocks) // NUM_SPLITS

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    q_offset = i_bh * K + offs_d
    b_q = tl.load(q + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    b_r = tl.load(r + q_offset, mask=mask_d, other=0.0).to(tl.float32)
    scale_log2 = scale * _LOG2E

    m = tl.full((1,), -float("inf"), tl.float32)
    d1 = tl.zeros((1,), tl.float32)
    d2 = tl.zeros((1,), tl.float32)
    o1 = tl.zeros((BK,), tl.float32)
    o2 = tl.zeros((BK,), tl.float32)

    for block_id in range(first_block, last_block):
        cols = block_id * BN + tl.arange(0, BN)
        mask_n = cols < Skv
        kv_offsets = ((i_b * Skv + cols[:, None]) * H + i_h) * K + offs_d[None, :]
        mask_kv = mask_n[:, None] & mask_d[None, :]
        b_k = tl.load(k + kv_offsets, mask=mask_kv, other=0.0)
        s1 = tl.sum(b_k.to(tl.float32) * b_q[None, :], axis=1) * scale_log2
        s2 = tl.sum(b_k.to(tl.float32) * b_r[None, :], axis=1)
        s1 = tl.where(mask_n, s1, -float("inf"))

        m_new = tl.maximum(m, tl.max(s1, axis=0))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = _exp2(m - m_safe)
        p1 = _exp2(s1 - m_safe)
        p2 = p1 * s2
        b_v = tl.load(v + kv_offsets, mask=mask_kv, other=0.0).to(tl.float32)
        d1 = d1 * alpha + tl.sum(p1, axis=0)
        d2 = d2 * alpha + tl.sum(p2, axis=0)
        o1 = o1 * alpha + tl.sum(p1[:, None] * b_v, axis=0)
        o2 = o2 * alpha + tl.sum(p2[:, None] * b_v, axis=0)
        m = m_new

    state_offset = i_bh * NUM_SPLITS + i_split
    scalar_offsets = state_offset + tl.arange(0, 1)
    tl.store(partial_m + scalar_offsets, m)
    tl.store(partial_d1 + scalar_offsets, d1)
    tl.store(partial_d2 + scalar_offsets, d2)
    vector_offsets = state_offset * K + offs_d
    tl.store(partial_o1 + vector_offsets, o1, mask=mask_d)
    tl.store(partial_o2 + vector_offsets, o2, mask=mask_d)


@triton.jit
def _decode_merge_kernel(
    partial_m,
    partial_d1,
    partial_d2,
    partial_o1,
    partial_o2,
    out,
    K: tl.constexpr,
    BK: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BSPLIT: tl.constexpr,
):
    """Merge split-local softmax states using a common log-sum-exp pivot."""
    i_bh = tl.program_id(0).to(tl.int64)
    offs_s = tl.arange(0, BSPLIT)
    mask_s = offs_s < NUM_SPLITS
    state_offsets = i_bh * NUM_SPLITS + offs_s
    m = tl.load(partial_m + state_offsets, mask=mask_s, other=-float("inf"))
    m_global = tl.max(m, axis=0)
    m_safe = tl.where(m_global == -float("inf"), 0.0, m_global)
    alpha = _exp2(m - m_safe)
    d1 = tl.sum(tl.load(partial_d1 + state_offsets, mask=mask_s, other=0.0) * alpha, axis=0)
    d2 = tl.sum(tl.load(partial_d2 + state_offsets, mask=mask_s, other=0.0) * alpha, axis=0)

    offs_d = tl.arange(0, BK)
    mask_d = offs_d < K
    vector_offsets = state_offsets[:, None] * K + offs_d[None, :]
    vector_mask = mask_s[:, None] & mask_d[None, :]
    o1 = tl.sum(tl.load(partial_o1 + vector_offsets, mask=vector_mask, other=0.0) * alpha[:, None], axis=0)
    o2 = tl.sum(tl.load(partial_o2 + vector_offsets, mask=vector_mask, other=0.0) * alpha[:, None], axis=0)
    inv_d1 = tl.where(d1 > 0.0, 1.0 / d1, 0.0)
    result = o1 * inv_d1 * (1.0 + d2 * inv_d1) - o2 * inv_d1
    tl.store(out + i_bh * K + offs_d, result, mask=mask_d)


_SPLIT_WORKSPACE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _split_workspace(device: torch.device, batch_heads: int, splits: int, dim: int):
    key = (device.type, device.index, batch_heads, splits, dim)
    workspace = _SPLIT_WORKSPACE.get(key)
    if workspace is None:
        scalar_shape = (batch_heads, splits)
        vector_shape = (batch_heads, splits, dim)
        workspace = (
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(scalar_shape, device=device, dtype=torch.float32),
            torch.empty(vector_shape, device=device, dtype=torch.float32),
            torch.empty(vector_shape, device=device, dtype=torch.float32),
        )
        _SPLIT_WORKSPACE[key] = workspace
    return workspace


def _validate_inputs(q, r, k, v, seqused_k, out):
    if q.ndim != 4 or r.shape != q.shape:
        raise ValueError("q and r must have the same (B, 1, H_q, D) shape")
    if q.shape[1] != 1:
        raise ValueError(f"decode requires seqlen_q=1, got {q.shape[1]}")
    if k.ndim != 4 or v.shape != k.shape:
        raise ValueError("k_cache and v_cache must have the same (B, L, H_kv, D) shape")
    batch, _, query_heads, dim = q.shape
    batch_k, length, kv_heads, dim_k = k.shape
    if batch != batch_k or dim != dim_k:
        raise ValueError("q/r and k/v batch size and head dimension must match")
    if batch < 1 or length < 1 or query_heads < 1 or kv_heads < 1 or dim < 1:
        raise ValueError("cache length, head counts, and head dimension must be positive")
    if query_heads % kv_heads:
        raise ValueError(f"H_q ({query_heads}) must be divisible by H_kv ({kv_heads}) for GQA")
    tensors = (q, r, k, v)
    if not all(t.device.type == "npu" for t in tensors):
        raise ValueError("q, r, k_cache, and v_cache must be NPU tensors")
    if not all(t.device == q.device for t in tensors):
        raise ValueError("q, r, k_cache, and v_cache must be on the same NPU device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q/r/k/v must be fp16 or bf16, got {q.dtype}")
    if not all(t.dtype == q.dtype for t in tensors):
        raise TypeError("q, r, k_cache, and v_cache must have the same dtype")
    if seqused_k is not None:
        if seqused_k.shape != (batch,) or seqused_k.dtype != torch.int32:
            raise ValueError(f"seqused_k must be int32 with shape ({batch},)")
        if seqused_k.device != q.device:
            raise ValueError("seqused_k must be on the same NPU device as q")
        lengths = seqused_k
        if not lengths.is_contiguous():
            lengths = lengths.contiguous()
        min_length = int(lengths.min().item())
        max_length = int(lengths.max().item())
        if min_length < 0 or max_length > length:
            raise ValueError(
                f"seqused_k values must be in [0, {length}], got [{min_length}, {max_length}]"
            )
        seqused_k = lengths
    if out is not None:
        if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
            raise ValueError("out must match q's shape, dtype, and device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    return batch, length, query_heads, kv_heads, dim, seqused_k


def _window_size_to_left(window_size) -> int:
    if window_size is None:
        return -1
    if isinstance(window_size, int):
        return window_size
    try:
        left, right = window_size
    except (TypeError, ValueError) as exc:
        raise ValueError("window_size must be None, an int, or a (left, right) pair") from exc
    if right is not None and right > 0:
        raise ValueError("single-token causal decode requires window_size right <= 0")
    return int(left)


def _choose_num_splits(batch_heads: int, kv_len: int, block_n: int) -> int:
    """Use split-KV for long, under-occupied decode launches."""
    blocks = triton.cdiv(kv_len, block_n)
    if kv_len < _SPLIT_THRESHOLD or batch_heads >= _AICORE_TARGET:
        return 1
    return max(1, min(triton.cdiv(2 * _AICORE_TARGET, batch_heads), blocks, _MAX_SPLITS))


def parallax_attn_with_kvcache(
    q: torch.Tensor,
    r: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    page_table: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    window_size=None,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute single-token Parallax attention against a dense KV cache."""
    if page_table is not None:
        raise NotImplementedError("paged KV (page_table) is not implemented")
    batch, length, query_heads, kv_heads, dim, seqused_k = _validate_inputs(
        q, r, k_cache, v_cache, seqused_k, out
    )
    window_left = _window_size_to_left(window_size)
    if window_left < -1:
        raise ValueError("window_size left must be -1 or non-negative")
    if scale is None:
        scale = 1.0 / math.sqrt(dim)

    q, r, k_cache, v_cache = (
        x if x.is_contiguous() else x.contiguous() for x in (q, r, k_cache, v_cache)
    )
    result = torch.empty_like(q) if out is None else out
    block_k = triton.next_power_of_2(dim)
    block_n = _BLOCK_N
    batch_heads = batch * query_heads
    num_splits = 1
    if seqused_k is None and window_left < 0:
        num_splits = _choose_num_splits(batch_heads, length, block_n)

    if num_splits == 1:
        _decode_single_kernel[(batch_heads,)](
            q,
            r,
            k_cache,
            v_cache,
            result,
            float(scale),
            seqused_k,
            length,
            HQ=query_heads,
            H=kv_heads,
            G=query_heads // kv_heads,
            K=dim,
            BK=block_k,
            BN=block_n,
            WINDOW_SIZE_LEFT=window_left,
            USE_SEQLEN=seqused_k is not None,
            num_warps=4,
            num_stages=2,
        )
        return result

    partial_m, partial_d1, partial_d2, partial_o1, partial_o2 = _split_workspace(
        q.device, batch_heads, num_splits, dim
    )
    _decode_split_kernel[(batch_heads, num_splits)](
        q,
        r,
        k_cache,
        v_cache,
        partial_m,
        partial_d1,
        partial_d2,
        partial_o1,
        partial_o2,
        float(scale),
        length,
        HQ=query_heads,
        H=kv_heads,
        G=query_heads // kv_heads,
        K=dim,
        BK=block_k,
        BN=block_n,
        NUM_SPLITS=num_splits,
        num_warps=4,
        num_stages=2,
    )
    _decode_merge_kernel[(batch_heads,)](
        partial_m,
        partial_d1,
        partial_d2,
        partial_o1,
        partial_o2,
        result,
        K=dim,
        BK=block_k,
        NUM_SPLITS=num_splits,
        BSPLIT=triton.next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return result


def parallax_decode(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float | None = None,
    *,
    window_size_left: int = -1,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility alias using the original decode argument names."""
    window_size = None if window_size_left < 0 else (window_size_left, 0)
    return parallax_attn_with_kvcache(
        q,
        r,
        k,
        v,
        window_size=window_size,
        scale=qk_scale,
        out=out,
    )


__all__ = ["parallax_attn_with_kvcache", "parallax_decode"]
