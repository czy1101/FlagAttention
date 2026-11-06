# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Packed causal Triton forward kernels, runtime guards and metadata."""

from __future__ import annotations

import inspect
import itertools
import math
import os
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver

try:
    if triton.__version__ != "3.6.0":
        raise ImportError(f"loaded Triton {triton.__version__}")
    from flagtree import _flagprism
    from triton import knobs
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.runtime._distributed import DistributedRtContext
except (ImportError, AttributeError) as exc:
    raise ImportError(
        "Hygon InfLLM-V2 requires the validated HCU FlagTree 0.7.0+hcu3.6 / Triton 3.6.0 runtime. "
        "Install that HCU runtime and check triton.__version__ and triton.__file__; "
        "the generic Triton package metadata alone does not establish compatibility."
    ) from exc

from flag_attn.runtime import autotune_cache_kwargs


# Base kernels and fallback
def _check_cuda_contiguous(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError("InfLLM-V2 Triton kernels require CUDA tensors")
        if not tensor.is_contiguous():
            raise ValueError("InfLLM-V2 Triton kernels require contiguous tensors")


@triton.jit
def _compress_k_kernel(
    k_ptr,
    out_ptr,
    cu_in_ptr,
    cu_out_ptr,
    hkv: tl.constexpr,
    d: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    block_window: tl.constexpr,
    block_d: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    in_start = tl.load(cu_in_ptr + batch)
    in_end = tl.load(cu_in_ptr + batch + 1)
    out_start = tl.load(cu_out_ptr + batch)
    out_end = tl.load(cu_out_ptr + batch + 1)
    if out_start + chunk >= out_end:
        return
    offs_w = tl.arange(0, block_window)
    offs_d = tl.arange(0, block_d)
    token = in_start + chunk * stride + offs_w
    ptrs = k_ptr + token[:, None] * hkv * d + head * d + offs_d[None, :]
    values = tl.load(
        ptrs, mask=(offs_w[:, None] < kernel_size) & (token[:, None] < in_end) & (offs_d[None, :] < d), other=0.0
    ).to(tl.float32)
    mean = tl.sum(values, axis=0) / kernel_size
    out_ptrs = out_ptr + (out_start + chunk) * hkv * d + head * d + offs_d
    tl.store(out_ptrs, mean, mask=offs_d < d)


@triton.autotune(
    configs=[
        triton.Config({"block_q": 1, "block_n": 128}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 4, "block_n": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 8, "block_n": 128}, num_warps=4, num_stages=2),
    ],
    key=["d", "group", "tune_key"],
    **autotune_cache_kwargs(),
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
):
    q_local_start = tl.program_id(0) * block_q
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
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
        q_ptrs, mask=q_mask[:, None, None] & (offs_g[None, :, None] < group) & (offs_d[None, None, :] < d), other=0.0
    )
    q = tl.reshape(q, [block_q * block_g, block_d])
    m = tl.full((block_q, block_g), float("-inf"), tl.float32)
    l = tl.zeros((block_q, block_g), tl.float32)
    coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
    coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
    coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
    offs_n = tl.arange(0, block_n)
    for start_n in tl.range(0, k2_len, block_n):
        pos = start_n + offs_n
        k_ptrs = k2_ptr + (k2_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k2_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid = q_mask[:, None] & (pos[None, :] < k2_len)
        if causal:
            valid = valid & (pos[None, :] < coarse_right[:, None])
        logits = tl.where(valid[:, None, :] & (offs_g[None, :, None] < group), logits, float("-inf"))
        tile_m = tl.max(logits, axis=2)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(valid[:, None, :] & (offs_g[None, :, None] < group), tl.exp(logits - new_m[:, :, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=2)
        m = new_m
    lse = m + tl.log(l)
    fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
    fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
    for start_n in tl.range(0, k1_len, block_n):
        pos = start_n + offs_n
        k_ptrs = k1_ptr + (k1_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k1_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid_n = q_mask[:, None] & (pos[None, :] < k1_len)
        if causal:
            valid_n = valid_n & (pos[None, :] < fine_right[:, None])
        valid_g = offs_g < group
        probs = tl.where(
            valid_g[None, :, None] & valid_n[:, None, :] & (l[:, :, None] > 0.0), tl.exp(logits - lse[:, :, None]), 0.0
        )
        score = tl.sum(probs, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
        out_ptrs = out_ptr + hk * total_q * max_k1 + q_index[:, None] * max_k1 + pos[None, :]
        tl.store(out_ptrs, score, mask=q_mask[:, None] & (pos[None, :] < k1_len))


@triton.jit
def _pool_scores_kernel(
    score_ptr,
    out_ptr,
    cu_q_ptr,
    cu_k1_ptr,
    cu_k_ptr,
    total_q,
    max_k1,
    max_blocks,
    hkv: tl.constexpr,
    block_size: tl.constexpr,
    block_stride: tl.constexpr,
    pool_window: tl.constexpr,
    pad: tl.constexpr,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    block_b: tl.constexpr,
    valid_only: tl.constexpr,
):
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k1_len = tl.load(cu_k1_ptr + batch + 1) - tl.load(cu_k1_ptr + batch)
    q_len = q_end - q_start
    k_len = tl.load(cu_k_ptr + batch + 1) - tl.load(cu_k_ptr + batch)
    q_position = k_len - q_len + q_local
    offs_b = tl.arange(0, block_b)
    active = offs_b < max_blocks
    if valid_only:
        valid_blocks = tl.minimum(
            max_blocks, tl.minimum((k_len + block_size - 1) // block_size, (q_position + block_size) // block_size)
        )
        active = active & (offs_b < valid_blocks)
    q_block = q_position // block_size
    forced = (offs_b < init_blocks) | (q_block >= offs_b) & (q_block <= offs_b + local_blocks)
    pooled = tl.full((block_b,), float("-inf"), tl.float32)
    for j in tl.static_range(0, pool_window):
        fine = offs_b * block_stride - pad + j
        ptrs = score_ptr + hk * total_q * max_k1 + (q_start + q_local) * max_k1 + fine
        value = tl.load(ptrs, mask=active & ~forced & (fine >= 0) & (fine < k1_len), other=float("-inf"))
        pooled = tl.maximum(pooled, value)
    pooled = tl.where(forced, float("inf"), pooled)
    out_ptrs = out_ptr + hk * total_q * max_blocks + (q_start + q_local) * max_blocks + offs_b
    tl.store(out_ptrs, pooled, mask=active)


@triton.jit
def _topk_float_to_ordered_key(x_bits):
    sign_bit = tl.full(x_bits.shape, 2147483648, dtype=tl.uint32)
    full_mask = tl.full(x_bits.shape, 4294967295, dtype=tl.uint32)
    return x_bits ^ tl.where(x_bits & sign_bit != 0, full_mask, sign_bit)


@triton.jit
def _topk_index_to_key(index):
    max_u16 = tl.full(index.shape, 65535, dtype=tl.uint32)
    return max_u16 - index.to(tl.uint32)


@triton.jit
def _topk_key_to_index(index_key):
    max_u16 = tl.full(index_key.shape, 65535, dtype=tl.uint32)
    return (max_u16 - index_key.to(tl.uint32)).to(tl.int32)


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
        max_blocks, tl.minimum((k_len + block_size - 1) // block_size, (q_position + block_size) // block_size)
    )
    q_index = q_start + q_local
    offs_t = tl.arange(0, topk)
    out_ptrs = out_ptr + hk * total_q * topk + q_index * topk + offs_t
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
    score = tl.where(score == score, score, float("-inf"))  # noqa: PLR0124 - preserve original NaN score handling
    score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
    index_key = _topk_index_to_key(block_idx)
    packed = score_key.to(tl.uint64) << 16 | index_key.to(tl.uint64)
    packed = tl.where(valid, packed, tl.zeros_like(packed))
    selected_keys = tl.topk(packed, topk)
    for _ in tl.range(0, num_tiles - 1):
        selected_keys = tl.bitonic_merge(selected_keys)
        tile_start -= block_b
        block_idx = tile_start + offs_b
        score = tl.load(score_row + block_idx).to(tl.float32)
        score = tl.where(score == score, score, float("-inf"))  # noqa: PLR0124 - preserve original NaN score handling
        score_key = _topk_float_to_ordered_key(score.to(tl.uint32, bitcast=True))
        index_key = _topk_index_to_key(block_idx)
        packed = score_key.to(tl.uint64) << 16 | index_key.to(tl.uint64)
        selected_keys = tl.maximum(selected_keys, tl.topk(packed, topk))
    selected_keys = selected_keys << 48 | selected_keys >> 16
    selected_keys = tl.sort(selected_keys, descending=True)
    selected = _topk_key_to_index((selected_keys >> 48).to(tl.uint32))
    selected = tl.where(selected < valid_blocks, selected, -1)
    tl.store(out_ptrs, selected)


@triton.jit
def _sparse_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_d: tl.constexpr,
):
    q_local = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    hk = head // group
    offs_d = tl.arange(0, block_d)
    offs_n = tl.arange(0, sparse_block)
    qv = tl.load(q_ptr + (q_start + q_local) * hq * d + head * d + offs_d, mask=offs_d < d, other=0.0)
    m = float("-inf")
    l = 0.0
    acc = tl.zeros((block_d,), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    for rank in tl.static_range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + (q_start + q_local) * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        valid = (block >= 0) & (pos < k_len)
        if causal:
            valid = valid & (pos < causal_limit)
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=valid[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        logits = tl.sum(kval * qv[None, :], axis=1) * scale
        logits = tl.where(valid, logits, float("-inf"))
        tile_m = tl.max(logits, axis=0)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        probs = tl.where(valid, tl.exp(logits - new_m), 0.0)
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        vval = tl.load(v_ptrs, mask=valid[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * vval, axis=0)
        l = l * alpha + tl.sum(probs, axis=0)
        m = new_m
    result = tl.where(l > 0.0, acc / l, 0.0)
    tl.store(out_ptr + (q_start + q_local) * hq * d + head * d + offs_d, result, mask=offs_d < d)
    tl.store(lse_ptr + (q_start + q_local) * hq + head, tl.where(l > 0.0, m + tl.log(l), float("-inf")))


@triton.jit
def _sparse_attention_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_d: tl.constexpr,
):
    q_local = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local
    hk = head // group
    offs_d = tl.arange(0, block_d)
    offs_n = tl.arange(0, sparse_block)
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + head * d + offs_d
    qv = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)
    outv = tl.load(out_ptr + q_index * hq * d + head * d + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dov = tl.load(do_ptr + q_index * hq * d + head * d + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + q_index * hq + head)
    delta = tl.sum(outv * dov, axis=0)
    causal_limit = k_len - q_len + q_local + 1
    dq = tl.zeros((block_d,), tl.float32)
    for rank in tl.static_range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        valid = (block >= 0) & (pos < k_len)
        if causal:
            valid = valid & (pos < causal_limit)
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        mask = valid[:, None] & d_mask[None, :]
        kval = tl.load(k_ptrs, mask=mask, other=0.0).to(tl.float32)
        vval = tl.load(v_ptrs, mask=mask, other=0.0).to(tl.float32)
        logits = tl.sum(kval * qv[None, :], axis=1) * scale
        probs = tl.where(valid & (lse != float("-inf")), tl.exp(logits - lse), 0.0)
        dp = tl.sum(vval * dov[None, :], axis=1)
        ds = probs * (dp - delta) * scale
        dq += tl.sum(ds[:, None] * kval, axis=0)
        tl.atomic_add(
            dk_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :], ds[:, None] * qv[None, :], mask=mask
        )
        tl.atomic_add(
            dv_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :],
            probs[:, None] * dov[None, :],
            mask=mask,
        )
    tl.store(dq_ptr + q_index * hq * d + head * d + offs_d, dq, mask=d_mask)


@triton.jit
def _fallback_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    store_lse: tl.constexpr,
):
    """Tensor-Core Stage-2: one query token and one complete GQA group."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local
    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    heads = hk * group + offs_m
    m_mask = offs_m < group
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
    qv = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    m_i = tl.full((block_m,), float("-inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, block_d), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    rank_count = topk
    if causal:
        rank_count = tl.minimum(topk, (causal_limit + sparse_block - 1) // sparse_block)
    for rank in tl.range(0, rank_count):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)
        kv_mask = n_mask[:, None] & d_mask[None, :]
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, float("-inf"))
        tile_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, tile_m)
        alpha = tl.where(new_m == float("-inf"), 0.0, tl.exp(m_i - new_m))
        probs = tl.where(m_mask[:, None] & n_mask[None, :], tl.exp(logits - new_m[:, None]), 0.0)
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        vval = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(vval.dtype), vval, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_m
    result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :],
        result,
        mask=m_mask[:, None] & d_mask[None, :],
    )
    if store_lse:
        tl.store(lse_ptr + q_index * hq + heads, tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf")), mask=m_mask)


@triton.jit
def _sparse_attention_gqa_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    do_ptr,
    lse_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Tensor-Core backward, reducing a full GQA group before dK/dV atomics."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local
    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    heads = hk * group + offs_m
    m_mask = offs_m < group
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
    md_mask = m_mask[:, None] & d_mask[None, :]
    qv = tl.load(q_ptrs, mask=md_mask, other=0.0)
    outv = tl.load(out_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :], mask=md_mask, other=0.0)
    dov = tl.load(do_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :], mask=md_mask, other=0.0)
    lse = tl.load(lse_ptr + q_index * hq + heads, mask=m_mask, other=float("-inf"))
    delta = tl.sum(outv.to(tl.float32) * dov.to(tl.float32), axis=1)
    dq = tl.zeros((block_m, block_d), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    for rank in tl.range(0, topk):
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)
        nd_mask = n_mask[:, None] & d_mask[None, :]
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=nd_mask, other=0.0)
        vval = tl.load(v_ptrs, mask=nd_mask, other=0.0)
        logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
        valid = m_mask[:, None] & n_mask[None, :] & (lse[:, None] != float("-inf"))
        probs = tl.where(valid, tl.exp(logits - lse[:, None]), 0.0)
        dp = tl.dot(dov, tl.trans(vval), out_dtype=tl.float32)
        ds = probs * (dp - delta[:, None]) * scale
        dq += tl.dot(ds.to(kval.dtype), kval, out_dtype=tl.float32)
        dk_tile = tl.dot(tl.trans(ds.to(qv.dtype)), qv, out_dtype=tl.float32)
        dv_tile = tl.dot(tl.trans(probs.to(dov.dtype)), dov, out_dtype=tl.float32)
        grad_ptrs = (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        tl.atomic_add(dk_ptr + grad_ptrs, dk_tile, mask=nd_mask)
        tl.atomic_add(dv_ptr + grad_ptrs, dv_tile, mask=nd_mask)
    tl.store(dq_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :], dq, mask=md_mask)


@triton.jit
def _cast_fp32_kernel(src_ptr, dst_ptr, n_elements, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(src_ptr + offsets, mask=offsets < n_elements, other=0.0)
    tl.store(dst_ptr + offsets, values, mask=offsets < n_elements)


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
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_cuda_contiguous(q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    if selected_blocks.ndim != 3 or selected_blocks.shape[:2] != (hkv, total_q):
        raise ValueError(
            "selected_blocks must have [Hkv, total_q_tokens, topk] layout; all query heads in one GQA group share the per-token Hkv selection"
        )
    topk = selected_blocks.shape[-1]
    out = torch.empty_like(q)
    lse = torch.empty((total_q, hq), dtype=torch.float32, device=q.device) if store_lse else out
    scale = softmax_scale if softmax_scale is not None else d ** (-0.5)
    block_m = max(16, triton.next_power_of_2(hq // hkv))
    stage2_num_warps = 2 if max_seqlen_q >= 1024 else 4 if max_seqlen_q >= 128 else 8
    if max_seqlen_q >= 1024:
        stage2_num_stages = 3 if d == 64 else 2
    else:
        stage2_num_stages = 2
    _fallback_gqa_kernel[max_seqlen_q, hkv, cu_seqlens_q.numel() - 1](
        q,
        k,
        v,
        selected_blocks,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        hq=hq,
        hkv=hkv,
        group=hq // hkv,
        d=d,
        total_q=total_q,
        scale=scale,
        topk=topk,
        sparse_block=block_size,
        causal=causal,
        block_m=block_m,
        block_n=block_size,
        block_d=triton.next_power_of_2(d),
        store_lse=store_lse,
        num_warps=stage2_num_warps,
        num_stages=stage2_num_stages,
    )
    return (out, lse)


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
    ) -> torch.Tensor:
        out, lse = _sparse_attention_forward(
            q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, block_size, softmax_scale, causal
        )
        ctx.save_for_backward(q, k, v, selected_blocks, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, selected, out, lse, cu_q, cu_k = ctx.saved_tensors
        dout = dout.contiguous()
        if ctx.max_seqlen_q >= 64:
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
            )
            return (dq, dk, dv, None, None, None, None, None, None, None)
        total_q, hq, d = q.shape
        hkv = k.shape[1]
        scale = ctx.softmax_scale if ctx.softmax_scale is not None else d ** (-0.5)
        dq = torch.empty_like(q)
        dk_acc = torch.zeros_like(k, dtype=torch.float32)
        dv_acc = torch.zeros_like(v, dtype=torch.float32)
        topk = selected.shape[-1]
        block_m = max(16, triton.next_power_of_2(hq // hkv))
        _sparse_attention_gqa_backward_kernel[ctx.max_seqlen_q, hkv, cu_q.numel() - 1](
            q,
            k,
            v,
            selected,
            out,
            dout,
            lse,
            dq,
            dk_acc,
            dv_acc,
            cu_q,
            cu_k,
            hq=hq,
            hkv=hkv,
            group=hq // hkv,
            d=d,
            total_q=total_q,
            scale=scale,
            topk=topk,
            sparse_block=ctx.block_size,
            causal=ctx.causal,
            block_m=block_m,
            block_n=ctx.block_size,
            block_d=triton.next_power_of_2(d),
            num_warps=8,
            num_stages=2,
        )
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        for source, target in ((dk_acc, dk), (dv_acc, dv)):
            n_elements = source.numel()
            _cast_fp32_kernel[triton.cdiv(n_elements, 256),](source, target, n_elements, block=256)
        return (dq, dk, dv, None, None, None, None, None, None, None)


def _fallback_sparse_attention(
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
        )[0]
    return _SparseAttentionFunction.apply(
        q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, block_size, softmax_scale, causal
    )


# Sparse GQA and split merge
@triton.jit
def _sparse_attention_gqa_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    selected_ptr,
    out_ptr,
    lse_ptr,
    cu_q_ptr,
    cu_k_ptr,
    hq: tl.constexpr,
    hkv: tl.constexpr,
    group: tl.constexpr,
    d: tl.constexpr,
    total_q,
    scale,
    topk: tl.constexpr,
    sparse_block: tl.constexpr,
    causal: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    store_lse: tl.constexpr,
    splits: tl.constexpr,
):
    """Tensor-Core Stage-2: one query token and one complete GQA group."""
    q_local = tl.program_id(0)
    hk = tl.program_id(1)
    batch = tl.program_id(2) // splits
    split = tl.program_id(2) % splits
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
    if q_start + q_local >= q_end:
        return
    k_start = tl.load(cu_k_ptr + batch)
    k_end = tl.load(cu_k_ptr + batch + 1)
    q_len = q_end - q_start
    k_len = k_end - k_start
    q_index = q_start + q_local
    offs_m = tl.arange(0, block_m)
    offs_n = tl.arange(0, block_n)
    offs_d = tl.arange(0, block_d)
    heads = hk * group + offs_m
    m_mask = offs_m < group
    d_mask = offs_d < d
    q_ptrs = q_ptr + q_index * hq * d + heads[:, None] * d + offs_d[None, :]
    qv = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)
    m_i = tl.full((block_m,), float("-inf"), tl.float32)
    l_i = tl.zeros((block_m,), tl.float32)
    acc = tl.zeros((block_m, block_d), tl.float32)
    causal_limit = k_len - q_len + q_local + 1
    rank_count = topk
    if causal:
        rank_count = tl.minimum(topk, (causal_limit + sparse_block - 1) // sparse_block)
    ranks_per_split = tl.cdiv(rank_count, splits)
    rank_begin = split * ranks_per_split
    rank_end = tl.minimum(rank_count, rank_begin + ranks_per_split)
    for tile in tl.range(rank_begin * (sparse_block // block_n), rank_end * (sparse_block // block_n)):
        rank = tile // (sparse_block // block_n)
        sub = tile % (sparse_block // block_n)
        block = tl.load(selected_ptr + hk * total_q * topk + q_index * topk + rank)
        safe_block = tl.maximum(block, 0)
        pos = safe_block * sparse_block + sub * block_n + offs_n
        n_mask = (block >= 0) & (pos < k_len)
        if causal:
            n_mask = n_mask & (pos < causal_limit)
        kv_mask = n_mask[:, None] & d_mask[None, :]
        k_ptrs = k_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        kval = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        logits = tl.dot(qv, tl.trans(kval), out_dtype=tl.float32) * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, float("-inf"))
        tile_m = tl.max(logits, axis=1)
        new_m = tl.maximum(m_i, tile_m)
        alpha = tl.where(new_m == float("-inf"), 0.0, tl.exp(m_i - new_m))
        probs = tl.where(m_mask[:, None] & n_mask[None, :], tl.exp(logits - new_m[:, None]), 0.0)
        v_ptrs = v_ptr + (k_start + pos[:, None]) * hkv * d + hk * d + offs_d[None, :]
        vval = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(probs.to(vval.dtype), vval, out_dtype=tl.float32)
        l_i = l_i * alpha + tl.sum(probs, axis=1)
        m_i = new_m
    result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr + split * total_q * hq * d + q_index * hq * d + heads[:, None] * d + offs_d[None, :],
        result,
        mask=m_mask[:, None] & d_mask[None, :],
    )
    if store_lse:
        tl.store(
            lse_ptr + split * total_q * hq + q_index * hq + heads,
            tl.where(l_i > 0.0, m_i + tl.log(l_i), float("-inf")),
            mask=m_mask,
        )


@triton.jit
def _merge_splits(
    partial,
    lse,
    out,
    TQ: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    S: tl.constexpr,
    BD: tl.constexpr,
    RS: tl.constexpr,
):
    row = tl.program_id(0)
    si = tl.arange(0, RS)
    dd = tl.arange(0, BD)
    ll = tl.load(lse + si * TQ * HQ + row, si < S, other=float("-inf"))
    mm = tl.max(ll, 0)
    weights = tl.where((si < S) & (ll != float("-inf")), tl.exp(ll - mm), 0.0)
    vv = tl.load(
        partial + si[:, None] * TQ * HQ * D + row * D + dd[None, :], (si[:, None] < S) & (dd[None, :] < D), other=0.0
    )
    den = tl.sum(weights, 0)
    oo = tl.sum(vv * weights[:, None], 0) / tl.where(den > 0, den, 1.0)
    tl.store(out + row * D + dd, oo, dd < D)


# Long prefill Stage1
@triton.autotune(
    configs=[
        triton.Config({"block_q": 1, "block_n": 128}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 4, "block_n": 64}, num_warps=4, num_stages=2),
        triton.Config({"block_q": 8, "block_n": 128}, num_warps=4, num_stages=2),
    ],
    key=["d", "group", "tune_key"],
    **autotune_cache_kwargs(),
)
@triton.jit
def _stage1_prefill_kernel(
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
):
    q_local_start = tl.program_id(0) * block_q
    hk = tl.program_id(1)
    batch = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + batch)
    q_end = tl.load(cu_q_ptr + batch + 1)
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
        q_ptrs, mask=q_mask[:, None, None] & (offs_g[None, :, None] < group) & (offs_d[None, None, :] < d), other=0.0
    )
    q = tl.reshape(q, [block_q * block_g, block_d])
    m = tl.full((block_q, block_g), float("-inf"), tl.float32)
    l = tl.zeros((block_q, block_g), tl.float32)
    coarse_stride = tl.where(k1_len == k2_len, k1_stride, k2_stride)
    coarse_q_len = tl.maximum(0, q_len - coarse_stride + 1) // coarse_stride
    coarse_right = tl.maximum(0, (q_local + 1) // coarse_stride - 1 + k2_len - coarse_q_len)
    offs_n = tl.arange(0, block_n)
    coarse_end = k2_len
    if causal:
        coarse_end = tl.minimum(k2_len, tl.max(tl.where(q_mask, coarse_right, 0), 0))
    for start_n in tl.range(0, coarse_end, block_n):
        pos = start_n + offs_n
        k_ptrs = k2_ptr + (k2_start + pos[None, :]) * hkv * d + hk * d + offs_d[:, None]
        kval = tl.load(k_ptrs, mask=(offs_d[:, None] < d) & (pos[None, :] < k2_len), other=0.0)
        logits = tl.reshape(tl.dot(q, kval) * scale, [block_q, block_g, block_n])
        valid = q_mask[:, None] & (pos[None, :] < k2_len)
        if causal:
            valid = valid & (pos[None, :] < coarse_right[:, None])
        logits = tl.where(valid[:, None, :] & (offs_g[None, :, None] < group), logits, float("-inf"))
        tile_m = tl.max(logits, axis=2)
        new_m = tl.maximum(m, tile_m)
        alpha = tl.where(m == float("-inf"), 0.0, tl.exp(m - new_m))
        p = tl.where(valid[:, None, :] & (offs_g[None, :, None] < group), tl.exp(logits - new_m[:, :, None]), 0.0)
        l = l * alpha + tl.sum(p, axis=2)
        m = new_m
    lse = m + tl.log(l)
    fine_q_len = tl.maximum(0, q_len - k1_stride + 1) // k1_stride
    fine_right = tl.maximum(0, (q_local + 1) // k1_stride - 1 + k1_len - fine_q_len)
    fine_end = k1_len
    if causal:
        fine_end = tl.minimum(k1_len, tl.max(tl.where(q_mask, fine_right, 0), 0))
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
            valid_g[None, :, None] & valid_n[:, None, :] & (l[:, :, None] > 0.0), tl.exp(logits - lse[:, :, None]), 0.0
        )
        score = tl.sum(probs, axis=1).to(q_ptr.dtype.element_ty).to(tl.float32)
        out_ptrs = out_ptr + hk * total_q * max_k1 + q_index[:, None] * max_k1 + pos[None, :]
        tl.store(out_ptrs, score, mask=q_mask[:, None] & (pos[None, :] < k1_len))


# Guarded runtime and length metadata
_tls = threading.local()
_kernels = []
_targets = {}
_missing = object()


def _has_hook(hook):
    if hook is None:
        return False
    if type(hook) is knobs.HookChain:
        return bool(hook.calls)
    return True


def _scope():
    return getattr(_tls, "current", None)


def stream_id(device):
    scope = _scope()
    if scope is not None and scope["enabled"] and (device.index == scope["device"]):
        return scope["stream"]
    return torch.cuda.current_stream(device).cuda_stream


def request_entry(cu):
    scope = _scope()
    if scope is None or not scope["metadata"]:
        return None
    entry = scope["lengths"].get((id(cu), stream_id(cu.device)))
    return entry[1] if entry is not None and entry[0]() is cu else None


def remember_request_entry(cu, result):
    scope = _scope()
    if scope is not None and scope["metadata"]:
        scope["lengths"][id(cu), stream_id(cu.device)] = (weakref.ref(cu), result)


@contextmanager
def decode_scope(q, k, v, maxq):
    previous = _scope()
    hooks = (
        _has_hook(knobs.runtime.launch_enter_hook)
        or _has_hook(knobs.runtime.launch_exit_hook)
        or any(x.fn.pre_run_hooks for x in _kernels)
    )
    enabled = (
        maxq == 1
        and triton.__version__ == "3.6.0"
        and q.is_cuda
        and (q.dtype == torch.bfloat16)
        and (q.shape[1] == 32)
        and (q.shape[-1] in (64, 128))
        and (k.shape[1] == 2)
        and (not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))))
        and (not hooks)
        and (os.environ.get("FLAG_ATTN_INFLLMV2_V3_RUNNER", "1") == "1")
    )
    device = q.device.index if q.is_cuda else None
    stream = None
    env = ()
    if enabled:
        if (
            torch.cuda.current_device() != device
            or torch.cuda.is_current_stream_capturing()
            or DistributedRtContext().is_lite_mode
        ):
            enabled = False
        else:
            target = _targets.get(device)
            if target is None:
                target = driver.active.get_current_target()
                _targets[device] = target
            enabled = target.backend == "hip" and target.arch == "gfx936" and (target.warp_size == 64)
            if enabled:
                stream = torch.cuda.current_stream(device).cuda_stream
                env = tuple(sorted(get_cache_invalidating_env_vars().items()))
    _tls.current = {
        "enabled": enabled,
        "metadata": not hooks,
        "device": device,
        "stream": stream,
        "env": env,
        "lengths": {},
        "traits": {},
    }
    try:
        yield
    finally:
        _tls.current = previous


def clear_launch_cache():
    for kernel in _kernels:
        kernel.cache.clear()
        kernel.hits = kernel.misses = kernel.fallbacks = 0


def launch_stats():
    return [
        {
            "kernel": k.fn.__name__,
            "hits": k.hits,
            "misses": k.misses,
            "fallbacks": k.fallbacks,
            "compiled_hashes": sorted({v.hash for v in k.cache.values()}),
        }
        for k in _kernels
    ]


class _GuardedKernel:
    def __init__(self, fn):
        self.fn = fn
        self.names = tuple(fn.signature.parameters)
        self.name_set = set(self.names)
        self.defaults = {
            n: p.default for n, p in fn.signature.parameters.items() if p.default is not inspect.Parameter.empty
        }
        self.cache = OrderedDict()
        self.hits = self.misses = self.fallbacks = 0
        _kernels.append(self)

    def __getitem__(self, grid):

        def call(*args, **kwargs):
            return self._launch(grid, args, kwargs)

        return call

    def _launch(self, grid, args, kwargs):
        scope = _scope()
        if scope is None or not scope["enabled"] or callable(grid) or self.fn.pre_run_hooks:
            self.fallbacks += 1
            return self.fn[grid](*args, **kwargs)
        raw = kwargs
        kw = dict(kwargs)
        kw["debug"] = kw.get("debug", self.fn.debug) or knobs.runtime.debug
        kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
        _flagprism.apply_compile_options(kw)
        try:
            if any(n in kw for n in self.names[: len(args)]):
                raise TypeError("duplicate argument")
            values = args + tuple(kw.pop(n) if n in kw else self.defaults[n] for n in self.names[len(args) :])
            traits = []
            for value in values:
                if type(value) is torch.Tensor:
                    cached = scope["traits"].get(id(value))
                    if cached is not None and cached[0] is value:
                        trait = cached[1]
                    else:
                        if value.get_device() != scope["device"] or not value.is_contiguous():
                            raise TypeError("unsupported tensor")
                        trait = (
                            "tensor",
                            value.dtype,
                            value.data_ptr() % 16 == 0,
                            value.untyped_storage().size() <= 2**31 - 1,
                        )
                        scope["traits"][id(value)] = (value, trait)
                    traits.append(trait)
                elif value is None or type(value) in (bool, int, float, str, bytes):
                    traits.append((type(value), value))
                else:
                    raise TypeError("unsupported argument")
            if any(v is not None and type(v) not in (bool, int, float, str, bytes) for v in kw.values()):
                raise TypeError("unsupported launch option")
            key = (
                scope["device"],
                scope["env"],
                tuple(traits),
                tuple(sorted(((k, type(v), v) for k, v in kw.items()))),
            )
        except (KeyError, TypeError, AttributeError):
            self.fallbacks += 1
            return self.fn[grid](*args, **raw)
        compiled = self.cache.get(key)
        if compiled is None:
            self.misses += 1
            compiled = self.fn[grid](*args, **raw)
            if compiled is not None:
                self.cache[key] = compiled
                while len(self.cache) > 128:
                    self.cache.popitem(last=False)
            return compiled
        for (name, _), (old, globals_dict) in self.fn.used_global_vals.items():
            if globals_dict.get(name, _missing) != old:
                raise RuntimeError(f"Global variable {name} changed after kernel compilation")
        self.hits += 1
        grid3 = (tuple(grid) + (1, 1, 1))[:3]
        compiled[grid3](*values, stream=scope["stream"])
        return compiled


@contextmanager
def chunk_scope(q, k, v, maxq):
    previous = _scope()
    hooks = (
        _has_hook(knobs.runtime.launch_enter_hook)
        or _has_hook(knobs.runtime.launch_exit_hook)
        or any(x.fn.pre_run_hooks for x in _kernels)
    )
    enabled = (
        maxq == 128
        and triton.__version__ == "3.6.0"
        and q.is_cuda
        and (q.dtype == torch.bfloat16)
        and (q.shape[1] == 32)
        and (q.shape[-1] in (64, 128))
        and (k.shape[1] == 2)
        and (not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v))))
        and (not hooks)
        and (os.environ.get("FLAG_ATTN_INFLLMV2_V5_RUNNER", "1") == "1")
    )
    device = q.device.index if q.is_cuda else None
    stream = None
    env = ()
    if enabled:
        if (
            torch.cuda.current_device() != device
            or torch.cuda.is_current_stream_capturing()
            or DistributedRtContext().is_lite_mode
        ):
            enabled = False
        else:
            target = _targets.get(device)
            if target is None:
                target = driver.active.get_current_target()
                _targets[device] = target
            enabled = target.backend == "hip" and target.arch == "gfx936" and (target.warp_size == 64)
            if enabled:
                stream = torch.cuda.current_stream(device).cuda_stream
                env = tuple(sorted(get_cache_invalidating_env_vars().items()))
    _tls.current = {
        "enabled": enabled,
        "metadata": not hooks,
        "device": device,
        "stream": stream,
        "env": env,
        "lengths": {},
        "traits": {},
    }
    try:
        yield
    finally:
        _tls.current = previous


_record_tls = threading.local()


@contextmanager
def record():
    previous = getattr(_record_tls, "records", None)
    records = []
    _record_tls.records = records
    try:
        yield records
    finally:
        _record_tls.records = previous


class CachedKernel(_GuardedKernel):
    def _launch(self, grid, args, kwargs):
        compiled = super()._launch(grid, args, kwargs)
        records = getattr(_record_tls, "records", None)
        if records is not None and compiled is not None:
            values = args + tuple(kwargs[n] if n in kwargs else self.defaults[n] for n in self.names[len(args) :])
            records.append((self.fn, compiled, (tuple(grid) + (1, 1, 1))[:3], values))
        return compiled


_entries = OrderedDict()
_decode = OrderedDict()
_content = OrderedDict()
_compress = CachedKernel(_compress_k_kernel)
_stage1_short = CachedKernel(_stage1_kernel.fn)
_select = CachedKernel(_select_blocks_streaming_kernel)
_pool = CachedKernel(_pool_scores_kernel)


def clear_metadata_cache():
    _entries.clear()
    _decode.clear()
    _content.clear()


def _entry(cu):
    try:
        version = cu._version
    except RuntimeError:
        version = None
    stream = stream_id(cu.device)
    key = (id(cu), version, stream)
    hit = _entries.get(key) if version is not None else None
    if hit is not None and hit["ref"]() is cu:
        _entries.move_to_end(key)
        return hit
    if version is None:
        hit = request_entry(cu)
        if hit is not None:
            return hit
    values = tuple(cu.detach().cpu().tolist())
    result = {"ref": weakref.ref(cu), "values": values, "derived": {}}
    if version is not None:
        _entries[key] = result
        while len(_entries) > 64:
            _entries.popitem(last=False)
    if version is None:
        remember_request_entry(cu, result)
    return result


def _remember(cu, values):
    try:
        version = cu._version
    except RuntimeError:
        return
    key = (id(cu), version, stream_id(cu.device))
    _entries[key] = {"ref": weakref.ref(cu), "values": tuple(values), "derived": {}}
    while len(_entries) > 64:
        _entries.popitem(last=False)


def max_length(cu):
    values = _entry(cu)["values"]
    return max((b - a for a, b in itertools.pairwise(values)), default=0)


def decode_cu(batch, device):
    key = (batch, str(device), stream_id(device))
    cached = _decode.get(key)
    if cached is not None and cached[0]._version == cached[1]:
        return cached[0]
    with torch.inference_mode(False):
        out = torch.arange(batch + 1, device=device, dtype=torch.int32)
    _remember(out, range(batch + 1))
    _decode[key] = (out, out._version)
    while len(_decode) > 64:
        _decode.popitem(last=False)
    return out


def compressed_lengths(cu, kernel_size, stride):
    e = _entry(cu)
    key = (kernel_size, stride)
    content_key = (cu.device.index, stream_id(cu.device), e["values"], kernel_size, stride)
    cached = e["derived"].get(key) or _content.get(content_key)
    if cached is not None:
        out, version = cached
        try:
            if out._version == version:
                return out
        except RuntimeError:
            pass
    values = e["values"]
    acc = [0]
    for a, b in itertools.pairwise(values):
        acc.append(acc[-1] + max(0, (b - a - kernel_size) // stride + 1))
    with torch.inference_mode(False):
        out = torch.tensor(acc, device=cu.device, dtype=torch.int32)
    _remember(out, acc)
    try:
        version = out._version
    except RuntimeError:
        version = None
    e["derived"][key] = (out, version)
    _content[content_key] = (out, version)
    while len(_content) > 64:
        _content.popitem(last=False)
    return out


def compress_k(k, cu_seqlens_k, kernel_size, stride):
    _check_cuda_contiguous(k, cu_seqlens_k)
    if cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_k must be int32")
    cu_out = compressed_lengths(cu_seqlens_k, kernel_size, stride)
    values = _entry(cu_out)["values"]
    total = values[-1]
    out = torch.empty((total, k.shape[1], k.shape[2]), device=k.device, dtype=k.dtype)
    if total:
        _compress[(max_length(cu_out), k.shape[1], cu_out.numel() - 1)](
            k,
            out,
            cu_seqlens_k,
            cu_out,
            hkv=k.shape[1],
            d=k.shape[2],
            kernel_size=kernel_size,
            stride=stride,
            block_window=triton.next_power_of_2(kernel_size),
            block_d=triton.next_power_of_2(k.shape[2]),
            num_warps=4,
        )
    return out, cu_out


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
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(q, k1, k2, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k2, cu_seqlens_k)
    total_q, hq, d = q.shape
    hkv = k1.shape[1]
    if hq % hkv:
        raise ValueError("number of query heads must be divisible by KV heads")
    group = hq // hkv
    max_k1 = max_length(cu_seqlens_k1)
    # Stage1 probabilities are already rounded to the Q dtype in the kernel.
    # Storing them as FP32 only doubles traffic and capacity without retaining
    # additional information.
    out = torch.zeros((hkv, total_q, max_k1), dtype=q.dtype, device=q.device)
    if max_k1 == 0:
        return out
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    # Short chunks need the untiled Q=1 grid to expose enough independent CTAs;
    # direct A/B shows Q=4 is about 20% slower at Q=128 even when the kernel
    # autotuner picks it.  Autotune only the long-prefill regime where K reuse
    # amortizes the smaller grid.
    kernel = _stage1_kernel
    if max_seqlen_q >= 1024 and os.environ.get("FLAG_ATTN_INFLLMV2_V2_CAUSAL_TRIM", "1") == "1":
        kernel = _stage1_prefill_kernel
    tune_key = triton.next_power_of_2(max_seqlen_q)
    if max_seqlen_q < 1024:
        block_q = 1
        _stage1_short[(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)](
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
            block_n=128,
            num_warps=4,
            num_stages=2,
        )
        return out

    def grid(meta):
        return (
            triton.cdiv(max_seqlen_q, meta["block_q"]),
            hkv,
            cu_seqlens_q.numel() - 1,
        )

    kernel[grid](
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
        block_g=max(16, triton.next_power_of_2(group)),
        block_d=triton.next_power_of_2(d),
    )
    return out


def _full_prefill(total_q, hkv, cuq, cuk, maxq):
    """Recognize measured full-prefill shapes from actual packed lengths."""
    if total_q != maxq or maxq not in (8192, 32768, 65536) or hkv != 2:
        return False
    if cuq.numel() != 2 or cuk.numel() != 2:
        return False
    if triton.__version__ != "3.6.0":
        return False
    device = cuq.device.index
    target = _targets.get(device)
    if target is None:
        target = driver.active.get_current_target()
        _targets[device] = target
    if target.backend != "hip" or target.arch != "gfx936" or target.warp_size != 64:
        return False
    lengths = (0, maxq)
    return tuple(_entry(cuq)["values"]) == lengths and tuple(_entry(cuk)["values"]) == lengths


def select_blocks(
    block_score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    block_size: int = 64,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    _allow_tle: bool = True,
) -> torch.Tensor:
    """Exact TopK with a shape-selected TLE radix or standard streaming kernel."""
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    _check_cuda_contiguous(block_score, cu_seqlens_q, cu_seqlens_k)
    hkv, total_q, max_blocks = block_score.shape
    if max_blocks > 4096:
        raise ValueError("TopK currently supports at most 4096 KV blocks")
    if topk > 2048 or (topk & (topk - 1)) != 0:
        raise ValueError("TopK must be a power of two no greater than 2048")
    out = torch.full((hkv, total_q, topk), -1, dtype=torch.int32, device=block_score.device)
    if max_blocks == 0:
        return out
    max_q = max_length(cu_seqlens_q)
    grid = (max_q, hkv, cu_seqlens_q.numel() - 1)
    block_b = max(64, topk, triton.next_power_of_2(min(max_blocks, 1024)))
    num_warps = 2 if block_b <= 64 else (4 if block_b <= 128 else 8)
    num_stages = 2
    if (
        block_score.dtype == torch.bfloat16
        and topk == 64
        and block_size == 64
        and _full_prefill(total_q, hkv, cu_seqlens_q, cu_seqlens_k, max_q)
    ):
        # A wave-local 64-key merge avoids the multi-wave sorting network.
        # The original packed score/index key and tie-breaking remain intact.
        block_b, num_warps, num_stages = 64, 1, 1
    _select[grid](
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
        num_stages=num_stages,
    )
    return out


def pool_scores(
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k1: torch.Tensor,
    max_seqlen_q: int,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    init_blocks: int = 1,
    local_blocks: int = 32,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    *,
    _valid_only: bool = False,
) -> torch.Tensor:
    cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
    max_seqlen_k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    _check_cuda_contiguous(score, cu_seqlens_q, cu_seqlens_k1, cu_seqlens_k)
    hkv, total_q, max_k1 = score.shape
    max_blocks = math.ceil(max_seqlen_k / block_size)
    out = torch.empty((hkv, total_q, max_blocks), dtype=score.dtype, device=score.device)
    if max_blocks == 0:
        return out
    pool_window = kernel_size // kernel_stride + block_size // kernel_stride - 1
    num_warps = 8
    if (
        score.dtype == torch.bfloat16
        and block_size == 64
        and max_seqlen_q == 8192
        and _full_prefill(total_q, hkv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)
    ):
        num_warps = 2
    _pool[(max_seqlen_q, hkv, cu_seqlens_q.numel() - 1)](
        score,
        out,
        cu_seqlens_q,
        cu_seqlens_k1,
        cu_seqlens_k,
        total_q,
        max_k1,
        max_blocks,
        hkv=hkv,
        block_size=block_size,
        block_stride=block_size // kernel_stride,
        pool_window=pool_window,
        pad=kernel_size // kernel_stride - 1,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        block_b=triton.next_power_of_2(max_blocks),
        valid_only=_valid_only,
        num_warps=num_warps,
    )
    return out


# Sparse launch and autograd
_attention_launcher = CachedKernel(_sparse_attention_gqa_kernel)
_merge_launcher = CachedKernel(_merge_splits)


def _forward(
    q,
    k,
    v,
    selected,
    cuq,
    cuk,
    maxq,
    bs=64,
    scale=None,
    causal=True,
    store_lse=False,
    *,
    tile=None,
    warps=None,
    stages=None,
    splits=1,
):
    _check_cuda_contiguous(q, k, v, selected, cuq, cuk)
    tq, hq, d = q.shape
    hkv = k.shape[1]
    g = hq // hkv
    if hq % hkv or selected.ndim != 3 or selected.shape[:2] != (hkv, tq):
        raise ValueError("selected must have [Hkv,total_q,topk] layout and Hq must be a multiple of Hkv")
    prefill_tile = 32
    if (
        tile is None
        and q.dtype == torch.bfloat16
        and d == 128
        and hq == 32
        and bs == 64
        and selected.shape[-1] == 64
        and causal
        and scale is None
        and splits == 1
        and not store_lse
        and _full_prefill(tq, hkv, cuq, cuk, maxq)
    ):
        prefill_tile = 16
    n = int(os.environ.get("FLAG_ATTN_INFLLMV2_V2_TILE", str(prefill_tile))) if tile is None else tile
    w = int(os.environ.get("FLAG_ATTN_INFLLMV2_V2_WARPS", "1")) if warps is None else warps
    st = int(os.environ.get("FLAG_ATTN_INFLLMV2_V2_STAGES", "1")) if stages is None else stages
    if n not in (16, 32, 64) or bs % n or w not in (1, 2, 4, 8) or (st not in (1, 2, 3)):
        raise ValueError("invalid V2 tile/waves/stages or incompatible sparse block")
    out = torch.empty_like(q)
    target = torch.empty((splits, tq, hq, d), device=q.device, dtype=torch.float32) if splits > 1 else out
    lse = (
        torch.empty((splits, tq, hq), device=q.device, dtype=torch.float32)
        if splits > 1
        else torch.empty((tq, hq), device=q.device, dtype=torch.float32)
        if store_lse
        else out
    )
    compiled = _attention_launcher[maxq, hkv, (cuq.numel() - 1) * splits](
        q,
        k,
        v,
        selected,
        target,
        lse,
        cuq,
        cuk,
        hq,
        hkv,
        g,
        d,
        tq,
        d ** (-0.5) if scale is None else scale,
        selected.shape[-1],
        bs,
        causal,
        max(16, triton.next_power_of_2(g)),
        n,
        triton.next_power_of_2(d),
        store_lse or splits > 1,
        splits,
        num_warps=w,
        num_stages=st,
    )
    if splits > 1:
        _merge_launcher[tq * hq,](
            target,
            lse,
            out,
            tq,
            hq,
            d,
            splits,
            triton.next_power_of_2(d),
            triton.next_power_of_2(splits),
            num_warps=1,
            num_stages=1,
        )
    return (out, lse, compiled)


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sel, cuq, cuk, maxq, bs, scale, causal):
        out, lse, _ = _forward(q, k, v, sel, cuq, cuk, maxq, bs, scale, causal, True)
        ctx.save_for_backward(q, k, v, sel, out, lse, cuq, cuk)
        ctx.max_seqlen_q = maxq
        ctx.block_size = bs
        ctx.softmax_scale = scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout):
        if ctx.max_seqlen_q >= 64:
            from .backward import sparse_backward

            q, k, v, sel, out, lse, cuq, cuk = ctx.saved_tensors
            grads = sparse_backward(
                q,
                k,
                v,
                out,
                dout.contiguous(),
                lse,
                sel,
                cuq,
                cuk,
                ctx.max_seqlen_q,
                ctx.block_size,
                ctx.softmax_scale,
                ctx.causal,
            )
            return (*grads, None, None, None, None, None, None, None)
        return _SparseAttentionFunction.backward(ctx, dout)


def sparse_attention(
    q,
    k,
    v,
    selected_blocks,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    block_size=64,
    softmax_scale=None,
    causal=True,
    *,
    _allow_tle=True,
):
    args = (q, k, v, selected_blocks, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, block_size, softmax_scale, causal)
    if (
        block_size not in (32, 64, 128)
        or q.dtype not in (torch.bfloat16, torch.float16)
        or q.shape[-1] not in (32, 64, 128)
    ):
        return _fallback_sparse_attention(*args, _allow_tle=False)
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _Attention.apply(*args)
    splits = (
        int(os.environ.get("FLAG_ATTN_INFLLMV2_V2_SPLITS", str(32 if cu_seqlens_q.numel() == 2 else 8)))
        if max_seqlen_q == 1
        else 1
    )
    if splits not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError("V2 splits must be a power of two <=64")
    return _forward(*args, splits=splits)[0]
