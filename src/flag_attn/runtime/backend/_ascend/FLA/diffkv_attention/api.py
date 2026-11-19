# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for paged DiffKV decode attention.

The Ascend implementation is a two-kernel split-KV decode:

1. ``_diffkv_decode_kernel`` — one program per (KV split, KV head, sequence).
   It owns all ``HQ // HKV`` query heads of that KV head, walks its token range
   one physical page at a time and keeps a running (max, sum, output) triple in
   fp32.
2. ``_diffkv_reduce_kernel`` — combines the per-split triples with the usual
   log-sum-exp weighting.

No device-to-host synchronisation: the split count is derived from the
``block_tables`` upper bound, and splits that fall past a sequence's length
simply produce an empty partial that the reduce drops.
"""

import torch
import triton

from .kernels import _diffkv_decode_kernel, _diffkv_reduce_kernel

#: Wanted number of programs in flight, roughly three waves of the AI cores.
_TARGET_PROGRAMS = 16
_MAX_SPLITS = 32


def _supported(query, key_cache, value_cache, context_lens, block_tables, window_size):
    if query.dim() != 3 or key_cache.dim() != 4 or value_cache.dim() != 4:
        return False
    if query.dtype not in (torch.bfloat16, torch.float16):
        return False
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        return False
    if not (query.is_contiguous() and key_cache.is_contiguous()
            and value_cache.is_contiguous() and block_tables.is_contiguous()
            and context_lens.is_contiguous()):
        return False
    batch, num_q_heads, head_qk = query.shape
    num_blocks, block_size, num_kv_heads, _ = key_cache.shape
    head_v = value_cache.shape[3]
    if key_cache.shape[:3] != value_cache.shape[:3]:
        return False
    if num_q_heads % num_kv_heads:
        return False
    group = num_q_heads // num_kv_heads
    if group & (group - 1):
        return False
    if head_qk > 256 or head_v not in (16, 32, 64, 128, 256):
        return False
    if block_size & (block_size - 1):
        return False
    if context_lens.numel() != batch or block_tables.shape[0] != batch:
        return False
    if window_size == 0 or window_size < -1:
        return False
    if block_tables.dtype not in (torch.int32, torch.int64):
        return False
    return True


def _choose_splits(batch, num_kv_heads, max_pages):
    """Split count from the block-table bound only (no device sync)."""
    base = max(batch * num_kv_heads, 1)
    want = (_TARGET_PROGRAMS + base - 1) // base
    return int(max(1, min(want, _MAX_SPLITS, max(max_pages, 1))))


def diffkv_attention(
    query,
    key_cache,
    value_cache,
    context_lens,
    block_tables,
    attn_scale=None,
    window_size=-1,
    path="2d",
    num_segments=None,
    backend=None,
):
    """Paged DiffKV decode attention on Ascend; ``None`` if unsupported."""
    if not _supported(query, key_cache, value_cache, context_lens,
                      block_tables, window_size):
        return None

    batch, num_q_heads, head_qk = query.shape
    num_kv_heads = key_cache.shape[2]
    head_v = value_cache.shape[3]
    block_size = key_cache.shape[1]
    group = num_q_heads // num_kv_heads
    max_blocks = block_tables.shape[1]

    if attn_scale is None:
        attn_scale = head_qk**-0.5

    lengths = context_lens.to(device=query.device, dtype=torch.int32)
    tables = block_tables.to(device=query.device, dtype=torch.int32)
    out = torch.empty((batch, num_q_heads, head_v),
                      device=query.device, dtype=query.dtype)

    if num_segments is not None:
        splits = int(max(1, min(int(num_segments), _MAX_SPLITS)))
    else:
        splits = _choose_splits(batch, num_kv_heads, max_blocks)
    d_pad = triton.next_power_of_2(head_qk)
    s_pad = triton.next_power_of_2(splits)

    o_part = torch.empty((batch, num_q_heads, splits, head_v),
                         device=query.device, dtype=torch.float32)
    ml_part = torch.empty((2, batch, num_q_heads, splits),
                          device=query.device, dtype=torch.float32)
    m_part, l_part = ml_part[0], ml_part[1]

    _diffkv_decode_kernel[(splits, num_kv_heads, batch)](
        query,
        key_cache,
        value_cache,
        tables,
        lengths,
        o_part,
        m_part,
        l_part,
        attn_scale,
        HQ=num_q_heads,
        HKV=num_kv_heads,
        GROUP=group,
        DQK=head_qk,
        DPAD=d_pad,
        DV=head_v,
        BS=block_size,
        SPLITS=splits,
        BT_STRIDE=tables.stride(0),
        WINDOW=window_size,
    )
    _diffkv_reduce_kernel[(batch * num_q_heads,)](
        o_part,
        m_part,
        l_part,
        out,
        DV=head_v,
        SPLITS=splits,
        SPAD=s_pad,
    )
    return out
