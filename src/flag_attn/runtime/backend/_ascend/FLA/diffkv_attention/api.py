# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for paged DiffKV decode attention.

Two decode paths, both split-KV and both finished by the same LSE reduce:

* ``paged`` — walks the KV range one physical page at a time straight out of
  the paged cache.  Every load stays affine and the page-table lookup is
  scalar, but an iteration only covers ``block_size`` tokens, and on this
  backend the per-iteration cost dominates.
* ``dense`` — de-pages the KV range into a contiguous buffer first (one cheap
  copy kernel), then decodes with ``BLOCK_N``-token tiles: fewer, wider
  iterations, at the price of one extra pass over the KV bytes.

No device-to-host synchronisation: the split count is derived from the
``block_tables`` bound, and splits that land past a sequence's length produce
an empty partial that the reduce drops.
"""

import torch
import triton

from .kernels import (
    _diffkv_decode_dense_kernel,
    _diffkv_decode_kernel,
    _diffkv_depage_kernel,
    _diffkv_reduce_kernel,
)

#: Wanted number of programs in flight; the decode kernel stops scaling well
#: beyond this, and every extra split also duplicates the program prologue.
_TARGET_PROGRAMS = 16
_MAX_SPLITS = 32
#: Pages de-paged per program in the de-page pass.
_PPG = 8
#: Tokens per loop iteration in the dense path.
_BLOCK_N = 256
#: Default decode path: "auto", "paged" or "dense".
_MODE = "auto"
#: With at most this many KV tokens to read, reading the paged cache directly
#: (one launch) beats paying for the de-page pass.  Measured crossover:
#: b1/l2048 (8k tokens) and b8/l512 (16k) favour paged, b8/l2048 (65k) dense.
_DENSE_MIN_TOKENS = 32768


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
    block_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
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
    """Split count.

    Measured on this backend: one split per (sequence, KV head) wins even at
    batch=1, because every extra split duplicates the program prologue and adds
    a reduce launch, while the decode kernel gets no extra parallelism back
    (full b1/l8192: 557 us at one split vs 1096 us at eight).  So the dense path
    always uses a single split and finalises in the decode kernel itself.
    """
    return 1


def _workspace(query, batch, num_q_heads, head_v, splits):
    o_part = torch.empty((batch, num_q_heads, splits, head_v),
                         device=query.device, dtype=torch.float32)
    ml_part = torch.empty((2, batch, num_q_heads, splits),
                          device=query.device, dtype=torch.float32)
    return o_part, ml_part[0], ml_part[1]


def _paged(query, key_cache, value_cache, lengths, tables, attn_scale,
           window_size, splits, out):
    batch, num_q_heads, head_qk = query.shape
    num_kv_heads = key_cache.shape[2]
    head_v = value_cache.shape[3]
    if splits == 1:
        o_part = out
        m_part = l_part = out
    else:
        o_part, m_part, l_part = _workspace(query, batch, num_q_heads, head_v, splits)
    _diffkv_decode_kernel[(splits, num_kv_heads, batch)](
        query, key_cache, value_cache, tables, lengths,
        o_part, m_part, l_part, attn_scale,
        HQ=num_q_heads, HKV=num_kv_heads, GROUP=num_q_heads // num_kv_heads,
        DQK=head_qk, DPAD=triton.next_power_of_2(head_qk), DV=head_v,
        BS=key_cache.shape[1], SPLITS=splits, BT_STRIDE=tables.stride(0),
        WINDOW=window_size, FINAL=splits == 1,
    )
    return o_part, m_part, l_part


def _dense(query, key_cache, value_cache, lengths, tables, attn_scale,
           window_size, splits, out):
    batch, num_q_heads, head_qk = query.shape
    num_kv_heads = key_cache.shape[2]
    head_v = value_cache.shape[3]
    block_size = key_cache.shape[1]
    # one BLOCK_N of slack: the decode loop reads whole blocks, and with a
    # short window the last block can run past the sequence's own pages
    len_alloc = tables.shape[1] * block_size + _BLOCK_N
    k_dense = torch.empty((batch, len_alloc, num_kv_heads, head_qk),
                          device=query.device, dtype=query.dtype)
    v_dense = torch.empty((batch, len_alloc, num_kv_heads, head_v),
                          device=query.device, dtype=query.dtype)
    max_pages = triton.cdiv(len_alloc, block_size)
    _diffkv_depage_kernel[(triton.cdiv(max_pages, _PPG), batch)](
        key_cache, value_cache, tables, lengths, k_dense, v_dense,
        HKV=num_kv_heads, DQK=head_qk,
        KFLAT=triton.next_power_of_2(num_kv_heads * head_qk),
        DV=head_v, VFLAT=triton.next_power_of_2(num_kv_heads * head_v),
        BS=block_size, BT_STRIDE=tables.stride(0), LSTRIDE=len_alloc, PPG=_PPG,
        WINDOW=window_size, BN=_BLOCK_N,
        NPAGES_MAX=max_pages,
    )
    final = splits == 1
    if final:
        o_part = out
        m_part = l_part = out
    else:
        o_part, m_part, l_part = _workspace(query, batch, num_q_heads, head_v, splits)
    _diffkv_decode_dense_kernel[(splits, num_kv_heads, batch)](
        query, k_dense, v_dense, lengths, o_part, m_part, l_part, attn_scale,
        HQ=num_q_heads, HKV=num_kv_heads, GROUP=num_q_heads // num_kv_heads,
        DQK=head_qk, DPAD=triton.next_power_of_2(head_qk), DV=head_v,
        BN=_BLOCK_N, BS_PAGE=block_size, SPLITS=splits, LSTRIDE=len_alloc,
        WINDOW=window_size, FINAL=final,
    )
    return o_part, m_part, l_part


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
    if attn_scale is None:
        attn_scale = head_qk**-0.5

    lengths = context_lens.to(device=query.device, dtype=torch.int32)
    tables = block_tables.to(device=query.device, dtype=torch.int32)
    out = torch.empty((batch, num_q_heads, head_v),
                      device=query.device, dtype=query.dtype)

    if num_segments is not None:
        splits = int(max(1, min(int(num_segments), _MAX_SPLITS)))
    elif _MODE == "dense":
        splits = 1
    else:
        splits = _choose_splits(batch, num_kv_heads, tables.shape[1])

    if _MODE == "dense" or (
        _MODE == "auto"
        and batch * num_kv_heads * tables.shape[1] * key_cache.shape[1] > _DENSE_MIN_TOKENS
    ):
        o_part, m_part, l_part = _dense(query, key_cache, value_cache, lengths,
                                        tables, attn_scale, window_size,
                                        splits, out)
        if splits == 1:
            return out
    else:
        o_part, m_part, l_part = _paged(query, key_cache, value_cache, lengths,
                                        tables, attn_scale, window_size, splits, out)
        if splits == 1:
            return out
    _diffkv_reduce_kernel[(batch * num_q_heads,)](
        o_part, m_part, l_part, out,
        DV=head_v, SPLITS=splits, SPAD=triton.next_power_of_2(splits),
    )
    return out
