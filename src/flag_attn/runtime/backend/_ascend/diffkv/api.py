# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for paged DiffKV decode attention."""

import torch
import triton

from .decode import (
    _diffkv_decode_dense_kernel,
    _diffkv_decode_kernel,
    _diffkv_decode_window_kernel,
    _diffkv_depage_kernel,
    _diffkv_reduce_kernel,
)

_BLOCK_N = 256
_PAGES_PER_PROGRAM = 8
_PAGED_MAX_WORK = 1024
_MAX_SPLITS = 32


def _validate_inputs(query, key_cache, value_cache, context_lens, block_tables, window_size):
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query [B,Hq,Dqk] and caches [pages,block_size,Hkv,D]")
    if context_lens.ndim != 1 or block_tables.ndim != 2:
        raise ValueError("expected context_lens [B] and block_tables [B,max_blocks]")
    if query.device.type != "npu" or any(
        cache.device != query.device for cache in (key_cache, value_cache)
    ):
        raise ValueError("query and KV caches must be on the same NPU device")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("DiffKV supports float16 and bfloat16 query tensors")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise TypeError("query, key_cache, and value_cache must have the same dtype")
    if context_lens.dtype not in (torch.int32, torch.int64) or block_tables.dtype not in (
        torch.int32, torch.int64
    ):
        raise TypeError("context_lens and block_tables must be integer tensors")

    batch, num_q_heads, head_qk = query.shape
    _, page_size, num_kv_heads, key_dim = key_cache.shape
    head_v = value_cache.shape[-1]
    if (
        not 0 < head_qk <= 256
        or key_dim != head_qk
        or head_v not in (16, 32, 64, 128, 256)
        or page_size <= 0
        or page_size & (page_size - 1)
        or num_kv_heads <= 0
        or num_q_heads <= 0
        or num_q_heads % num_kv_heads
        or value_cache.shape[:3] != key_cache.shape[:3]
        or context_lens.shape[0] != batch
        or block_tables.shape[0] != batch
    ):
        raise ValueError("invalid or unsupported DiffKV cache or batch layout")
    group = num_q_heads // num_kv_heads
    if group & (group - 1):
        raise ValueError("Ascend DiffKV requires a power-of-two query group size")
    if window_size == 0 or window_size < -1:
        raise ValueError("window_size must be -1 or a positive token count")


def _depage(query, key_cache, value_cache, lengths, tables, window_size, head_major, max_pages):
    batch = query.shape[0]
    _, page_size, num_kv_heads, head_qk = key_cache.shape
    head_v = value_cache.shape[-1]
    max_pages += triton.cdiv(_BLOCK_N, page_size)
    len_alloc = max_pages * page_size
    layout = (batch, num_kv_heads, len_alloc) if head_major else (batch, len_alloc, num_kv_heads)
    k_dense = torch.empty((*layout, head_qk), device=query.device, dtype=query.dtype)
    v_dense = torch.empty((*layout, head_v), device=query.device, dtype=query.dtype)

    grid = (triton.cdiv(max_pages, _PAGES_PER_PROGRAM), batch, num_kv_heads if head_major else 1)
    _diffkv_depage_kernel[grid](
        key_cache, value_cache, tables, lengths, k_dense, v_dense,
        HKV=num_kv_heads, DQK=head_qk, DV=head_v,
        K_STRIDES=key_cache.stride(), V_STRIDES=value_cache.stride(), HEAD_MAJOR=head_major,
        BS=page_size, BT_STRIDE=tables.stride(0), LSTRIDE=len_alloc, PPG=_PAGES_PER_PROGRAM,
        WINDOW=window_size, BN=_BLOCK_N, NPAGES_MAX=max_pages,
    )
    return k_dense, v_dense, len_alloc


def _prefer_raw_decode(batch, max_tokens, window_size):
    if window_size > 0:
        return min(max_tokens, window_size) >= 512
    if batch == 1:
        return max_tokens >= 4096
    return max_tokens >= 2048 or (batch >= 16 and max_tokens >= 512)


@torch.no_grad()
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
    """Run paged DiffKV decode attention on Ascend.

    The cache manager supplies valid sequence lengths and physical page IDs.
    Metadata may be on the host or device.
    The automatic backend selects TLE Raw for eligible long contexts and
    larger batches. Use ``backend="triton"`` to select the standard path.
    Select ``backend="tle"`` to gather paged BF16 K/V directly into L1
    on supported layouts with a single KV split.
    """
    _validate_inputs(query, key_cache, value_cache, context_lens, block_tables, window_size)
    path = path.strip().lower()
    if path not in ("2d", "3d"):
        raise ValueError("path must be 2d or 3d")
    backend = "auto" if backend is None else backend.strip().lower()
    if backend not in ("auto", "triton", "tle"):
        raise ValueError("backend must be auto, triton or tle")
    batch, num_q_heads, head_qk = query.shape
    _, page_size, num_kv_heads, _ = key_cache.shape
    head_v = value_cache.shape[-1]
    group = num_q_heads // num_kv_heads
    out = torch.empty((batch, num_q_heads, head_v), device=query.device, dtype=query.dtype)
    if batch == 0:
        return out
    if block_tables.shape[1] == 0:
        return out.zero_()

    query = query.contiguous()
    if key_cache.stride(-1) != 1:
        key_cache = key_cache.contiguous()
    if value_cache.stride(-1) != 1:
        value_cache = value_cache.contiguous()
    lengths = context_lens.to(device=query.device, dtype=torch.int32).contiguous()
    tables = block_tables.to(device=query.device, dtype=torch.int32).contiguous()
    scale = head_qk**-0.5 if attn_scale is None else float(attn_scale)
    splits = 1
    if path == "3d" and num_segments is not None:
        splits = max(1, min(int(num_segments), _MAX_SPLITS))

    if backend == "tle":
        from .raw_decode import launch_raw_decode, supports_raw_decode

        if splits != 1 or not supports_raw_decode(query, key_cache, value_cache):
            raise ValueError("the TLE Raw backend does not support this DiffKV configuration")
        launch_raw_decode(query, key_cache, value_cache, lengths, tables, out, scale, window_size)
        return out

    if (
        splits == 1
        and 0 < window_size <= 128
        and page_size == 16
        and head_qk == 192
        and head_v == 128
        and group in (8, 16)
    ):
        _diffkv_decode_window_kernel[(batch, num_kv_heads)](
            query, key_cache, value_cache, lengths, tables, out, scale,
            HQ=num_q_heads, GROUP=group, BS=page_size, BT_STRIDE=tables.stride(0),
            K_STRIDES=key_cache.stride(), V_STRIDES=value_cache.stride(),
            WINDOW=window_size, DQK=head_qk, DPAD=256, DV=head_v,
            BN=max(16, triton.next_power_of_2(window_size)),
        )
        return out

    if (
        backend == "auto"
        and splits == 1
        and query.dtype == torch.bfloat16
        and page_size == 16
        and head_qk == 192
        and head_v == 128
        and group in (8, 16)
        and _prefer_raw_decode(batch, tables.shape[1] * page_size, window_size)
    ):
        try:
            from .raw_decode import try_raw_decode
        except ImportError:
            pass
        else:
            if try_raw_decode(query, key_cache, value_cache, lengths, tables, out, scale, window_size):
                return out

    if splits == 1:
        o_part = m_part = l_part = out
    else:
        o_part = torch.empty(
            (batch, num_q_heads, splits, head_v), device=query.device, dtype=torch.float32
        )
        m_part = torch.empty(o_part.shape[:-1], device=query.device, dtype=torch.float32)
        l_part = torch.empty_like(m_part)

    max_pages = tables.shape[1]
    if window_size > 0:
        max_pages = min(max_pages, triton.cdiv(window_size + page_size - 1, page_size))
    use_paged = batch * num_kv_heads * max_pages <= _PAGED_MAX_WORK and group >= 16
    grid = (splits, num_kv_heads, batch)
    decode = dict(
        HQ=num_q_heads, HKV=num_kv_heads, GROUP=group, DQK=head_qk,
        DPAD=triton.next_power_of_2(head_qk), DV=head_v,
        SPLITS=splits, WINDOW=window_size, FINAL=splits == 1,
    )
    if use_paged:
        _diffkv_decode_kernel[grid](
            query, key_cache, value_cache, tables, lengths, o_part, m_part, l_part, scale,
            BS=page_size, BT_STRIDE=tables.stride(0),
            K_STRIDES=key_cache.stride(), V_STRIDES=value_cache.stride(), **decode,
        )
    else:
        head_major = not (key_cache.is_contiguous() and value_cache.is_contiguous())
        k_dense, v_dense, len_alloc = _depage(
            query, key_cache, value_cache, lengths, tables, window_size, head_major, max_pages
        )
        _diffkv_decode_dense_kernel[grid](
            query, k_dense, v_dense, lengths, o_part, m_part, l_part, scale,
            BN=_BLOCK_N, BS_PAGE=page_size, LSTRIDE=len_alloc, HEAD_MAJOR=head_major, **decode,
        )
    if splits > 1:
        _diffkv_reduce_kernel[(batch * num_q_heads,)](
            o_part, m_part, l_part, out,
            DV=head_v, SPLITS=splits, SPAD=triton.next_power_of_2(splits),
        )
    return out
