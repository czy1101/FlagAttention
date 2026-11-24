# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public Hygon DiffKV decode API and workload configuration."""
import importlib

import torch
# Support the upstream module rename without changing the public operator.
try:
    from flag_attn.diffkv import api
except ModuleNotFoundError as error:
    if error.name != "flag_attn.diffkv":
        raise
    from flag_attn.diffkv_attention import api

_fallback = importlib.import_module(__package__ + ".diffkv")
_fallback_attention = _fallback.fallback_attention
_DIFFKV_AUTOTUNE = _fallback._DIFFKV_AUTOTUNE
_attention = _fallback._attention
_wide_attention = _fallback._wide_attention
_reduce = _fallback._reduce

@_fallback.lru_cache(maxsize=128)
def _decode_segments(capacity, programs, cus):
    count = (4 * cus + max(programs, 1) - 1) // max(programs, 1)
    parallel = 1 << (max(count, 1) - 1).bit_length()
    useful = 1 << max((max(capacity // 512, 1)).bit_length() - 1, 0)
    segments = min(parallel, useful, 128)
    # Avoid additional splitting once attention is hidden by reducer setup.
    tiles = (capacity + 15) // 16
    if programs * segments < cus and (tiles + segments - 1) // segments > 40:
        segments = min(segments * 2, 128)
    return segments


def configuration(q, k, table, window, path, num_segments):
    capacity = table.shape[1] * 16
    programs = q.shape[0] * k.shape[2]
    cus = _fallback._device_num_sms(q.device)
    segments = 1
    if path != '2d':
        if num_segments is not None:
            if not isinstance(num_segments, int) or num_segments < 1 or num_segments & (num_segments - 1):
                raise ValueError('num_segments must be a positive power of two')
            segments = num_segments
        elif not 0 < window < capacity and capacity > 1024:
            segments = _decode_segments(capacity, programs, cus)
    if segments > 1:
        return 16, 1, 1, segments
    if window == -1 and capacity > 1024 and q.shape[1] // k.shape[2] == 16:
        return (16, 4, 2, segments) if programs < cus else (32, 4, 1, segments)
    if 0 < window <= 256 and capacity > 1024 and programs > 3 * cus and q.shape[1] // k.shape[2] == 8:
        return 32, 1, 1, segments
    return 32, 2, 2, segments


@torch.no_grad()
def diffkv_attention(query, key_cache, value_cache, context_lens, block_tables,
                     attn_scale=None, window_size=-1, path='2d', num_segments=None, backend=None):
    selected = api._resolve_backend(backend)
    normalized = path.strip().lower()
    if selected != 'triton' or _DIFFKV_AUTOTUNE or _fallback._DIFFKV_AUTOTUNE:
        return api.diffkv_attention(query, key_cache, value_cache, context_lens, block_tables,
            attn_scale=attn_scale, window_size=window_size,
            path='2d' if normalized == 'auto' else normalized, num_segments=num_segments, backend=selected)
    if normalized not in {'2d', '3d', 'auto'}:
        raise ValueError('path must be 2d, 3d, or auto')
    q, k, v = query, key_cache, value_cache
    if not (isinstance(window_size, int)
            and isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
            and q.ndim == 3 and k.ndim == v.ndim == 4 and q.shape[-1] == k.shape[-1] == 192
            and v.shape[-1] == 128 and k.shape[1] == 16 and k.shape[2] > 0
            and q.shape[1] // k.shape[2] <= 16
            and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
            and context_lens.stride(0) == block_tables.stride(1) == 1):
        return _fallback_attention(q, k, v, context_lens, block_tables, attn_scale=attn_scale,
                  window_size=window_size, path=normalized, num_segments=num_segments, backend=selected)
    batch, hq, _, _ = api._validate_public_inputs(q, k, v, context_lens, block_tables, window_size)
    # Lengths are int32 under the existing API contract. A larger window
    # includes every valid token; avoid narrowing its Python integer in IR.
    effective_window = -1 if window_size > 2147483647 else window_size
    tile, warps, stages, segments = configuration(q, k, block_tables, effective_window, normalized, num_segments)
    if segments > 128:
        return _fallback_attention(q, k, v, context_lens, block_tables, attn_scale=attn_scale,
                  window_size=window_size, path=normalized, num_segments=num_segments, backend=selected)
    context_lens = context_lens.to(dtype=torch.int32)
    block_tables = block_tables.to(dtype=torch.int32)
    out = torch.empty((batch, hq, 128), device=q.device, dtype=q.dtype)
    if batch == 0:
        return out
    if segments > 1:
        partial = torch.empty((batch, hq, segments, 128), device=q.device, dtype=torch.float32)
        maxima = torch.empty((batch, hq, segments), device=q.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
    else:
        partial = maxima = sums = out
    strides = (q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
               v.stride(0), v.stride(1), v.stride(2), block_tables.stride(0))
    wide = (segments == 1 and effective_window == -1
            and block_tables.shape[1] * 16 > 1024 and hq // k.shape[2] == 16)
    launcher = _wide_attention if wide else _attention
    if wide:
        launcher[(batch, k.shape[2], segments)](
            Q=q, K=k, V=v, Lens=context_lens, Table=block_tables, Out=out,
            Partial=partial, Maxima=maxima, Sums=sums, STRIDES=strides,
            HQ=hq, GROUP=hq // k.shape[2], SCALE=float(attn_scale) if attn_scale is not None else 192**-.5,
            WINDOW=max(effective_window, 0), SEGMENTS=segments, TILE=tile, STAGES=stages, WAVES=warps,
            num_warps=warps, num_stages=stages)
    else:
        tensors = (q, k, v, context_lens, block_tables, out, partial, maxima, sums)
        constants = (strides, hq, hq // k.shape[2],
                     float(attn_scale) if attn_scale is not None else 192**-.5,
                     max(effective_window, 0), segments, tile, stages, warps)
        state = []
        launcher.launch((batch, k.shape[2], segments), tensors, constants, warps, stages, state)
    if segments > 1:
        _reduce.launch((batch * hq,), (out, partial, maxima, sums), (segments,), 4, 1, state)
    return out
