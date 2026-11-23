# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from triton.experimental.tle.language import raw
from triton.experimental.tle.raw import dialect
from triton.experimental.tle.raw.cann.runtime import _find_ccec, _find_llvm_link, _template_include_dirs
from triton.compiler.errors import CompilationError, MLIRCompilationError


_PAGE_SIZE = 16
_TILE_TOKENS = 256
_KEY_DIM = 192
_VALUE_DIM = 128
_SOURCE = Path(__file__).with_name("raw_gather_pages.cpp")
if not _SOURCE.is_file():
    raise ImportError("DiffKV TLE Raw source is unavailable")


@dialect(
    name="cann",
    file=str(_SOURCE),
    extern_func_name="diffkv_gather_pages_bf16",
    pipeline={"core": "cube", "pipe": "PIPE_MTE2"},
)
def _gather_pages(*args):
    ...


@triton.jit
def _load_pages(source, page_ids, zeros, first, end, WIDTH: tl.constexpr,
                TILE: tl.constexpr, STRIDES: tl.constexpr):
    return raw.call(
        _gather_pages,
        [
            source, page_ids, zeros, tl.full((), TILE, tl.int64),
            tl.full((), WIDTH, tl.int64), tl.full((), STRIDES[0], tl.int64),
            tl.full((), STRIDES[1], tl.int64), first.to(tl.int64), end.to(tl.int64),
            tl.full((TILE, WIDTH), 0, tl.bfloat16),
        ],
        output_indices=[9],
    )


@triton.jit
def _decode_raw(
    q, k, v, tables, lengths, zeros, out, scale,
    CACHE_ROWS: tl.constexpr, TABLE_WIDTH: tl.constexpr, KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr, TILE: tl.constexpr, K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr, WINDOW: tl.constexpr,
):
    kv_head = tl.program_id(0)
    batch = tl.program_id(1)
    length = tl.load(lengths + batch)
    start = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    first_page = start // 16
    first_token = first_page * 16

    k_source = tl.make_block_ptr(
        k + kv_head.to(tl.int64) * K_STRIDES[2], (CACHE_ROWS, 192),
        (K_STRIDES[1], 1), (0, 0), (TILE, 192), (1, 0),
    )
    v_source = tl.make_block_ptr(
        v + kv_head.to(tl.int64) * V_STRIDES[2], (CACHE_ROWS, 128),
        (V_STRIDES[1], 1), (0, 0), (TILE, 128), (1, 0),
    )
    zero_page = tl.make_block_ptr(zeros, (16, 192), (192, 1), (0, 0), (16, 192), (1, 0))
    row = (batch * KV_HEADS + kv_head) * GROUP
    q_source = tl.make_block_ptr(q + row * 192, (GROUP, 192), (192, 1), (0, 0), (GROUP, 192), (1, 0))
    query = tl.load(q_source)
    table = tables + batch.to(tl.int64) * TABLE_WIDTH

    row_max = tl.full((GROUP,), -1.0e30, tl.float32)
    denominator = tl.full((GROUP,), 0, tl.float32)
    accumulator = tl.full((GROUP, 128), 0, tl.float32)
    tokens = tl.arange(0, TILE)
    for tile in tl.range(tl.cdiv(length - first_token, TILE)):
        tile_start = first_token + tile * TILE
        lo = tl.maximum(0, start - tile_start)
        hi = tl.minimum(TILE, length - tile_start)
        valid = (tokens >= lo) & (tokens < hi)
        page_ids = tl.make_block_ptr(
            table, (TABLE_WIDTH, 1), (1, 1), (first_page + tile * (TILE // 16), 0),
            (TILE // 16, 1), (1, 0),
        )
        key = _load_pages(k_source, page_ids, zero_page, lo, hi, 192, TILE, K_STRIDES)
        scores = tl.dot(query, tl.trans(key)) * scale
        scores = tl.where(valid[None, :], scores, -1.0e30)
        value = _load_pages(v_source, page_ids, zero_page, lo, hi, 128, TILE, V_STRIDES)

        next_max = tl.maximum(row_max, tl.max(scores, 1))
        rescale = tl.exp(row_max - next_max)
        prob = tl.exp(scores - next_max[:, None])
        prob = tl.where(valid[None, :], prob, 0.0)
        denominator = denominator * rescale + tl.sum(prob, 1)
        accumulator = tl.dot(prob.to(tl.bfloat16), value, accumulator * rescale[:, None])
        row_max = next_max

    rows = row + tl.arange(0, GROUP)
    features = tl.arange(0, 128)
    result = accumulator / tl.where(denominator > 0.0, denominator, 1.0)[:, None]
    tl.store(out + rows[:, None] * 128 + features[None, :], result)


def supports_raw_decode(query, key_cache, value_cache):
    if (
        query.dtype != torch.bfloat16
        or key_cache.shape[1] != _PAGE_SIZE
        or key_cache.shape[2] == 0
        or query.shape[1] // key_cache.shape[2] not in (8, 16)
        or key_cache.shape[-1] != _KEY_DIM
        or value_cache.shape[-1] != _VALUE_DIM
    ):
        return False

    for cache in (key_cache, value_cache):
        strides = cache.stride()
        # BF16 ND2NZ reads 32-byte blocks and takes a 16-bit source row stride.
        if (
            strides[-1] != 1
            or cache.data_ptr() % 32
            or not cache.shape[-1] <= strides[1] <= 65535
            or any(stride % 16 for stride in strides[:3])
        ):
            return False
    return True


@lru_cache(maxsize=1)
def _raw_toolchain_available():
    if not any((Path(directory) / "Utils.h").is_file() for directory in _template_include_dirs()):
        return False
    try:
        _find_llvm_link(_find_ccec())
    except RuntimeError:
        return False
    return True


def try_raw_decode(query, key_cache, value_cache, lengths, tables, out, scale, window_size):
    if not supports_raw_decode(query, key_cache, value_cache) or not _raw_toolchain_available():
        return False
    try:
        launch_raw_decode(query, key_cache, value_cache, lengths, tables, out, scale, window_size)
    except (CompilationError, MLIRCompilationError):
        return False
    return True


def launch_raw_decode(query, key_cache, value_cache, lengths, tables, out, scale, window_size):
    batch, query_heads, _ = query.shape
    cache_pages, _, kv_heads, _ = key_cache.shape
    if cache_pages == 0:
        out.zero_()
        return
    # Both operands use this page for lanes outside the active sequence or window.
    zeros = torch.zeros((_PAGE_SIZE, _KEY_DIM), dtype=query.dtype, device=query.device)
    _decode_raw[(kv_heads, batch)](
        query, key_cache, value_cache, tables, lengths, zeros, out, scale,
        CACHE_ROWS=cache_pages * _PAGE_SIZE, TABLE_WIDTH=tables.shape[1],
        KV_HEADS=kv_heads, GROUP=query_heads // kv_heads, TILE=_TILE_TOKENS,
        K_STRIDES=key_cache.stride(), V_STRIDES=value_cache.stride(), WINDOW=window_size,
        enable_legacy_insert_load_store_for_mix_cv=True,
    )
