# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import torch
import triton

from flag_attn.inkling_fa4.relative_attention import _validate_inputs

from .kernels import _combine_split_kv_metax_kernel, _fa4_rel_attn_paged_kernel

_DOT_TILE = 16
_TLE_UNAVAILABLE = "TLE is unavailable on MetaX"


@lru_cache
def _select_num_splits(
    *,
    kv_tiles: int,
    query_groups: int,
    single_query: bool,
    wide_heads: bool,
    num_sms: int,
) -> int:
    groups_per_sm = 1 if wide_heads else 4
    parallel_limit = max(1, groups_per_sm * num_sms // query_groups)
    tiles_per_group = 1 if single_query else 2
    work_limit = max(1, kv_tiles // tiles_per_group)
    split_limit = min(parallel_limit, work_limit, 128)
    tiles_per_split = triton.cdiv(kv_tiles, split_limit)
    return triton.cdiv(kv_tiles, tiles_per_split)


def _combine_config(*, output_rows: int, head_dim_v: int, num_splits: int) -> tuple[int, int, int]:
    block_s = min(triton.next_power_of_2(num_splits), 128)
    dimension_limit = 16 if output_rows < 32 and num_splits >= 16 else 128
    block_d = min(triton.next_power_of_2(head_dim_v), dimension_limit, 16384 // block_s)
    num_warps = min(4, triton.next_power_of_2(triton.cdiv(block_s * block_d, 4096)))
    return block_s, block_d, num_warps


@torch.no_grad()
def inkling_fa4_rel_attention_triton(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply paged varlen attention with Inkling relative-position logits.

    Args mirror vLLM's ``inkling_fa4_rel_attention`` operator. ``q`` and
    ``rel_logits`` are packed by Query sequence, while ``key_cache`` and
    ``value_cache`` use the logical layout
    ``[num_blocks, block_size, num_kv_heads, head_dim]``. ``block_table`` maps
    each sequence's logical KV blocks to physical cache blocks.

    This is an inference-only implementation.
    """
    _validate_inputs(
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max_seqlen_q,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        num_splits=1 if num_splits is None else num_splits,
        out=out,
    )

    # Match FA4 interface window canonicalization. The vLLM wrapper only turns
    # exactly (-1, -1) into (None, None); mixed -1 values remain real local
    # window widths when their sum is non-negative.
    window_left, window_right = (None, None) if window_size == (-1, -1) else window_size
    if causal:
        window_right = 0
    if window_left is not None and window_right is not None and window_left + window_right < 0:
        window_left = None
        window_right = None
    if window_left is None and window_right == 0:
        resolved_causal = True
        is_local = False
        window_right = None
    elif window_left is not None or window_right is not None:
        resolved_causal = False
        is_local = True
    else:
        resolved_causal = causal
        is_local = False

    rel_logits = rel_logits.contiguous()
    batch_size = cache_seqlens.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim_q = q.shape[2]
    head_dim_v = value_cache.shape[3]
    block_size = key_cache.shape[1]
    if out is None:
        out = torch.empty(
            (q.shape[0], num_heads, head_dim_v),
            dtype=q.dtype,
            device=q.device,
        )

    # FA4 explicitly zeroes empty-Q output instead of returning uninitialized
    # storage. No scheduler coordinate exists for an empty batch or query.
    if batch_size == 0 or q.shape[0] == 0:
        return out.zero_()
    if max_seqlen_q == 0:
        raise ValueError("max_seqlen_q must be positive for non-empty q")

    q_heads_per_kv_head = num_heads // num_kv_heads
    # Use the page-table capacity as a synchronization-free upper bound for KV
    # work.  Reading cache_seqlens.max().item() here would serialize every
    # serving invocation with the CPU.
    max_seqlen_k_bound = block_table.shape[1] * block_size
    dot_tile = _DOT_TILE
    wide_heads = max(head_dim_q, head_dim_v) > 128
    block_k = 16 if wide_heads else 32
    main_warps = 4 if wide_heads else 2
    pack_gqa = q_heads_per_kv_head > 1
    scheduler_heads = num_kv_heads if pack_gqa else num_heads
    rows_per_q_token = q_heads_per_kv_head if pack_gqa else 1
    max_scheduler_rows = max_seqlen_q * rows_per_q_token
    max_q_blocks = triton.cdiv(max_scheduler_rows, dot_tile)
    if num_splits is None:
        kv_tiles = triton.cdiv(max_seqlen_k_bound, block_k)
        if is_local and window_left is not None and window_right is not None:
            window_tokens = window_left + window_right + min(max_seqlen_q, dot_tile)
            kv_tiles = min(kv_tiles, triton.cdiv(window_tokens + block_k - 1, block_k))
        if kv_tiles <= (16 if is_local else 8):
            num_splits = 1
        else:
            num_splits = _select_num_splits(
                kv_tiles=kv_tiles,
                query_groups=max_q_blocks * batch_size * scheduler_heads,
                single_query=max_seqlen_q == 1,
                wide_heads=wide_heads,
                num_sms=torch.cuda.get_device_properties(q.device).multi_processor_count,
            )
    split_kv = num_splits > 1
    single_tile_shortcut = num_splits >= 16
    kernel_options = {}
    if (head_dim_q <= 128 or max(head_dim_q, head_dim_v) <= 256) and (split_kv or max_seqlen_q > 1):
        kernel_options["pipeline"] = ""
    if split_kv:
        partial_out = torch.empty_strided(
            (num_splits, q.shape[0], num_heads, head_dim_v),
            (head_dim_v, num_heads * num_splits * head_dim_v, num_splits * head_dim_v, 1),
            dtype=torch.float32,
            device=q.device,
        )
        partial_max = torch.empty_strided(
            (num_splits, q.shape[0], num_heads),
            (1, num_heads * num_splits, num_splits),
            dtype=torch.float32,
            device=q.device,
        )
        partial_sum = torch.empty_like(partial_max)
    else:
        # Unused placeholders keep a single compiled kernel signature.
        partial_out = out
        partial_max = out
        partial_sum = out
    grid = (max_q_blocks, batch_size * scheduler_heads, num_splits)
    _fa4_rel_attn_paged_kernel[grid](
        q_ptr=q,
        k_ptr=key_cache,
        v_ptr=value_cache,
        rel_ptr=rel_logits,
        out_ptr=out,
        partial_out_ptr=partial_out,
        partial_max_ptr=partial_max,
        partial_sum_ptr=partial_sum,
        block_table_ptr=block_table,
        cache_seqlens_ptr=cache_seqlens,
        cu_seqlens_q_ptr=cu_seqlens_q,
        stride_q_t=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_k_block=key_cache.stride(0),
        stride_k_t=key_cache.stride(1),
        stride_k_h=key_cache.stride(2),
        stride_k_d=key_cache.stride(3),
        stride_v_block=value_cache.stride(0),
        stride_v_t=value_cache.stride(1),
        stride_v_h=value_cache.stride(2),
        stride_v_d=value_cache.stride(3),
        stride_rel_t=rel_logits.stride(0),
        stride_rel_h=rel_logits.stride(1),
        stride_rel_d=rel_logits.stride(2),
        stride_out_t=out.stride(0),
        stride_out_h=out.stride(1),
        stride_out_d=out.stride(2),
        stride_partial_s=partial_out.stride(0),
        stride_partial_t=partial_out.stride(1) if split_kv else 0,
        stride_partial_h=partial_out.stride(2) if split_kv else 0,
        stride_partial_d=partial_out.stride(3) if split_kv else 0,
        stride_stats_s=partial_max.stride(0) if split_kv else 0,
        stride_stats_t=partial_max.stride(1) if split_kv else 0,
        stride_stats_h=partial_max.stride(2) if split_kv else 0,
        stride_bt_b=block_table.stride(0),
        stride_bt_block=block_table.stride(1),
        softmax_scale=softmax_scale,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        SCHEDULER_HEADS=scheduler_heads,
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        PACK_GQA=pack_gqa,
        NUM_SPLITS=num_splits,
        SPLIT_KV=split_kv,
        HEAD_DIM_Q=head_dim_q,
        HEAD_DIM_V=head_dim_v,
        BLOCK_D_Q=max(dot_tile, triton.next_power_of_2(head_dim_q)),
        BLOCK_D_V=max(dot_tile, triton.next_power_of_2(head_dim_v)),
        REL_EXTENT=rel_extent,
        REL_COVERS_VISIBLE_PAST=rel_extent >= max_seqlen_k_bound or (is_local and window_left < rel_extent),
        BLOCK_SIZE=block_size,
        BLOCK_Q=dot_tile,
        BLOCK_K=block_k,
        CAUSAL=resolved_causal,
        LOCAL=is_local,
        WINDOW_LEFT=0 if window_left is None else window_left,
        WINDOW_RIGHT=0 if window_right is None else window_right,
        SINGLE_QUERY=max_seqlen_q == 1,
        SINGLE_TILE_SHORTCUT=single_tile_shortcut,
        num_warps=main_warps,
        num_stages=1,
        **kernel_options,
    )
    if split_kv:
        combine_block_s, combine_block_d, combine_warps = _combine_config(
            output_rows=q.shape[0] * num_heads,
            head_dim_v=head_dim_v,
            num_splits=num_splits,
        )
        combine_grid = (q.shape[0] * num_heads, triton.cdiv(head_dim_v, combine_block_d))
        _combine_split_kv_metax_kernel[combine_grid](
            partial_out_ptr=partial_out,
            partial_max_ptr=partial_max,
            partial_sum_ptr=partial_sum,
            out_ptr=out,
            stride_partial_s=partial_out.stride(0),
            stride_partial_t=partial_out.stride(1),
            stride_partial_h=partial_out.stride(2),
            stride_partial_d=partial_out.stride(3),
            stride_stats_s=partial_max.stride(0),
            stride_stats_t=partial_max.stride(1),
            stride_stats_h=partial_max.stride(2),
            stride_out_t=out.stride(0),
            stride_out_h=out.stride(1),
            stride_out_d=out.stride(2),
            NUM_HEADS=num_heads,
            NUM_SPLITS=num_splits,
            BLOCK_S=combine_block_s,
            SINGLE_TILE_SHORTCUT=single_tile_shortcut,
            HEAD_DIM_V=head_dim_v,
            BLOCK_D_V=combine_block_d,
            num_warps=combine_warps,
        )
    return out


def inkling_fa4_rel_attention_tle(*args: Any, **kwargs: Any) -> torch.Tensor:
    raise RuntimeError(_TLE_UNAVAILABLE)


def tle_available() -> tuple[bool, str]:
    return False, _TLE_UNAVAILABLE


def _require_triton(name: str | None, tle_error: type[Exception]) -> None:
    selected = (name or os.getenv("INKLING_FA4_BACKEND", "auto")).strip().lower()
    if selected in ("auto", "triton", "base"):
        return
    if selected in ("tle", "triton_tle", "triton-tle"):
        raise tle_error(_TLE_UNAVAILABLE)
    raise ValueError(f"unknown Inkling FA4 backend={selected!r}; choose from auto, triton, base")


def get_backend(name: str | None = None):
    _require_triton(name, ImportError)
    return inkling_fa4_rel_attention_triton


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    _require_triton(backend, RuntimeError)
    return inkling_fa4_rel_attention_triton(*args, **kwargs)
