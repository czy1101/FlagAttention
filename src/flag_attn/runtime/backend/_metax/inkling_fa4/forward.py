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

import torch
import triton

from flag_attn.inkling_fa4.relative_attention import _validate_inputs

from . import launch_configs
from .kernels import _combine_split_kv_metax_kernel, _fa4_rel_attn_paged_kernel


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

    This is an inference-only implementation. On H100 the upstream Inkling
    heuristic normally selects one split; larger ``num_splits`` values execute
    real split-KV attention followed by a numerically stable FP32 combine.
    """
    auto_split = num_splits is None
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
    cfg = launch_configs.get_preset(head_dim_q=head_dim_q, head_dim_v=head_dim_v)
    block_q = cfg["BLOCK_Q"]
    pack_gqa = q_heads_per_kv_head > 1
    scheduler_heads = num_kv_heads if pack_gqa else num_heads
    rows_per_q_token = q_heads_per_kv_head if pack_gqa else 1
    max_scheduler_rows = max_seqlen_q * rows_per_q_token
    max_q_blocks = triton.cdiv(max_scheduler_rows, block_q)
    num_splits = launch_configs.select_num_splits(
        num_splits=num_splits,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k_bound,
        block_k=cfg["BLOCK_K"],
        query_blocks=max_q_blocks,
        batch_size=batch_size,
        scheduler_heads=scheduler_heads,
        head_dim_q=head_dim_q,
        head_dim_v=head_dim_v,
        is_local=is_local,
        device=q.device,
    )
    allow_direct_output = auto_split and num_splits >= 16
    split_kv = num_splits > 1
    kernel_options = launch_configs.get_kernel_options(
        head_dim_q=head_dim_q,
        head_dim_v=head_dim_v,
        split_kv=split_kv,
        max_seqlen_q=max_seqlen_q,
    )
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
    # A regular grid makes the task coordinate directly available through
    # program_id. Shorter sequences launch masked programs, but the wrapper no
    # longer needs CUDA elementwise/div/cumsum kernels to construct task_ends.
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
        BLOCK_D_Q=max(16, triton.next_power_of_2(head_dim_q)),
        BLOCK_D_V=max(16, triton.next_power_of_2(head_dim_v)),
        REL_EXTENT=rel_extent,
        BLOCK_SIZE=block_size,
        BLOCK_Q=block_q,
        BLOCK_K=cfg["BLOCK_K"],
        CAUSAL=resolved_causal,
        LOCAL=is_local,
        WINDOW_LEFT=0 if window_left is None else window_left,
        WINDOW_RIGHT=0 if window_right is None else window_right,
        ALLOW_DIRECT_OUTPUT=allow_direct_output,
        SINGLE_QUERY=max_seqlen_q == 1,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        **kernel_options,
    )
    if split_kv:
        combine_block_d, combine_warps = launch_configs.get_combine_config(
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
            SPLITS_PAD=triton.next_power_of_2(num_splits),
            HEAD_DIM_V=head_dim_v,
            BLOCK_D_V=combine_block_d,
            num_warps=combine_warps,
            ALLOW_DIRECT_OUTPUT=allow_direct_output,
        )
    return out
