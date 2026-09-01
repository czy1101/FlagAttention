      
"""Triton implementation of Inkling paged relative attention.

The public function mirrors the vLLM ``inkling_fa4_rel_attention`` operator.
The forward path supports Hopper-compatible split-KV, different Q/K and V
head dimensions, and packed GQA. Packed variable-length queries use a regular
3D task grid over ``(query block, sequence/head, split)``. Each program checks
its query block against the sequence length from ``cu_seqlens_q``; this avoids
launching eager PyTorch kernels to build a compact prefix-sum task table.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from inkling_fa4 import autotune_configs

@torch.no_grad()
def inkling_fa4_rel_attention(
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
    num_splits: int = 1,
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
        num_splits=num_splits,
        out=out,
    )

    # Match FA4 interface window canonicalization. The vLLM wrapper only turns
    # exactly (-1, -1) into (None, None); mixed -1 values remain real local
    # window widths when their sum is non-negative.
    window_left, window_right = (
        (None, None) if window_size == (-1, -1) else window_size
    )
    if causal:
        window_right = 0
    if (
        window_left is not None
        and window_right is not None
        and window_left + window_right < 0
    ):
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
    cfg = autotune_configs.get_preset(
        max_seqlen_q,
        max_seqlen_k=max_seqlen_k_bound,
        head_dim_q=head_dim_q,
        head_dim_v=head_dim_v,
        q_heads_per_kv_head=q_heads_per_kv_head,
    )
    block_q = cfg["BLOCK_Q"]
    pack_gqa = q_heads_per_kv_head > 1
    scheduler_heads = num_kv_heads if pack_gqa else num_heads
    rows_per_q_token = q_heads_per_kv_head if pack_gqa else 1
    max_scheduler_rows = max_seqlen_q * rows_per_q_token
    max_q_blocks = triton.cdiv(max_scheduler_rows, block_q)
    split_kv = num_splits > 1
    if split_kv:
        partial_out = torch.empty(
            (num_splits, *out.shape), dtype=torch.float32, device=q.device
        )
        partial_max = torch.empty(
            (num_splits, q.shape[0], num_heads),
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
        q,
        key_cache,
        value_cache,
        rel_logits,
        out,
        partial_out,
        partial_max,
        partial_sum,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        partial_out.stride(0),
        partial_out.stride(1) if split_kv else 0,
        partial_out.stride(2) if split_kv else 0,
        partial_out.stride(3) if split_kv else 0,
        partial_max.stride(0) if split_kv else 0,
        partial_max.stride(1) if split_kv else 0,
        partial_max.stride(2) if split_kv else 0,
        block_table.stride(0),
        block_table.stride(1),
        softmax_scale,
        NUM_HEADS=num_heads,
        NUM_KV_HEADS=num_kv_heads,
        SCHEDULER_HEADS=scheduler_heads,
        Q_HEADS_PER_KV_HEAD=q_heads_per_kv_head,
        PACK_GQA=pack_gqa,
        NUM_SPLITS=num_splits,
        SPLIT_KV=split_kv,
        HEAD_DIM_Q=head_dim_q,
        HEAD_DIM_V=head_dim_v,
        BLOCK_D_Q=triton.next_power_of_2(head_dim_q),
        BLOCK_D_V=triton.next_power_of_2(head_dim_v),
        REL_EXTENT=rel_extent,
        BLOCK_SIZE=block_size,
        BLOCK_Q=block_q,
        BLOCK_K=cfg["BLOCK_K"],
        CAUSAL=resolved_causal,
        LOCAL=is_local,
        WINDOW_LEFT=0 if window_left is None else window_left,
        WINDOW_RIGHT=0 if window_right is None else window_right,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    if split_kv:
        combine_grid = (q.shape[0] * num_heads,)
        _combine_split_kv_kernel[combine_grid](
            partial_out,
            partial_max,
            partial_sum,
            out,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_max.stride(0),
            partial_max.stride(1),
            partial_max.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            NUM_HEADS=num_heads,
            NUM_SPLITS=num_splits,
            SPLITS_PAD=triton.next_power_of_2(num_splits),
            HEAD_DIM_V=head_dim_v,
            BLOCK_D_V=triton.next_power_of_2(head_dim_v),
            num_warps=4,
        )
    return out

def _validate_inputs(
    *,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    window_size: tuple[int, int],
    rel_extent: int,
    rel_logits: torch.Tensor,
    num_splits: int,
    out: torch.Tensor | None,
) -> None:
    tensors = (
        q,
        key_cache,
        value_cache,
        block_table,
        cache_seqlens,
        cu_seqlens_q,
        rel_logits,
    )
    if not all(t.is_cuda for t in tensors):
        raise ValueError("all attention inputs must be CUDA tensors")
    if len({t.device for t in tensors}) != 1:
        raise ValueError("all attention inputs must be on the same CUDA device")
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, num_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            "key_cache and value_cache must have shape "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    if q.shape[1] <= 0 or key_cache.shape[2] <= 0:
        raise ValueError("Q and KV head counts must be positive")
    if key_cache.shape[1] <= 0:
        raise ValueError("KV cache block_size must be positive")
    if key_cache.shape[:3] != value_cache.shape[:3]:
        raise ValueError(
            "key_cache and value_cache block, token, and head shapes must match"
        )
    if block_table.ndim != 2:
        raise ValueError("block_table must have shape [batch, max_num_blocks]")
    if cache_seqlens.ndim != 1:
        raise ValueError("cache_seqlens must have shape [batch]")
    if cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must have shape [batch + 1]")
    batch_size = cache_seqlens.shape[0]
    if block_table.shape[0] != batch_size:
        raise ValueError("block_table and cache_seqlens batch sizes differ")
    if cu_seqlens_q.shape[0] != batch_size + 1:
        raise ValueError("cu_seqlens_q must contain batch_size + 1 entries")
    if rel_logits.shape != (q.shape[0], q.shape[1], rel_extent):
        raise ValueError(
            "rel_logits must have shape [total_q, num_heads, rel_extent]"
        )
    if key_cache.shape[3] != q.shape[2]:
        raise ValueError("Q and K head dimensions must match")
    alignment = 16 // q.element_size()
    if not (
        8 <= q.shape[2] <= 512
        and 8 <= value_cache.shape[3] <= 512
        and q.shape[2] % alignment == 0
        and value_cache.shape[3] % alignment == 0
    ):
        raise ValueError(
            "Q/K and V head dimensions must be in [8, 512] and aligned "
            f"to {alignment} elements"
        )
    if q.shape[1] % key_cache.shape[2] != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("q must use torch.bfloat16 or torch.float16")
    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        raise TypeError("the current Triton path requires Q, K, and V to match")
    if rel_logits.dtype != q.dtype:
        raise TypeError("rel_logits and q must have the same dtype")
    if block_table.dtype != torch.int32:
        raise TypeError("block_table must use torch.int32")
    if cache_seqlens.dtype != torch.int32 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("sequence metadata must use torch.int32")
    if cache_seqlens.stride(0) != 1 or cu_seqlens_q.stride(0) != 1:
        raise ValueError("sequence metadata must be contiguous")
    if max_seqlen_q < 0:
        raise ValueError("max_seqlen_q must be non-negative")
    if rel_extent <= 0:
        raise ValueError("rel_extent must be positive")
    if (
        not isinstance(window_size, tuple)
        or len(window_size) != 2
        or not all(isinstance(v, int) and v >= -1 for v in window_size)
    ):
        raise ValueError("window_size must be a pair of integers >= -1")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")
    if out is not None:
        if not out.is_cuda or out.device != q.device:
            raise ValueError("out must be on the same CUDA device as q")
        expected_out_shape = (q.shape[0], q.shape[1], value_cache.shape[3])
        if out.shape != expected_out_shape or out.dtype != q.dtype:
            raise ValueError(
                "out must have shape [total_q, num_heads, value_head_dim] "
                "and the same dtype as q"
            )

@triton.jit
def _fa4_rel_attn_paged_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    rel_ptr,
    out_ptr,
    partial_out_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    block_table_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_block,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_v_block,
    stride_v_t,
    stride_v_h,
    stride_v_d,
    stride_rel_t,
    stride_rel_h,
    stride_rel_d,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    stride_partial_s,
    stride_partial_t,
    stride_partial_h,
    stride_partial_d,
    stride_stats_s,
    stride_stats_t,
    stride_stats_h,
    stride_bt_b,
    stride_bt_block,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    SCHEDULER_HEADS: tl.constexpr,
    Q_HEADS_PER_KV_HEAD: tl.constexpr,
    PACK_GQA: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_KV: tl.constexpr,
    HEAD_DIM_Q: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    LOCAL: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    WINDOW_RIGHT: tl.constexpr,
):
    # The regular 3D grid directly encodes (Q block, sequence/head, split).
    # This replaces the wrapper-side task_ends prefix sum and the per-program
    # lower-bound search used by the previous compact scheduler.
    pid_q_grid = tl.program_id(0)
    seq_head_id = tl.program_id(1)
    split_idx = tl.program_id(2)
    seq_id = seq_head_id // SCHEDULER_HEADS
    scheduler_head = seq_head_id - seq_id * SCHEDULER_HEADS

    q_start = tl.load(cu_seqlens_q_ptr + seq_id)
    q_end = tl.load(cu_seqlens_q_ptr + seq_id + 1)
    q_len = q_end - q_start
    scheduler_rows = q_len * (Q_HEADS_PER_KV_HEAD if PACK_GQA else 1)
    num_q_blocks = (scheduler_rows + BLOCK_Q - 1) // BLOCK_Q
    task_valid = pid_q_grid < num_q_blocks
    pid_q = pid_q_grid

    # FA4 uses long-processing-time order for causal/local attention: later Q
    # tiles generally see more KV work, so schedule them first.
    if CAUSAL or LOCAL:
        pid_q = tl.where(task_valid, num_q_blocks - 1 - pid_q_grid, 0)
    else:
        pid_q = tl.where(task_valid, pid_q, 0)

    q_start = tl.where(task_valid, q_start, 0)
    q_len = tl.where(task_valid, q_len, 0)
    # Setting k_len to zero is essential: an invalid regular-grid program must
    # not walk the full KV range in non-causal attention before masked stores.
    k_len = tl.where(
        task_valid,
        tl.load(cache_seqlens_ptr + seq_id),
        0,
    )

    off_q = tl.arange(0, BLOCK_Q)
    off_k = tl.arange(0, BLOCK_K)
    off_d_q = tl.arange(0, BLOCK_D_Q)
    off_d_v = tl.arange(0, BLOCK_D_V)
    d_q_valid = off_d_q < HEAD_DIM_Q
    d_v_valid = off_d_v < HEAD_DIM_V
    scheduler_row = pid_q * BLOCK_Q + off_q
    if PACK_GQA:
        q_local = scheduler_row // Q_HEADS_PER_KV_HEAD
        q_head = (
            scheduler_head * Q_HEADS_PER_KV_HEAD
            + scheduler_row % Q_HEADS_PER_KV_HEAD
        )
        kv_head = scheduler_head
    else:
        q_local = scheduler_row
        q_head = scheduler_head
        kv_head = scheduler_head
    if PACK_GQA:
        q_head_offset = q_head[:, None]
    else:
        q_head_offset = q_head
    q_valid = task_valid & (scheduler_row < scheduler_rows)
    q_global = q_start + q_local
    q_pos = q_local + k_len - q_len

    q_ptrs = (
        q_ptr
        + q_global[:, None] * stride_q_t
        + q_head_offset * stride_q_h
        + off_d_q[None, :] * stride_q_d
    )
    q_value = tl.load(
        q_ptrs,
        mask=q_valid[:, None] & d_q_valid[None, :],
        other=0.0,
    )

    m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_D_V], dtype=tl.float32)

    # Prune KV blocks that cannot contain a valid score for any row in this Q
    # tile. The per-token masks below are still required for boundary blocks.
    q_pos_min = tl.min(tl.where(q_valid, q_pos, k_len), axis=0)
    q_pos_max = tl.max(tl.where(q_valid, q_pos, -1), axis=0)
    valid_k_start = 0
    valid_k_end = k_len
    if CAUSAL:
        valid_k_end = tl.minimum(valid_k_end, q_pos_max + 1)
    if LOCAL:
        valid_k_start = tl.maximum(
            valid_k_start, q_pos_min - WINDOW_LEFT
        )
        valid_k_end = tl.minimum(
            valid_k_end, q_pos_max + WINDOW_RIGHT + 1
        )
    valid_k_start = tl.maximum(valid_k_start, 0)
    valid_k_end = tl.maximum(tl.minimum(valid_k_end, k_len), 0)
    valid_block_start = valid_k_start // BLOCK_K
    valid_block_end = (valid_k_end + BLOCK_K - 1) // BLOCK_K

    num_k_blocks = (k_len + BLOCK_K - 1) // BLOCK_K
    blocks_per_split = (num_k_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    split_block_start = split_idx * blocks_per_split
    split_block_end = tl.minimum(
        split_block_start + blocks_per_split, num_k_blocks
    )
    block_start = tl.maximum(split_block_start, valid_block_start)
    block_end = tl.minimum(split_block_end, valid_block_end)
    for block_k in range(block_start, block_end):
        k_local = block_k * BLOCK_K + off_k
        k_valid = k_local < k_len
        logical_block = k_local // BLOCK_SIZE
        offset_in_block = k_local % BLOCK_SIZE
        # A page-aligned tile needs one page-table scalar instead of BLOCK_K
        # identical entries. Keep the vector gather for the common 16-token
        # page / 64-or-128-token compute tile case.
        if BLOCK_K == BLOCK_SIZE:
            physical_page = tl.load(
                block_table_ptr
                + seq_id * stride_bt_b
                + block_k * stride_bt_block,
                mask=block_k < num_k_blocks,
                other=0,
            )
            physical_block = physical_page + tl.zeros([BLOCK_K], tl.int32)
        else:
            physical_block = tl.load(
                block_table_ptr
                + seq_id * stride_bt_b
                + logical_block * stride_bt_block,
                mask=k_valid,
                other=0,
            )

        k_ptrs = (
            k_ptr
            + physical_block[None, :] * stride_k_block
            + offset_in_block[None, :] * stride_k_t
            + kv_head * stride_k_h
            + off_d_q[:, None] * stride_k_d
        )
        k_value = tl.load(
            k_ptrs,
            mask=d_q_valid[:, None] & k_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q_value, k_value).to(tl.float32) * softmax_scale

        mask = q_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            mask = mask & (k_local[None, :] <= q_pos[:, None])
        if LOCAL:
            mask = mask & (
                k_local[None, :] >= q_pos[:, None] - WINDOW_LEFT
            )
        if LOCAL:
            mask = mask & (
                k_local[None, :] <= q_pos[:, None] + WINDOW_RIGHT
            )

        rel_dist = q_pos[:, None] - k_local[None, :]
        rel_in_range = (rel_dist >= 0) & (rel_dist < REL_EXTENT)
        safe_rel_idx = tl.where(rel_in_range, rel_dist, 0)
        rel_ptrs = (
            rel_ptr
            + q_global[:, None] * stride_rel_t
            + q_head_offset * stride_rel_h
            + safe_rel_idx * stride_rel_d
        )
        rel_bias = tl.load(
            rel_ptrs,
            mask=mask & rel_in_range,
            other=0.0,
        ).to(tl.float32)
        # Hopper's approximate exp2 path is cheaper. Convert the *complete*
        # score, including relative bias, so the operator remains unchanged.
        scores = tl.where(
            mask,
            (scores + rel_bias) * 1.4426950408889634,
            -float("inf"),
        )

        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        row_has_value = m_new != -float("inf")
        m_safe = tl.where(row_has_value, m_new, 0.0)
        alpha = tl.where(
            m_i == -float("inf"),
            0.0,
            tl.exp2(m_i - m_safe),
        )
        probabilities = tl.exp2(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(probabilities, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = (
            v_ptr
            + physical_block[:, None] * stride_v_block
            + offset_in_block[:, None] * stride_v_t
            + kv_head * stride_v_h
            + off_d_v[None, :] * stride_v_d
        )
        v_value = tl.load(
            v_ptrs,
            mask=k_valid[:, None] & d_v_valid[None, :],
            other=0.0,
        )
        acc += tl.dot(probabilities.to(v_value.dtype), v_value).to(tl.float32)
        m_i = tl.where(row_has_value, m_new, m_i)

    if SPLIT_KV:
        partial_ptrs = (
            partial_out_ptr
            + split_idx * stride_partial_s
            + q_global[:, None] * stride_partial_t
            + q_head_offset * stride_partial_h
            + off_d_v[None, :] * stride_partial_d
        )
        stats_ptrs = (
            split_idx * stride_stats_s
            + q_global * stride_stats_t
            + q_head * stride_stats_h
        )
        tl.store(partial_ptrs, acc, mask=q_valid[:, None] & d_v_valid[None, :])
        tl.store(partial_max_ptr + stats_ptrs, m_i, mask=q_valid)
        tl.store(partial_sum_ptr + stats_ptrs, l_i, mask=q_valid)
    else:
        denominator = tl.where(l_i > 0.0, l_i, 1.0)
        output = acc / denominator[:, None]
        out_ptrs = (
            out_ptr
            + q_global[:, None] * stride_out_t
            + q_head_offset * stride_out_h
            + off_d_v[None, :] * stride_out_d
        )
        tl.store(out_ptrs, output, mask=q_valid[:, None] & d_v_valid[None, :])

@triton.jit
def _combine_split_kv_kernel(
    partial_out_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    out_ptr,
    stride_partial_s,
    stride_partial_t,
    stride_partial_h,
    stride_partial_d,
    stride_stats_s,
    stride_stats_t,
    stride_stats_h,
    stride_out_t,
    stride_out_h,
    stride_out_d,
    NUM_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
):
    output_id = tl.program_id(0)
    q_global = output_id // NUM_HEADS
    q_head = output_id - q_global * NUM_HEADS

    off_d = tl.arange(0, BLOCK_D_V)
    d_valid = off_d < HEAD_DIM_V

    # Pass 1: scalar maximum across splits. The split loop is compile-time
    # unrolled and never constructs a [SPLITS_PAD, BLOCK_D_V] tensor.
    global_max = -float("inf")
    for split in tl.static_range(0, NUM_SPLITS):
        stats_offset = (
            split * stride_stats_s
            + q_global * stride_stats_t
            + q_head * stride_stats_h
        )
        split_max = tl.load(partial_max_ptr + stats_offset)
        global_max = tl.maximum(global_max, split_max)

    has_value = global_max != -float("inf")
    max_safe = tl.where(has_value, global_max, 0.0)

    # Pass 2: keep only one [BLOCK_D_V] partial vector live per iteration.
    numerator = tl.zeros([BLOCK_D_V], dtype=tl.float32)
    denominator = 0.0
    for split in tl.static_range(0, NUM_SPLITS):
        stats_offset = (
            split * stride_stats_s
            + q_global * stride_stats_t
            + q_head * stride_stats_h
        )
        split_max = tl.load(partial_max_ptr + stats_offset)
        split_sum = tl.load(partial_sum_ptr + stats_offset)
        weight = tl.where(
            split_max != -float("inf"),
            tl.exp2(split_max - max_safe),
            0.0,
        )
        partial = tl.load(
            partial_out_ptr
            + split * stride_partial_s
            + q_global * stride_partial_t
            + q_head * stride_partial_h
            + off_d * stride_partial_d,
            mask=d_valid,
            other=0.0,
        )
        numerator += partial * weight
        denominator += split_sum * weight

    denominator_safe = tl.where(denominator > 0.0, denominator, 1.0)
    output = tl.where(
        denominator > 0.0,
        numerator / denominator_safe,
        0.0,
    )

    out_ptrs = (
        out_ptr
        + q_global * stride_out_t
        + q_head * stride_out_h
        + off_d * stride_out_d
    )

    tl.store(out_ptrs, output, mask=d_valid)

# Backward-compatible name used by this repository before the public wrapper
# was aligned with vLLM's operator name. Both names reference the same wrapper
# and therefore have exactly the same signature and behavior.
inkling_fa4_rel_attention_triton = inkling_fa4_rel_attention

__all__ = ["inkling_fa4_rel_attention", "inkling_fa4_rel_attention_triton"]
