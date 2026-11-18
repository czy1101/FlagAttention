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

import triton
import triton.language as tl


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
    ALLOW_DIRECT_OUTPUT: tl.constexpr = False,
    SINGLE_QUERY: tl.constexpr = False,
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
    if SINGLE_QUERY:
        scheduler_rows = Q_HEADS_PER_KV_HEAD if PACK_GQA else 1
        num_q_blocks = (scheduler_rows + BLOCK_Q - 1) // BLOCK_Q
        task_valid = (q_len > 0) & (pid_q_grid < num_q_blocks)
    else:
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
    if SINGLE_QUERY:
        q_len = tl.where(task_valid, 1, 0)
    else:
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
    if SINGLE_QUERY:
        q_local = tl.full([BLOCK_Q], 0, tl.int32)
        q_head = scheduler_head
        if PACK_GQA:
            q_head = scheduler_head * Q_HEADS_PER_KV_HEAD + scheduler_row
        kv_head = scheduler_head
    elif PACK_GQA:
        q_local = scheduler_row // Q_HEADS_PER_KV_HEAD
        q_head = scheduler_head * Q_HEADS_PER_KV_HEAD + scheduler_row % Q_HEADS_PER_KV_HEAD
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

    if SPLIT_KV and ALLOW_DIRECT_OUTPUT:
        if k_len <= BLOCK_K and split_idx > 0:
            return

    q_ptrs = q_ptr + q_global[:, None] * stride_q_t + q_head_offset * stride_q_h + off_d_q[None, :] * stride_q_d
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
    if SINGLE_QUERY:
        q_pos_min = k_len - q_len
        q_pos_max = q_pos_min
    else:
        q_pos_min = tl.min(tl.where(q_valid, q_pos, k_len), axis=0)
        q_pos_max = tl.max(tl.where(q_valid, q_pos, -1), axis=0)
    valid_k_start = 0
    valid_k_end = k_len
    if CAUSAL:
        valid_k_end = tl.minimum(valid_k_end, q_pos_max + 1)
    if LOCAL:
        valid_k_start = tl.maximum(valid_k_start, q_pos_min - WINDOW_LEFT)
        valid_k_end = tl.minimum(valid_k_end, q_pos_max + WINDOW_RIGHT + 1)
    valid_k_start = tl.maximum(valid_k_start, 0)
    valid_k_end = tl.maximum(tl.minimum(valid_k_end, k_len), 0)
    valid_block_start = valid_k_start // BLOCK_K
    valid_block_end = (valid_k_end + BLOCK_K - 1) // BLOCK_K

    num_k_blocks = (k_len + BLOCK_K - 1) // BLOCK_K
    blocks_per_split = (num_k_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    split_block_start = split_idx * blocks_per_split
    split_block_end = tl.minimum(split_block_start + blocks_per_split, num_k_blocks)
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
                block_table_ptr + seq_id * stride_bt_b + block_k * stride_bt_block,
                mask=block_k < num_k_blocks,
                other=0,
            )
            physical_block = physical_page + tl.zeros([BLOCK_K], tl.int32)
        else:
            physical_block = tl.load(
                block_table_ptr + seq_id * stride_bt_b + logical_block * stride_bt_block,
                mask=k_valid,
                other=0,
            )

        mask = q_valid[:, None] & k_valid[None, :]
        if CAUSAL:
            mask = mask & (k_local[None, :] <= q_pos[:, None])
        if LOCAL:
            mask = mask & (k_local[None, :] >= q_pos[:, None] - WINDOW_LEFT)
        if LOCAL:
            mask = mask & (k_local[None, :] <= q_pos[:, None] + WINDOW_RIGHT)
        rel_in_range = (
            (k_local[None, :] <= q_pos[:, None])
            & (k_local[None, :] > q_pos[:, None] - REL_EXTENT)
        )
        rel_ptrs = (
            rel_ptr + q_global[:, None] * stride_rel_t + q_head_offset * stride_rel_h
            + q_pos[:, None] * stride_rel_d - k_local[None, :] * stride_rel_d
        )
        rel_bias = tl.load(rel_ptrs, mask=mask & rel_in_range, other=0.0)

        k_ptrs = (
            k_ptr
            + physical_block[:, None] * stride_k_block
            + offset_in_block[:, None] * stride_k_t
            + kv_head * stride_k_h
            + off_d_q[None, :] * stride_k_d
        )
        k_value = tl.load(
            k_ptrs,
            mask=k_valid[:, None] & d_q_valid[None, :],
            other=0.0,
        )
        scores = tl.dot(q_value, tl.trans(k_value)).to(tl.float32) * softmax_scale

        # Hopper's approximate exp2 path is cheaper. Convert the *complete*
        # score, including relative bias, so the operator remains unchanged.
        scores = tl.where(
            mask,
            (scores + rel_bias.to(tl.float32)) * 1.4426950408889634,
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
        acc = tl.dot(probabilities.to(v_value.dtype), v_value, acc)
        if SINGLE_QUERY and NUM_SPLITS >= 16 and (BLOCK_D_Q != 256 or BLOCK_D_V != 512):
            m_i = m_new
        else:
            m_i = tl.where(row_has_value, m_new, m_i)

    if SPLIT_KV:
        partial_ptrs = (
            partial_out_ptr
            + split_idx * stride_partial_s
            + q_global[:, None] * stride_partial_t
            + q_head_offset * stride_partial_h
            + off_d_v[None, :] * stride_partial_d
        )
        stats_ptrs = split_idx * stride_stats_s + q_global * stride_stats_t + q_head * stride_stats_h
        stored_max = m_i
        stored_sum = l_i
        if ALLOW_DIRECT_OUTPUT:
            stored_max = tl.where(k_len <= BLOCK_K, l_i, m_i)
            stored_sum = tl.where(k_len <= BLOCK_K, -1.0, l_i)
        tl.store(partial_ptrs, acc, mask=q_valid[:, None] & d_v_valid[None, :])
        tl.store(partial_max_ptr + stats_ptrs, stored_max, mask=q_valid)
        tl.store(partial_sum_ptr + stats_ptrs, stored_sum, mask=q_valid)
    else:
        denominator = tl.where(l_i > 0.0, l_i, 1.0)
        output = acc / denominator[:, None]
        out_ptrs = (
            out_ptr + q_global[:, None] * stride_out_t + q_head_offset * stride_out_h + off_d_v[None, :] * stride_out_d
        )
        tl.store(out_ptrs, output, mask=q_valid[:, None] & d_v_valid[None, :])


@triton.jit
def _combine_split_kv_metax_kernel(
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
    ALLOW_DIRECT_OUTPUT: tl.constexpr = False,
):
    output_id = tl.program_id(0)
    q_global = output_id // NUM_HEADS
    q_head = output_id % NUM_HEADS
    off_d = tl.program_id(1) * BLOCK_D_V + tl.arange(0, BLOCK_D_V)
    d_valid = off_d < HEAD_DIM_V
    if ALLOW_DIRECT_OUTPUT:
        first_stats_offset = q_global * stride_stats_t + q_head * stride_stats_h
        first_sum = tl.load(partial_sum_ptr + first_stats_offset)
        if first_sum < 0.0:
            denominator = tl.load(partial_max_ptr + first_stats_offset)
            partial = tl.load(
                partial_out_ptr + q_global * stride_partial_t + q_head * stride_partial_h
                + off_d * stride_partial_d,
                mask=d_valid,
                other=0.0,
            )
            output = partial / tl.where(denominator > 0.0, denominator, 1.0)
            tl.store(
                out_ptr + q_global * stride_out_t + q_head * stride_out_h + off_d * stride_out_d,
                output,
                mask=d_valid,
            )
            return
    split = tl.arange(0, SPLITS_PAD)
    split_valid = split < NUM_SPLITS
    stats_offset = split * stride_stats_s + q_global * stride_stats_t + q_head * stride_stats_h
    split_max = tl.load(partial_max_ptr + stats_offset, mask=split_valid, other=-float("inf"))
    split_sum = tl.load(partial_sum_ptr + stats_offset, mask=split_valid, other=0.0)
    global_max = tl.max(split_max, axis=0)
    max_safe = tl.where(global_max != -float("inf"), global_max, 0.0)
    split_active = split_valid & (split_max != -float("inf")) & (split_sum > 0.0)
    weight = tl.where(split_active, tl.exp2(split_max - max_safe), 0.0)
    denominator = tl.sum(split_sum * weight, axis=0)
    partial_ptrs = (
        partial_out_ptr
        + split[:, None] * stride_partial_s
        + q_global * stride_partial_t
        + q_head * stride_partial_h
        + off_d[None, :] * stride_partial_d
    )
    partial = tl.load(partial_ptrs, mask=split_active[:, None] & d_valid[None, :], other=0.0)
    numerator = tl.sum(partial * weight[:, None], axis=0)
    denominator_safe = tl.where(denominator > 0.0, denominator, 1.0)
    output = tl.where(denominator > 0.0, numerator / denominator_safe, 0.0)
    out_ptrs = out_ptr + q_global * stride_out_t + q_head * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, output, mask=d_valid)
