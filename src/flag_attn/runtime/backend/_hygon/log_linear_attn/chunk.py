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

"""Forward-only Hygon implementation of dense Log-Linear Attention.

Uses chunk-local attention and hierarchical KV states with standard Triton loads.
"""

import math
from functools import cache

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64
BLOCK_KEY = 64
BLOCK_VALUE = 64
BLOCK_T = tl.constexpr(64)


@cache
def _level_lut(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    lut = torch.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.int32, device=device)
    rows = torch.arange(CHUNK_SIZE, device=device)[:, None]
    cols = torch.arange(CHUNK_SIZE, device=device)[None, :]
    for level in range(1, int(math.log2(CHUNK_SIZE)) + 1):
        half = 1 << (level - 1)
        anchor = rows - rows % half
        selected = (rows % (2 * half) >= half) & (cols + half >= anchor) & (cols < anchor)
        lut = torch.where(selected, level, lut)
    return lut


def _chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    batch, sequence, heads = g.shape
    return g.view(batch, sequence // CHUNK_SIZE, CHUNK_SIZE, heads).cumsum(dim=2).view_as(g)


@triton.jit
def _qk_fwd_kernel(
    q, k, scores, T: tl.constexpr, K: tl.constexpr, NT: tl.constexpr, NK: tl.constexpr, BK: tl.constexpr
):
    chunk = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    key_tile = tl.program_id(2)
    rows = tl.arange(0, 64)
    keys = key_tile * BK + tl.arange(0, BK)
    offsets = (batch * T + chunk * 64 + rows[:, None]) * K + keys[None, :]
    b_q = tl.load(q + offsets)
    b_k = tl.load(k + offsets)
    score = tl.dot(b_q, tl.trans(b_k))
    target = ((batch * NT + chunk) * NK + key_tile) * 4096
    tl.store(scores + target + rows[:, None] * 64 + rows[None, :], score)


@triton.jit
def _local_fwd_kernel(
    q,
    k,
    v,
    g,
    raw_g,
    level_scales,
    level_lut,
    local_output,
    states,
    chunk_gates,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    FUSED_SCAN: tl.constexpr,
    DIRECT_OUTPUT: tl.constexpr = False,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t

    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    k_ptrs = k + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    v_ptrs = v + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    b_q = tl.load(q_ptrs).to(tl.float32)
    b_k = tl.load(k_ptrs).to(tl.float32)
    b_v = tl.load(v_ptrs).to(tl.float32)
    if FUSED_SCAN:
        gate_ptrs = (batch_index * T + token_offsets) * H + head_index
        b_g = tl.cumsum(tl.load(raw_g + gate_ptrs), axis=0)
        if not DIRECT_OUTPUT:
            tl.store(g + gate_ptrs, b_g)
    else:
        b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

    row = offsets_t[:, None]
    col = offsets_t[None, :]
    b_level = tl.load(level_lut + row * BLOCK_T + col)
    scale_ptrs = level_scales + ((batch_index * T + token_offsets[:, None]) * H + head_index) * L + b_level
    b_level_scale = tl.load(scale_ptrs).to(tl.float32)

    b_score = tl.dot(b_q.to(q.dtype.element_ty), tl.trans(b_k.to(k.dtype.element_ty)))
    b_score *= tl.exp(b_g[:, None] - b_g[None, :]) * b_level_scale
    b_score = tl.where(row >= col, b_score, 0.0)
    b_output = tl.dot(b_score.to(v.dtype.element_ty), b_v.to(v.dtype.element_ty))

    output_ptrs = (
        local_output
        + ((((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V)
        + offsets_v[None, :]
    )
    if DIRECT_OUTPUT and T == 128:  # noqa: SIM102 - keep the Triton compile-time guard separate
        # The second query chunk has exactly one preceding leaf.
        if chunk_index == 1:
            k0_ptrs = k + (batch_index * T + offsets_t[:, None]) * K + offsets_k[None, :]
            v0_ptrs = v + ((batch_index * T + offsets_t[:, None]) * H + head_index) * V + offsets_v[None, :]
            gate0_ptrs = raw_g + (batch_index * T + offsets_t) * H + head_index
            end_gate0 = tl.cumsum(tl.load(gate0_ptrs), axis=0)
            last0 = tl.sum(tl.where(offsets_t == 63, end_gate0, 0.0), axis=0)
            v0 = tl.load(v0_ptrs).to(tl.float32) * tl.exp(last0 - end_gate0)[:, None]
            state0 = tl.dot(tl.trans(tl.load(k0_ptrs)), v0.to(v.dtype.element_ty))
            level7 = tl.load(level_scales + ((batch_index * T + token_offsets) * H + head_index) * L + 7).to(tl.float32)
            current = b_output.to(local_output.dtype.element_ty).to(tl.float32)
            current += (
                tl.dot(b_q.to(q.dtype.element_ty), state0.to(q.dtype.element_ty)) * (level7 * tl.exp(b_g))[:, None]
            )
            b_output = current
    tl.store(output_ptrs, b_output.to(local_output.dtype.element_ty))

    if not DIRECT_OUTPUT:
        if FUSED_SCAN:
            gate_last = tl.sum(tl.where(offsets_t == BLOCK_T - 1, b_g, 0.0), axis=0)
        else:
            gate_last = tl.load(g + (batch_index * T + chunk_index * BLOCK_T + BLOCK_T - 1) * H + head_index)
        b_v_scaled = b_v * tl.exp(gate_last - b_g)[:, None]
        b_state = tl.dot(
            tl.trans(b_k.to(k.dtype.element_ty)),
            b_v_scaled.to(v.dtype.element_ty),
        )
        state_base = (
            (((batch_index * NT + chunk_index) * H + head_index) * K) * V + offsets_k[:, None] * V + offsets_v[None, :]
        )
        tl.store(states + state_base, b_state)
        if key_tile == 0 and value_tile == 0:
            tl.store(
                chunk_gates + (batch_index * NT + chunk_index) * H + head_index,
                gate_last,
            )


@triton.jit
def _local_shared_fwd_kernel(
    q,
    k,
    v,
    g,
    raw_g,
    shared_scores,
    level_scales,
    level_lut,
    local_output,
    states,
    chunk_gates,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
    FUSED_SCAN: tl.constexpr,
    SHARED_QK: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t

    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    k_ptrs = k + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    v_ptrs = v + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    b_q = tl.load(q_ptrs).to(tl.float32)
    b_k = tl.load(k_ptrs).to(tl.float32)
    b_v = tl.load(v_ptrs).to(tl.float32)
    if FUSED_SCAN:
        gate_ptrs = (batch_index * T + token_offsets) * H + head_index
        b_g = tl.cumsum(tl.load(raw_g + gate_ptrs), axis=0)
        tl.store(g + gate_ptrs, b_g)
    else:
        b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

    row = offsets_t[:, None]
    col = offsets_t[None, :]
    b_level = tl.load(level_lut + row * BLOCK_T + col)
    scale_ptrs = level_scales + ((batch_index * T + token_offsets[:, None]) * H + head_index) * L + b_level
    b_level_scale = tl.load(scale_ptrs).to(tl.float32)

    if SHARED_QK:
        score_base = ((batch_index * NT + chunk_index) * NK + key_tile) * 4096
        b_score = tl.load(shared_scores + score_base + row * 64 + col)
    else:
        b_score = tl.dot(b_q.to(q.dtype.element_ty), tl.trans(b_k.to(k.dtype.element_ty)))
    b_score *= tl.exp(b_g[:, None] - b_g[None, :]) * b_level_scale
    b_score = tl.where(row >= col, b_score, 0.0)
    b_output = tl.dot(b_score.to(v.dtype.element_ty), b_v.to(v.dtype.element_ty))

    output_ptrs = (
        local_output
        + ((((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V)
        + offsets_v[None, :]
    )
    tl.store(output_ptrs, b_output.to(local_output.dtype.element_ty))

    if FUSED_SCAN:
        gate_last = tl.sum(tl.where(offsets_t == BLOCK_T - 1, b_g, 0.0), axis=0)
    else:
        gate_last = tl.load(g + (batch_index * T + chunk_index * BLOCK_T + BLOCK_T - 1) * H + head_index)
    b_v_scaled = b_v * tl.exp(gate_last - b_g)[:, None]
    b_state = tl.dot(
        tl.trans(b_k.to(k.dtype.element_ty)),
        b_v_scaled.to(v.dtype.element_ty),
    )
    state_base = (
        (((batch_index * NT + chunk_index) * H + head_index) * K) * V + offsets_k[:, None] * V + offsets_v[None, :]
    )
    tl.store(states + state_base, b_state)
    if key_tile == 0 and value_tile == 0:
        tl.store(
            chunk_gates + (batch_index * NT + chunk_index) * H + head_index,
            gate_last,
        )


@triton.jit
def _merge_states_kernel(
    states,
    chunk_gates,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    LEVEL: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    STATE_OFFSETS: tl.constexpr = (),
    query_states=None,
    TOP_QUERY: tl.constexpr = -1,
):
    node_index = tl.program_id(2).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(0)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    right_end = (node_index + 1) * BLOCK - 1
    left_end = right_end - HALF
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)

    previous_level_base = tl.full((), LEVEL - 1, tl.int64) * B * NT * H * K * V
    left_base = (
        previous_level_base
        + (((batch_index * NT + left_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    right_base = (
        previous_level_base
        + (((batch_index * NT + right_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    if len(STATE_OFFSETS) > 0:
        pitch = H * K * V
        level_base = tl.full((), STATE_OFFSETS[LEVEL - 1], tl.int64)
        position = level_base + batch_index * (NT // HALF) + 2 * node_index
        left_base = position * pitch + head_index * K * V + offsets_k[:, None] * V + offsets_v[None, :]
        right_base = left_base + pitch
    left_state = tl.load(states + left_base)
    right_state = tl.load(states + right_base)

    offsets_chunk = tl.arange(0, HALF)
    gate_ptrs = chunk_gates + (batch_index * NT + right_end - HALF + 1 + offsets_chunk) * H + head_index
    right_decay = tl.exp(tl.sum(tl.load(gate_ptrs), axis=0))
    merged = left_state * right_decay + right_state

    output_level_base = tl.full((), LEVEL, tl.int64) * B * NT * H * K * V
    output_base = (
        output_level_base
        + (((batch_index * NT + right_end) * H + head_index) * K) * V
        + offsets_k[:, None] * V
        + offsets_v[None, :]
    )
    if len(STATE_OFFSETS) > 0:
        position = tl.full((), STATE_OFFSETS[LEVEL], tl.int64) + batch_index * (NT // BLOCK) + node_index
        output_base = position * H * K * V + head_index * K * V + offsets_k[:, None] * V + offsets_v[None, :]
    if TOP_QUERY >= 0:
        if node_index == 0:
            query_ptr = (
                ((TOP_QUERY * B + batch_index) * H + head_index) * K * V + offsets_k[:, None] * V + offsets_v[None, :]
            )
            tl.store(query_states + query_ptr, merged.to(tl.bfloat16))
    else:
        tl.store(states + output_base, merged)


@triton.jit
def _merge_two_levels_kernel(
    states,
    chunk_gates,
    query_states,
    QUERY_BASE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    LEVEL: tl.constexpr,
    HALF: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    STATE_OFFSETS: tl.constexpr,
    TOP_QUERY: tl.constexpr,
    LEVEL2_QUERY: tl.constexpr,
):
    tile = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    node = tl.program_id(2).to(tl.int64)
    batch = bh // H
    head = bh % H
    kk = (tile // NV) * BK + tl.arange(0, BK)
    vv = (tile % NV) * BV + tl.arange(0, BV)
    end = (node + 1) * (4 * HALF) - 1
    stride = B * NT * H * K * V
    offsets = kk[:, None] * V + vv[None, :]
    root = ((batch * NT + end) * H + head) * K * V + offsets
    step = HALF * H * K * V
    base = tl.full((), LEVEL - 1, tl.int64) * stride + root
    if len(STATE_OFFSETS) > 0:
        position = tl.full((), STATE_OFFSETS[LEVEL - 1], tl.int64) + batch * (NT // HALF) + 4 * node + 3
        base = (position * H + head) * K * V + offsets
        step = H * K * V
    a = tl.load(states + base - 3 * step)
    b = tl.load(states + base - 2 * step)
    c = tl.load(states + base - step)
    d = tl.load(states + base)
    if LEVEL2_QUERY >= 0:
        # At LEVEL == 3, the four inputs are consecutive level-2 nodes.
        # The first and third are the ones selected by the query decomposition.
        query_index = LEVEL2_QUERY + 2 * node
        query_ptr = ((query_index * B + batch) * H + head) * K * V + offsets
        tl.store(query_states + query_ptr, a.to(tl.bfloat16))
        tl.store(query_states + query_ptr + B * H * K * V, c.to(tl.bfloat16))
    i = tl.arange(0, HALF)
    g1 = chunk_gates + (batch * NT + end - 3 * HALF + 1 + i) * H + head
    g2 = chunk_gates + (batch * NT + end - HALF + 1 + i) * H + head
    left = a * tl.exp(tl.sum(tl.load(g1), axis=0)) + b
    right = c * tl.exp(tl.sum(tl.load(g2), axis=0)) + d
    mid = tl.full((), LEVEL, tl.int64) * stride + root
    if QUERY_BASE >= 0:
        query_index = QUERY_BASE + node
        query_ptr = ((query_index * B + batch) * H + head) * K * V + offsets
        tl.store(query_states + query_ptr, left.to(tl.bfloat16))
    else:
        tl.store(states + mid - 2 * step, left)
        tl.store(states + mid, right)
    j = tl.arange(0, 2 * HALF)
    g3 = chunk_gates + (batch * NT + end - 2 * HALF + 1 + j) * H + head
    top = left * tl.exp(tl.sum(tl.load(g3), axis=0)) + right
    top_base = tl.full((), LEVEL + 1, tl.int64) * stride + root
    if len(STATE_OFFSETS) > 0:
        position = tl.full((), STATE_OFFSETS[LEVEL + 1], tl.int64) + batch * (NT // (4 * HALF)) + node
        top_base = (position * H + head) * K * V + offsets
    if TOP_QUERY >= 0:
        if node == 0:
            query_ptr = ((TOP_QUERY * B + batch) * H + head) * K * V + offsets
            tl.store(query_states + query_ptr, top.to(tl.bfloat16))
    else:
        tl.store(states + top_base, top)


@triton.jit
def _inter_fwd_kernel(
    q,
    g,
    level_scales,
    local_output,
    states,
    chunk_gates,
    output,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    key_tile = tile_index // NV
    value_tile = tile_index % NV
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_k = key_tile * BK + tl.arange(0, BK)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t

    q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
    local_ptrs = (
        local_output
        + ((((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V)
        + offsets_v[None, :]
    )
    b_q = tl.load(q_ptrs).to(tl.float32)
    b_output = tl.load(local_ptrs).to(tl.float32)
    b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

    current_end = chunk_index - 1
    later_decay = tl.zeros([], dtype=tl.float32)
    for level in tl.static_range(NUM_LEVELS):
        block_size = 1 << level
        if (chunk_index & block_size) != 0:
            level_base = tl.full((), level, tl.int64) * B * NT * H * K * V
            state_base = (
                level_base
                + (((batch_index * NT + current_end) * H + head_index) * K) * V
                + offsets_k[:, None] * V
                + offsets_v[None, :]
            )
            b_state = tl.load(states + state_base)
            scale_ptrs = level_scales + ((batch_index * T + token_offsets) * H + head_index) * L + 7 + level
            b_scale = tl.load(scale_ptrs).to(tl.float32)
            # Apply the row scale after the dot to avoid rounding scaled Q
            # back to BF16; isolated inter-level inputs otherwise exceed the
            # public reference tolerance on HCU.
            b_row_scale = b_scale * tl.exp(b_g + later_decay)
            b_output += (
                tl.dot(
                    b_q.to(q.dtype.element_ty),
                    b_state.to(q.dtype.element_ty),
                )
                * b_row_scale[:, None]
            )

            offsets_chunk = tl.arange(0, MAX_BLOCK)
            gate_ptrs = chunk_gates + (batch_index * NT + current_end - block_size + 1 + offsets_chunk) * H + head_index
            later_decay += tl.sum(
                tl.load(
                    gate_ptrs,
                    mask=offsets_chunk < block_size,
                    other=0.0,
                ),
                axis=0,
            )
            current_end -= block_size

    output_ptrs = (
        output
        + ((((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V)
        + offsets_v[None, :]
    )
    tl.store(output_ptrs, b_output.to(output.dtype.element_ty))


@triton.jit
def _inter_reduce_fwd_kernel(
    q,
    g,
    level_scales,
    local_output,
    states,
    chunk_gates,
    output,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    value_tile = tile_index
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t
    accumulator = tl.zeros((BLOCK_T, BV), dtype=tl.float32)
    for key_tile in range(NK):
        offsets_k = key_tile * BK + tl.arange(0, BK)

        q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
        local_ptrs = (
            local_output
            + ((((batch_index * T + token_offsets[:, None]) * H + head_index) * NK + key_tile) * V)
            + offsets_v[None, :]
        )
        b_q = tl.load(q_ptrs).to(tl.float32)
        b_output = tl.load(local_ptrs).to(tl.float32)
        b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)

        current_end = chunk_index - 1
        later_decay = tl.zeros([], dtype=tl.float32)
        for level in tl.static_range(NUM_LEVELS):
            block_size = 1 << level
            if (chunk_index & block_size) != 0:
                level_base = tl.full((), level, tl.int64) * B * NT * H * K * V
                state_base = (
                    level_base
                    + (((batch_index * NT + current_end) * H + head_index) * K) * V
                    + offsets_k[:, None] * V
                    + offsets_v[None, :]
                )
                b_state = tl.load(states + state_base)
                scale_ptrs = level_scales + ((batch_index * T + token_offsets) * H + head_index) * L + 7 + level
                b_scale = tl.load(scale_ptrs).to(tl.float32)
                # Apply the row scale after the dot to avoid rounding scaled Q
                # back to BF16; isolated inter-level inputs otherwise exceed the
                # public reference tolerance on HCU.
                b_row_scale = b_scale * tl.exp(b_g + later_decay)
                b_output += (
                    tl.dot(
                        b_q.to(q.dtype.element_ty),
                        b_state.to(q.dtype.element_ty),
                    )
                    * b_row_scale[:, None]
                )

                offsets_chunk = tl.arange(0, MAX_BLOCK)
                gate_ptrs = (
                    chunk_gates + (batch_index * NT + current_end - block_size + 1 + offsets_chunk) * H + head_index
                )
                later_decay += tl.sum(
                    tl.load(
                        gate_ptrs,
                        mask=offsets_chunk < block_size,
                        other=0.0,
                    ),
                    axis=0,
                )
                current_end -= block_size

        accumulator += b_output.to(output.dtype.element_ty).to(tl.float32)
    output_ptrs = output + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    tl.store(output_ptrs, accumulator.to(output.dtype.element_ty))


@triton.jit
def _fullk_local_fwd_kernel(
    q,
    k,
    v,
    g,
    level_scales,
    level_lut,
    local_output,
    shared_scores,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SHARED_FULL_QK: tl.constexpr,
):
    chunk = tl.program_id(0).to(tl.int64)
    bh = tl.program_id(1).to(tl.int64)
    batch = bh // H
    head = bh % H
    value_tile = tl.program_id(2)
    t = chunk * 64 + tl.arange(0, 64)
    vv = value_tile * BV + tl.arange(0, BV)
    if SHARED_FULL_QK:
        rows = tl.arange(0, 64)
        score_base = (batch * (T // 64) + chunk) * 4096
        score = tl.load(shared_scores + score_base + rows[:, None] * 64 + rows[None, :])
    else:
        score = tl.zeros((64, 64), dtype=tl.float32)
        for kt in range(K // BK):
            kk = kt * BK + tl.arange(0, BK)
            ptrs = (batch * T + t[:, None]) * K + kk[None, :]
            qq = tl.load(q + ptrs)
            key = tl.load(k + ptrs)
            score = tl.dot(qq, tl.trans(key), score)
    gg = tl.load(g + (batch * T + t) * H + head)
    rr = tl.arange(0, 64)[:, None]
    cc = tl.arange(0, 64)[None, :]
    level = tl.load(level_lut + rr * 64 + cc)
    scale = tl.load(level_scales + ((batch * T + t[:, None]) * H + head) * L + level).to(tl.float32)
    score *= tl.exp(gg[:, None] - gg[None, :]) * scale
    score = tl.where(rr >= cc, score, 0.0)
    vp = ((batch * T + t[:, None]) * H + head) * V + vv[None, :]
    values = tl.load(v + vp)
    result = tl.dot(score.to(v.dtype.element_ty), values)
    tl.store(local_output + vp, result.to(local_output.dtype.element_ty))


@triton.jit
def _leaf_only_fwd_kernel(
    k,
    v,
    g,
    states,
    chunk_gates,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    NT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
    HEAD_FIRST: tl.constexpr = False,
):
    if HEAD_FIRST:
        bh = tl.program_id(0).to(tl.int64)
        chunk = tl.program_id(1).to(tl.int64)
    else:
        chunk = tl.program_id(0).to(tl.int64)
        bh = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(2)
    batch = bh // H
    head = bh % H
    kk = (tile // NV) * BK + tl.arange(0, BK)
    vv = (tile % NV) * BV + tl.arange(0, BV)
    t = chunk * 64 + tl.arange(0, 64)
    key = tl.load(k + (batch * T + t[:, None]) * K + kk[None, :])
    values = tl.load(v + ((batch * T + t[:, None]) * H + head) * V + vv[None, :]).to(tl.float32)
    gg = tl.load(g + (batch * T + t) * H + head)
    last = tl.load(g + (batch * T + chunk * 64 + 63) * H + head)
    values *= tl.exp(last - gg)[:, None]
    state = tl.dot(tl.trans(key), values.to(v.dtype.element_ty))
    addr = ((batch * NT + chunk) * H + head) * K * V + kk[:, None] * V + vv[None, :]
    tl.store(states + addr, state)
    if tile == 0:
        tl.store(chunk_gates + (batch * NT + chunk) * H + head, last)


@triton.jit
def _inter_fullk_fwd_kernel(
    q,
    g,
    level_scales,
    local_output,
    states,
    query_states,
    QUERY_OFFSETS: tl.constexpr,
    STATE_OFFSETS: tl.constexpr,
    chunk_gates,
    output,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    L: tl.constexpr,
    NT: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    MAX_BLOCK: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NK: tl.constexpr,
    NV: tl.constexpr,
):
    chunk_index = tl.program_id(0).to(tl.int64)
    batch_head = tl.program_id(1).to(tl.int64)
    tile_index = tl.program_id(2)
    value_tile = tile_index
    batch_index = batch_head // H
    head_index = batch_head % H

    offsets_t = tl.arange(0, BLOCK_T)
    offsets_v = value_tile * BV + tl.arange(0, BV)
    token_offsets = chunk_index * BLOCK_T + offsets_t
    local_ptrs = local_output + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    accumulator = tl.load(local_ptrs).to(tl.float32)
    # Reuse FP32 row scales across K tiles; preserve the original decay order.
    b_g = tl.load(g + (batch_index * T + token_offsets) * H + head_index)
    row_scales = ()
    current_end = chunk_index - 1
    later_decay = tl.zeros([], dtype=tl.float32)
    for level in tl.static_range(NUM_LEVELS):
        block_size = 1 << level
        row_scale = tl.zeros((BLOCK_T,), dtype=tl.float32)
        if (chunk_index & block_size) != 0:
            scale_ptrs = level_scales + ((batch_index * T + token_offsets) * H + head_index) * L + 7 + level
            b_scale = tl.load(scale_ptrs).to(tl.float32)
            row_scale = b_scale * tl.exp(b_g + later_decay)
            offsets_chunk = tl.arange(0, MAX_BLOCK)
            gate_ptrs = chunk_gates + (batch_index * NT + current_end - block_size + 1 + offsets_chunk) * H + head_index
            later_decay += tl.sum(
                tl.load(
                    gate_ptrs,
                    mask=offsets_chunk < block_size,
                    other=0.0,
                ),
                axis=0,
            )
            current_end -= block_size
        row_scales += (row_scale,)
    for key_tile in range(NK):
        offsets_k = key_tile * BK + tl.arange(0, BK)

        q_ptrs = q + (batch_index * T + token_offsets[:, None]) * K + offsets_k[None, :]
        b_q = tl.load(q_ptrs).to(tl.float32)
        b_output = tl.zeros((BLOCK_T, BV), dtype=tl.float32)

        for level in tl.static_range(NUM_LEVELS):
            block_size = 1 << level
            if (chunk_index & block_size) != 0:
                current_end = chunk_index - 1 - (chunk_index & (block_size - 1))
                level_base = tl.full((), level, tl.int64) * B * NT * H * K * V
                state_base = (
                    level_base
                    + (((batch_index * NT + current_end) * H + head_index) * K) * V
                    + offsets_k[:, None] * V
                    + offsets_v[None, :]
                )
                if QUERY_OFFSETS[level] >= 0:
                    node = (current_end + 1) // (2 * block_size)
                    query_index = QUERY_OFFSETS[level] + node
                    query_ptr = (
                        ((query_index * B + batch_index) * H + head_index) * K * V
                        + offsets_k[:, None] * V
                        + offsets_v[None, :]
                    )
                    b_state = tl.load(query_states + query_ptr)
                else:
                    if len(STATE_OFFSETS) > 0:
                        position = (
                            tl.full((), STATE_OFFSETS[level], tl.int64)
                            + batch_index * (NT // block_size)
                            + current_end // block_size
                        )
                        state_base = (position * H + head_index) * K * V + offsets_k[:, None] * V + offsets_v[None, :]
                    b_state = tl.load(states + state_base)
                b_row_scale = row_scales[level]
                b_output += (
                    tl.dot(
                        b_q.to(q.dtype.element_ty),
                        b_state.to(q.dtype.element_ty),
                    )
                    * b_row_scale[:, None]
                )

        accumulator += b_output
    output_ptrs = output + ((batch_index * T + token_offsets[:, None]) * H + head_index) * V + offsets_v[None, :]
    tl.store(output_ptrs, accumulator.to(output.dtype.element_ty))


def _validate_inputs(q, k, v, g, level_scales):
    tensors = (q, k, v, g, level_scales)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("chunk_log_linear_attn requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError("chunk_log_linear_attn is forward-only and does not support autograd")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("chunk_log_linear_attn supports BF16 q, k, and v")
    if g.dtype != torch.float32:
        raise ValueError("g must use float32")
    if level_scales.dtype != q.dtype:
        raise ValueError("level_scales must have the same dtype as q")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with matching q/k shapes")
    if q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("chunk_log_linear_attn requires one shared Q/K head")
    batch, sequence, _, key_dim = q.shape
    heads, value_dim = v.shape[2], v.shape[-1]
    if v.shape[:2] != (batch, sequence):
        raise ValueError("v must have shape [B, T, H, V]")
    if g.shape != (batch, sequence, heads):
        raise ValueError("g must have shape [B, T, H]")
    if sequence % CHUNK_SIZE != 0 or sequence & (sequence - 1):
        raise ValueError("sequence must be a power of two and divisible by 64")
    if key_dim not in (64, 128, 256) or value_dim not in (64, 128, 256):
        raise ValueError("the Hygon path supports K/V in {64, 128, 256}")
    if level_scales.ndim != 4:
        raise ValueError("level_scales must be rank 4")
    levels = level_scales.shape[-1]
    if level_scales.shape[:3] != (batch, sequence, heads):
        raise ValueError("level_scales must have shape [B, T, H, L]")
    if levels != math.ceil(math.log2(sequence)) + 1:
        raise ValueError("invalid level_scales level count")
    return batch, sequence, heads, key_dim, value_dim, levels


def _prepare_forward(
    q,
    k,
    v,
    g,
    level_scales,
    block_value=BLOCK_VALUE,
    local_warps=4,
    merge_warps=4,
    inter_warps=4,
    reduction_warps=4,
    num_stages=2,
):
    batch, sequence, heads, key_dim, value_dim, levels = _validate_inputs(q, k, v, g, level_scales)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    level_scales = level_scales.contiguous()
    if key_dim == value_dim == block_value == 64 and sequence in (64, 128):
        output = torch.empty_like(v)
        level_lut = _level_lut(q.device.index or 0)

        def run_direct():
            _local_fwd_kernel[(sequence // 64, batch * heads, 1)](
                q,
                k,
                v,
                g,
                g,
                level_scales,
                level_lut,
                output,
                v,
                g,
                T=sequence,
                H=heads,
                K=64,
                V=64,
                L=levels,
                NT=sequence // 64,
                BK=64,
                BV=64,
                NK=1,
                NV=1,
                FUSED_SCAN=True,
                DIRECT_OUTPUT=True,
                num_warps=local_warps,
                num_stages=num_stages,
            )

        return run_direct, output
    if key_dim == 256 and value_dim == 256 and 512 <= sequence <= 2048 and batch * heads >= 32:
        block_value = 128
    num_chunks = sequence // CHUNK_SIZE
    num_levels = max(1, math.ceil(math.log2(num_chunks)))
    num_key_tiles = key_dim // BLOCK_KEY
    num_value_tiles = value_dim // block_value
    # The separate leaf launch pays off only with enough work. Keep the
    # existing shared-QK path and low-parallelism D128 inputs on their old path.
    high_head_fullk = key_dim == value_dim == 128 and heads in (64, 96) and batch * num_chunks >= 128
    shared_full_qk = high_head_fullk
    fullk = high_head_fullk or (
        key_dim > 64
        and batch * heads * sequence >= 32768
        and (key_dim != 128 or (batch * heads >= 32 and heads * num_value_tiles < 64))
    )
    wide_leaf = fullk and key_dim == value_dim == 256
    leaf_block_key = 256 if wide_leaf else BLOCK_KEY
    leaf_block_value = 64 if wide_leaf else block_value
    leaf_value_tiles = value_dim // leaf_block_value
    leaf_warps = 4 if wide_leaf else local_warps
    num_tiles = num_key_tiles * num_value_tiles
    inter_block_value = min(block_value, BLOCK_VALUE)
    inter_value_tiles = value_dim // inter_block_value
    shared_qk = not fullk and key_dim == 128 and heads * num_value_tiles >= 64
    score_tiles = 1 if shared_full_qk else num_key_tiles
    shared_scores = (
        torch.empty((batch, num_chunks, score_tiles, 64, 64), device=v.device, dtype=torch.float32)
        if shared_qk or shared_full_qk
        else q
    )
    fused_scan = key_dim == 64 and value_dim == 64 and block_value == 64
    local_g = torch.empty_like(g) if fused_scan else _chunk_local_cumsum(g)
    level_lut = _level_lut(q.device.index or 0)
    partial_shape = (
        batch,
        sequence,
        heads,
        num_key_tiles,
        value_dim,
    )
    local_output = torch.empty_like(v) if fullk else torch.empty(partial_shape, device=v.device, dtype=v.dtype)
    output_partial = torch.empty_like(local_output) if num_key_tiles == 1 else None
    output = output_partial.squeeze(3) if num_key_tiles == 1 else torch.empty_like(v)
    chunk_gates = torch.empty(batch, num_chunks, heads, device=v.device, dtype=torch.float32)

    query_offsets = [-1] * num_levels
    query_count = 0
    if fullk and key_dim == value_dim == 256:
        for level in range(1, num_levels, 2):
            if level + 1 < num_levels:
                query_offsets[level] = query_count
                query_count += num_chunks // (2 << level)
    # The last level has two nodes. Only node zero is queried, and
    # neither node is consumed by a subsequent FP32 merge.
    if query_count:
        query_offsets[-1] = query_count
        query_count += 1
    query_offsets = tuple(query_offsets)
    state_offsets = ()
    if query_count:
        offsets = []
        node_count = 0
        for level, query_offset in enumerate(query_offsets):
            offsets.append(node_count if query_offset < 0 else -1)
            if query_offset < 0:
                node_count += batch * (num_chunks >> level)
        state_offsets = tuple(offsets)
        states = torch.empty((node_count, heads, key_dim, value_dim), device=v.device, dtype=torch.float32)
    else:
        states = torch.empty(
            num_levels,
            batch,
            num_chunks,
            heads,
            key_dim,
            value_dim,
            device=v.device,
            dtype=torch.float32,
        )
    level2_query = -1
    if query_count and num_levels >= 5:
        # Keep the FP32 layer allocated for merge. Publish its BF16 copy only
        # during the two-level merge that loads it for the last time.
        level2_query = query_count
        query_count += num_chunks // 8
        query_offsets = query_offsets[:2] + (level2_query,) + query_offsets[3:]
    query_states = (
        torch.empty((query_count, batch, heads, key_dim, value_dim), device=v.device, dtype=torch.bfloat16)
        if query_count
        else states
    )

    def run():
        if shared_qk or shared_full_qk:
            _qk_fwd_kernel[(num_chunks, batch, score_tiles)](
                q,
                k,
                shared_scores,
                T=sequence,
                K=key_dim,
                NT=num_chunks,
                NK=score_tiles,
                BK=key_dim if shared_full_qk else BLOCK_KEY,
                num_warps=local_warps,
                num_stages=num_stages,
            )
        if fullk:
            _fullk_local_fwd_kernel[(num_chunks, batch * heads, num_value_tiles)](
                q,
                k,
                v,
                local_g,
                level_scales,
                level_lut,
                local_output,
                shared_scores,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                L=levels,
                BK=BLOCK_KEY,
                BV=block_value,
                SHARED_FULL_QK=shared_full_qk,
                num_warps=local_warps,
                num_stages=num_stages,
            )
            # Q/K are shared across heads. Group the states using the same K
            # tile before advancing to the next chunk.
            leaf_head_first = key_dim == value_dim == 128 and (batch * heads) % 8 == 0
            leaf_tiles = (key_dim // leaf_block_key) * leaf_value_tiles
            leaf_grid = (
                (batch * heads, num_chunks, leaf_tiles) if leaf_head_first else (num_chunks, batch * heads, leaf_tiles)
            )
            _leaf_only_fwd_kernel[leaf_grid](
                k,
                v,
                local_g,
                states,
                chunk_gates,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                NT=num_chunks,
                BK=leaf_block_key,
                BV=leaf_block_value,
                NV=leaf_value_tiles,
                HEAD_FIRST=leaf_head_first,
                num_warps=leaf_warps,
                num_stages=num_stages,
            )
        elif shared_qk:
            _local_shared_fwd_kernel[(num_chunks, batch * heads, num_tiles)](
                q,
                k,
                v,
                local_g,
                g,
                shared_scores,
                level_scales,
                level_lut,
                local_output,
                states,
                chunk_gates,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                L=levels,
                NT=num_chunks,
                BK=BLOCK_KEY,
                BV=block_value,
                NK=num_key_tiles,
                NV=num_value_tiles,
                FUSED_SCAN=fused_scan,
                SHARED_QK=True,
                num_warps=8,
                num_stages=num_stages,
            )
        else:
            _local_fwd_kernel[(num_chunks, batch * heads, num_tiles)](
                q,
                k,
                v,
                local_g,
                g,
                level_scales,
                level_lut,
                local_output,
                states,
                chunk_gates,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                L=levels,
                NT=num_chunks,
                BK=BLOCK_KEY,
                BV=block_value,
                NK=num_key_tiles,
                NV=num_value_tiles,
                FUSED_SCAN=fused_scan,
                num_warps=local_warps,
                num_stages=num_stages,
            )
        use_d64_pair = key_dim == value_dim == 64 and block_value == 64 and 512 <= sequence <= 2048
        first_pair_level = 1
        if key_dim == value_dim == 256 or use_d64_pair:
            for level in range(1, num_levels):
                if level >= first_pair_level and (level - first_pair_level) % 2 == 0 and level + 1 < num_levels:
                    half = 1 << (level - 1)
                    _merge_two_levels_kernel[(num_tiles, batch * heads, num_chunks // (4 * half))](
                        states,
                        chunk_gates,
                        query_states,
                        query_offsets[level],
                        B=batch,
                        H=heads,
                        K=key_dim,
                        V=value_dim,
                        NT=num_chunks,
                        LEVEL=level,
                        HALF=half,
                        BK=BLOCK_KEY,
                        BV=block_value,
                        NV=num_value_tiles,
                        STATE_OFFSETS=state_offsets,
                        TOP_QUERY=query_offsets[level + 1] if level + 1 == num_levels - 1 else -1,
                        LEVEL2_QUERY=level2_query if level == 3 else -1,
                        num_warps=merge_warps,
                        num_stages=num_stages,
                    )
                elif level > first_pair_level and (level - first_pair_level) % 2 == 1:
                    continue
                else:
                    block = 1 << level
                    half = block >> 1
                    _merge_states_kernel[(num_tiles, batch * heads, num_chunks // block)](
                        states,
                        chunk_gates,
                        B=batch,
                        H=heads,
                        K=key_dim,
                        V=value_dim,
                        NT=num_chunks,
                        LEVEL=level,
                        HALF=half,
                        BLOCK=block,
                        BK=BLOCK_KEY,
                        BV=block_value,
                        NV=num_value_tiles,
                        STATE_OFFSETS=state_offsets,
                        query_states=query_states,
                        TOP_QUERY=query_offsets[level] if level == num_levels - 1 else -1,
                        num_warps=merge_warps,
                        num_stages=num_stages,
                    )
        else:
            for level in range(1, num_levels):
                block = 1 << level
                half = block >> 1
                _merge_states_kernel[(num_tiles, batch * heads, num_chunks // block)](
                    states,
                    chunk_gates,
                    B=batch,
                    H=heads,
                    K=key_dim,
                    V=value_dim,
                    NT=num_chunks,
                    LEVEL=level,
                    HALF=half,
                    BLOCK=block,
                    BK=BLOCK_KEY,
                    BV=block_value,
                    NV=num_value_tiles,
                    num_warps=merge_warps,
                    num_stages=num_stages,
                )
        inter_kernel = (
            _inter_fullk_fwd_kernel if fullk else (_inter_reduce_fwd_kernel if num_key_tiles > 1 else _inter_fwd_kernel)
        )
        inter_kernel[
            (num_chunks, batch * heads, inter_value_tiles if num_key_tiles > 1 else num_key_tiles * inter_value_tiles)
        ](
            q,
            local_g,
            level_scales,
            local_output,
            states,
            *((query_states, query_offsets, state_offsets) if fullk else ()),
            chunk_gates,
            output if num_key_tiles > 1 else output_partial,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            BK=BLOCK_KEY,
            BV=inter_block_value,
            NK=num_key_tiles,
            NV=inter_value_tiles,
            num_warps=inter_warps,
            num_stages=num_stages,
        )

    return run, output


def chunk_log_linear_attn(q, k, v, g, level_scales):
    """Run dense, forward-only Log-Linear Attention with Hygon-compatible Triton kernels."""
    run, output = _prepare_forward(q, k, v, g, level_scales)
    run()
    return output


log_linear_attn = chunk_log_linear_attn


__all__ = [
    "chunk_log_linear_attn",
    "log_linear_attn",
]
