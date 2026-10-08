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

"""Shared host-side utilities for attention decode implementations."""

from __future__ import annotations


from dataclasses import dataclass, replace
import os

import torch
import triton
import triton.language as tl


try:
    import triton.experimental.tle.language as tle
    from triton.experimental.tle.language.gpu import types as gpu_types
except (ImportError, AttributeError):
    tle = None
    gpu_types = None
    HAS_TLE = False
else:
    HAS_TLE = True

USE_TLE = HAS_TLE and torch.version.hip is None
ENABLE_TLE_LOAD = (
    HAS_TLE
    and torch.version.hip is not None
    and os.environ.get("FLAG_ATTN_DECODE_USE_TLE_LOAD", "1") == "1"
)


@dataclass
class PureTritonMTP1Workspace:
    """Portable GQA8 split-K workspace used when TLE is unavailable."""

    task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    quant_type: str
    num_ctas: int
    max_tasks: int
    sched_ints: int
    heads_per_group: int
    padded_heads_per_group: int
    min_process_len: int

    uniform_length: int | None = None
    uniform_chunks: int = 0

@dataclass
class PureTritonMTPWorkspace:
    """Collection of MTP=1 Triton workspaces used for MTP>1 fallback."""

    workspaces: tuple[PureTritonMTP1Workspace, ...]
    mtp1_inputs: tuple | None = None


@triton.jit
def _pure_triton_bf16_gqa8_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    USE_TLE_LOAD: tl.constexpr,
):
    """BF16 MTP1 producer: GQA8 reuse, tl.dot, and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h
        q = tl.load(
            Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((GROUP_M, D), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + offs_n
            logical = seq_start + local_n
            page = logical // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            valid_n = local_n < seq_len
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=valid_n,
                other=0,
            )
            k_ptrs = K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD
            if USE_TLE_LOAD:
                k = tle.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                )
            else:
                k = tl.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                )
            v_ptrs = V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD
            if USE_TLE_LOAD:
                v = tle.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                )
            else:
                v = tl.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
            p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
            alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                v,
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
            lse,
            mask=valid_h,
        )



@triton.jit
def _pure_triton_bf16_uniform_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
    CHUNK_TOKENS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    VALID_LENGTH: tl.constexpr,
    USE_TLE_LOAD: tl.constexpr,
):
    task_id = tl.program_id(0)
    group_block = tl.program_id(1)

    program_chunks = (
        (NUM_CHUNKS + CHUNKS_PER_PROGRAM - 1)
        // CHUNKS_PER_PROGRAM
    )
    tasks_per_batch = program_chunks * (H_Q // HEADS_PER_GROUP)
    batch = task_id // tasks_per_batch
    task_in_batch = task_id - batch * tasks_per_batch
    hkv = task_in_batch // program_chunks
    chunk_base = (
        task_in_batch - hkv * program_chunks
    ) * CHUNKS_PER_PROGRAM

    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))

    hq = hkv * HEADS_PER_GROUP + offs_h
    q = tl.load(
        Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_h[:, None],
        other=0.0,
    )

    chunk_offset = 0
    while chunk_offset < CHUNKS_PER_PROGRAM:
        chunk = chunk_base + chunk_offset
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((GROUP_M, D), tl.float32)

        seq_start = chunk * CHUNK_TOKENS
        start = 0

        while start < CHUNK_TOKENS:
            local_n = start + offs_n
            logical = seq_start + local_n
            page = logical // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            if VALID_LENGTH % CHUNK_TOKENS == 0:
                valid_n = tl.full((BLOCK_N,), 1, tl.int1)
            else:
                valid_n = local_n < (VALID_LENGTH - seq_start)

            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=valid_n,
                other=0,
            )

            k_ptrs = (
                K
                + physical[:, None] * K_SBLOCK
                + pos[:, None] * K_STOKEN
                + hkv * K_SHEAD
                + offs_d[None, :] * K_SD
            )

            v_ptrs = (
                V
                + physical[:, None] * V_SBLOCK
                + pos[:, None] * V_STOKEN
                + hkv * V_SHEAD
                + offs_d[None, :] * V_SD
            )

            if USE_TLE_LOAD:
                k = tle.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                )
                v = tle.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                )
            else:
                k = tl.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                )
                v = tl.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                )

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )

            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)

            p = tl.where(
                active[:, None],
                tl.exp(scores - safe_new[:, None]),
                0.0,
            )
            alpha = tl.where(
                active,
                tl.exp(safe_old - safe_new),
                0.0,
            )

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                v,
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N

        partial = acc / l_i[:, None]
        lse = tl.log(l_i) + m_i

        if NUM_CHUNKS == 1:
            tl.store(
                OUT
                + batch * O_SB
                + hq[:, None] * O_SH
                + offs_d[None, :],
                partial,
                mask=valid_h[:, None],
            )
        else:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + chunk * SO_SC
                + hq[:, None] * SO_SH
                + offs_d[None, :],
                partial,
                mask=valid_h[:, None],
            )
            tl.store(
                SPLIT_LSE
                + batch * SL_SB
                + chunk * SL_SC
                + hkv * SL_SK
                + offs_h * SL_SH,
                lse,
                mask=valid_h,
            )
        chunk_offset += 1


@triton.jit
def _pure_triton_fp8_gqa8_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    TASK_MAP,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    SPLIT_OUT,
    SPLIT_LSE,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    QS_SB: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    USE_TLE_LOAD: tl.constexpr,
):
    """FP8 MTP1 producer matching the v3 pure-Triton GQA8 structure."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    dot_scale = tl.rsqrt(tl.full((), D, tl.float32))
    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task + 0)
        batch = tl.load(TASK_MAP + task + 1)
        if hkv < 0:
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + offs_h
        q = tl.load(
            Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        ).to(tl.float32)
        q_scale = tl.load(
            Q_SCALE + batch * QS_SB + hq * QS_SH,
            mask=valid_h,
            other=1.0,
        ).to(tl.float32)
        value_scale = tl.load(V_SCALE + (hkv if QUANT_TYPE == 0 else 0)).to(tl.float32) / 256.0
        tensor_k_scale = tl.load(K_SCALE).to(tl.float32) if QUANT_TYPE == 1 else 1.0
        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((GROUP_M, D), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + offs_n
            logical = seq_start + local_n
            page = logical // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            valid_n = local_n < seq_len
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=valid_n,
                other=0,
            )
            k_ptrs = K + physical[:, None] * K_SBLOCK + pos[:, None] * K_STOKEN + hkv * K_SHEAD + offs_d[None, :] * K_SD
            if USE_TLE_LOAD:
                k = tle.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                ).to(tl.float32)
            else:
                k = tl.load(
                    k_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                ).to(tl.float32)
            v_ptrs = V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD
            if USE_TLE_LOAD:
                v = tle.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                    is_async=False,
                ).to(tl.float32)
            else:
                v = tl.load(
                    v_ptrs,
                    mask=valid_n[:, None],
                    other=0.0,
                ).to(tl.float32)
            scores = tl.dot(q, tl.trans(k))
            if QUANT_TYPE == 0:
                byte_offset = physical * KS_SBLOCK + (pos // 32) * KS_STOKEN + hkv * KS_SHEAD + (pos % 32) * 4 * KS_SD
                scale_ptr = (K_SCALE + byte_offset).to(tl.pointer_type(tl.float32))
                k_scale = tl.load(scale_ptr, mask=valid_n, other=0.0).to(tl.float32)
                scores *= q_scale[:, None] * k_scale[None, :] * dot_scale
            else:
                scores *= q_scale[:, None] * tensor_k_scale * dot_scale
            scores = tl.where(
                valid_h[:, None] & valid_n[None, :],
                scores,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, tile_max)
            active = m_new != -float("inf")
            safe_new = tl.where(active, m_new, 0.0)
            safe_old = tl.where(m_i == -float("inf"), safe_new, m_i)
            p = tl.where(active[:, None], tl.exp(scores - safe_new[:, None]), 0.0)
            alpha = tl.where(active, tl.exp(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p_scaled = (p * 256.0).to(tl.float8e4nv).to(tl.float32)
            acc = acc * alpha[:, None] + tl.dot(p_scaled, v)
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(
            has_value[:, None],
            acc / l_i[:, None] * value_scale,
            0.0,
        )
        lse = tl.where(has_value, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq[:, None] * SO_SH + offs_d[None, :],
            partial,
            mask=valid_h[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + offs_h * SL_SH,
            lse,
            mask=valid_h,
        )


@triton.jit
def _pure_triton_splitk_combine_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    running_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC + hkv * SL_SK + h_in_group * SL_SH,
        )
        partial = tl.load(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC + hq * SO_SH + offs_d,
        )

        new_max = tl.maximum(running_max, value)
        active = new_max != -float("inf")
        safe_new = tl.where(active, new_max, 0.0)
        safe_old = tl.where(
            running_max == -float("inf"),
            safe_new,
            running_max,
        )

        rescale = tl.where(
            active,
            tl.exp(safe_old - safe_new),
            0.0,
        )
        weight = tl.where(
            active,
            tl.exp(value - safe_new),
            0.0,
        )

        out = out * rescale + partial * weight
        denom = denom * rescale + weight
        running_max = new_max
        chunk += 1

    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + batch * O_SB + hq * O_SH + offs_d,
        tl.where(denom > 0.0, out / safe_denom, 0.0),
    )



@triton.jit
def _pure_triton_uniform_splitk_combine_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    OUT,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SH: tl.constexpr,
):
    batch = tl.program_id(0)
    hq = tl.program_id(1)
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)

    running_max = tl.full((), -float("inf"), tl.float32)
    denom = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)

    chunk = 0
    while chunk < N_CHUNKS:
        value = tl.load(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + hkv * SL_SK
            + h_in_group * SL_SH,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + chunk * SO_SC
            + hq * SO_SH
            + offs_d,
        )

        new_max = tl.maximum(running_max, value)
        rescale = tl.exp(running_max - new_max)
        weight = tl.exp(value - new_max)

        out = out * rescale + partial * weight
        denom = denom * rescale + weight
        running_max = new_max
        chunk += 1

    tl.store(
        OUT + batch * O_SB + hq * O_SH + offs_d,
        out / denom,
    )


def prepare_pure_triton_mtp1_workspace(inputs, quant_type: str) -> PureTritonMTP1Workspace:
    if inputs.mtp != 1:
        raise NotImplementedError("the no-TLE fallback intentionally supports MTP=1 only")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != 128:
        raise ValueError("q must have shape [batch, Hq, 128] for MTP=1")
    if inputs.k_cache.ndim != 4 or inputs.v_cache.ndim != 4:
        raise ValueError("K/V caches must be rank-4 tensors")
    if inputs.k_cache.shape[1] != 64 or inputs.v_cache.shape[1] != 64:
        raise ValueError("K/V caches must use 64-token blocks")
    if inputs.k_cache.shape[-1] != 128 or inputs.v_cache.shape[-1] != 128:
        raise ValueError("K/V cache head dimension must be 128")
    if inputs.block_ids.ndim != 2 or inputs.block_ids.dtype != torch.int32:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.ndim != 1 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("kv_lens must be rank-1 int32")
    if inputs.q.shape[0] != inputs.kv_lens.numel():
        raise ValueError("MTP1 q batch must match kv_lens")
    if quant_type == "bf16":
        if any(t.dtype != torch.bfloat16 for t in (inputs.q, inputs.k_cache, inputs.v_cache)):
            raise ValueError("BF16 fallback requires BF16 Q/K/V")
    elif quant_type not in {
        "qkpertoken_perhead_vperhead",
        "qpertoken_perhead_kvpertensor",
    }:
        raise ValueError(f"unsupported fallback quantization: {quant_type}")
    elif any(t.element_size() != 1 for t in (inputs.q, inputs.k_cache, inputs.v_cache)):
        raise ValueError("FP8 fallback requires one-byte Q/K/V storage")
    hkv = int(inputs.k_cache.shape[2])
    hq = int(inputs.q.shape[1])
    heads_per_group = hq // hkv
    padded_heads = triton.cdiv(heads_per_group, 8) * 8
    properties = torch.cuda.get_device_properties(inputs.q.device)
    num_ctas = properties.multi_processor_count * 4
    min_process_len = 512

    length_values = tuple(
        int(value)
        for value in inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    )
    uniform_length = (
        length_values[0]
        if length_values and all(value == length_values[0] for value in length_values)
        else None
    )
    uniform_chunks = (
        (uniform_length + 511) // 512
        if uniform_length in (510, 511, 512, 4094, 4095, 4096)
        else 0
    )
    use_uniform = uniform_chunks > 0 and uniform_chunks <= num_ctas

    if use_uniform:
        max_tasks = 1
        sched_ints = 1
        workspace_ints = 1
    else:
        from .assign_task import pure_triton_task_map_metadata

        max_tasks, sched_ints, workspace_ints = pure_triton_task_map_metadata(
            inputs.kv_lens,
            num_head_kv=hkv,
            num_ctas=num_ctas,
            min_process_len=min_process_len,
        )

    task_map = torch.full(
        (workspace_ints,),
        -1,
        dtype=torch.int32,
        device=inputs.q.device,
    )
    split_out = torch.empty(
        (inputs.batch, num_ctas, hq, inputs.q.shape[-1]),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    split_lse = torch.empty(
        (inputs.batch, num_ctas, hkv, padded_heads),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    return PureTritonMTP1Workspace(
        task_map=task_map,
        split_out=split_out,
        split_lse=split_lse,
        out=torch.empty_like(inputs.q, dtype=torch.bfloat16),
        quant_type=quant_type,
        num_ctas=num_ctas,
        max_tasks=max_tasks,
        sched_ints=sched_ints,
        heads_per_group=heads_per_group,
        padded_heads_per_group=padded_heads,
        min_process_len=min_process_len,
        uniform_length=uniform_length if use_uniform else None,
        uniform_chunks=uniform_chunks if use_uniform else 0,
    )


def _pure_triton_mtp1_inputs(inputs, query_index: int):
    """View one MTP query as an independent MTP=1 workload."""
    batch = inputs.batch
    mtp = inputs.mtp
    q = inputs.q.reshape(batch, mtp, *inputs.q.shape[1:])[:, query_index]
    effective_kv_lens = inputs.kv_lens - (mtp - 1 - query_index)
    fields = {"q": q, "kv_lens": effective_kv_lens}
    q_scale = getattr(inputs, "q_scale", None)
    if q_scale is not None:
        fields["q_scale"] = q_scale.reshape(batch, mtp, -1)[:, query_index]
    return replace(inputs, **fields)


def prepare_pure_triton_mtp_workspace(inputs, quant_type: str) -> PureTritonMTPWorkspace:
    """Prepare MTP=2/4 by reusing the existing MTP=1 Triton path."""
    if inputs.mtp < 2:
        raise ValueError("the MTP workspace requires MTP >= 2")
    mtp1_inputs = tuple(
        _pure_triton_mtp1_inputs(inputs, query_index)
        for query_index in range(inputs.mtp)
    )
    workspaces = tuple(
        prepare_pure_triton_mtp1_workspace(mtp1_input, quant_type)
        for mtp1_input in mtp1_inputs
    )
    return PureTritonMTPWorkspace(
        workspaces=workspaces,
        mtp1_inputs=mtp1_inputs,
    )


def attention_decode_pure_triton_mtp(
    inputs,
    workspace: PureTritonMTPWorkspace,
):
    """Run each MTP query through the existing MTP=1 Triton implementation."""
    outputs = []
    cached_inputs = workspace.mtp1_inputs
    if cached_inputs is None:
        cached_inputs = tuple(
            _pure_triton_mtp1_inputs(inputs, query_index)
            for query_index in range(inputs.mtp)
        )

    for mtp1_input, mtp1_workspace in zip(
        cached_inputs,
        workspace.workspaces,
    ):
        output = attention_decode_pure_triton_mtp1(
            mtp1_input,
            mtp1_workspace,
        )
        outputs.append(output.unsqueeze(1))
    return torch.cat(outputs, dim=1).reshape_as(inputs.q).to(torch.bfloat16)


def attention_decode_pure_triton_mtp1(inputs, workspace: PureTritonMTP1Workspace):
    if inputs.mtp != 1:
        raise NotImplementedError("the no-TLE fallback intentionally supports MTP=1 only")
    batch = inputs.batch
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    from .assign_task import launch_pure_triton_task_map

    use_uniform_bf16 = (
        workspace.quant_type == "bf16"
        and workspace.uniform_length is not None
        and workspace.uniform_chunks > 0
    )
    if not use_uniform_bf16:
        launch_pure_triton_task_map(
            inputs.kv_lens,
            workspace.task_map,
            num_head_kv=hkv,
            num_ctas=workspace.num_ctas,
            min_process_len=workspace.min_process_len,
            max_tasks=workspace.max_tasks,
            sched_ints=workspace.sched_ints,
        )
    common = dict(
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        GROUP_M=8,
        D=inputs.q.shape[-1],
        BLOCK_SIZE=64,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        MAX_TASKS=workspace.max_tasks,
        BLOCK_N=64,
        Q_SB=inputs.q.stride(0),
        Q_SH=inputs.q.stride(1),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SH=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SK=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        num_warps=4,
        num_stages=3,
    )
    if use_uniform_bf16:
        uniform_common = {
            key: value
            for key, value in common.items()
            if key != "MAX_TASKS"
        }
        uniform_common["GROUP_M"] = 4 if workspace.uniform_chunks == 1 else 8
        uniform_common["O_SB"] = workspace.out.stride(0)
        uniform_common["O_SH"] = workspace.out.stride(1)
        chunks_per_program = (
            4 if workspace.uniform_chunks == 8 else 1
        )
        program_chunks = (
            workspace.uniform_chunks + chunks_per_program - 1
        ) // chunks_per_program
        uniform_grid = (
            batch * hkv * program_chunks,
            triton.cdiv(workspace.heads_per_group, 4 if workspace.uniform_chunks == 1 else 8),
        )
        _pure_triton_bf16_uniform_splitk_kernel[uniform_grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.split_out,
            workspace.split_lse,
            workspace.out,
            CHUNK_TOKENS=512,
            NUM_CHUNKS=workspace.uniform_chunks,
            CHUNKS_PER_PROGRAM=chunks_per_program,
            VALID_LENGTH=workspace.uniform_length,
            USE_TLE_LOAD=ENABLE_TLE_LOAD,
            **uniform_common,
        )
        if workspace.uniform_chunks > 1:
            _pure_triton_uniform_splitk_combine_kernel[(batch, hq)](
                workspace.split_out,
                workspace.split_lse,
                workspace.out,
                H_Q=hq,
                HEADS_PER_GROUP=workspace.heads_per_group,
                D=inputs.q.shape[-1],
                N_CHUNKS=workspace.uniform_chunks,
                SO_SB=workspace.split_out.stride(0),
                SO_SC=workspace.split_out.stride(1),
                SO_SH=workspace.split_out.stride(2),
                SL_SB=workspace.split_lse.stride(0),
                SL_SC=workspace.split_lse.stride(1),
                SL_SK=workspace.split_lse.stride(2),
                SL_SH=workspace.split_lse.stride(3),
                O_SB=workspace.out.stride(0),
                O_SH=workspace.out.stride(1),
                num_warps=4,
                num_stages=1,
            )
        return workspace.out

    grid = (
        workspace.num_ctas,
        triton.cdiv(workspace.heads_per_group, 8),
    )
    if workspace.quant_type == "bf16":
        _pure_triton_bf16_gqa8_splitk_kernel[grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.task_map,
            workspace.split_out,
            workspace.split_lse,
            USE_TLE_LOAD=ENABLE_TLE_LOAD,
            **common,
        )
    else:
        quant_type = 0 if workspace.quant_type == "qkpertoken_perhead_vperhead" else 1
        _pure_triton_fp8_gqa8_splitk_kernel[grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.task_map,
            inputs.q_scale,
            inputs.k_scale,
            inputs.v_scale,
            workspace.split_out,
            workspace.split_lse,
            QUANT_TYPE=quant_type,
            QS_SB=inputs.q_scale.stride(0),
            QS_SH=inputs.q_scale.stride(1),
            KS_SBLOCK=inputs.k_scale.stride(0),
            KS_STOKEN=inputs.k_scale.stride(1) if inputs.k_scale.ndim > 1 else 0,
            KS_SHEAD=inputs.k_scale.stride(2) if inputs.k_scale.ndim > 2 else 0,
            KS_SD=inputs.k_scale.stride(3) if inputs.k_scale.ndim > 3 else 0,
            USE_TLE_LOAD=ENABLE_TLE_LOAD,
            **common,
        )
    _pure_triton_splitk_combine_kernel[(batch, hq)](
        workspace.split_out,
        workspace.split_lse,
        workspace.task_map,
        workspace.out,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        D=inputs.q.shape[-1],
        MAX_SPLIT=workspace.num_ctas,
        MAX_TASKS=workspace.max_tasks,
        NUM_CTAS=workspace.num_ctas,
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SH=workspace.split_out.stride(2),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SK=workspace.split_lse.stride(2),
        SL_SH=workspace.split_lse.stride(3),
        O_SB=workspace.out.stride(0),
        O_SH=workspace.out.stride(1),
        num_warps=4,
        num_stages=1,
    )
    return workspace.out


@dataclass(frozen=True)
class DecodeWorkload:
    """Order-independent KV-length features used by policy selection."""

    histogram: tuple[tuple[int, int], ...]
    batch_size: int
    min_length: int
    max_length: int

    @classmethod
    def from_lengths(cls, lengths) -> "DecodeWorkload":
        values = tuple(int(value) for value in lengths)
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return cls(
            histogram=tuple(sorted(counts.items())),
            batch_size=len(values),
            min_length=min(values, default=0),
            max_length=max(values, default=0),
        )

    def count(self, length: int) -> int:
        for value, count in self.histogram:
            if value == length:
                return count
        return 0

    @property
    def distinct_lengths(self) -> int:
        return len(self.histogram)

    @property
    def uniform_length(self) -> int | None:
        if len(self.histogram) == 1:
            return self.histogram[0][0]
        return None

    def is_uniform(self, length: int | None = None) -> bool:
        uniform = self.uniform_length
        return uniform is not None and (length is None or uniform == length)

    def is_mix(
        self,
        first_length: int,
        first_count: int,
        second_length: int,
        second_count: int,
    ) -> bool:
        return (
            self.batch_size == first_count + second_count
            and self.distinct_lengths == 2
            and self.count(first_length) == first_count
            and self.count(second_length) == second_count
        )

    def is_one_long_tail(self, long_length: int, short_length: int) -> bool:
        return (
            self.distinct_lengths == 2
            and self.count(long_length) == 1
            and self.count(short_length) == self.batch_size - 1
        )

    def is_two_long_tail(self, long_length: int, short_length: int) -> bool:
        return (
            self.distinct_lengths == 2
            and self.count(long_length) == 2
            and self.count(short_length) == self.batch_size - 2
        )

    @property
    def signature(self) -> tuple[int, tuple[tuple[int, int], ...]]:
        return self.batch_size, self.histogram


__all__ = [
    "DecodeWorkload",
    "HAS_TLE",
    "USE_TLE",
    "PureTritonMTP1Workspace",
    "PureTritonMTPWorkspace",
    "attention_decode_pure_triton_mtp",
    "attention_decode_pure_triton_mtp1",
    "prepare_pure_triton_mtp_workspace",
    "prepare_pure_triton_mtp1_workspace",
    "gpu_types",
    "tle",
]