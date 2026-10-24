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
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from .mtp_fused import FusedMTPWorkspace

from .scheduler import (
    DEFAULT_BLOCKS_PER_CHUNK,
    DEFAULT_PROGRAMS_PER_CTA_TARGET,
    derive_chunks_per_program,
    derive_blocks_per_chunk,
    derive_group_m,
    derive_hardware_policy,
    derive_policy,
    derive_uniform_group_m,
    choose_decode_path,
)


def _env_candidates(name: str) -> tuple[str, ...]:
    if name.startswith("HY3_"):
        return (name, f"FLAG_ATTN_{name}", f"FLA_{name}")
    return (name,)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read a Hy3 tuning knob with stable backend-prefixed aliases.

    ``HY3_*`` remains the canonical spelling.  The ``FLAG_ATTN_HY3_*`` and
    ``FLA_HY3_*`` aliases follow the naming convention used by the other
    Hygon FLA operators, so benchmark scripts can share one environment
    namespace without changing the kernel implementation.
    """
    raw = default
    for candidate in _env_candidates(name):
        if candidate in os.environ:
            raw = os.environ[candidate]
            break
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_int_any(names: tuple[str, ...], default: int, minimum: int = 1) -> int:
    """Read the first set tuning override, keeping old names as aliases."""
    for name in names:
        if any(candidate in os.environ for candidate in _env_candidates(name)):
            return _env_int(name, default, minimum)
    return max(minimum, default)


def _validate_decode_indirection(inputs) -> None:
    """Validate device-side indirection before launching a decode kernel.

    A bad page table or stale sequence length used to turn into a Hygon
    VMFault several instructions after the original error.  The validation is
    performed once while preparing the workspace, not in the timed kernel
    path, so valid workloads retain the same launch geometry and latency.
    """
    block_size = int(inputs.k_cache.shape[1])
    num_blocks = int(inputs.k_cache.shape[0])
    max_pages = int(inputs.block_ids.shape[1])
    lengths = [
        int(value)
        for value in inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    ]
    if any(value < 0 for value in lengths):
        raise ValueError("kv_lens must be non-negative")
    required_pages = [
        (value + block_size - 1) // block_size for value in lengths
    ]
    if any(pages > max_pages for pages in required_pages):
        raise ValueError(
            "block_ids has fewer pages than required by kv_lens; refusing "
            "to launch an unsafe decode kernel"
        )
    # Only the prefix consumed by the kernel is inspected.  Tail entries are
    # intentionally allowed to remain uninitialized for paged-cache callers.
    block_ids = inputs.block_ids.detach().cpu().to(torch.int64)
    for batch, pages in enumerate(required_pages):
        if pages == 0:
            continue
        selected = block_ids[batch, :pages]
        if bool((selected < 0).any()) or bool((selected >= num_blocks).any()):
            raise ValueError(
                "block_ids contains an out-of-range physical block; refusing "
                "to launch an unsafe decode kernel"
            )


@dataclass
class PureTritonMTP1Workspace:
    """Workspace for the pure-Triton GQA split-K path."""

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
    # The task map depends only on kv_lens and the workspace policy.  Decode
    # calls normally reuse both, so avoid rebuilding it on every launch.  The
    # tensor version lets us invalidate the cache after an in-place update.
    task_map_ready: bool = False
    kv_lens_version: int = -1
    kv_lens_ptr: int = 0
    task_map_stream: int = 0

@dataclass
class PureTritonMTPWorkspace:
    """Collection of single-query Triton workspaces used by the MTP wrapper."""

    workspaces: tuple[PureTritonMTP1Workspace, ...]
    mtp1_inputs: tuple | None = None
    # Optional fused MTP workspace for BF16 and INT8 paths.
    mtp_fused_workspace: FusedMTPWorkspace | None = None


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
    INT8_INPUT: tl.constexpr,
    QK_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
):
    """BF16/INT8-input MTP1 producer: GQA reuse and dynamic split-K."""
    cta = tl.program_id(0)
    group_block = tl.program_id(1)
    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    scale = tl.rsqrt(tl.full((), D, tl.float32))
    if INT8_INPUT:
        scale = scale * QK_SCALE
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
        if INT8_INPUT:
            q = q.to(tl.int8)
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
            k = tl.load(
                k_ptrs,
                mask=valid_n[:, None],
                other=0.0,
            )
            if INT8_INPUT:
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32) * scale
            else:
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
            # V participates only in the PV dot.  Delay this load until after
            # QK softmax so K/V tiles do not remain live together across the
            # score path on gfx936.
            v_ptrs = V + physical[:, None] * V_SBLOCK + pos[:, None] * V_STOKEN + hkv * V_SHEAD + offs_d[None, :] * V_SD
            v = tl.load(
                v_ptrs,
                mask=valid_n[:, None],
                other=0.0,
            )
            if INT8_INPUT:
                v = v.to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                v,
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N
        has_value = l_i > 0.0
        partial = tl.where(has_value[:, None], acc / l_i[:, None], 0.0)
        if INT8_INPUT:
            partial = partial * V_SCALE
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
def _pure_triton_uniform_splitk_kernel(
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
    INT8_INPUT: tl.constexpr,
    QK_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
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
    if INT8_INPUT:
        scale = scale * QK_SCALE

    hq = hkv * HEADS_PER_GROUP + offs_h
    q = tl.load(
        Q + batch * Q_SB + hq[:, None] * Q_SH + offs_d[None, :],
        mask=valid_h[:, None],
        other=0.0,
    )
    if INT8_INPUT:
        q = q.to(tl.int8)

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

            k = tl.load(
                k_ptrs,
                mask=valid_n[:, None],
                other=0.0,
            )

            if INT8_INPUT:
                scores = tl.dot(
                    q, tl.trans(k), out_dtype=tl.int32
                ).to(tl.float32) * scale
            else:
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
            v_ptrs = (
                V
                + physical[:, None] * V_SBLOCK
                + pos[:, None] * V_STOKEN
                + hkv * V_SHEAD
                + offs_d[None, :] * V_SD
            )
            v = tl.load(
                v_ptrs,
                mask=valid_n[:, None],
                other=0.0,
            )
            if INT8_INPUT:
                v = v.to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16),
                v,
                out_dtype=tl.float32,
            )
            m_i = m_new
            start += BLOCK_N

        partial = acc / l_i[:, None]
        if INT8_INPUT:
            partial = partial * V_SCALE
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


def prepare_pure_triton_mtp1_workspace(
    inputs, quant_type: str, *, force_split_k: bool = False
) -> PureTritonMTP1Workspace:
    if inputs.mtp != 1:
        raise NotImplementedError("the single-query Triton producer requires MTP=1")
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
    num_kv_heads = int(inputs.k_cache.shape[2])
    if num_kv_heads <= 0 or int(inputs.q.shape[1]) % num_kv_heads:
        raise ValueError("q head count must be divisible by the K/V head count")
    if quant_type not in ("bf16", "int8"):
        raise ValueError("Hygon Hy3 decode supports BF16 and INT8")
    expected_dtype = torch.bfloat16 if quant_type == "bf16" else torch.int8
    if any(t.dtype != expected_dtype for t in (inputs.q, inputs.k_cache, inputs.v_cache)):
        raise ValueError(f"{quant_type.upper()} decode requires matching Q/K/V dtypes")
    _validate_decode_indirection(inputs)
    hkv = int(inputs.k_cache.shape[2])
    hq = int(inputs.q.shape[1])
    heads_per_group = hq // hkv
    padded_heads = triton.cdiv(heads_per_group, 8) * 8
    length_values = tuple(
        int(value)
        for value in inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    )
    properties = torch.cuda.get_device_properties(inputs.q.device)
    ragged_lengths = bool(length_values) and not all(
        value == length_values[0] for value in length_values
    )
    block_size = int(inputs.k_cache.shape[1])
    uniform_length = (
        length_values[0]
        if length_values and all(value == length_values[0] for value in length_values)
        else None
    )
    derived_blocks_per_chunk = DEFAULT_BLOCKS_PER_CHUNK
    blocks_per_chunk = _env_int(
        "HY3_BLOCKS_PER_CHUNK",
        derived_blocks_per_chunk,
    )
    if "HY3_BLOCKS_PER_CHUNK" not in os.environ:
        outer_programs_per_sequence = hkv * triton.cdiv(
            heads_per_group, derive_group_m(heads_per_group)
        )
        one_page_programs = sum(
            triton.cdiv(length, block_size) for length in length_values
        ) * outer_programs_per_sequence
        # Allow 64-token page chunks only if the resulting grid does not exceed
        # one CTA per CU. Finer splitting beyond that point increases partial
        # output/reduction work without adding CU-level parallelism.
        min_blocks_per_chunk = (
            1
            if 0 < one_page_programs <= properties.multi_processor_count
            else 2
        )
        blocks_per_chunk = derive_blocks_per_chunk(
            length_values,
            block_size=block_size,
            outer_programs_per_sequence=outer_programs_per_sequence,
            num_cus=properties.multi_processor_count,
            default_blocks_per_chunk=derived_blocks_per_chunk,
            min_blocks_per_chunk=min_blocks_per_chunk,
        )
    chunk_tokens = block_size * blocks_per_chunk
    approximate_uniform_chunks = (
        triton.cdiv(uniform_length, chunk_tokens)
        if uniform_length is not None
        else 0
    )
    hardware_policy = derive_hardware_policy(
        quant_type=quant_type,
        head_dim=int(inputs.q.shape[-1]),
        blocks_per_chunk=blocks_per_chunk,
        ragged=ragged_lengths,
        uniform_chunks=approximate_uniform_chunks,
    )
    tuning_prefix = f"HY3_{quant_type.upper()}"
    cta_waves = _env_int_any(
        ("HY3_CTA_WAVES", f"{tuning_prefix}_CTA_WAVES"),
        hardware_policy.cta_waves,
    )
    max_ctas = properties.multi_processor_count * cta_waves
    min_process_len = chunk_tokens
    min_tiles_per_cta = max(1, triton.cdiv(min_process_len, block_size))
    total_tiles = sum(
        triton.cdiv(length, block_size) for length in length_values
    ) * hkv
    num_ctas = min(
        max_ctas,
        max(1, triton.cdiv(total_tiles, min_tiles_per_cta)),
    )
    geometry = derive_policy(
        block_size=block_size,
        heads_per_group=heads_per_group,
        uniform_length=uniform_length,
        ragged=ragged_lengths,
        num_ctas=num_ctas,
        blocks_per_chunk=blocks_per_chunk,
    )
    uniform_chunks = geometry.uniform_chunks
    # Uniform workloads use a direct grid and do not need task-map metadata.
    mtp_fusion_candidate = _supports_mtp_fusion(inputs, quant_type)
    decode_path = choose_decode_path(
        quant_type=quant_type,
        uniform_chunks=uniform_chunks,
        num_ctas=num_ctas,
        fusion_candidate=mtp_fusion_candidate or force_split_k,
    )
    # A/B on gfx936 shows the INT8 direct-uniform kernel is modestly faster
    # for one chunk, while multi-chunk cases lose to the task-map producer.
    # Keep the fast path for its measured regime and retain split-K otherwise.
    if quant_type == "int8" and uniform_chunks > 1:
        decode_path = "split_k"
    use_uniform = decode_path == "uniform"


    if use_uniform:
        max_tasks = 1
        sched_ints = 1
        workspace_ints = 1
    else:
        from .scheduler import pure_triton_task_map_metadata

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
    # Uniform producers index partials by uniform chunk rather than CTA. The
    # one-chunk kernel writes final output directly and does not touch partials.
    split_slots = uniform_chunks if use_uniform else num_ctas
    split_out = torch.empty(
        (inputs.batch, split_slots, hq, inputs.q.shape[-1]),
        dtype=torch.float32,
        device=inputs.q.device,
    )
    split_lse = torch.empty(
        (inputs.batch, split_slots, hkv, padded_heads),
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
        kv_lens_version=int(getattr(inputs.kv_lens, "_version", -1)),
        kv_lens_ptr=int(inputs.kv_lens.data_ptr()),
    )


def _pure_triton_mtp1_inputs(inputs, query_index: int):
    """View one MTP query as an independent MTP=1 workload."""
    batch = inputs.batch
    mtp = inputs.mtp
    q = inputs.q.reshape(batch, mtp, *inputs.q.shape[1:])[:, query_index]
    effective_kv_lens = inputs.kv_lens - (mtp - 1 - query_index)
    fields = {"q": q, "kv_lens": effective_kv_lens}
    return replace(inputs, **fields)


def _effective_mtp_rows(inputs) -> int:
    """Return matrix rows available after fusing MTP and one GQA group."""
    num_kv_heads = int(inputs.k_cache.shape[2])
    num_query_heads = int(inputs.q.shape[1])
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads:
        raise ValueError("query heads must form an integral GQA group")
    return int(inputs.mtp) * (num_query_heads // num_kv_heads)


def _supports_mtp_fusion(inputs, quant_type: str) -> bool:
    """Check semantic eligibility, independent of a tuned MTP/GQA pair."""
    expected_dtype = torch.bfloat16 if quant_type == "bf16" else torch.int8
    return (
        int(inputs.mtp) >= 2
        and int(inputs.k_cache.shape[2]) > 0
        and int(inputs.q.shape[1]) % int(inputs.k_cache.shape[2]) == 0
        and inputs.q.dtype == expected_dtype
        and inputs.k_cache.dtype == expected_dtype
        and inputs.v_cache.dtype == expected_dtype
    )

def prepare_pure_triton_mtp_workspace(inputs, quant_type: str) -> PureTritonMTPWorkspace:
    """Prepare MTP workspaces and an optional fused MTP/GQA path."""
    if quant_type not in ("bf16", "int8"):
        raise ValueError("Hygon Hy3 decode supports BF16 and INT8")
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
    workspace = PureTritonMTPWorkspace(
        workspaces=workspaces,
        mtp1_inputs=mtp1_inputs,
    )
    if not _supports_mtp_fusion(inputs, quant_type):
        return workspace

    lengths = tuple(
        int(value)
        for value in inputs.kv_lens.detach().cpu().to(torch.int64).tolist()
    )
    is_uniform = bool(lengths) and all(value == lengths[0] for value in lengths)
    q1_workspace = workspaces[-1]
    properties = torch.cuda.get_device_properties(inputs.q.device)
    min_split_tasks = triton.cdiv(3 * int(properties.multi_processor_count), 2)

    if q1_workspace.uniform_length is not None and is_uniform:
        split_tasks = (
            inputs.batch
            * int(inputs.k_cache.shape[2])
            * triton.cdiv(lengths[0], q1_workspace.min_process_len)
        )
        if split_tasks >= min_split_tasks:
            q1_workspace = prepare_pure_triton_mtp1_workspace(
                mtp1_inputs[-1], quant_type, force_split_k=True
            )
            workspaces = workspaces[:-1] + (q1_workspace,)
            workspace.workspaces = workspaces

    if q1_workspace.uniform_length is None:
        if is_uniform:
            split_tasks = (
                inputs.batch
                * int(inputs.k_cache.shape[2])
                * triton.cdiv(lengths[0], q1_workspace.min_process_len)
            )
        else:
            split_tasks = int(inputs.k_cache.shape[2]) * sum(
                triton.cdiv(length, q1_workspace.min_process_len)
                for length in lengths
            )
        if split_tasks >= min_split_tasks:
            from .mtp_fused import prepare_fused_mtp_workspace

            fused_workspace = prepare_fused_mtp_workspace(inputs, workspace, quant_type)
            workspace.mtp_fused_workspace = fused_workspace
    return workspace


def attention_decode_pure_triton_mtp(
    inputs,
    workspace: PureTritonMTPWorkspace,
):
    """Run MTP through fused or independent pure-Triton kernels."""
    fused_workspace = workspace.mtp_fused_workspace
    if fused_workspace is not None:
        from .mtp_fused import run_fused_mtp

        return run_fused_mtp(inputs, fused_workspace)

    batch = inputs.batch
    output = torch.empty(
        (batch, inputs.mtp, inputs.q.shape[1], inputs.q.shape[-1]),
        dtype=torch.bfloat16,
        device=inputs.q.device,
    )
    cached_inputs = workspace.mtp1_inputs
    if cached_inputs is None:
        cached_inputs = tuple(
            _pure_triton_mtp1_inputs(inputs, query_index)
            for query_index in range(inputs.mtp)
        )
    for query_index, (mtp1_input, mtp1_workspace) in enumerate(
        zip(cached_inputs, workspace.workspaces)
    ):
        attention_decode_pure_triton_mtp1(
            mtp1_input,
            mtp1_workspace,
            output=output[:, query_index],
        )
    return output.reshape_as(inputs.q)


def attention_decode_pure_triton_mtp1(
    inputs,
    workspace: PureTritonMTP1Workspace,
    *,
    output: torch.Tensor | None = None,
):
    if inputs.mtp != 1:
        raise NotImplementedError("the single-query Triton producer requires MTP=1")
    batch = inputs.batch
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    out = workspace.out if output is None else output
    if out.shape != (batch, hq, inputs.q.shape[-1]):
        raise ValueError("output must have shape [batch, Hq, D]")
    if out.dtype != torch.bfloat16 or out.device != inputs.q.device:
        raise ValueError("output must be BF16 on the query device")
    from .scheduler import launch_pure_triton_task_map

    use_uniform = (
        workspace.uniform_length is not None
        and workspace.uniform_chunks > 0
    )
    if not use_uniform:
        kv_lens_version = int(getattr(inputs.kv_lens, "_version", -1))
        kv_lens_ptr = int(inputs.kv_lens.data_ptr())
        stream_id = int(torch.cuda.current_stream(inputs.q.device).cuda_stream)
        task_map_stale = (
            not workspace.task_map_ready
            or (
                workspace.kv_lens_version >= 0
                and kv_lens_version != workspace.kv_lens_version
            )
            or (
                workspace.kv_lens_ptr != 0
                and kv_lens_ptr != workspace.kv_lens_ptr
            )
            or (
                workspace.task_map_stream != 0
                and stream_id != workspace.task_map_stream
            )
        )
        if task_map_stale:
            launch_pure_triton_task_map(
                inputs.kv_lens,
                workspace.task_map,
                num_head_kv=hkv,
                num_ctas=workspace.num_ctas,
                min_process_len=workspace.min_process_len,
                max_tasks=workspace.max_tasks,
                sched_ints=workspace.sched_ints,
            )
            workspace.task_map_ready = True
            workspace.kv_lens_version = kv_lens_version
            workspace.kv_lens_ptr = kv_lens_ptr
            workspace.task_map_stream = stream_id
    tuning_prefix = f"HY3_{workspace.quant_type.upper()}"
    block_size = int(inputs.k_cache.shape[1])
    # Chunk geometry is captured when the matching workspace is prepared.
    # Re-read the environment here only if it was also used to prepare that
    # workspace; using the recorded extent avoids mismatched split buffers.
    blocks_per_chunk = workspace.min_process_len // block_size
    # For BF16 MTP=1/GQA=8, padding eight valid head rows to the gfx936 MMAC
    # M=16 tile outperformed the M=8 FMA path on the measured workloads.
    # Select it from shape metadata, without an experiment-only environment flag.
    use_mtp1_padded_mmac = (
        int(getattr(inputs, "mtp", 1)) == 1
        and workspace.heads_per_group == 8
    )
    runtime_group_m = 16 if use_mtp1_padded_mmac else derive_group_m(
        workspace.heads_per_group
    )
    runtime_chunk_tokens = block_size * blocks_per_chunk
    hardware_policy = derive_hardware_policy(
        quant_type=workspace.quant_type,
        head_dim=int(inputs.q.shape[-1]),
        blocks_per_chunk=blocks_per_chunk,
        ragged=workspace.uniform_length is None,
        uniform_chunks=workspace.uniform_chunks,
    )
    runtime_block_n = derive_policy(
        block_size=block_size,
        heads_per_group=workspace.heads_per_group,
        uniform_length=workspace.uniform_length,
        ragged=workspace.uniform_length is None,
        num_ctas=workspace.num_ctas,
        blocks_per_chunk=blocks_per_chunk,
    ).block_n
    # One producer warp is the measured choice for BF16 uniform and ragged
    # workloads. Keep the hardware-policy default for INT8 uniform; explicit
    # environment overrides remain available. Ragged combine retains 2 warps.
    default_num_warps = (
        1
        if workspace.uniform_length is None or workspace.quant_type == "bf16"
        else hardware_policy.num_warps
    )
    # BF16 uniform A/B favors two combine warps on multi-chunk reductions;
    # ragged combines retain two, while INT8 uniform keeps the hardware policy.
    default_combine_warps = (
        2
        if workspace.uniform_length is None or workspace.quant_type == "bf16"
        else hardware_policy.combine_warps
    )
    # BLOCK_N is derived from page geometry and the number of uniform
    # chunks; the environment variable remains an experiment hook.
    default_block_n = runtime_block_n
    common = dict(
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        GROUP_M=runtime_group_m,
        D=inputs.q.shape[-1],
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        MAX_TASKS=workspace.max_tasks,
        BLOCK_N=_env_int(f"{tuning_prefix}_BLOCK_N", default_block_n),
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
        num_warps=_env_int_any(
            ("HY3_NUM_WARPS", f"{tuning_prefix}_NUM_WARPS"),
            default_num_warps,
        ),
        num_stages=_env_int_any(
            ("HY3_NUM_STAGES", f"{tuning_prefix}_NUM_STAGES"),
            hardware_policy.num_stages,
        ),
    )
    if use_uniform:
        uniform_common = {
            key: value
            for key, value in common.items()
            if key != "MAX_TASKS"
        }
        default_group_m = derive_uniform_group_m(
            workspace.heads_per_group,
            workspace.uniform_chunks,
            blocks_per_chunk=blocks_per_chunk,
        )
        # Deep reductions need not split heads when KV chunks already expose
        # enough parallelism. Keep K/V reuse within the full head tile, then
        # let the chunk-group policy preserve the target launch grid below.
        full_group_m = derive_group_m(workspace.heads_per_group)
        full_group_programs = (
            batch
            * hkv
            * triton.cdiv(workspace.heads_per_group, full_group_m)
            * workspace.uniform_chunks
        )
        if (
            workspace.uniform_chunks >= blocks_per_chunk * 2
            and full_group_programs >= max(
                workspace.num_ctas,
                torch.cuda.get_device_properties(inputs.q.device).multi_processor_count,
            )
        ):
            default_group_m = full_group_m
        uniform_group_m = _env_int(
            f"{tuning_prefix}_GROUP_M", default_group_m
        )
        if use_mtp1_padded_mmac:
            uniform_group_m = 16
        else:
            uniform_group_m = min(uniform_group_m, workspace.heads_per_group)
        uniform_common["GROUP_M"] = uniform_group_m
        uniform_common["O_SB"] = out.stride(0)
        uniform_common["O_SH"] = out.stride(1)
        chunks_per_program = _env_int(
            f"{tuning_prefix}_CHUNKS_PER_PROGRAM",
            derive_chunks_per_program(
                workspace.uniform_chunks,
                blocks_per_chunk=blocks_per_chunk,
                outer_programs=(
                    batch
                    * hkv
                    * triton.cdiv(
                        workspace.heads_per_group, uniform_group_m
                    )
                ),
                target_programs=(
                    workspace.num_ctas * DEFAULT_PROGRAMS_PER_CTA_TARGET
                ),
            ),
        )
        chunks_per_program = min(chunks_per_program, workspace.uniform_chunks)
        if workspace.uniform_chunks % chunks_per_program:
            raise ValueError(
                f"{tuning_prefix}_CHUNKS_PER_PROGRAM must evenly divide the number "
                "of uniform chunks"
            )
        program_chunks = (
            workspace.uniform_chunks + chunks_per_program - 1
        ) // chunks_per_program
        uniform_grid = (
            batch * hkv * program_chunks,
            triton.cdiv(workspace.heads_per_group, uniform_group_m),
        )
        _pure_triton_uniform_splitk_kernel[uniform_grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.split_out,
            workspace.split_lse,
            out,
            CHUNK_TOKENS=runtime_chunk_tokens,
            NUM_CHUNKS=workspace.uniform_chunks,
            CHUNKS_PER_PROGRAM=chunks_per_program,
            VALID_LENGTH=workspace.uniform_length,
            INT8_INPUT=workspace.quant_type == "int8",
            QK_SCALE=(
                float(getattr(inputs, "q_scale", 1.0))
                * float(getattr(inputs, "k_scale", 1.0))
            ),
            V_SCALE=float(getattr(inputs, "v_scale", 1.0)),
            **uniform_common,
        )
        if workspace.uniform_chunks > 1:
            _pure_triton_uniform_splitk_combine_kernel[(batch, hq)](
                workspace.split_out,
                workspace.split_lse,
                out,
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
                O_SB=out.stride(0),
                O_SH=out.stride(1),
                num_warps=_env_int_any(
                    ("HY3_COMBINE_WARPS", f"{tuning_prefix}_COMBINE_WARPS"),
                    hardware_policy.combine_warps,
                ),
                num_stages=_env_int_any(
                    ("HY3_COMBINE_STAGES", f"{tuning_prefix}_COMBINE_STAGES"),
                    hardware_policy.combine_stages,
                ),
            )
        return out

    grid = (
        workspace.num_ctas,
        triton.cdiv(workspace.heads_per_group, runtime_group_m),
    )
    if workspace.quant_type in ("bf16", "int8"):
        _pure_triton_bf16_gqa8_splitk_kernel[grid](
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.block_ids,
            workspace.task_map,
            workspace.split_out,
            workspace.split_lse,
            **common,
            INT8_INPUT=workspace.quant_type == "int8",
            QK_SCALE=(
                float(getattr(inputs, "q_scale", 1.0))
                * float(getattr(inputs, "k_scale", 1.0))
            ),
            V_SCALE=float(getattr(inputs, "v_scale", 1.0)),
        )
    _pure_triton_splitk_combine_kernel[(batch, hq)](
        workspace.split_out,
        workspace.split_lse,
        workspace.task_map,
        out,
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
        O_SB=out.stride(0),
        O_SH=out.stride(1),
        num_warps=_env_int_any(
            ("HY3_COMBINE_WARPS", f"{tuning_prefix}_COMBINE_WARPS"),
            default_combine_warps,
        ),
        num_stages=_env_int_any(
            ("HY3_COMBINE_STAGES", f"{tuning_prefix}_COMBINE_STAGES"),
            hardware_policy.combine_stages,
        ),
    )
    return out


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
    "PureTritonMTP1Workspace",
    "PureTritonMTPWorkspace",
    "attention_decode_pure_triton_mtp",
    "attention_decode_pure_triton_mtp1",
    "prepare_pure_triton_mtp_workspace",
    "prepare_pure_triton_mtp1_workspace",
]
