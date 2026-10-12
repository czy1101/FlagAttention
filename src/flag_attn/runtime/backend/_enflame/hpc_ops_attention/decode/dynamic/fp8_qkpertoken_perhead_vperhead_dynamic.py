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


from dataclasses import dataclass
from typing import Literal

import torch
import triton
import triton.language as tl

from .bf16_dynamic import (
    _launch_s60_dynamic_task_map,
    _s60_compute_units,
    _s60_dynamic_task_map_metadata,
)


_TASK_STRIDE = 12
_BLOCK_SIZE = 64
_HEAD_DIM = 128
_TILE_N = 128
_BLOCK_N = 64
_CTAS_PER_CU = 128
_MAX_CTAS = 2048
_MIN_PROCESS_LEN = 128
_WORKSPACE_BUDGET_BYTES = 8 * 1024**3
_TASK_STRIDE_JIT = tl.constexpr(_TASK_STRIDE)

@dataclass(repr=False)
class S60DynamicFP8Workspace:
    """Buffers and launch metadata for the S60 dynamic FP8 route."""

    task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    quant_type: int
    mtp: int
    num_ctas: int
    total_tiles: int
    uniform_seq_tiles: int
    max_tasks: int
    sched_ints: int
    heads_per_group: int
    padded_heads_per_group: int
    min_process_len: int
    route: str = "s60-fp8-dynamic-page64-blockptr-split128-w1"


@triton.jit
def _s60_decode_fp8e4m3fn(bits):
    """Decode raw E4M3FN bytes without exposing an FP8 type to GCU."""
    value = bits.to(tl.int32)
    exponent = (value >> 3) & 0xF
    mantissa = value & 0x7
    sign = tl.where((value & 0x80) != 0, -1.0, 1.0)
    subnormal = mantissa.to(tl.float32) * 0.001953125
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    decoded = tl.where(exponent == 0, subnormal, normal) * sign
    is_nan = (exponent == 0xF) & (mantissa == 0x7)
    return tl.where(is_nan, float("nan"), decoded)


@triton.jit
def _s60_roundtrip_fp8e4m3fn(value):
    """Apply E4M3FN round-to-nearest-even and return the decoded value."""
    magnitude = tl.abs(value)
    safe_magnitude = tl.maximum(magnitude, 0.001953125)
    exponent = tl.floor(tl.log2(safe_magnitude))
    exponent = tl.maximum(-6.0, tl.minimum(exponent, 8.0))
    normal_step = tl.exp2(exponent - 3.0)
    step = tl.where(magnitude < 0.015625, 0.001953125, normal_step)
    scaled = magnitude / step
    lower = tl.floor(scaled)
    fraction = scaled - lower
    lower_is_odd = (lower.to(tl.int32) & 1) != 0
    round_up = (fraction > 0.5) | ((fraction == 0.5) & lower_is_odd)
    rounded = lower + round_up.to(tl.float32)
    quantized = tl.minimum(rounded * step, 448.0)
    quantized = tl.where(value < 0.0, -quantized, quantized)
    return tl.where(magnitude == 0.0, 0.0, quantized)


@triton.jit
def _s60_dynamic_fp8_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    TASK_MAP,
    Q_SCALE,
    K_SCALE,
    V_SCALE,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_GROUP_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS,
    BLOCK_N: tl.constexpr,
    QUANT_TYPE: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
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
    QS_SM: tl.constexpr,
    QS_SH: tl.constexpr,
    KS_SBLOCK: tl.constexpr,
    KS_STOKEN: tl.constexpr,
    KS_SHEAD: tl.constexpr,
    KS_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """Compute S60 FP8 split-K partials from raw byte storage."""
    pid = tl.program_id(0).to(tl.int32)
    seq_m = pid % NUM_SEQ_Q
    work = pid // NUM_SEQ_Q
    group_block = work % NUM_GROUP_BLOCKS
    cta = work // NUM_GROUP_BLOCKS

    offs_h = group_block * GROUP_M + tl.arange(0, GROUP_M)
    valid_h = offs_h < HEADS_PER_GROUP
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    dot_scale_log2 = (
        tl.rsqrt(tl.full((), D, tl.float32)) * 1.4426950408889634
    )

    slot = 0
    task = ((MAX_TASKS * cta + slot) + 1) * _TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task + 0)
    while (slot < MAX_TASKS) & (hkv >= 0):
        batch = tl.load(TASK_MAP + task + 1)
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        total_len = tl.load(KV_LENS + batch)
        effective_len = total_len - (NUM_SEQ_Q - 1 - seq_m)

        hq = hkv * HEADS_PER_GROUP + offs_h
        valid_hq = valid_h & (hq < H_Q)
        q_bits = tl.load(
            Q
            + batch * Q_SB
            + seq_m * Q_SM
            + hq[:, None] * Q_SH
            + offs_d[None, :],
            mask=valid_hq[:, None],
            other=0,
        )
        q = _s60_decode_fp8e4m3fn(q_bits).to(tl.bfloat16)
        q_scale = tl.load(
            Q_SCALE
            + batch * QS_SB
            + seq_m * QS_SM
            + hq * QS_SH,
            mask=valid_hq,
            other=1.0,
        ).to(tl.float32)
        value_scale = (
            tl.load(V_SCALE + (hkv if QUANT_TYPE == 0 else 0)).to(
                tl.float32
            )
            / 256.0
        )
        tensor_k_scale = (
            tl.load(K_SCALE).to(tl.float32) if QUANT_TYPE == 1 else 1.0
        )

        m_i = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((GROUP_M,), tl.float32)
        acc = tl.zeros((GROUP_M, D), tl.float32)
        start = 0
        while start < seq_len:
            logical_start = seq_start + start
            logical = logical_start + offs_n
            page = logical_start // BLOCK_SIZE
            pos = logical - page * BLOCK_SIZE
            local_n = start + offs_n
            valid_n = (local_n < seq_len) & (logical < total_len)
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=logical_start < total_len,
                other=0,
            )
            k_page = tl.make_block_ptr(
                base=K + physical * K_SBLOCK + hkv * K_SHEAD,
                shape=(BLOCK_SIZE, D),
                strides=(K_STOKEN, K_SD),
                offsets=(0, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0),
            )
            k_bits = tl.load(
                k_page,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            k = _s60_decode_fp8e4m3fn(k_bits).to(tl.bfloat16)
            v_page = tl.make_block_ptr(
                base=V + physical * V_SBLOCK + hkv * V_SHEAD,
                shape=(BLOCK_SIZE, D),
                strides=(V_STOKEN, V_SD),
                offsets=(0, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0),
            )
            v_bits = tl.load(
                v_page,
                boundary_check=(0, 1),
                padding_option="zero",
            )
            v = _s60_decode_fp8e4m3fn(v_bits).to(tl.bfloat16)

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            if QUANT_TYPE == 0:
                byte_offset = (
                    physical * KS_SBLOCK
                    + (pos // 32) * KS_STOKEN
                    + hkv * KS_SHEAD
                    + (pos % 32) * 4 * KS_SD
                )
                scale_ptr = (K_SCALE + byte_offset).to(
                    tl.pointer_type(tl.float32)
                )
                k_scale = tl.load(
                    scale_ptr, mask=valid_n, other=0.0
                ).to(tl.float32)
                scores *= (
                    q_scale[:, None]
                    * k_scale[None, :]
                    * dot_scale_log2
                )
            else:
                scores *= (
                    q_scale[:, None]
                    * tensor_k_scale
                    * dot_scale_log2
                )

            causal_n = valid_n & (logical < effective_len)
            scores = tl.where(
                valid_hq[:, None] & causal_n[None, :],
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
                tl.exp2(scores - safe_new[:, None]),
                0.0,
            )
            alpha = tl.where(active, tl.exp2(safe_old - safe_new), 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p_fp8 = _s60_roundtrip_fp8e4m3fn(p * 256.0).to(
                tl.bfloat16
            )
            acc = acc * alpha[:, None]
            acc = tl.dot(p_fp8, v, acc=acc, out_dtype=tl.float32)
            m_i = m_new
            start += BLOCK_N

        has_value = l_i > 0.0
        partial = tl.where(
            has_value[:, None],
            acc / l_i[:, None] * value_scale,
            0.0,
        )
        lse = tl.where(has_value, tl.log2(l_i) + m_i, -float("inf"))
        tl.store(
            SPLIT_OUT
            + batch * SO_SB
            + chunk * SO_SC
            + seq_m * SO_SM
            + hq[:, None] * SO_SH
            + offs_d[None, :],
            partial,
            mask=valid_hq[:, None],
        )
        tl.store(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + seq_m * SL_SM
            + hkv * SL_SK
            + offs_h * SL_SH,
            lse,
            mask=valid_hq,
        )

        slot += 1
        task = ((MAX_TASKS * cta + slot) + 1) * _TASK_STRIDE_JIT
        hkv = tl.load(
            TASK_MAP + task + 0,
            mask=slot < MAX_TASKS,
            other=-1,
        )


@triton.jit
def _s60_dynamic_fp8_combine_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    D: tl.constexpr,
    MAX_TASKS,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Combine FP8 split-K partials using their base-2 LSE values."""
    pid = tl.program_id(0).to(tl.int32)
    hq = pid % H_Q
    work = pid // H_Q
    seq_m = work % NUM_SEQ_Q
    batch = work // NUM_SEQ_Q
    hkv = hq // HEADS_PER_GROUP
    h_in_group = hq - hkv * HEADS_PER_GROUP
    offs_d = tl.arange(0, D)

    counts = (MAX_TASKS * NUM_CTAS + 1) * _TASK_STRIDE_JIT
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(
        TASK_MAP + counts + hkv * batch_count + batch
    )

    max_lse = tl.full((), -float("inf"), tl.float32)
    denom = tl.zeros((), tl.float32)
    out = tl.zeros((D,), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        value = tl.load(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + seq_m * SL_SM
            + hkv * SL_SK
            + h_in_group * SL_SH,
        )
        next_max = tl.maximum(max_lse, value)
        active = next_max != -float("inf")
        safe_next = tl.where(active, next_max, 0.0)
        safe_old = tl.where(max_lse == -float("inf"), safe_next, max_lse)
        alpha = tl.where(active, tl.exp2(safe_old - safe_next), 0.0)
        weight = tl.where(
            value != -float("inf"),
            tl.exp2(value - safe_next),
            0.0,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + chunk * SO_SC
            + seq_m * SO_SM
            + hq * SO_SH
            + offs_d,
        )
        out = out * alpha + weight * partial
        denom = denom * alpha + weight
        max_lse = next_max
        chunk += 1

    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT
        + batch * O_SB
        + seq_m * O_SM
        + hq * O_SH
        + offs_d,
        tl.where(denom > 0.0, out / safe_denom, 0.0),
    )


def prepare_s60_dynamic_fp8_workspace(
    inputs,
    quant_type: int,
) -> S60DynamicFP8Workspace:
    """Allocate an S60 workspace for MTP1/2/4 dynamic FP8 decode."""
    if quant_type not in (0, 1):
        raise ValueError(f"unsupported FP8 quantization id: {quant_type}")
    mtp = inputs.mtp
    if mtp not in (1, 2, 4):
        raise ValueError("S60 dynamic FP8 decode supports MTP 1, 2, or 4")
    batch = inputs.batch
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if hq % hkv:
        raise ValueError("query heads must be divisible by KV heads")
    if int(inputs.kv_lens.min().item()) < mtp:
        raise ValueError("each KV length must be at least MTP")

    heads_per_group = hq // hkv
    padded_heads = triton.cdiv(heads_per_group, 8) * 8
    lengths = inputs.kv_lens.detach().cpu().to(torch.int32).tolist()
    seq_tiles = [
        (int(length) + _TILE_N - 1) // _TILE_N for length in lengths
    ]
    total_tiles = sum(seq_tiles) * hkv
    uniform_seq_tiles = (
        seq_tiles[0]
        if seq_tiles and all(value == seq_tiles[0] for value in seq_tiles)
        else 0
    )
    workspace_bytes_per_cta = (
        batch * mtp * (hq * _HEAD_DIM + hkv * padded_heads) * 4
    )
    max_ctas_by_memory = max(
        1,
        _WORKSPACE_BUDGET_BYTES // workspace_bytes_per_cta,
    )
    num_ctas = max(
        1,
        min(
            total_tiles,
            _s60_compute_units(inputs.q.device) * _CTAS_PER_CU,
            _MAX_CTAS,
            max_ctas_by_memory,
        ),
    )
    min_process_len = _MIN_PROCESS_LEN
    max_tasks, sched_ints, workspace_ints = (
        _s60_dynamic_task_map_metadata(
            inputs.kv_lens,
            num_head_kv=hkv,
            num_ctas=num_ctas,
            min_process_len=min_process_len,
        )
    )
    return S60DynamicFP8Workspace(
        task_map=torch.full(
            (workspace_ints,),
            -1,
            dtype=torch.int32,
            device=inputs.q.device,
        ),
        split_out=torch.empty(
            (batch, num_ctas, mtp, hq, _HEAD_DIM),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        split_lse=torch.empty(
            (batch, num_ctas, mtp, hkv, padded_heads),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        out=torch.empty(
            (batch, mtp, hq, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=inputs.q.device,
        ),
        quant_type=quant_type,
        mtp=mtp,
        num_ctas=num_ctas,
        total_tiles=total_tiles,
        uniform_seq_tiles=uniform_seq_tiles,
        max_tasks=max_tasks,
        sched_ints=sched_ints,
        heads_per_group=heads_per_group,
        padded_heads_per_group=padded_heads,
        min_process_len=min_process_len,
    )


def attention_decode_s60_dynamic_fp8(
    inputs,
    workspace: S60DynamicFP8Workspace,
) -> torch.Tensor:
    """Run the native S60 dynamic FP8 producer and split-K combiner."""
    mtp = inputs.mtp
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (
        mtp != workspace.mtp
        or hq // hkv != workspace.heads_per_group
    ):
        raise ValueError("S60 FP8 workspace does not match inputs")

    _launch_s60_dynamic_task_map(
        inputs.kv_lens,
        workspace.task_map,
        num_head_kv=hkv,
        num_ctas=workspace.num_ctas,
        min_process_len=workspace.min_process_len,
        total_tiles=workspace.total_tiles,
        uniform_seq_tiles=workspace.uniform_seq_tiles,
        max_tasks=workspace.max_tasks,
        sched_ints=workspace.sched_ints,
    )

    q_4d = inputs.q.reshape(inputs.batch, mtp, hq, _HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(inputs.batch, mtp, hq)
    group_m = 8
    num_group_blocks = triton.cdiv(workspace.heads_per_group, group_m)
    # MTP stays on the program axis: a wider fused Q tile is not stable on
    # GCU300 and can raise a device-level SIP exception.
    producer_grid = workspace.num_ctas * num_group_blocks * mtp
    q_storage = q_4d.view(torch.uint8)
    k_storage = inputs.k_cache.view(torch.uint8)
    v_storage = inputs.v_cache.view(torch.uint8)
    k_scale_storage = (
        inputs.k_scale.view(torch.uint8)
        if workspace.quant_type == 0
        else inputs.k_scale
    )
    _s60_dynamic_fp8_splitk_kernel[(producer_grid,)](
        q_storage,
        k_storage,
        v_storage,
        inputs.block_ids,
        inputs.kv_lens,
        workspace.task_map,
        q_scale_3d,
        k_scale_storage,
        inputs.v_scale,
        workspace.split_out,
        workspace.split_lse,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        GROUP_M=group_m,
        NUM_GROUP_BLOCKS=num_group_blocks,
        D=_HEAD_DIM,
        BLOCK_SIZE=_BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        MAX_TASKS=workspace.max_tasks,
        BLOCK_N=_BLOCK_N,
        QUANT_TYPE=workspace.quant_type,
        Q_SB=q_4d.stride(0),
        Q_SM=q_4d.stride(1),
        Q_SH=q_4d.stride(2),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
        QS_SB=q_scale_3d.stride(0),
        QS_SM=q_scale_3d.stride(1),
        QS_SH=q_scale_3d.stride(2),
        KS_SBLOCK=inputs.k_scale.stride(0),
        KS_STOKEN=(
            inputs.k_scale.stride(1) if inputs.k_scale.ndim > 1 else 0
        ),
        KS_SHEAD=(
            inputs.k_scale.stride(2) if inputs.k_scale.ndim > 2 else 0
        ),
        KS_SD=(
            inputs.k_scale.stride(3) if inputs.k_scale.ndim > 3 else 0
        ),
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SK=workspace.split_lse.stride(3),
        SL_SH=workspace.split_lse.stride(4),
        num_warps=1,
        num_stages=1,
    )

    combine_grid = inputs.batch * mtp * hq
    _s60_dynamic_fp8_combine_kernel[(combine_grid,)](
        workspace.split_out,
        workspace.split_lse,
        workspace.task_map,
        workspace.out,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        D=_HEAD_DIM,
        MAX_TASKS=workspace.max_tasks,
        NUM_CTAS=workspace.num_ctas,
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SK=workspace.split_lse.stride(3),
        SL_SH=workspace.split_lse.stride(4),
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return workspace.out.reshape_as(inputs.q)


def s60_dynamic_fp8_workspace_is_reset(
    workspace: S60DynamicFP8Workspace,
) -> bool:
    """The S60 implementation has no cooperative completion counters."""
    return True

QUANT_TYPE = "qkpertoken_perhead_vperhead"
QUANT_TYPE_ID = 0
BLOCK_SIZE = _BLOCK_SIZE
HEAD_DIM = _HEAD_DIM
SUPPORTED_MTP = (1, 2, 4)
QUANT_TYPES = ("qkpertoken_perhead_vperhead",)

OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}

_DYNAMIC_CT = {
    1: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (8, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (4, 256),
        "one_64k_31x4k": (8, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 512),
    },
    2: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 256),
        "one_64k_15x4k": (4, 512),
        "one_64k_31x4k": (4, 512),
        "one_128k_31x4k": (4, 1024),
        "two_32k_30x4k": (4, 512),
    },
    4: {
        "uniform_512": (2, 256),
        "uniform_4096": (2, 1024),
        "skewed_mix": (2, 512),
        "skewed_extreme": (4, 128),
        "one_64k_7x4k": (4, 512),
        "one_64k_15x4k": (4, 512),
        "one_64k_31x4k": (4, 1024),
        "one_128k_31x4k": (2, 1024),
        "two_32k_30x4k": (4, 1024),
    },
}


@dataclass(repr=False)
class FP8DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


def _layout(cache: torch.Tensor) -> str:
    return "HND" if cache.stride(2) > cache.stride(1) else "NHD"


def _validate(
    inputs: FP8DecodeInputs,
    *,
    check_kv_values: bool = True,
) -> None:
    if inputs.mtp not in SUPPORTED_MTP:
        raise ValueError("S60 dynamic FP8 decode supports MTP 1, 2, or 4")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must have shape [batch * MTP,Hq,128]")
    if inputs.q.dtype != torch.float8_e4m3fn:
        raise ValueError("q must be float8_e4m3fn")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.element_size() != 1:
            raise ValueError(f"{name} must use a 1-byte FP8 storage dtype")
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.shape[-1] != HEAD_DIM or cache.stride(-1) != 1:
            raise ValueError(f"{name} head dimension must be contiguous 128")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if (num_head_kv, num_head_q) not in ((1, 8), (4, 32)):
        raise ValueError(
            "S60 dynamic FP8 decode requires GQA8 heads "
            "(Hkv,Hq)=(1,8) or (4,32)"
        )
    if inputs.block_ids.dtype != torch.int32 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError("block_ids and kv_lens must be int32")
    if check_kv_values and int(inputs.kv_lens.min().item()) < inputs.mtp:
        raise ValueError("each KV length must be at least MTP")


def _case(inputs: FP8DecodeInputs) -> str:
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    for name, official in OFFICIAL_CASES.items():
        if lengths == official:
            return name
    return "uniform_512" if max(lengths) <= 1024 else "uniform_4096"


@dataclass(frozen=True)
class FP8DecodePolicy:
    schedule: Literal["dynamic"]
    mtp: int
    workload: object
    layout: str
    cluster_size: int
    chunk_tokens: int
    route: str

    @property
    def label(self) -> str:
        return f"c{self.cluster_size}t{self.chunk_tokens}/{self.route}"


def select_decode_policy(inputs: FP8DecodeInputs) -> FP8DecodePolicy:
    _validate(inputs)
    case = _case(inputs)
    cluster_size, chunk_tokens = _DYNAMIC_CT[inputs.mtp][case]
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return FP8DecodePolicy(
        "dynamic",
        inputs.mtp,
        lengths,
        _layout(inputs.k_cache),
        cluster_size,
        chunk_tokens,
        "s60-dynamic-task-map",
    )

FP8DecodeWorkspace = S60DynamicFP8Workspace


def prepare_decode_workspace(inputs: FP8DecodeInputs) -> S60DynamicFP8Workspace:
    _validate(inputs)
    return prepare_s60_dynamic_fp8_workspace(inputs, QUANT_TYPE_ID)


def attention_decode_fp8(
    inputs: FP8DecodeInputs,
    workspace: S60DynamicFP8Workspace,
) -> torch.Tensor:
    # Device-value validation is completed while preparing the workspace.
    # Avoid .item() here because this entrypoint is captured by GCU Graph.
    _validate(inputs, check_kv_values=False)
    if not isinstance(workspace, S60DynamicFP8Workspace):
        raise TypeError("workspace must be prepared by the S60 dynamic FP8 path")
    return attention_decode_s60_dynamic_fp8(inputs, workspace)


def workspace_is_reset(workspace: S60DynamicFP8Workspace) -> bool:
    if not isinstance(workspace, S60DynamicFP8Workspace):
        return False
    return s60_dynamic_fp8_workspace_is_reset(workspace)


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "QUANT_TYPE",
    "QUANT_TYPE_ID",
    "QUANT_TYPES",
    "SUPPORTED_MTP",
    "FP8DecodeInputs",
    "FP8DecodePolicy",
    "FP8DecodeWorkspace",
    "attention_decode_fp8",
    "prepare_decode_workspace",
    "select_decode_policy",
    "workspace_is_reset",
]
