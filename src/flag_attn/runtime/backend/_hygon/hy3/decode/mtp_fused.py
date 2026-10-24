"""BF16 and INT8 MTP fusion for the portable Hygon Triton decode path.

The fused producer tiles MTP * GQA_group rows in 16-row dot tiles.
Fusion is prepared when split-K parallelism is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .core import _effective_mtp_rows
from .scheduler import launch_pure_triton_task_map
from .combine import _combine


@dataclass
class FusedMTPWorkspace:
    q1_inputs: object
    q1_workspace: object
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    quant_type: str


@triton.jit
def _producer(
    Q, K, V, BLOCK_IDS, KV_LENS, TASK_MAP, SPLIT_OUT, SPLIT_LSE,
    H_Q: tl.constexpr, BATCH_SIZE: tl.constexpr, MTP: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr,
    NUM_BLOCKS: tl.constexpr, MAX_TASKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_SB: tl.constexpr, Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr, K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr, K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr, V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr, V_SD: tl.constexpr,
    SO_SB: tl.constexpr, SO_SC: tl.constexpr, SO_SH: tl.constexpr,
    SL_SB: tl.constexpr, SL_SC: tl.constexpr,
    SL_SK: tl.constexpr, SL_SH: tl.constexpr,
    ROWS: tl.constexpr, ROW_GROUPS: tl.constexpr,
    INT8_INPUT: tl.constexpr, QK_SCALE: tl.constexpr, V_SCALE: tl.constexpr,
):
    """Split-K producer that fuses MTP and GQA rows for BF16 or INT8 input."""
    cta = tl.program_id(0)
    row_group = tl.program_id(1) if ROW_GROUPS > 1 else 0
    rows = tl.arange(0, ROWS)
    logical_rows = row_group * ROWS + rows
    query = logical_rows // HEADS_PER_GROUP
    head_in_group = logical_rows % HEADS_PER_GROUP
    d = tl.arange(0, D)
    n = tl.arange(0, BLOCK_N)
    qk_scale = tl.rsqrt(tl.full((), D, tl.float32))

    for slot in tl.static_range(0, MAX_TASKS):
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(TASK_MAP + task)
        batch = tl.load(TASK_MAP + task + 1)
        if (hkv < 0) | (batch < 0) | (batch >= BATCH_SIZE):
            return
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        hq = hkv * HEADS_PER_GROUP + head_in_group
        row_valid = logical_rows < MTP * HEADS_PER_GROUP
        q_batch = batch * MTP + query
        q = tl.load(
            Q + q_batch[:, None] * Q_SB + hq[:, None] * Q_SH + d[None, :],
            mask=row_valid[:, None],
            other=0.0,
        )
        if INT8_INPUT:
            q = q.to(tl.int8)
        else:
            q = q.to(tl.bfloat16)
        kv_len = tl.load(KV_LENS + batch)
        m = tl.full((ROWS,), -float("inf"), tl.float32)
        l = tl.zeros((ROWS,), tl.float32)
        acc = tl.zeros((ROWS, D), tl.float32)
        start = 0
        while start < seq_len:
            local_n = start + n
            valid_n = local_n < seq_len
            tile_start = seq_start + start
            page = tile_start // BLOCK_SIZE
            pos = tile_start - page * BLOCK_SIZE + n
            page_valid = (page >= 0) & (page < MAX_BLOCKS)
            physical = tl.load(
                BLOCK_IDS + batch * MAX_BLOCKS + page,
                mask=page_valid,
                other=0,
            )
            valid = (
                valid_n & page_valid & (physical >= 0) & (physical < NUM_BLOCKS)
            )
            k = tl.load(
                K + physical * K_SBLOCK + pos[:, None] * K_STOKEN
                + hkv * K_SHEAD + d[None, :] * K_SD,
                mask=valid[:, None],
                other=0.0,
            )
            if INT8_INPUT:
                k = k.to(tl.int8)
                scores = (
                    tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
                    * qk_scale
                    * QK_SCALE
                )
            else:
                k = k.to(tl.bfloat16)
                scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * qk_scale
            key_pos = page * BLOCK_SIZE + pos
            query_kv_len = kv_len - (MTP - 1) + query
            key_valid = valid[None, :] & (
                key_pos[None, :] < query_kv_len[:, None]
            )
            scores = tl.where(row_valid[:, None] & key_valid, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_m = tl.maximum(m, tile_max)
            active = new_m != -float("inf")
            safe_m = tl.where(active, new_m, 0.0)
            old_m = tl.where(m == -float("inf"), safe_m, m)
            p = tl.where(
                active[:, None], tl.exp(scores - safe_m[:, None]), 0.0
            )
            alpha = tl.where(active, tl.exp(old_m - safe_m), 0.0)
            l = l * alpha + tl.sum(p, axis=1)
            v = tl.load(
                V + physical * V_SBLOCK + pos[:, None] * V_STOKEN
                + hkv * V_SHEAD + d[None, :] * V_SD,
                mask=valid[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(
                p.to(tl.bfloat16), v, out_dtype=tl.float32
            )
            m = new_m
            start += BLOCK_N

        has_value = l > 0.0
        safe_l = tl.where(has_value, l, 1.0)
        partial = tl.where(
            has_value[:, None], acc / safe_l[:, None], 0.0
        )
        if INT8_INPUT:
            partial = partial * V_SCALE
        lse = tl.where(has_value, tl.log(safe_l) + m, -float("inf"))
        flat_hq = query * H_Q + hq
        tl.store(
            SPLIT_OUT + batch * SO_SB + chunk * SO_SC
            + flat_hq[:, None] * SO_SH + d[None, :],
            partial,
            mask=row_valid[:, None],
        )
        tl.store(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC
            + hkv * SL_SK + logical_rows * SL_SH,
            lse,
            mask=row_valid,
        )




def prepare_fused_mtp_workspace(inputs, serial_workspace, quant_type: str):
    if quant_type not in ("bf16", "int8"):
        raise ValueError("fused decode supports BF16 and INT8")
    expected_dtype = torch.bfloat16 if quant_type == "bf16" else torch.int8
    if any(
        tensor.dtype != expected_dtype
        for tensor in (inputs.q, inputs.k_cache, inputs.v_cache)
    ):
        raise ValueError(
            f"fused {quant_type.upper()} decode requires matching Q/K/V dtypes"
        )
    num_kv_heads = int(inputs.k_cache.shape[2])
    if num_kv_heads <= 0 or inputs.q.shape[1] % num_kv_heads != 0:
        raise ValueError("fused decode requires an integral query-head group")
    if int(inputs.mtp) < 2:
        raise ValueError("fused decode requires MTP >= 2")
    effective_rows = _effective_mtp_rows(inputs)
    q1, q1ws = serial_workspace.mtp1_inputs[-1], serial_workspace.workspaces[-1]
    if q1ws.uniform_length is not None:
        raise ValueError("fused decode requires split-K task-map mode")
    batch = int(inputs.batch)
    hq, hkv, d = (
        int(inputs.q.shape[1]), int(inputs.k_cache.shape[2]), int(inputs.q.shape[-1])
    )
    ctas = int(q1ws.num_ctas)
    return FusedMTPWorkspace(
        q1_inputs=q1,
        q1_workspace=q1ws,
        split_out=torch.empty(
            (batch, ctas, int(inputs.mtp) * hq, d),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        split_lse=torch.empty(
            (batch, ctas, hkv, effective_rows),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        out=torch.empty_like(inputs.q, dtype=torch.bfloat16),
        quant_type=quant_type,
    )


def run_fused_mtp(inputs, workspace: FusedMTPWorkspace):
    q1, ws = workspace.q1_inputs, workspace.q1_workspace
    stream = int(torch.cuda.current_stream(inputs.q.device).cuda_stream)
    stale = (
        not ws.task_map_ready
        or int(getattr(q1.kv_lens, "_version", -1)) != ws.kv_lens_version
        or int(q1.kv_lens.data_ptr()) != ws.kv_lens_ptr
        or (ws.task_map_stream != 0 and stream != ws.task_map_stream)
    )
    if stale:
        launch_pure_triton_task_map(
            q1.kv_lens,
            ws.task_map,
            num_head_kv=inputs.k_cache.shape[2],
            num_ctas=ws.num_ctas,
            min_process_len=ws.min_process_len,
            max_tasks=ws.max_tasks,
            sched_ints=ws.sched_ints,
        )
        ws.task_map_ready = True
        ws.kv_lens_version = int(getattr(q1.kv_lens, "_version", -1))
        ws.kv_lens_ptr = int(q1.kv_lens.data_ptr())
        ws.task_map_stream = stream

    q, k, v = inputs.q, inputs.k_cache, inputs.v_cache
    so, sl, out = workspace.split_out, workspace.split_lse, workspace.out
    hq, hkv, d = int(q.shape[1]), int(k.shape[2]), int(q.shape[-1])
    group = hq // hkv
    mtp = int(inputs.mtp)
    effective_rows = _effective_mtp_rows(inputs)
    tile_rows = 16
    row_groups = triton.cdiv(effective_rows, tile_rows)
    producer_grid = (ws.num_ctas, row_groups)
    default_block_n = (
        32
        if mtp == 2 and effective_rows == 16 and ws.max_tasks >= 4
        else 64
    )
    block_n = default_block_n
    _producer[producer_grid](
        q, k, v, inputs.block_ids, inputs.kv_lens, ws.task_map, so, sl,
        H_Q=hq,
        BATCH_SIZE=inputs.batch,
        MTP=mtp,
        HEADS_PER_GROUP=group,
        D=d,
        BLOCK_SIZE=k.shape[1],
        MAX_BLOCKS=inputs.block_ids.shape[1],
        NUM_BLOCKS=k.shape[0],
        MAX_TASKS=ws.max_tasks,
        BLOCK_N=block_n,
        Q_SB=q.stride(0),
        Q_SH=q.stride(1),
        K_SBLOCK=k.stride(0),
        K_STOKEN=k.stride(1),
        K_SHEAD=k.stride(2),
        K_SD=k.stride(3),
        V_SBLOCK=v.stride(0),
        V_STOKEN=v.stride(1),
        V_SHEAD=v.stride(2),
        V_SD=v.stride(3),
        SO_SB=so.stride(0),
        SO_SC=so.stride(1),
        SO_SH=so.stride(2),
        SL_SB=sl.stride(0),
        SL_SC=sl.stride(1),
        SL_SK=sl.stride(2),
        SL_SH=sl.stride(3),
        ROWS=tile_rows,
        ROW_GROUPS=row_groups,
        INT8_INPUT=workspace.quant_type == "int8",
        QK_SCALE=(
            float(getattr(inputs, "q_scale", 1.0))
            * float(getattr(inputs, "k_scale", 1.0))
        ),
        V_SCALE=float(getattr(inputs, "v_scale", 1.0)),
        num_warps=4,
        num_stages=2,
    )
    _combine[(inputs.batch, mtp * hq)](
        so,
        sl,
        ws.task_map,
        out,
        H_Q=hq,
        HEADS_PER_GROUP=group,
        MTP=mtp,
        D=d,
        MAX_TASKS=ws.max_tasks,
        NUM_CTAS=ws.num_ctas,
        SO_SB=so.stride(0),
        SO_SC=so.stride(1),
        SO_SH=so.stride(2),
        SL_SB=sl.stride(0),
        SL_SC=sl.stride(1),
        SL_SK=sl.stride(2),
        SL_SH=sl.stride(3),
        O_SB=out.stride(0),
        O_SH=out.stride(1),
        num_warps=4,
        num_stages=2,
    )
    return out
