"""Shared BF16 MTP split-K combine kernel for Hygon hy3."""

from __future__ import annotations

import triton
import triton.language as tl

@triton.jit
def _combine(
    SPLIT_OUT, SPLIT_LSE, TASK_MAP, OUT,
    H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, MTP: tl.constexpr,
    D: tl.constexpr,
    MAX_TASKS: tl.constexpr, NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr, SO_SC: tl.constexpr, SO_SH: tl.constexpr,
    SL_SB: tl.constexpr, SL_SC: tl.constexpr, SL_SK: tl.constexpr, SL_SH: tl.constexpr,
    O_SB: tl.constexpr, O_SH: tl.constexpr,
):
    batch = tl.program_id(0)
    flat_hq = tl.program_id(1)
    query = flat_hq // H_Q
    hq = flat_hq - query * H_Q
    hkv = hq // HEADS_PER_GROUP
    row = query * HEADS_PER_GROUP + hq - hkv * HEADS_PER_GROUP
    d = tl.arange(0, D)
    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)
    m = tl.full((), -float("inf"), tl.float32)
    denom = tl.zeros((), tl.float32)
    acc = tl.zeros((D,), tl.float32)
    chunk = 0
    while chunk < chunks:
        lse = tl.load(
            SPLIT_LSE + batch * SL_SB + chunk * SL_SC
            + hkv * SL_SK + row * SL_SH
        )
        part = tl.load(SPLIT_OUT + batch * SO_SB + chunk * SO_SC + flat_hq * SO_SH + d)
        new_m = tl.maximum(m, lse)
        active = new_m != -float("inf")
        safe_m = tl.where(active, new_m, 0.0)
        old_m = tl.where(m == -float("inf"), safe_m, m)
        alpha = tl.where(active, tl.exp(old_m - safe_m), 0.0)
        weight = tl.where(active, tl.exp(lse - safe_m), 0.0)
        acc = acc * alpha + part * weight
        denom = denom * alpha + weight
        m = new_m
        chunk += 1
    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT + (batch * MTP + query) * O_SB + hq * O_SH + d,
        tl.where(denom > 0.0, acc / safe_denom, 0.0),
    )

