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

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .compat import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE_FLASHMLA_SPARSE = True
    except ImportError:
        tle = None
        HAS_TLE_FLASHMLA_SPARSE = False
else:
    tle = None
    HAS_TLE_FLASHMLA_SPARSE = False


TLE_FLASHMLA_PREFILL_BK = 64
TLE_FLASHMLA_PREFILL_BH = 64
TLE_FLASHMLA_PREFILL_PAIR_BLOCKS = 2
TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS = 4

# MetaX C550 serial TLE path.
# Keep tiles small to stay below the 64 KiB shared-memory limit.
METAX_SERIAL_TLE_BK = 16
METAX_SERIAL_TLE_BH = 16
METAX_SERIAL_TLE_NUM_WARPS = 2


# ============================================================
# V6.2: adaptive split-TOPK policy
#
# Result of empirical sweep on MetaX C550.
#
# split=1 means: use original V1 kernel.
#
# Only install policies with meaningful measured gain.
# ~1% noise-level wins are intentionally left as split=1.
# ============================================================

_V6_SPLIT_POLICY = {
    # --------------------------------------------------------
    # HQ = 64
    # --------------------------------------------------------
    64: {
        # SQ : {TOPK: NUM_SPLITS}
        1: {
            512: 16,
            1024: 16,
            2048: 16,
        },
        2: {
            512: 8,
            1024: 8,
            2048: 8,
        },
        4: {
            512: 32,
            1024: 16,
            2048: 16,
        },
        8: {
            512: 32,
            1024: 16,
            2048: 16,
        },
        16: {
            512: 32,
            1024: 8,
            2048: 8,
        },
        32: {
            512: 1,
            1024: 4,
            2048: 4,
        },
    },

    # --------------------------------------------------------
    # HQ = 128
    # --------------------------------------------------------
    128: {
        1: {
            512: 8,
            1024: 8,
            2048: 8,
        },
        2: {
            512: 32,
            1024: 64,
            2048: 16,
        },
        4: {
            512: 32,
            1024: 16,
            2048: 16,
        },
        8: {
            512: 32,
            1024: 8,
            2048: 8,
        },
        16: {
            512: 32,
            1024: 4,
            2048: 4,
        },
    },
}


def _v6_choose_num_splits(
    sq: int,
    hq: int,
    topk: int,
) -> int:
    """Return measured best split count.

    Unknown shapes deliberately fall back to split=1.
    """

    hq_policy = _V6_SPLIT_POLICY.get(hq)

    if hq_policy is None:
        return 1

    sq_policy = hq_policy.get(sq)

    if sq_policy is None:
        return 1

    return sq_policy.get(topk, 1)


# ============================================================
# V6.2 reusable workspace
#
# key:
#   (device, NUM_SPLITS, HQ, DV)
#
# Each entry remembers the largest SQ allocated so far.
# Smaller subsequent SQ values reuse a slice.
#
# Experimental path is intended for the normal single-stream
# attention execution used by this benchmark.
#
# A production integration with concurrent streams should use
# vLLM's workspace manager or a caller-owned workspace.
# ============================================================

_V6_WORKSPACE_CACHE = {}


def _v6_get_workspace(
    q,
    sq: int,
    hq: int,
    dv: int,
    num_splits: int,
):

    device_index = (
        q.device.index
        if q.device.index is not None
        else 0
    )

    key = (
        device_index,
        num_splits,
        hq,
        dv,
    )

    entry = _V6_WORKSPACE_CACHE.get(key)

    need_new = (
        entry is None
        or entry["capacity_sq"] < sq
    )

    if need_new:

        # Grow geometrically so slightly larger SQ does not
        # repeatedly cause reallocations.
        if entry is None:
            capacity_sq = sq
        else:
            capacity_sq = max(
                sq,
                entry["capacity_sq"] * 2,
            )

        partial_acc = torch.empty(
            (
                num_splits,
                capacity_sq,
                hq,
                dv,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        partial_max = torch.empty(
            (
                num_splits,
                capacity_sq,
                hq,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        partial_sum = torch.empty_like(
            partial_max
        )

        entry = {
            "capacity_sq": capacity_sq,
            "partial_acc": partial_acc,
            "partial_max": partial_max,
            "partial_sum": partial_sum,
        }

        _V6_WORKSPACE_CACHE[key] = entry

    # Views preserve the parent strides, which is fine:
    # all strides are passed explicitly to Triton.
    partial_acc = entry["partial_acc"][:, :sq]
    partial_max = entry["partial_max"][:, :sq]
    partial_sum = entry["partial_sum"][:, :sq]

    return (
        partial_acc,
        partial_max,
        partial_sum,
    )




# ============================================================
# V6: split-TOPK path for low-SQ sparse prefill
# ============================================================

V6_SPLIT_MAX_SQ = 16
V6_NUM_SPLITS = 4
V6_SPLIT_BK = 16
V6_SPLIT_BH = 16


@triton.autotune(
    configs=[
        # ====================================================
        # V0 baseline
        # ====================================================
        triton.Config(
            {"BK": 16, "BH": 16},
            num_warps=2,
            num_stages=1,
        ),

        # ====================================================
        # Increase BK:
        # reduce the number of serial TOPK iterations.
        # ====================================================
        triton.Config(
            {"BK": 32, "BH": 16},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BK": 64, "BH": 16},
            num_warps=4,
            num_stages=1,
        ),

        # ====================================================
        # Increase BH:
        # process more Q heads per program and reduce repeated
        # sparse-KV gathers when HKV == 1.
        # ====================================================
        triton.Config(
            {"BK": 16, "BH": 32},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BK": 32, "BH": 32},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BK": 64, "BH": 32},
            num_warps=4,
            num_stages=1,
        ),

        # ====================================================
        # Probe BH=64.
        # MetaX official sparse-prefill uses BlockM=64.
        # These may hit register/resource limits in Triton;
        # autotune can discard invalid configs.
        # ====================================================
        triton.Config(
            {"BK": 16, "BH": 64},
            num_warps=4,
            num_stages=1,
        ),
        triton.Config(
            {"BK": 32, "BH": 64},
            num_warps=8,
            num_stages=1,
        ),
        triton.Config(
            {"BK": 64, "BH": 64},
            num_warps=8,
            num_stages=1,
        ),
    ],
    key=[
        "SQ",
        "HQ",
        "DQK",
        "SKV",
        "TOPK",
        "HAVE_ATTN_SINK",
        "HAVE_TOPK_LENGTH",
    ],
)

@triton.jit
def triton_flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    attn_sink,
    topk_length,
    sm_scale: tl.constexpr,
    output,
    max_logits,
    lse,
    stride_qh,
    stride_qm,
    stride_kvg,
    stride_kvn,
    stride_tg,
    stride_tm,
    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,
    SQ,  # s_q
    HQ: tl.constexpr,  # h_q=64 or 128
    DQK: tl.constexpr,  # d_qk=512 or 576
    SKV,  # s_kv
    TOPK: tl.constexpr,  # topk
    HAVE_ATTN_SINK: tl.constexpr,
    HAVE_TOPK_LENGTH: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH
    pid = tl.program_id(0)
    i_sq = pid // num_head_blocks
    i_sq = i_sq.to(tl.int64)  # prevent mul overflow
    i_gbh = pid % num_head_blocks
    gbh_base = i_gbh * BH
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    q_base = q + i_sq * stride_qm + gbh_base * stride_qh
    kv_base = kv
    tkv_base = kv + DP
    t_base = indices + i_sq * stride_tm
    attn_sink_ptr = attn_sink + gbh_base if HAVE_ATTN_SINK else 0
    topk_length_ptr = topk_length + i_sq if HAVE_TOPK_LENGTH else 0
    o_base = output + i_sq * stride_om + gbh_base * stride_oh
    max_log_base = max_logits + i_sq * stride_mm + gbh_base
    l_base = lse + i_sq * stride_lm + gbh_base

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    if DQK == 576:
        offs_td = tl.arange(0, 64)
    offs_t = tl.arange(0, BK)

    # `[BH, 256] x 2` delivers better performance than `[BH, 512]` when BH=64
    q_ptr = q_base + offs_h[:, None] * stride_qh + offs_d[None, :]
    q_blk0 = tl.load(q_ptr, eviction_policy="evict_first")
    q_blk1 = tl.load(q_ptr + BDP, eviction_policy="evict_first")
    if DQK == 576:
        tq_ptr = q_base + DP + offs_h[:, None] * stride_qh + offs_td[None, :]
        tq_blk = tl.load(tq_ptr, eviction_policy="evict_first")

    max_log = tl.full([BH], float("-inf"), dtype=tl.float32)
    sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
    acc0 = tl.zeros([BH, BDP], dtype=tl.float32)
    acc1 = tl.zeros([BH, BDP], dtype=tl.float32)

    topk_len = tl.load(topk_length_ptr) if HAVE_TOPK_LENGTH else TOPK
    NK = tl.cdiv(topk_len, BK)
    for ck in range(NK):
        # step1: load indices
        t_ptr = BK * ck + offs_t  # [BK]
        t_msk = t_ptr < topk_len
        t_ptr += t_base
        kv_ids = tl.load(t_ptr, t_msk, other=-1)
        mask_ids = (kv_ids < SKV) & (kv_ids >= 0)
        # filter invalid index that may cause overflow in mul
        kv_ids = tl.where(mask_ids, kv_ids, 0)

        # step2: gather KV as [BK, D]
        #
        # Each sparse token owns one row and the D dimension
        # is contiguous in memory:
        #
        #   [token0_d0, token0_d1, ...]
        #   [token1_d0, token1_d1, ...]
        #
        # This is more natural for coalesced/vectorized global
        # loads than the original [D, BK] gather.
        kv_ptr = (
            kv_base
            + kv_ids[:, None] * stride_kvn
            + offs_d[None, :]
        )

        kv_blk0 = tl.load(
            kv_ptr,
            cache_modifier=".cg",
        )  # [BK, BDP]

        kv_blk1 = tl.load(
            kv_ptr + BDP,
            cache_modifier=".cg",
        )  # [BK, BDP]

        # step3: Q @ K^T
        qk = tl.dot(
            q_blk0,
            kv_blk0.trans(),
            out_dtype=tl.float32,
        )  # [BH, BDP]@[BDP, BK] -> [BH, BK]

        qk = tl.dot(
            q_blk1,
            kv_blk1.trans(),
            qk,
            out_dtype=tl.float32,
        )

        if DQK == 576:
            tkv_ptr = (
                tkv_base
                + kv_ids[:, None] * stride_kvn
                + offs_td[None, :]
            )

            tkv_blk = tl.load(
                tkv_ptr,
                cache_modifier=".cg",
            )  # [BK, 64]

            qk = tl.dot(
                tq_blk,
                tkv_blk.trans(),
                qk,
                out_dtype=tl.float32,
            )
        qk *= sm_scale

        # step4: preprocess for logsumexp
        qk = tl.where(mask_ids[None, :], qk, float("-inf"))  # [BH, BK]
        # step5: lse=logsumexp(qk), loop part
        new_max = tl.maximum(max_log, tl.max(qk, axis=1))  # [BH]
        exp_qk = tl.math.exp(qk - new_max[:, None])  # [BH, BK]
        sum_qk = tl.sum(exp_qk, axis=1)  # [BH]
        alpha = tl.math.exp(max_log - new_max)  # [BH]
        sum_exp = sum_exp * alpha + sum_qk  # [BH]
        # step6: exp(qk-lse) @ gathered_kv.trans(), loop part
        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0,
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP] -> [BH, BDP]

        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1,
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )  # [BH, BK]@[BK, BDP] -> [BH, BDP]
        max_log = new_max

    # step7: store max_logits
    valid_mask = max_log != float("-inf")
    max_log = tl.where(valid_mask, max_log, float("-inf"))
    tl.store(max_log_base + offs_h, max_log)  # [BH], float32

    # step8: lse=logsumexp(qk) final part, store lse
    orig_lse = max_log + tl.math.log(sum_exp)
    lse_out = tl.where(valid_mask, orig_lse, float("inf"))
    tl.store(l_base + offs_h, lse_out)  # [BH], float32

    # step9: exp(qk-lse) @ gathered_kv.trans(), final part
    if HAVE_ATTN_SINK:
        # step10: attn_sink
        sink = tl.load(attn_sink_ptr + offs_h)  # [BH]
        sum_exp_new_lse = tl.math.exp(orig_lse) + tl.math.exp(sink)
        factor = tl.math.exp(max_log) / sum_exp_new_lse
    else:
        factor = 1.0 / sum_exp

    out_vals0 = tl.where(valid_mask[:, None], acc0 * factor[:, None], 0.0)
    out_vals1 = tl.where(valid_mask[:, None], acc1 * factor[:, None], 0.0)
    # step11: store output
    o_ptr = o_base + offs_h[:, None] * stride_oh + offs_d[None, :]  # [BH, BDP]
    tl.store(o_ptr, out_vals0.to(tl.bfloat16))
    tl.store(o_ptr + BDP, out_vals1.to(tl.bfloat16))




# ============================================================
# V6 Stage 1
#
# Split TOPK into NUM_SPLITS independent ranges.
#
# Each program owns:
#
#   one query
#   one BH=16 head block
#   one TOPK split
#
# It produces an unnormalized online-softmax state:
#
#   partial_max
#   partial_sum
#   partial_acc[..., 512]
#
# Stage 2 merges these states exactly.
# ============================================================


@triton.jit
def triton_flash_mla_sparse_split_fwd(
    q,
    kv,
    indices,
    topk_length,
    sm_scale: tl.constexpr,

    partial_acc,
    partial_max,
    partial_sum,

    stride_qh,
    stride_qm,
    stride_kvn,
    stride_tm,

    stride_pa_s,
    stride_pa_q,
    stride_pa_h,

    stride_ps_s,
    stride_ps_q,

    SQ,
    HQ: tl.constexpr,
    DQK: tl.constexpr,
    SKV,
    TOPK: tl.constexpr,

    HAVE_TOPK_LENGTH: tl.constexpr,

    NUM_SPLITS: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
):
    DP: tl.constexpr = 512
    BDP: tl.constexpr = 256

    num_head_blocks: tl.constexpr = HQ // BH
    programs_per_q: tl.constexpr = (
        num_head_blocks * NUM_SPLITS
    )

    pid = tl.program_id(0)

    i_sq = pid // programs_per_q
    local_pid = pid % programs_per_q

    i_gbh = local_pid // NUM_SPLITS
    i_split = local_pid % NUM_SPLITS

    i_sq64 = i_sq.to(tl.int64)

    gbh_base = i_gbh * BH

    # --------------------------------------------------------
    # Each split owns a contiguous TOPK range.
    #
    # For our current benchmark shapes:
    #
    # TOPK 512  -> 128 tokens/split ->  8 BK loops
    # TOPK 1024 -> 256 tokens/split -> 16 BK loops
    # TOPK 2048 -> 512 tokens/split -> 32 BK loops
    #
    # versus V1:
    #
    # 32 / 64 / 128 loops.
    # --------------------------------------------------------

    SPLIT_TOKENS: tl.constexpr = TOPK // NUM_SPLITS
    SPLIT_NK: tl.constexpr = SPLIT_TOKENS // BK

    split_start = i_split * SPLIT_TOKENS

    # --------------------------------------------------------
    # Base pointers
    # --------------------------------------------------------

    q_base = (
        q
        + i_sq64 * stride_qm
        + gbh_base * stride_qh
    )

    kv_base = kv
    tkv_base = kv + DP

    t_base = (
        indices
        + i_sq64 * stride_tm
    )

    topk_len_ptr = (
        topk_length + i_sq64
        if HAVE_TOPK_LENGTH
        else 0
    )

    # --------------------------------------------------------
    # Q
    # --------------------------------------------------------

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_t = tl.arange(0, BK)

    q_ptr = (
        q_base
        + offs_h[:, None] * stride_qh
        + offs_d[None, :]
    )

    q_blk0 = tl.load(
        q_ptr,
        eviction_policy="evict_first",
    )

    q_blk1 = tl.load(
        q_ptr + BDP,
        eviction_policy="evict_first",
    )

    if DQK == 576:
        offs_td = tl.arange(0, 64)

        tq_ptr = (
            q_base
            + DP
            + offs_h[:, None] * stride_qh
            + offs_td[None, :]
        )

        tq_blk = tl.load(
            tq_ptr,
            eviction_policy="evict_first",
        )

    # --------------------------------------------------------
    # Partial online-softmax state
    # --------------------------------------------------------

    max_log = tl.full(
        [BH],
        float("-inf"),
        dtype=tl.float32,
    )

    sum_exp = tl.zeros(
        [BH],
        dtype=tl.float32,
    )

    acc0 = tl.zeros(
        [BH, BDP],
        dtype=tl.float32,
    )

    acc1 = tl.zeros(
        [BH, BDP],
        dtype=tl.float32,
    )

    topk_len = (
        tl.load(topk_len_ptr)
        if HAVE_TOPK_LENGTH
        else TOPK
    )

    # ========================================================
    # Only this split's TOPK segment
    # ========================================================

    for ck_rel in tl.range(0, SPLIT_NK):

        t_off = (
            split_start
            + ck_rel * BK
            + offs_t
        )

        t_mask = t_off < topk_len

        kv_ids = tl.load(
            t_base + t_off,
            mask=t_mask,
            other=-1,
        )

        valid_ids = (
            t_mask
            & (kv_ids >= 0)
            & (kv_ids < SKV)
        )

        safe_kv_ids = tl.where(
            valid_ids,
            kv_ids,
            0,
        ).to(tl.int64)

        kv_offsets = (
            safe_kv_ids * stride_kvn
        )

        # ----------------------------------------------------
        # Gather KV[0:256]
        # ----------------------------------------------------

        kv_ptr = (
            kv_base
            + offs_d[:, None]
            + kv_offsets[None, :]
        )

        kv_blk0 = tl.load(
            kv_ptr,
            cache_modifier=".cg",
        )

        # ----------------------------------------------------
        # Gather KV[256:512]
        # ----------------------------------------------------

        kv_blk1 = tl.load(
            kv_ptr + BDP,
            cache_modifier=".cg",
        )

        # ----------------------------------------------------
        # QK: first 512 dims
        # ----------------------------------------------------

        qk = tl.dot(
            q_blk0,
            kv_blk0,
            out_dtype=tl.float32,
        )

        qk = tl.dot(
            q_blk1,
            kv_blk1,
            qk,
            out_dtype=tl.float32,
        )

        # ----------------------------------------------------
        # QK: optional RoPE 64 dims
        # ----------------------------------------------------

        if DQK == 576:

            tkv_ptr = (
                tkv_base
                + offs_td[:, None]
                + kv_offsets[None, :]
            )

            tkv_blk = tl.load(
                tkv_ptr,
                cache_modifier=".cg",
            )

            qk = tl.dot(
                tq_blk,
                tkv_blk,
                qk,
                out_dtype=tl.float32,
            )

        qk *= sm_scale

        qk = tl.where(
            valid_ids[None, :],
            qk,
            float("-inf"),
        )

        # ----------------------------------------------------
        # Partial online softmax
        # ----------------------------------------------------

        new_max = tl.maximum(
            max_log,
            tl.max(qk, axis=1),
        )

        exp_qk = tl.math.exp(
            qk - new_max[:, None]
        )

        sum_qk = tl.sum(
            exp_qk,
            axis=1,
        )

        alpha = tl.math.exp(
            max_log - new_max
        )

        sum_exp = (
            sum_exp * alpha
            + sum_qk
        )

        acc0 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk0.trans(),
            acc0 * alpha[:, None],
            out_dtype=tl.float32,
        )

        acc1 = tl.dot(
            exp_qk.to(tl.bfloat16),
            kv_blk1.trans(),
            acc1 * alpha[:, None],
            out_dtype=tl.float32,
        )

        max_log = new_max

    # --------------------------------------------------------
    # A completely empty split can encounter -inf - -inf
    # internally. Do NOT let an empty split poison Stage 2.
    # --------------------------------------------------------

    valid_row = (
        max_log != float("-inf")
    )

    partial_max_out = tl.where(
        valid_row,
        max_log,
        float("-inf"),
    )

    partial_sum_out = tl.where(
        valid_row,
        sum_exp,
        0.0,
    )

    partial_acc0_out = tl.where(
        valid_row[:, None],
        acc0,
        0.0,
    )

    partial_acc1_out = tl.where(
        valid_row[:, None],
        acc1,
        0.0,
    )

    # --------------------------------------------------------
    # Store partial max/sum
    #
    # layouts:
    #
    # partial_max [split, SQ, HQ]
    # partial_sum [split, SQ, HQ]
    # --------------------------------------------------------

    ps_base = (
        i_split * stride_ps_s
        + i_sq64 * stride_ps_q
        + gbh_base
    )

    tl.store(
        partial_max
        + ps_base
        + offs_h,
        partial_max_out,
    )

    tl.store(
        partial_sum
        + ps_base
        + offs_h,
        partial_sum_out,
    )

    # --------------------------------------------------------
    # Store partial accumulator
    #
    # partial_acc [split, SQ, HQ, 512]
    # --------------------------------------------------------

    pa_base = (
        partial_acc
        + i_split * stride_pa_s
        + i_sq64 * stride_pa_q
        + gbh_base * stride_pa_h
    )

    pa_ptr = (
        pa_base
        + offs_h[:, None] * stride_pa_h
        + offs_d[None, :]
    )

    tl.store(
        pa_ptr,
        partial_acc0_out,
    )

    tl.store(
        pa_ptr + BDP,
        partial_acc1_out,
    )


# ============================================================
# V6 Stage 2
#
# Exact online-softmax merge:
#
# M = max_i(m_i)
#
# w_i = exp(m_i - M)
#
# L = sum_i(l_i * w_i)
#
# A = sum_i(A_i * w_i)
#
# output = A / L
# ============================================================


@triton.jit
def triton_flash_mla_sparse_split_combine(
    partial_acc,
    partial_max,
    partial_sum,

    attn_sink,

    output,
    max_logits,
    lse,

    stride_pa_s,
    stride_pa_q,
    stride_pa_h,

    stride_ps_s,
    stride_ps_q,

    stride_oh,
    stride_om,
    stride_mm,
    stride_lm,

    SQ,
    HQ: tl.constexpr,

    HAVE_ATTN_SINK: tl.constexpr,

    NUM_SPLITS: tl.constexpr,
    BH: tl.constexpr,
):
    BDP: tl.constexpr = 256

    num_head_blocks: tl.constexpr = HQ // BH

    pid = tl.program_id(0)

    i_sq = pid // num_head_blocks
    i_gbh = pid % num_head_blocks

    i_sq64 = i_sq.to(tl.int64)

    gbh_base = i_gbh * BH

    offs_h = tl.arange(0, BH)
    offs_d = tl.arange(0, BDP)
    offs_s = tl.arange(0, NUM_SPLITS)

    # --------------------------------------------------------
    # First find the global row max.
    # --------------------------------------------------------

    pm_ptr = (
        partial_max
        + offs_s[:, None] * stride_ps_s
        + i_sq64 * stride_ps_q
        + gbh_base
        + offs_h[None, :]
    )

    split_max = tl.load(
        pm_ptr
    )

    global_max = tl.max(
        split_max,
        axis=0,
    )

    valid_row = (
        global_max != float("-inf")
    )

    # Avoid -inf - -inf in empty rows.
    safe_global_max = tl.where(
        valid_row,
        global_max,
        0.0,
    )

    total_sum = tl.zeros(
        [BH],
        dtype=tl.float32,
    )

    final_acc0 = tl.zeros(
        [BH, BDP],
        dtype=tl.float32,
    )

    final_acc1 = tl.zeros(
        [BH, BDP],
        dtype=tl.float32,
    )

    # --------------------------------------------------------
    # Merge all 4 partial states.
    # --------------------------------------------------------

    for s in tl.static_range(0, NUM_SPLITS):

        m_s = tl.load(
            partial_max
            + s * stride_ps_s
            + i_sq64 * stride_ps_q
            + gbh_base
            + offs_h
        )

        sum_s = tl.load(
            partial_sum
            + s * stride_ps_s
            + i_sq64 * stride_ps_q
            + gbh_base
            + offs_h
        )

        weight = tl.math.exp(
            m_s - safe_global_max
        )

        total_sum += (
            sum_s * weight
        )

        pa_base = (
            partial_acc
            + s * stride_pa_s
            + i_sq64 * stride_pa_q
            + gbh_base * stride_pa_h
        )

        pa_ptr = (
            pa_base
            + offs_h[:, None] * stride_pa_h
            + offs_d[None, :]
        )

        acc0_s = tl.load(
            pa_ptr
        )

        acc1_s = tl.load(
            pa_ptr + BDP
        )

        final_acc0 += (
            acc0_s * weight[:, None]
        )

        final_acc1 += (
            acc1_s * weight[:, None]
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    orig_lse = (
        safe_global_max
        + tl.math.log(total_sum)
    )

    max_logits_out = tl.where(
        valid_row,
        global_max,
        float("-inf"),
    )

    lse_out = tl.where(
        valid_row,
        orig_lse,
        float("inf"),
    )

    max_base = (
        max_logits
        + i_sq64 * stride_mm
        + gbh_base
    )

    lse_base = (
        lse
        + i_sq64 * stride_lm
        + gbh_base
    )

    tl.store(
        max_base + offs_h,
        max_logits_out,
    )

    tl.store(
        lse_base + offs_h,
        lse_out,
    )

    # --------------------------------------------------------
    # Final normalization
    # --------------------------------------------------------

    if HAVE_ATTN_SINK:

        sink = tl.load(
            attn_sink
            + gbh_base
            + offs_h
        )

        denom = (
            tl.math.exp(orig_lse)
            + tl.math.exp(sink)
        )

        factor = (
            tl.math.exp(safe_global_max)
            / denom
        )

    else:

        factor = (
            1.0 / total_sum
        )

    out0 = tl.where(
        valid_row[:, None],
        final_acc0 * factor[:, None],
        0.0,
    )

    out1 = tl.where(
        valid_row[:, None],
        final_acc1 * factor[:, None],
        0.0,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    o_base = (
        output
        + i_sq64 * stride_om
        + gbh_base * stride_oh
    )

    o_ptr = (
        o_base
        + offs_h[:, None] * stride_oh
        + offs_d[None, :]
    )

    tl.store(
        o_ptr,
        out0.to(tl.bfloat16),
    )

    tl.store(
        o_ptr + BDP,
        out1.to(tl.bfloat16),
    )


if HAS_TLE_FLASHMLA_SPARSE:

    @triton.jit
    def _tle_flashmla_prefill_producer(
        k0_l_writer,
        k0_r_writer,
        k1_l_writer,
        k1_r_writer,
        valid_writer,
        kv_base,
        tkv_base,
        t_base,
        topk_len_ptr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        VG: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        max_col = SKV - 1
        stride_kvn: tl.constexpr = VG * (TD + D)
        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        offs_t = tl.arange(0, BK)
        offs_tile = tl.arange(0, 64)
        kv_tile_rows = tl.broadcast_to(offs_t[:, None], (BK, 64))
        for pair in tl.range(NPAIRS):
            ck0 = pair * 2
            ck1 = ck0 + 1
            t_offs0 = BK * ck0 + offs_t
            t_msk0 = t_offs0 < topk_len
            kv_ids0 = tl.load(t_base + t_offs0, t_msk0, other=-1)
            valid0 = t_msk0 & (kv_ids0 <= max_col) & (kv_ids0 >= 0)
            kv_offsets0 = tl.where(valid0, kv_ids0, 0).to(tl.int64) * stride_kvn

            t_offs1 = BK * ck1 + offs_t
            t_msk1 = t_offs1 < topk_len
            kv_ids1 = tl.load(t_base + t_offs1, t_msk1, other=-1)
            valid1 = t_msk1 & (kv_ids1 <= max_col) & (kv_ids1 >= 0)
            kv_offsets1 = tl.where(valid1, kv_ids1, 0).to(tl.int64) * stride_kvn

            k0_l_slot = k0_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_l_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k0_l_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_l_blk = tl.load(
                    k0_l_ptr,
                    mask=k0_l_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_l_blk,
                    mask=k0_l_msk,
                )
            k0_l_writer.commit(pair)

            k1_r_slot = k1_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_r_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k1_r_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_r_blk = tl.load(
                    k1_r_ptr,
                    mask=k1_r_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_r_blk,
                    mask=k1_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k1_r_tail_ptr = tkv_base + kv_offsets1[:, None] + offs_td[None, :]
                k1_r_tail_msk = valid1[:, None] & (offs_td < TD)[None, :]
                k1_r_tail_blk = tl.load(
                    k1_r_tail_ptr,
                    mask=k1_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_r_slot.sK_tail),
                    k1_r_tail_blk,
                    mask=k1_r_tail_msk,
                )
            k1_r_writer.commit(pair)

            k0_r_slot = k0_r_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = DPH + tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k0_r_ptr = kv_base + kv_offsets0[:, None] + k_cols[None, :]
                k0_r_msk = valid0[:, None] & (k_cols < D)[None, :]
                k0_r_blk = tl.load(
                    k0_r_ptr,
                    mask=k0_r_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK, (kv_tile_rows, k_cols_b)),
                    k0_r_blk,
                    mask=k0_r_msk,
                )
            if HAVE_TAIL:
                offs_td = tl.arange(0, TDP)
                k0_r_tail_ptr = tkv_base + kv_offsets0[:, None] + offs_td[None, :]
                k0_r_tail_msk = valid0[:, None] & (offs_td < TD)[None, :]
                k0_r_tail_blk = tl.load(
                    k0_r_tail_ptr,
                    mask=k0_r_tail_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k0_r_slot.sK_tail),
                    k0_r_tail_blk,
                    mask=k0_r_tail_msk,
                )
            k0_r_writer.commit(pair)

            k1_l_slot = k1_l_writer.acquire(pair)
            for tile in tl.static_range(0, DPH, 64):
                k_cols = tile + offs_tile
                k_cols_b = tl.broadcast_to(k_cols[None, :], (BK, 64))
                k1_l_ptr = kv_base + kv_offsets1[:, None] + k_cols[None, :]
                k1_l_msk = valid1[:, None] & (k_cols < D)[None, :]
                k1_l_blk = tl.load(
                    k1_l_ptr,
                    mask=k1_l_msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                tl.store(
                    tle.gpu.local_ptr(k1_l_slot.sK, (kv_tile_rows, k_cols_b)),
                    k1_l_blk,
                    mask=k1_l_msk,
                )
            k1_l_writer.commit(pair)

            valid_slot = valid_writer.acquire(pair)
            valid_row0 = tl.full([BK], 0, dtype=tl.int32)
            valid_row1 = tl.full([BK], 1, dtype=tl.int32)
            valid_ptr0 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row0, offs_t))
            valid_ptr1 = tle.gpu.local_ptr(valid_slot.is_kv_valid, (valid_row1, offs_t))
            tl.store(valid_ptr0, valid0.to(tl.int8))
            tl.store(valid_ptr1, valid1.to(tl.int8))
            valid_writer.commit(pair)

    @triton.jit
    def _tle_flashmla_prefill_consumer0(
        q_writer,
        q_reader,
        q_desc,
        tq_desc,
        k0_l_reader,
        k0_r_qk_reader,
        k1_l_remote_reader,
        valid_reader,
        sM_wg0_writer,
        sM_wg1_reader,
        sS0_writer,
        sS1_reader,
        sL_wg0_writer,
        sL_wg1_reader,
        output_desc,
        output_row,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_l = offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))

        q_write_slot = q_writer.acquire(0)
        tle.gpu.copy(q_desc, q_write_slot.sQ_l, [BH, DPH], [output_row, 0])
        tle.gpu.copy(q_desc, q_write_slot.sQ_r, [BH, DPH], [output_row, DPH])
        if HAVE_TAIL:
            tle.gpu.copy(tq_desc, q_write_slot.sQ_tail, [BH, TDP], [output_row, D])
        q_writer.commit(0)

        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_l = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)

        for pair in tl.range(NPAIRS):
            k0_l_wait = k0_l_reader.wait(pair)
            k0_l_slot = k0_l_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))

            qk0 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk0 = tl.dot(q_l_blk, tl.trans(k0_l_blk), qk0, out_dtype=tl.float32)

            k0_r_wait = k0_r_qk_reader.wait(pair)
            k0_r_slot = k0_r_wait.slot
            k0_r_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK, (kv_rows, kv_cols_r)))
            qk0 = tl.dot(q_r_blk, tl.trans(k0_r_blk), qk0, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k0_t_blk = tl.load(tle.gpu.local_ptr(k0_r_slot.sK_tail))
                qk0 = tl.dot(q_tail_blk, tl.trans(k0_t_blk), qk0, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row0 = tl.full([BK], 0, dtype=tl.int32)
            valid0 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row0, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk0 = tl.where(valid0[None, :], qk0, float("-inf"))
            valid_reader.release(pair)

            local_max = tl.maximum(max_prev, tl.max(qk0, axis=1))
            alpha = tl.math.exp2((max_prev - local_max) * log_scale)
            prob0 = tl.math.exp2(qk0 * log_scale - local_max[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob0, axis=1)
            acc_l = acc_l * alpha[:, None]
            prob0_b = prob0.to(OUT_DTYPE)

            sM_wg0_slot = sM_wg0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg0_slot.sM), local_max)
            sM_wg0_writer.commit(pair)

            k0_l_blk = tl.load(tle.gpu.local_ptr(k0_l_slot.sK, (kv_rows, kv_cols_l)))
            acc_l = tl.dot(prob0_b, k0_l_blk, acc_l, out_dtype=tl.float32)
            k0_l_reader.release(pair)
            k0_r_qk_reader.release(pair)

            sM_wg1_wait = sM_wg1_reader.wait(pair)
            max_next = tl.load(tle.gpu.local_ptr(sM_wg1_wait.slot.sM))
            sM_wg1_reader.release(pair)

            final_scale = tl.math.exp2((local_max - max_next) * log_scale)
            sum_exp = sum_exp * final_scale
            acc_l = acc_l * final_scale[:, None]

            prob0_scaled = prob0 * final_scale[:, None]
            sS0_slot = sS0_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS0_slot.sS0), prob0_scaled.to(OUT_DTYPE))
            sS0_writer.commit(pair)

            sS1_wait = sS1_reader.wait(pair)
            prob1 = tl.load(tle.gpu.local_ptr(sS1_wait.slot.sS1))
            k1_l_wait = k1_l_remote_reader.wait(pair)
            k1_l_blk = tl.load(
                tle.gpu.local_ptr(k1_l_wait.slot.sK, (kv_rows, kv_cols_l))
            )
            acc_l = tl.dot(prob1, k1_l_blk, acc_l, out_dtype=tl.float32)
            sS1_reader.release(pair)
            k1_l_remote_reader.release(pair)

            max_prev = max_next

        sL_wg0_slot = sL_wg0_writer.acquire(0)
        tl.store(tle.gpu.local_ptr(sL_wg0_slot.sL), sum_exp)
        sL_wg0_writer.commit(0)
        sL_wg1_wait = sL_wg1_reader.wait(1)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg1_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg1_reader.release(1)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_l_vals = acc_l * inv_total_sum[:, None]
        if HAVE_ATTN_SINK:
            fin_log = (
                max_prev * log_scale + tl.math.log2(total_sum)
            ) * 0.6931471805599453
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_l_vals = out_l_vals * sink_scale[:, None]
        out_l_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_l_vals)
        o_l_msk = mask_h[:, None] & mask_od_l[None, :]
        tl.store(q_l_smem_ptr, out_l_vals.to(OUT_DTYPE), o_l_msk)
        tle.gpu.copy(q_slot.sQ_l, output_desc, [BH, DPH], [output_row, 0])

    @triton.jit
    def _tle_flashmla_prefill_consumer1(
        q_reader,
        k1_r_reader,
        k1_l_qk_reader,
        k0_r_remote_reader,
        valid_reader,
        sM_wg1_writer,
        sM_wg0_reader,
        sS1_writer,
        sS0_reader,
        sL_wg1_writer,
        sL_wg0_reader,
        final_max_logits_smem,
        final_lse_smem,
        output_desc,
        output_row,
        max_logits_base,
        l_base,
        h_base,
        topk_len_ptr,
        attn_sink_base,
        log_scale: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        TOPK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        DPH: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
    ):
        topk_len = tl.load(topk_len_ptr) if HAVE_TOPK_LENGTH else TOPK
        offs_h = tl.arange(0, BH)
        offs_dh = tl.arange(0, DPH)
        mask_h = h_base + offs_h < G
        mask_od_r = DPH + offs_dh < D
        kv_rows = tl.broadcast_to(tl.arange(0, BK)[:, None], (BK, DPH))
        kv_cols_l = tl.broadcast_to(offs_dh[None, :], (BK, DPH))
        kv_cols_r = tl.broadcast_to((DPH + offs_dh)[None, :], (BK, DPH))
        q_slot = q_reader.wait(0).slot
        q_l_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_l)
        q_r_smem_ptr = tle.gpu.local_ptr(q_slot.sQ_r)
        max_prev = tl.full([BH], -1.0e30, dtype=tl.float32)
        sum_exp = tl.full([BH], 0.0, dtype=tl.float32)
        acc_r = tl.zeros([BH, DPH], dtype=tl.float32)

        NK = tl.cdiv(topk_len, BK)
        NPAIRS = tl.cdiv(NK, 2)
        for pair in tl.range(NPAIRS):
            k1_r_wait = k1_r_reader.wait(pair)
            k1_r_slot = k1_r_wait.slot

            q_l_blk = tl.load(q_l_smem_ptr)
            q_r_blk = tl.load(q_r_smem_ptr)
            k1_r_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK, (kv_rows, kv_cols_r)))

            qk1 = tl.full([BH, BK], 0.0, dtype=tl.float32)
            qk1 = tl.dot(q_r_blk, tl.trans(k1_r_blk), qk1, out_dtype=tl.float32)
            if HAVE_TAIL:
                q_tail_blk = tl.load(tle.gpu.local_ptr(q_slot.sQ_tail))
                k1_t_blk = tl.load(tle.gpu.local_ptr(k1_r_slot.sK_tail))
                qk1 = tl.dot(q_tail_blk, tl.trans(k1_t_blk), qk1, out_dtype=tl.float32)
            k1_l_wait = k1_l_qk_reader.wait(pair)
            k1_l_slot = k1_l_wait.slot
            k1_l_blk = tl.load(tle.gpu.local_ptr(k1_l_slot.sK, (kv_rows, kv_cols_l)))
            qk1 = tl.dot(q_l_blk, tl.trans(k1_l_blk), qk1, out_dtype=tl.float32)

            valid_wait = valid_reader.wait(pair)
            row1 = tl.full([BK], 1, dtype=tl.int32)
            valid1 = (
                tl.load(
                    tle.gpu.local_ptr(
                        valid_wait.slot.is_kv_valid, (row1, tl.arange(0, BK))
                    )
                )
                != 0
            )
            qk1 = tl.where(valid1[None, :], qk1, float("-inf"))
            valid_reader.release(pair)

            sM_wg0_wait = sM_wg0_reader.wait(pair)
            candidate0 = tl.load(tle.gpu.local_ptr(sM_wg0_wait.slot.sM))
            sM_wg0_reader.release(pair)

            candidate1 = tl.maximum(max_prev, tl.max(qk1, axis=1))
            max_next = tl.maximum(candidate1, candidate0)
            sM_wg1_slot = sM_wg1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sM_wg1_slot.sM), max_next)
            sM_wg1_writer.commit(pair)

            alpha = tl.math.exp2((max_prev - max_next) * log_scale)
            prob1 = tl.math.exp2(qk1 * log_scale - max_next[:, None] * log_scale)
            sum_exp = sum_exp * alpha + tl.sum(prob1, axis=1)
            acc_r = acc_r * alpha[:, None]
            prob1_b = prob1.to(OUT_DTYPE)

            k1_l_qk_reader.release(pair)

            acc_r = tl.dot(prob1_b, k1_r_blk, acc_r, out_dtype=tl.float32)

            sS1_slot = sS1_writer.acquire(pair)
            tl.store(tle.gpu.local_ptr(sS1_slot.sS1), prob1_b)
            sS1_writer.commit(pair)

            sS0_wait = sS0_reader.wait(pair)
            prob0 = tl.load(tle.gpu.local_ptr(sS0_wait.slot.sS0))
            k0_r_wait = k0_r_remote_reader.wait(pair)
            k0_r_blk = tl.load(
                tle.gpu.local_ptr(k0_r_wait.slot.sK, (kv_rows, kv_cols_r))
            )
            acc_r = tl.dot(prob0, k0_r_blk, acc_r, out_dtype=tl.float32)
            k1_r_reader.release(pair)
            sS0_reader.release(pair)
            k0_r_remote_reader.release(pair)
            max_prev = max_next

        sL_wg1_slot = sL_wg1_writer.acquire(1)
        tl.store(tle.gpu.local_ptr(sL_wg1_slot.sL), sum_exp)
        sL_wg1_writer.commit(1)
        sL_wg0_wait = sL_wg0_reader.wait(0)
        peer_sum = tl.load(tle.gpu.local_ptr(sL_wg0_wait.slot.sL))
        total_sum = sum_exp + peer_sum
        sL_wg0_reader.release(0)

        is_no_valid_tokens = total_sum == 0.0
        inv_total_sum = tl.fdiv(1.0, total_sum)
        out_r_vals = acc_r * inv_total_sum[:, None]
        final_max_logits_log2 = max_prev * log_scale
        final_max_logits = final_max_logits_log2 * 0.6931471805599453
        fin_log = (final_max_logits_log2 + tl.math.log2(total_sum)) * 0.6931471805599453
        if HAVE_ATTN_SINK:
            sink = tl.load(attn_sink_base + h_base + offs_h, mask_h, other=0.0)
            sink_scale = tl.fdiv(1.0, 1.0 + tl.math.exp(sink - fin_log))
            out_r_vals = out_r_vals * sink_scale[:, None]
        out_r_vals = tl.where(is_no_valid_tokens[:, None], 0.0, out_r_vals)
        o_r_msk = mask_h[:, None] & mask_od_r[None, :]
        tl.store(q_r_smem_ptr, out_r_vals.to(OUT_DTYPE), o_r_msk)
        tle.gpu.copy(q_slot.sQ_r, output_desc, [BH, DPH], [output_row, DPH])

        final_max_logits = tl.where(is_no_valid_tokens, float("-inf"), final_max_logits)
        fin_log = tl.where(is_no_valid_tokens, float("inf"), fin_log)
        tl.store(tle.gpu.local_ptr(final_max_logits_smem), final_max_logits, mask_h)
        tl.store(tle.gpu.local_ptr(final_lse_smem), fin_log, mask_h)
        final_max_logits = tl.load(
            tle.gpu.local_ptr(final_max_logits_smem),
            mask_h,
            other=float("-inf"),
        )
        fin_log = tl.load(tle.gpu.local_ptr(final_lse_smem), mask_h, other=float("inf"))
        tl.store(max_logits_base + offs_h, final_max_logits, mask_h)
        tl.store(l_base + offs_h, fin_log, mask_h)

    @triton.jit
    def _tle_flashmla_prefill_fwd(
        q_desc,
        tq_desc,
        output_desc,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale: tl.constexpr,
        output,
        max_logits,
        lse,
        SQ,
        H: tl.constexpr,
        DQK: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        D: tl.constexpr,
        TD: tl.constexpr,
        DP: tl.constexpr,
        TDP: tl.constexpr,
        G: tl.constexpr,
        VG: tl.constexpr,
        RH: tl.constexpr,
        HAVE_TAIL: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
        PAIR_BLOCKS: tl.constexpr,
    ):
        DPH: tl.constexpr = DP // 2
        stride_kvg: tl.constexpr = TD + D
        stride_tg = TOPK
        stride_tm = VG * stride_tg
        stride_lm = H
        stride_mm = H

        pid = tl.program_id(0)
        programs_per_q: tl.constexpr = VG * RH
        i_sq = pid // programs_per_q
        i_grh = pid % programs_per_q
        i_g = i_grh // RH
        i_rh = i_grh % RH
        h_base = i_rh * BH
        q_head_base = i_g * G + h_base
        i_sq64 = i_sq.to(tl.int64)
        i_g64 = i_g.to(tl.int64)
        q_head_base64 = q_head_base.to(tl.int64)
        kv_base = kv + i_g64 * stride_kvg
        tkv_base = kv_base + D
        t_base = indices + i_sq64 * stride_tm + i_g64 * stride_tg
        topk_len_ptr = topk_length + i_sq64 if HAVE_TOPK_LENGTH else indices
        attn_sink_base = attn_sink if HAVE_ATTN_SINK else max_logits
        max_logits_base = max_logits + i_sq64 * stride_mm + q_head_base64
        l_base = lse + i_sq64 * stride_lm + q_head_base64
        q_row = i_sq * H + q_head_base
        _ = output
        _ = SQ
        _ = DQK

        sQ_l_smem = tle.gpu.alloc(
            [1, BH, DPH],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sQ_r_smem = tle.gpu.alloc(
            [1, BH, DPH],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        if HAVE_TAIL:
            sQ_tail_smem = tle.gpu.alloc(
                [1, BH, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
                sQ_tail=sQ_tail_smem,
            )
        else:
            q_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sQ",
                readers=("wg0", "wg1"),
                one_shot=True,
                sQ_l=sQ_l_smem,
                sQ_r=sQ_r_smem,
            )

        sK0_smem = tle.gpu.alloc(
            [1, BK, DP],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sK1_smem = tle.gpu.alloc(
            [1, BK, DP],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        if HAVE_TAIL:
            sK0_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sK1_tail_smem = tle.gpu.alloc(
                [1, BK, TDP],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
            sS0_smem = sK0_tail_smem
        else:
            sS0_smem = tle.gpu.alloc(
                [1, BH, BK],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
            )
        is_kv_valid_smem = tle.gpu.alloc(
            [1, PAIR_BLOCKS, BK],
            dtype=tl.int8,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        k0_l_pipe = tle.pipe(
            capacity=1, scope="cta", name="flashmla_sK0_l", sK=sK0_smem
        )
        if HAVE_TAIL:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
                sK_tail=sK0_tail_smem,
            )
        else:
            k0_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK0_r",
                readers=("qk", "remote"),
                sK=sK0_smem,
            )
        k1_l_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_sK1_l",
            readers=("qk", "remote"),
            sK=sK1_smem,
        )
        if HAVE_TAIL:
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK1_r",
                sK=sK1_smem,
                sK_tail=sK1_tail_smem,
            )
        else:
            k1_r_pipe = tle.pipe(
                capacity=1,
                scope="cta",
                name="flashmla_sK1_r",
                sK=sK1_smem,
            )
        is_kv_valid_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_is_kv_valid_ready",
            readers=("wg0", "wg1"),
            is_kv_valid=is_kv_valid_smem,
        )

        sM_smem = tle.gpu.alloc(
            [1, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sS1_smem = tle.gpu.alloc(
            [1, BH, BK],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
        )
        sL_smem = tle.gpu.alloc(
            [2, BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_max_logits_smem = tle.gpu.alloc(
            [BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        final_lse_smem = tle.gpu.alloc(
            [BH],
            dtype=tl.float32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        sM_wg0_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_wg0_bunch_0_ready",
            sM=sM_smem,
        )
        sM_wg1_pipe = tle.pipe(
            capacity=1,
            scope="cta",
            name="flashmla_wg1_bunch_0_ready",
            sM=sM_smem,
        )
        sS0_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS0", sS0=sS0_smem)
        sS1_pipe = tle.pipe(capacity=1, scope="cta", name="flashmla_sS1", sS1=sS1_smem)
        sL_wg0_pipe = tle.pipe(
            capacity=2, scope="cta", name="flashmla_sL_wg0", sL=sL_smem
        )
        sL_wg1_pipe = tle.pipe(
            capacity=2, scope="cta", name="flashmla_sL_wg1", sL=sL_smem
        )

        log_scale: tl.constexpr = sm_scale * 1.4426950408889634

        tle.gpu.warp_specialize(
            [
                (
                    _tle_flashmla_prefill_consumer0,
                    (
                        q_pipe.writer(),
                        q_pipe.reader("wg0"),
                        q_desc,
                        tq_desc,
                        k0_l_pipe.reader(),
                        k0_r_pipe.reader("qk"),
                        k1_l_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg0"),
                        sM_wg0_pipe.writer(),
                        sM_wg1_pipe.reader(),
                        sS0_pipe.writer(),
                        sS1_pipe.reader(),
                        sL_wg0_pipe.writer(),
                        sL_wg1_pipe.reader(),
                        output_desc,
                        q_row,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_flashmla_prefill_consumer1,
                    (
                        q_pipe.reader("wg1"),
                        k1_r_pipe.reader(),
                        k1_l_pipe.reader("qk"),
                        k0_r_pipe.reader("remote", fields=("sK",)),
                        is_kv_valid_pipe.reader("wg1"),
                        sM_wg1_pipe.writer(),
                        sM_wg0_pipe.reader(),
                        sS1_pipe.writer(),
                        sS0_pipe.reader(),
                        sL_wg1_pipe.writer(),
                        sL_wg0_pipe.reader(),
                        final_max_logits_smem,
                        final_lse_smem,
                        output_desc,
                        q_row,
                        max_logits_base,
                        l_base,
                        h_base,
                        topk_len_ptr,
                        attn_sink_base,
                        log_scale,
                        D,
                        TD,
                        kv.dtype.element_ty,
                        HAVE_ATTN_SINK,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                        BH,
                        DPH,
                        TDP,
                        G,
                    ),
                ),
                (
                    _tle_flashmla_prefill_producer,
                    (
                        k0_l_pipe.writer(),
                        k0_r_pipe.writer(),
                        k1_l_pipe.writer(),
                        k1_r_pipe.writer(),
                        is_kv_valid_pipe.writer(),
                        kv_base,
                        tkv_base,
                        t_base,
                        topk_len_ptr,
                        D,
                        TD,
                        DPH,
                        TDP,
                        VG,
                        SKV,
                        TOPK,
                        HAVE_TOPK_LENGTH,
                        HAVE_TAIL,
                        BK,
                    ),
                ),
            ],
            [4, 4],
            [216, 72],
        )



    @triton.jit
    def _tle_flashmla_prefill_fwd_metax_serial(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale: tl.constexpr,
        output,
        max_logits,
        lse,
        stride_qh,
        stride_qm,
        stride_kvn,
        stride_tm,
        stride_oh,
        stride_om,
        stride_mm,
        stride_lm,
        SQ,
        HQ: tl.constexpr,
        DQK: tl.constexpr,
        SKV,
        TOPK: tl.constexpr,
        HAVE_ATTN_SINK: tl.constexpr,
        HAVE_TOPK_LENGTH: tl.constexpr,
        BK: tl.constexpr,
        BH: tl.constexpr,
    ):
        """
        MetaX C550 serial TLE implementation.

        This path intentionally does not use:
          - tle.pipe
          - tle.gpu.warp_specialize

        Sparse KV gather is serialized through small TLE shared-memory
        tiles using tle.gpu.alloc/local_ptr.
        """

        num_head_blocks: tl.constexpr = (HQ + BH - 1) // BH

        pid = tl.program_id(0)

        i_sq = pid // num_head_blocks
        i_hb = pid % num_head_blocks

        # Global address arithmetic may become large.
        i_sq64 = i_sq.to(tl.int64)

        h_base = i_hb * BH

        DP: tl.constexpr = 512
        BDP: tl.constexpr = 256
        HAVE_TAIL: tl.constexpr = DQK == 576

        # ----------------------------------------------------
        # Base pointers
        # ----------------------------------------------------

        q_base = (
            q
            + i_sq64 * stride_qm
            + h_base * stride_qh
        )

        kv_base = kv
        tkv_base = kv + DP

        t_base = indices + i_sq64 * stride_tm

        o_base = (
            output
            + i_sq64 * stride_om
            + h_base * stride_oh
        )

        max_log_base = (
            max_logits
            + i_sq64 * stride_mm
            + h_base
        )

        l_base = (
            lse
            + i_sq64 * stride_lm
            + h_base
        )

        if HAVE_ATTN_SINK:
            attn_sink_base = attn_sink + h_base

        if HAVE_TOPK_LENGTH:
            topk_len_ptr = topk_length + i_sq64

        # ----------------------------------------------------
        # Q tile
        # ----------------------------------------------------

        offs_h = tl.arange(0, BH)
        offs_d = tl.arange(0, BDP)
        offs_t = tl.arange(0, BK)

        mask_h = h_base + offs_h < HQ

        q_ptr = (
            q_base
            + offs_h[:, None] * stride_qh
            + offs_d[None, :]
        )

        q_blk0 = tl.load(
            q_ptr,
            mask=mask_h[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )

        q_blk1 = tl.load(
            q_ptr + BDP,
            mask=mask_h[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )

        if HAVE_TAIL:
            offs_td = tl.arange(0, 64)

            tq_ptr = (
                q_base
                + DP
                + offs_h[:, None] * stride_qh
                + offs_td[None, :]
            )

            tq_blk = tl.load(
                tq_ptr,
                mask=mask_h[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )

        # ----------------------------------------------------
        # TLE shared memory
        #
        # BK=16:
        # K0   = 16 x 256 x 2B = 8 KiB
        # K1   = 16 x 256 x 2B = 8 KiB
        # tail = 16 x  64 x 2B = 2 KiB
        # ----------------------------------------------------

        sK0 = tle.gpu.alloc(
            [BK, BDP],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        sK1 = tle.gpu.alloc(
            [BK, BDP],
            dtype=kv.dtype.element_ty,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )

        kv_rows = tl.broadcast_to(
            offs_t[:, None],
            (BK, BDP),
        )

        kv_cols = tl.broadcast_to(
            offs_d[None, :],
            (BK, BDP),
        )

        sK0_ptr = tle.gpu.local_ptr(
            sK0,
            (kv_rows, kv_cols),
        )

        sK1_ptr = tle.gpu.local_ptr(
            sK1,
            (kv_rows, kv_cols),
        )

        if HAVE_TAIL:
            sKT = tle.gpu.alloc(
                [BK, 64],
                dtype=kv.dtype.element_ty,
                layout=None,
                scope=tle.gpu.smem,
                nv_mma_shared_layout=False,
            )

            kt_rows = tl.broadcast_to(
                offs_t[:, None],
                (BK, 64),
            )

            kt_cols = tl.broadcast_to(
                offs_td[None, :],
                (BK, 64),
            )

            sKT_ptr = tle.gpu.local_ptr(
                sKT,
                (kt_rows, kt_cols),
            )

        # ----------------------------------------------------
        # Online softmax state
        # ----------------------------------------------------

        max_log = tl.full(
            [BH],
            float("-inf"),
            dtype=tl.float32,
        )

        sum_exp = tl.zeros(
            [BH],
            dtype=tl.float32,
        )

        acc0 = tl.zeros(
            [BH, BDP],
            dtype=tl.float32,
        )

        acc1 = tl.zeros(
            [BH, BDP],
            dtype=tl.float32,
        )

        topk_len = (
            tl.load(topk_len_ptr)
            if HAVE_TOPK_LENGTH
            else TOPK
        )

        NK = tl.cdiv(topk_len, BK)

        # ====================================================
        # Serial sparse-KV loop
        # ====================================================

        for ck in range(NK):

            # -----------------------------------------------
            # 1. Load sparse indices
            # -----------------------------------------------

            t_offs = ck * BK + offs_t

            t_mask = t_offs < topk_len

            kv_ids = tl.load(
                t_base + t_offs,
                mask=t_mask,
                other=-1,
            )

            valid = (
                t_mask
                & (kv_ids >= 0)
                & (kv_ids < SKV)
            )

            # Important:
            # promote to int64 BEFORE multiplying by stride.
            safe_kv_ids = tl.where(
                valid,
                kv_ids,
                0,
            ).to(tl.int64)

            kv_offsets = safe_kv_ids * stride_kvn

            # -----------------------------------------------
            # 2. Gather first 256 dims -> TLE shared memory
            # -----------------------------------------------

            g_k0_ptr = (
                kv_base
                + kv_offsets[:, None]
                + offs_d[None, :]
            )

            g_k0 = tl.load(
                g_k0_ptr,
                mask=valid[:, None],
                other=0.0,
                cache_modifier=".cg",
            )

            tl.store(
                sK0_ptr,
                g_k0,
            )

            # -----------------------------------------------
            # 3. Gather second 256 dims -> shared memory
            # -----------------------------------------------

            g_k1_ptr = (
                kv_base
                + kv_offsets[:, None]
                + BDP
                + offs_d[None, :]
            )

            g_k1 = tl.load(
                g_k1_ptr,
                mask=valid[:, None],
                other=0.0,
                cache_modifier=".cg",
            )

            tl.store(
                sK1_ptr,
                g_k1,
            )

            # -----------------------------------------------
            # 4. Optional 64-dim RoPE tail
            # -----------------------------------------------

            if HAVE_TAIL:
                g_kt_ptr = (
                    tkv_base
                    + kv_offsets[:, None]
                    + offs_td[None, :]
                )

                g_kt = tl.load(
                    g_kt_ptr,
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )

                tl.store(
                    sKT_ptr,
                    g_kt,
                )

            # -----------------------------------------------
            # 5. Read serialized shared-memory tiles
            # -----------------------------------------------

            k0 = tl.load(sK0_ptr)  # [BK, 256]
            k1 = tl.load(sK1_ptr)  # [BK, 256]

            # -----------------------------------------------
            # 6. QK
            # -----------------------------------------------

            qk = tl.dot(
                q_blk0,
                tl.trans(k0),
                out_dtype=tl.float32,
            )

            qk = tl.dot(
                q_blk1,
                tl.trans(k1),
                qk,
                out_dtype=tl.float32,
            )

            if HAVE_TAIL:
                kt = tl.load(sKT_ptr)

                qk = tl.dot(
                    tq_blk,
                    tl.trans(kt),
                    qk,
                    out_dtype=tl.float32,
                )

            qk *= sm_scale

            qk = tl.where(
                valid[None, :],
                qk,
                float("-inf"),
            )

            # -----------------------------------------------
            # 7. Online softmax
            # -----------------------------------------------

            new_max = tl.maximum(
                max_log,
                tl.max(qk, axis=1),
            )

            alpha = tl.math.exp(
                max_log - new_max
            )

            p = tl.math.exp(
                qk - new_max[:, None]
            )

            sum_exp = (
                sum_exp * alpha
                + tl.sum(p, axis=1)
            )

            acc0 *= alpha[:, None]
            acc1 *= alpha[:, None]

            p_bf16 = p.to(tl.bfloat16)

            # -----------------------------------------------
            # 8. P @ V
            #
            # First 512 dims of KV are also V.
            # -----------------------------------------------

            acc0 = tl.dot(
                p_bf16,
                k0,
                acc0,
                out_dtype=tl.float32,
            )

            acc1 = tl.dot(
                p_bf16,
                k1,
                acc1,
                out_dtype=tl.float32,
            )

            max_log = new_max

        # ====================================================
        # Finalize
        # ====================================================

        valid_row = max_log != float("-inf")

        max_log_out = tl.where(
            valid_row,
            max_log,
            float("-inf"),
        )

        tl.store(
            max_log_base + offs_h,
            max_log_out,
            mask=mask_h,
        )

        orig_lse = (
            max_log
            + tl.math.log(sum_exp)
        )

        lse_out = tl.where(
            valid_row,
            orig_lse,
            float("inf"),
        )

        tl.store(
            l_base + offs_h,
            lse_out,
            mask=mask_h,
        )

        # ----------------------------------------------------
        # Attention sink
        # ----------------------------------------------------

        if HAVE_ATTN_SINK:
            sink = tl.load(
                attn_sink_base + offs_h,
                mask=mask_h,
                other=0.0,
            )

            denom = (
                tl.math.exp(orig_lse)
                + tl.math.exp(sink)
            )

            factor = (
                tl.math.exp(max_log)
                / denom
            )

        else:
            factor = 1.0 / sum_exp

        out0 = tl.where(
            valid_row[:, None],
            acc0 * factor[:, None],
            0.0,
        )

        out1 = tl.where(
            valid_row[:, None],
            acc1 * factor[:, None],
            0.0,
        )

        # ----------------------------------------------------
        # Store O
        # ----------------------------------------------------

        o_ptr = (
            o_base
            + offs_h[:, None] * stride_oh
            + offs_d[None, :]
        )

        tl.store(
            o_ptr,
            out0.to(tl.bfloat16),
            mask=mask_h[:, None],
        )

        tl.store(
            o_ptr + BDP,
            out1.to(tl.bfloat16),
            mask=mask_h[:, None],
        )


def _flash_mla_sparse_tle_enabled() -> bool:
    value = os.environ.get("FLAGGEMS_FLASHMLA_SPARSE_TLE", "1").lower()
    return value not in {"0", "false", "off", "no"}



def _is_metax_device(device: torch.device) -> bool:
    try:
        return "metax" in torch.cuda.get_device_name(device).lower()
    except Exception:
        return False


def _can_use_tle_flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    topk_length: Optional[torch.Tensor] = None,
) -> bool:
    if not (HAS_TLE_FLASHMLA_SPARSE and _flash_mla_sparse_tle_enabled()):
        return False
    if q.device.type != "cuda":
        return False
    SQ, HQ, DQK = q.shape
    _ = SQ
    HKV = kv.shape[1]
    TOPK = indices.shape[-1]
    return (
        d_v == 512
        and HKV == 1
        and DQK in (512, 576)
        and HQ % TLE_FLASHMLA_PREFILL_BH == 0
        and TOPK > 0
        and TOPK % 128 == 0
    )


def _set_triton_descriptor_allocator(device: torch.device) -> None:
    def alloc_fn(size: int, align: int, stream):
        _ = align
        _ = stream
        return torch.empty(size, dtype=torch.int8, device=device)

    triton.set_allocator(alloc_fn)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be set to -1 or numbers >= s_kv
        sm_scale: float
        d_v: The dimension of value vectors. Can only be 512
        attn_sink: optional, [h_q], float32.
            If attn_sink is provided, when computing output, output will be additionally multiplied by
            exp(lse) / (exp(lse) + exp(attn_sink)). +-inf in attn_sink will be handled normally (i.e., -inf has no
            effect, +inf will make corresponding output all zeros).
            This argument has no effect on lse and max_logits.
        topk_length: optional, [s_q], int32.
            If provided, the i-th q token will only attend to k tokens specified by indices[i, :, :topk_length[i]],
            ignoring later k/v tokens (even if provided in indices). In extremely rare cases (topk_length provided,
            there is a valid topk index between topk_length[i] ~ s_kv, and that topk index points to a k token
            containing NaN), operator output will contain NaN, so please avoid this situation.

    Returns:
        (output, max_logits, lse)
        Please refer to tests/ref.py for the precise definitions of these parameters.
        - output: [s_q, h_q, d_v], bfloat16
        - max_logits:  [s_q, h_q], float
        - lse: [s_q, h_q], float, log-sum-exp of attention scores
    """
    assert q.is_contiguous() and kv.is_contiguous() and indices.is_contiguous()
    assert (
        q.dtype == torch.bfloat16
        and kv.dtype == torch.bfloat16
        and indices.dtype == torch.int32
    )
    SQ, HQ, DQK = q.shape
    SKV, HKV, _ = kv.shape

    assert d_v == 512, "Unsupported d_v"
    DV = d_v

    assert kv.shape[-1] == DQK
    _, _, TOPK = indices.shape
    assert indices.shape == (SQ, HKV, TOPK)
    if attn_sink is not None:
        assert attn_sink.is_contiguous()
        assert attn_sink.dtype == torch.float32
        assert attn_sink.shape == (HQ,), "attn_sink error shape"
    if topk_length is not None:
        assert topk_length.is_contiguous()
        assert topk_length.dtype == torch.int32
        assert topk_length.shape == (SQ,), "topk_length error shape"

    # check from FlashMLA
    assert HKV == 1, "h_kv is expected to be 1"
    assert HQ == 64 or HQ == 128, "Unsupported h_q"
    assert DQK == 576 or DQK == 512, "Unsupported d_qk"

    _ = SKV
    D = DV
    TD = DQK - D
    DP = triton.next_power_of_2(D)
    HAVE_TAIL = TD > 0
    TDP = triton.next_power_of_2(TD) if HAVE_TAIL else 1
    G = HQ // HKV
    BH = TLE_FLASHMLA_PREFILL_BH
    RH = G // BH
    BK = TLE_FLASHMLA_PREFILL_BK
    output = torch.empty((SQ, HQ, DV), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)
    lse = torch.empty((SQ, HQ), device=q.device, dtype=torch.float32)

    def triton_grid(META):
        return (triton.cdiv(HQ, META["BH"]) * SQ,)

    if _can_use_tle_flash_mla_sparse_fwd(q, kv, indices, d_v, topk_length):
        # MetaX currently lacks the TLE pipe IR builder
        # (create_pipe_create), so use a serial TLE path.
        if _is_metax_device(q.device):
            serial_grid = (
                triton.cdiv(HQ, METAX_SERIAL_TLE_BH) * SQ,
            )

            _tle_flashmla_prefill_fwd_metax_serial[serial_grid](
                q,
                kv,
                indices,
                attn_sink,
                topk_length,
                sm_scale,
                output,
                max_logits,
                lse,
                q.stride(1),
                q.stride(0),
                kv.stride(0),
                indices.stride(0),
                output.stride(1),
                output.stride(0),
                max_logits.stride(0),
                lse.stride(0),
                SQ,
                HQ,
                DQK,
                SKV,
                TOPK,
                attn_sink is not None,
                topk_length is not None,
                METAX_SERIAL_TLE_BK,
                METAX_SERIAL_TLE_BH,
                num_warps=METAX_SERIAL_TLE_NUM_WARPS,
                num_stages=1,
            )

            return output, max_logits, lse

        # Keep the original pipe + warp-specialized TLE path
        # for backends that support it.
        from triton.tools.tensor_descriptor import TensorDescriptor

        _set_triton_descriptor_allocator(q.device)
        q_desc = TensorDescriptor(
            q,
            shape=[SQ * HQ, DQK],
            strides=[DQK, 1],
            block_shape=[BH, DP // 2],
        )
        if HAVE_TAIL:
            tq_desc = TensorDescriptor(
                q,
                shape=[SQ * HQ, DQK],
                strides=[DQK, 1],
                block_shape=[BH, TDP],
            )
        else:
            tq_desc = q_desc
        output_desc = TensorDescriptor(
            output,
            shape=[SQ * HQ, D],
            strides=[D, 1],
            block_shape=[BH, DP // 2],
        )
        _tle_flashmla_prefill_fwd[triton_grid](
            q_desc,
            tq_desc,
            output_desc,
            kv,
            indices,
            attn_sink,
            topk_length,
            sm_scale,
            output,
            max_logits,
            lse,
            SQ,
            HQ,
            DQK,
            SKV,
            TOPK,
            attn_sink is not None,
            topk_length is not None,
            D,
            TD,
            DP,
            TDP,
            G,
            HKV,
            RH,
            HAVE_TAIL,
            BK,
            BH,
            TLE_FLASHMLA_PREFILL_PAIR_BLOCKS,
            num_warps=TLE_FLASHMLA_PREFILL_WORKER_NUM_WARPS,
            num_stages=1,
        )
        return output, max_logits, lse


    # ========================================================
    # V6 split-TOPK path
    #
    # Only low-SQ cases use this path.
    #
    # Larger SQ falls through to the exact V1 implementation
    # below.
    # ========================================================

    # ========================================================
    # V6.2 adaptive split-TOPK dispatch
    # ========================================================

    v6_num_splits = _v6_choose_num_splits(
        SQ,
        HQ,
        TOPK,
    )

    use_v6_split = (
        v6_num_splits > 1
        and HQ % V6_SPLIT_BH == 0
        and TOPK % (
            v6_num_splits * V6_SPLIT_BK
        ) == 0
    )

    if use_v6_split:

        # Reuse persistent FP32 partial state instead of
        # allocating it on every attention invocation.
        (
            partial_acc,
            partial_max,
            partial_sum,
        ) = _v6_get_workspace(
            q=q,
            sq=SQ,
            hq=HQ,
            dv=DV,
            num_splits=v6_num_splits,
        )

        split_grid = (
            SQ
            * (HQ // V6_SPLIT_BH)
            * v6_num_splits,
        )

        triton_flash_mla_sparse_split_fwd[
            split_grid
        ](
            q,
            kv,
            indices,
            topk_length,
            sm_scale,

            partial_acc,
            partial_max,
            partial_sum,

            q.stride(1),
            q.stride(0),
            kv.stride(0),
            indices.stride(0),

            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),

            partial_max.stride(0),
            partial_max.stride(1),

            SQ,
            HQ,
            DQK,
            SKV,
            TOPK,

            topk_length is not None,

            v6_num_splits,
            V6_SPLIT_BK,
            V6_SPLIT_BH,

            num_warps=2,
            num_stages=1,
        )

        combine_grid = (
            SQ
            * (HQ // V6_SPLIT_BH),
        )

        triton_flash_mla_sparse_split_combine[
            combine_grid
        ](
            partial_acc,
            partial_max,
            partial_sum,

            attn_sink,

            output,
            max_logits,
            lse,

            partial_acc.stride(0),
            partial_acc.stride(1),
            partial_acc.stride(2),

            partial_max.stride(0),
            partial_max.stride(1),

            output.stride(1),
            output.stride(0),
            max_logits.stride(0),
            lse.stride(0),

            SQ,
            HQ,

            attn_sink is not None,

            v6_num_splits,
            V6_SPLIT_BH,

            num_warps=4,
            num_stages=1,
        )

        return output, max_logits, lse

    # ========================================================
    # V6 fallback:
    # bypass FlagTree autotuner and directly use the V1 winner
    # for all SQ > 16 cases in our benchmark matrix.
    #
    # V1 measured winner:
    #   BK=16, BH=32, num_warps=4, num_stages=1
    # ========================================================

    fallback_BK = 16

    # V1 measured crossover.
    #
    # Low parallelism:
    #   BH16 / w2
    #
    # Higher parallelism:
    #   BH32 / w4
    if (
        SQ < 16
        or (SQ == 16 and HQ == 64)
    ):
        fallback_BH = 16
        fallback_warps = 2
    else:
        fallback_BH = 32
        fallback_warps = 4

    fallback_grid = (
        triton.cdiv(HQ, fallback_BH) * SQ,
    )

    # triton_flash_mla_sparse_fwd is an Autotuner object.
    # .fn is the wrapped @triton.jit JITFunction, so this
    # bypasses autotune while preserving exactly the V1 math.
    triton_flash_mla_sparse_fwd.fn[fallback_grid](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        sm_scale,
        output,
        max_logits,
        lse,
        q.stride(1),
        q.stride(0),
        kv.stride(1),
        kv.stride(0),
        indices.stride(1),
        indices.stride(0),
        output.stride(1),
        output.stride(0),
        max_logits.stride(0),
        lse.stride(0),
        SQ,
        HQ,
        DQK,
        SKV,
        TOPK,
        attn_sink is not None,
        topk_length is not None,
        fallback_BK,
        fallback_BH,
        num_warps=4,
        num_stages=1,
    )
    return output, max_logits, lse
