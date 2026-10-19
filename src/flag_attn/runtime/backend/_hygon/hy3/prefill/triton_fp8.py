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

"""Paged FP8 block-sparse prefill attention for the Hygon backend.

This module contains the backend-local portable Triton implementation.  The
Hygon public entry point intentionally uses this implementation only; the
legacy Hopper/TLE names are compatibility aliases and do not dispatch to
CUDA/Hopper kernels.
"""

from dataclasses import dataclass
import os
from typing import Optional

import torch
import triton
import triton.language as tl


_BLOCK_M = 128
_BLOCK_N = 128
_HEAD_DIM = 128
_FP8_P_SCALE = 256.0
_LOG2E = 1.4426950408889634
# One segment covers 64 BLOCK_N tiles (8192 tokens).  On gfx936 this keeps
# enough independent partials in flight without the excessive reduce launch
# count caused by smaller segments.  The launcher accepts 128/256 through
# HY3_PREFILL_SEGMENT_TILES for shape-specific experiments.
_SEGMENT_TILES = 64
_AUTO_GQA_MIN_KV_TOKENS = 16384
_AUTO_SPLIT_MIN_KV_TOKENS = 32768
_DEFAULT_MAX_SPLIT_WORKSPACE_BYTES = 1 << 30
_SPLIT_PARTIAL_BYTES_PER_ROW = 8 + (_HEAD_DIM * 2)  # FP32 m/l + BF16 acc



def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _segment_tile_config(
    max_seqlens_q: Optional[int] = None,
    max_kv_tokens: Optional[int] = None,
    num_head_q: Optional[int] = None,
    num_batch: Optional[int] = None,
) -> int:
    """Return the number of 128-token tiles handled by one split segment.

    The default remains 64 tiles.  When the launch metadata is available, a
    larger segment is selected only to fit the split workspace budget; this
    reduces partial-state memory for large batch/head combinations without
    changing the tuned default for ordinary requests.
    """
    forced = os.environ.get("HY3_PREFILL_SEGMENT_TILES")
    if forced is not None:
        try:
            value = int(forced)
        except ValueError as exc:
            raise ValueError("HY3_PREFILL_SEGMENT_TILES must be an integer") from exc
        if value not in (64, 128, 256):
            raise ValueError("HY3_PREFILL_SEGMENT_TILES must be one of: 64, 128, 256")
        return value

    if None in (max_seqlens_q, max_kv_tokens, num_head_q, num_batch):
        return _SEGMENT_TILES

    limit = _split_workspace_limit()
    if limit is None:
        return _SEGMENT_TILES
    for candidate in (64, 128, 256):
        if _split_workspace_bytes(
            max_seqlens_q,
            max_kv_tokens,
            num_head_q,
            num_batch,
            candidate,
        ) <= limit:
            return candidate
    # Even the largest supported segment cannot fit the budget.  Returning
    # 256 lets the selector safely fall back to the base kernel.
    return 256


def _split_workspace_limit() -> Optional[int]:
    """Return the per-launch split-KV workspace budget.

    A bounded default prevents automatic dispatch from allocating multi-GB
    partial state for large batches.  ``0`` disables the guard for controlled
    experiments; explicit split-KV remains otherwise subject to the same
    safety limit as automatic dispatch.
    """
    forced = os.environ.get("HY3_PREFILL_MAX_SPLIT_BYTES")
    if forced is None:
        return _DEFAULT_MAX_SPLIT_WORKSPACE_BYTES
    try:
        value = int(forced)
    except ValueError as exc:
        raise ValueError("HY3_PREFILL_MAX_SPLIT_BYTES must be an integer") from exc
    if value < 0:
        raise ValueError("HY3_PREFILL_MAX_SPLIT_BYTES must be >= 0")
    return None if value == 0 else value


def _split_workspace_bytes(
    max_seqlens_q: int,
    max_kv_tokens: int,
    num_head_q: int,
    num_batch: int,
    segment_tiles: int,
) -> int:
    """Estimate split-KV partial-state bytes without touching device tensors."""
    q_tiles = (max_seqlens_q + _BLOCK_M - 1) // _BLOCK_M
    kv_tiles = (max_kv_tokens + _BLOCK_N - 1) // _BLOCK_N
    num_segments = (kv_tiles + segment_tiles - 1) // segment_tiles
    rows = num_batch * num_head_q * q_tiles * num_segments * _BLOCK_M
    return rows * _SPLIT_PARTIAL_BYTES_PER_ROW


def _normalize_sparsity_bucket(block_mask: Optional[torch.Tensor], sparsity_bucket: Optional[int]) -> int:
    # Exact mask density is deliberately not read from the GPU here: doing so
    # would synchronize every production launch.  Workload generators that
    # already know the density (including our benchmark) can supply buckets
    # 0..3; an unspecified masked workload uses a separate generic bucket.
    if block_mask is None:
        return 0
    if sparsity_bucket is None:
        return 4
    value = int(sparsity_bucket)
    if value not in (0, 1, 2, 3):
        raise ValueError("sparsity_bucket must be one of 0, 1, 2, 3")
    return value


def _load_autotune_configs(name, fallback):
    """Use the shared backend config loader, with local defaults as a fallback."""
    try:
        from flag_attn.runtime import get_tuned_config

        configs = get_tuned_config(name)
    except (ImportError, OSError, TypeError, ValueError):
        return fallback()
    return configs or fallback()


def _portable_autotune_configs():
    """Return the portable launch configurations used by the migrated op."""
    configs = [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_stages in (2, 3, 4)
        for num_warps in (4, 8)
    ]
    forced = os.environ.get("HPC_BSA_TRITON_FORCE_CONFIG")
    if forced:
        try:
            num_stages, num_warps = map(int, forced.split(","))
        except ValueError as exc:
            raise ValueError("HPC_BSA_TRITON_FORCE_CONFIG must be STAGES,WARPS") from exc
        configs = [config for config in configs if config.num_stages == num_stages and config.num_warps == num_warps]
        if not configs:
            raise ValueError(f"unsupported forced portable BSA config: {forced}")
    return configs


def _autotune_configs():
    return _load_autotune_configs("attention_blocksparse_prefill_fp8", _portable_autotune_configs)



@triton.autotune(
    configs=_autotune_configs(),
    key=[
        "Q_LEN_BUCKET",
        "KV_LEN_BUCKET",
        "SPARSITY_BUCKET",
        "PAGE_SIZE",
        "HAS_BLOCK_MASK",
        "K_SCALE_PER_TOKEN",
        "KV_LAYOUT",
        "GROUP_SIZE",
        "SEGMENTED",
    ],
)
@triton.jit
def _bsa_fp8_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qscale_ptr,
    kscale_ptr,
    vscale_ptr,
    cu_seqlens_q_ptr,
    block_ids_ptr,
    seqlens_kv_ptr,
    block_mask_ptr,
    out_ptr,
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_qs_batch,
    stride_qs_head,
    stride_qs_token,
    stride_ks_page,
    stride_ks_group,
    stride_ks_head,
    stride_ks_token,
    stride_block_ids_batch,
    stride_block_ids_page,
    stride_mask_batch,
    stride_mask_head,
    stride_mask_qtile,
    stride_mask_kvtile,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    num_head_q,
    num_head_kv,
    num_mask_kv_tiles,
    PAGE_SIZE: tl.constexpr,
    MAX_KV_TOKENS: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    Q_LEN_BUCKET: tl.constexpr,
    KV_LEN_BUCKET: tl.constexpr,
    SPARSITY_BUCKET: tl.constexpr,
    KV_LAYOUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    Q_TILES: tl.constexpr,
    SEGMENTED: tl.constexpr,
    SEGMENT_TILES: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    SEGMENT_ACC_BF16: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    """One program computes one [BLOCK_M, HEAD_DIM] output tile."""
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch_segment = tl.program_id(2)
    # Triton on the Hygon backend exposes a 3-D launch grid.  Flatten the
    # batch/segment axes instead of relying on a fourth program_id dimension.
    batch = batch_segment // NUM_SEGMENTS
    segment_id = batch_segment % NUM_SEGMENTS

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)

    q_local_start = q_tile * BLOCK_M
    program_valid = q_local_start < q_len
    # GQA mapping is a compile-time specialization.  The old runtime divide
    # was cheap in isolation but remained in the pointer arithmetic for every
    # Q tile; making the group size a constexpr lets Triton fold it and keeps
    # the Hygon address-generation path integer-only.
    kv_head = q_head // GROUP_SIZE
    q_start_in_kv = kv_len - q_len

    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < q_len

    q_offsets = (q_begin + offs_m[:, None]) * stride_q_token + q_head * stride_q_head + offs_d[None, :] * stride_q_dim
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)

    qs_offsets = batch * stride_qs_batch + q_head * stride_qs_head + offs_m * stride_qs_token
    q_scale = tl.load(qscale_ptr + qs_offsets, mask=q_valid, other=0.0).to(tl.float32)

    # The CUDA implementation accumulates both softmax state and PV in FP32.
    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    # Only tiles up to the causal end of this Q tile can contribute.  The loop
    # still has a compile-time maximum so Triton can pipeline it.
    q_tile_end = tl.minimum(q_local_start + BLOCK_M, q_len)
    causal_kv_end = q_start_in_kv + q_tile_end
    num_causal_kv_tiles = (causal_kv_end + BLOCK_N - 1) // BLOCK_N

    offs_n = tl.arange(0, BLOCK_N)
    max_kv_tiles = (MAX_KV_TOKENS + BLOCK_N - 1) // BLOCK_N
    # Keep the indirect page-table -> K/V load loop at the proven-safe depth.
    # Config.num_stages is still autotuned for dot-feeding loads.  Feeding the
    # same value into tl.range would additionally pipeline most pointer-chasing
    # loads and produced NaNs / intermittent illegal accesses at depth 4 on
    # the Triton 3.6 FlagTree backend.
    # Keep loop pipeline depth independent from the launch stage count:
    # the long-KV Hygon path uses one stage to reduce scheduler pressure,
    # while short/masked workloads retain the two-stage overlap setting.
    if SEGMENTED:
        kv_tile_begin = segment_id * SEGMENT_TILES
        kv_tile_end = tl.minimum(kv_tile_begin + SEGMENT_TILES, max_kv_tiles)
    else:
        kv_tile_begin = 0
        kv_tile_end = max_kv_tiles
    for kv_tile in tl.range(kv_tile_begin, kv_tile_end, 1, num_stages=LOOP_STAGES):
        tile_active = program_valid & (kv_tile < num_causal_kv_tiles)
        if HAS_BLOCK_MASK:
            mask_in_range = kv_tile < num_mask_kv_tiles
            mask_offset = (
                batch * stride_mask_batch
                + q_head * stride_mask_head
                + q_tile * stride_mask_qtile
                + kv_tile * stride_mask_kvtile
            )
            selected = (
                tl.load(
                    block_mask_ptr + mask_offset,
                    mask=mask_in_range,
                    other=0,
                )
                != 0
            )
            # Match the CUDA compatibility rule: if the supplied mask is
            # shorter than the causal range, retain exactly the first tile
            # after the represented range.
            selected = selected | (kv_tile == num_mask_kv_tiles)
            tile_active = tile_active & selected

        # This is a program-uniform branch.  A skipped BSA tile performs no
        # page-table loads, K/V loads, or Tensor-Core work.
        if tile_active:
            kv_tokens = kv_tile * BLOCK_N + offs_n
            kv_valid = kv_tokens < kv_len
            kv_tile_full = kv_tile * BLOCK_N + BLOCK_N <= kv_len
            # BLOCK_N is page-aligned on the supported 32/64-token page sizes.
            # Load each page-table entry once per tile for both scale layouts;
            # selecting a page id with static predicates avoids the dynamic
            # tensor indexing that the Hygon Triton backend cannot lower.
            page_base = kv_tile * (BLOCK_N // PAGE_SIZE)
            page_slot = offs_n // PAGE_SIZE
            page0 = tl.load(
                block_ids_ptr
                + batch * stride_block_ids_batch
                + page_base * stride_block_ids_page,
                mask=kv_tile * BLOCK_N < kv_len,
                other=0,
            )
            page1 = tl.load(
                block_ids_ptr
                + batch * stride_block_ids_batch
                + (page_base + 1) * stride_block_ids_page,
                mask=kv_tile * BLOCK_N + PAGE_SIZE < kv_len,
                other=0,
            )
            if PAGE_SIZE == 32:
                page2 = tl.load(
                    block_ids_ptr
                    + batch * stride_block_ids_batch
                    + (page_base + 2) * stride_block_ids_page,
                    mask=kv_tile * BLOCK_N + 2 * PAGE_SIZE < kv_len,
                    other=0,
                )
                page3 = tl.load(
                    block_ids_ptr
                    + batch * stride_block_ids_batch
                    + (page_base + 3) * stride_block_ids_page,
                    mask=kv_tile * BLOCK_N + 3 * PAGE_SIZE < kv_len,
                    other=0,
                )
                physical_pages = tl.where(
                    page_slot == 0,
                    page0,
                    tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
                )
            else:
                physical_pages = tl.where(page_slot == 0, page0, page1)
            # At a page-aligned tile boundary, kv_tokens and offs_n have the
            # same in-page offset for both scale layouts.
            tokens_in_page = kv_tokens % PAGE_SIZE

            if KV_LAYOUT == 0:
                # Contiguous [page, token, head, dim] cache: specialize the
                # token/head strides instead of carrying two runtime multiplies.
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * (num_head_kv * HEAD_DIM)
                    + kv_head * HEAD_DIM
                    + offs_d[None, :] * stride_k_dim
                )
            elif KV_LAYOUT == 1:
                # Contiguous [page, head, token, dim] cache.
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * HEAD_DIM
                    + kv_head * (PAGE_SIZE * HEAD_DIM)
                    + offs_d[None, :] * stride_k_dim
                )
            else:
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * stride_k_token
                    + kv_head * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )
            if kv_tile_full:
                k = tl.load(k_ptr + k_offsets)
            else:
                k = tl.load(k_ptr + k_offsets, mask=kv_valid[:, None], other=0.0)

            # FP8 x FP8 -> FP32.  On NVIDIA targets Triton lowers this to the
            # architecture-appropriate Tensor-Core operation.
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

            if K_SCALE_PER_TOKEN:
                # K scale storage is logically
                # [page, token//32, kv_head, token%32] in FP32.  The public API
                # also accepts its byte-identical FP8 view; the launcher views
                # that tensor back to FP32 without copying.
                ks_offsets = (
                    physical_pages * stride_ks_page
                    + (tokens_in_page // 32) * stride_ks_group
                    + kv_head * stride_ks_head
                    + (tokens_in_page % 32) * stride_ks_token
                )
                k_scale = tl.load(
                    kscale_ptr + ks_offsets,
                    mask=kv_valid,
                    other=0.0,
                ).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale[None, :]
            else:
                k_scale = tl.load(kscale_ptr).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale

            # Every tile strictly before the diagonal is causally valid for
            # the whole Q tile. Avoid materializing the [BLOCK_M, BLOCK_N]
            # comparison for those tiles; only the diagonal tile needs the
            # element-wise causal predicate.
            if MAX_KV_TOKENS > BLOCK_N:
                causal_full = (
                    kv_tile * BLOCK_N + (BLOCK_N - 1)
                    <= q_start_in_kv + q_local_start
                )
                if causal_full:
                    score_valid = q_valid[:, None] & kv_valid[None, :]
                else:
                    q_positions = q_start_in_kv + offs_m
                    causal_valid = kv_tokens[None, :] <= q_positions[:, None]
                    score_valid = q_valid[:, None] & kv_valid[None, :] & causal_valid
            else:
                q_positions = q_start_in_kv + offs_m
                causal_valid = kv_tokens[None, :] <= q_positions[:, None]
                score_valid = q_valid[:, None] & kv_valid[None, :] & causal_valid
            scores = tl.where(
                score_valid,
                scores * score_scale * SCALE_LOG2E_OVER_SQRT_D,
                -float("inf"),
            )

            # Online softmax in base 2, equivalent to the CUDA implementation.
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            alpha = tl.exp2(m_i - new_max)
            p = tl.exp2(scores - new_max[:, None])
            tile_sum = tl.sum(p, axis=1)

            l_i = l_i * alpha + tile_sum
            acc *= alpha[:, None]

            # This quantization is part of the original BSA algorithm, not an
            # input Q/K/V quantization step.
            p_fp8 = (p * FP8_P_SCALE).to(tl.float8e4nv)

            if KV_LAYOUT == 0:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * (num_head_kv * HEAD_DIM)
                    + kv_head * HEAD_DIM
                    + offs_d[None, :] * stride_v_dim
                )
            elif KV_LAYOUT == 1:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * HEAD_DIM
                    + kv_head * (PAGE_SIZE * HEAD_DIM)
                    + offs_d[None, :] * stride_v_dim
                )
            else:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * stride_v_token
                    + kv_head * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )
            if kv_tile_full:
                v = tl.load(v_ptr + v_offsets)
            else:
                v = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0.0)
            acc = tl.dot(p_fp8, v, acc, out_dtype=tl.float32)
            m_i = new_max

    if SEGMENTED:
        # Split-KV programs retain their local log-sum-exp state.  A second
        # Triton kernel merges these states across segments, avoiding a long
        # serial KV loop in one program.
        state_base = (
            (((batch * num_head_q + q_head) * Q_TILES + q_tile) * NUM_SEGMENTS + segment_id)
            * BLOCK_M
        )
        tl.store(partial_m_ptr + state_base + tl.arange(0, BLOCK_M), m_i)
        tl.store(partial_l_ptr + state_base + tl.arange(0, BLOCK_M), l_i)
        acc_base = state_base * HEAD_DIM
        acc_offsets = acc_base + tl.arange(0, BLOCK_M)[:, None] * HEAD_DIM + offs_d[None, :]
        if SEGMENT_ACC_BF16:
            tl.store(partial_acc_ptr + acc_offsets, acc.to(tl.bfloat16))
        else:
            tl.store(partial_acc_ptr + acc_offsets, acc)
    else:
        if K_SCALE_PER_TOKEN:
            v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
        else:
            v_scale = tl.load(vscale_ptr).to(tl.float32)

        # A completely empty sparse row intentionally follows the CUDA contract
        # and produces NaN (0 / 0).  Callers are expected to retain the diagonal
        # BSA tile for every valid Q tile.
        output = acc * (v_scale / FP8_P_SCALE) / l_i[:, None]
        out_offsets = (
            (q_begin + offs_m[:, None]) * stride_out_token + q_head * stride_out_head + offs_d[None, :] * stride_out_dim
        )
        tl.store(out_ptr + out_offsets, output.to(tl.bfloat16), mask=q_valid[:, None])


@triton.jit
def _bsa_fp8_prefill_gqa_reuse_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qscale_ptr,
    kscale_ptr,
    vscale_ptr,
    cu_seqlens_q_ptr,
    block_ids_ptr,
    seqlens_kv_ptr,
    out_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_qs_batch,
    stride_qs_head,
    stride_qs_token,
    stride_ks_page,
    stride_ks_group,
    stride_ks_head,
    stride_ks_token,
    stride_block_ids_batch,
    stride_block_ids_page,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    num_head_q,
    num_head_kv,
    PAGE_SIZE: tl.constexpr,
    MAX_KV_TOKENS: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    KV_LAYOUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GROUP_M: tl.constexpr,
    GQA_ROWS: tl.constexpr,
    Q_LEN_BUCKET: tl.constexpr,
    KV_LEN_BUCKET: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GQA_BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    """Unmasked GQA path: one K/V tile serves GROUP_M Q heads."""
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 128)
    # 64 rows per Q head is the best of the currently validated gfx936
    # variants.  A 32-row tile increases the number of programs and was
    # consistently slower in the long-KV measurements.  GROUP_M may be 2 or
    # 4; keeping GQA_ROWS bounded by the caller's choice makes group-4 an
    # explicit experiment without changing the production default.
    tl.static_assert(GQA_BLOCK_M == 64)
    tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
    q_tile = tl.program_id(0)
    group_block = tl.program_id(1)
    batch = tl.program_id(2)
    q_head_base = group_block * GROUP_M
    kv_head = q_head_base // GROUP_SIZE

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)
    q_local_start = q_tile * GQA_BLOCK_M
    program_valid = q_local_start < q_len
    offs_hm = tl.arange(0, GQA_ROWS)
    offs_d = tl.arange(0, HEAD_DIM)
    q_head_offsets = q_head_base + offs_hm // GQA_BLOCK_M
    q_local = q_local_start + offs_hm % GQA_BLOCK_M
    q_valid = program_valid & (q_local < q_len)
    q_offsets = (
        (q_begin + q_local[:, None]) * stride_q_token
        + q_head_offsets[:, None] * stride_q_head
        + offs_d[None, :] * stride_q_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0.0)
    qs_offsets = (
        batch * stride_qs_batch
        + q_head_offsets * stride_qs_head
        + q_local * stride_qs_token
    )
    q_scale = tl.load(qscale_ptr + qs_offsets, mask=q_valid, other=0.0).to(tl.float32)
    m_i = tl.full((GQA_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((GQA_ROWS,), tl.float32)
    acc = tl.zeros((GQA_ROWS, HEAD_DIM), tl.float32)

    q_start_in_kv = kv_len - q_len
    q_tile_end = tl.minimum(q_local_start + GQA_BLOCK_M, q_len)
    causal_kv_end = q_start_in_kv + q_tile_end
    num_causal_kv_tiles = (causal_kv_end + BLOCK_N - 1) // BLOCK_N
    offs_n = tl.arange(0, BLOCK_N)
    max_kv_tiles = (MAX_KV_TOKENS + BLOCK_N - 1) // BLOCK_N
    for kv_tile in tl.range(0, max_kv_tiles, 1, num_stages=LOOP_STAGES):
        tile_active = program_valid & (kv_tile < num_causal_kv_tiles)
        if tile_active:
            kv_tokens = kv_tile * BLOCK_N + offs_n
            kv_valid = kv_tokens < kv_len
            kv_tile_full = kv_tile * BLOCK_N + BLOCK_N <= kv_len
            page_base = kv_tile * (BLOCK_N // PAGE_SIZE)
            page_slot = offs_n // PAGE_SIZE
            page0 = tl.load(
                block_ids_ptr
                + batch * stride_block_ids_batch
                + page_base * stride_block_ids_page,
                mask=kv_tile * BLOCK_N < kv_len,
                other=0,
            )
            page1 = tl.load(
                block_ids_ptr
                + batch * stride_block_ids_batch
                + (page_base + 1) * stride_block_ids_page,
                mask=kv_tile * BLOCK_N + PAGE_SIZE < kv_len,
                other=0,
            )
            if PAGE_SIZE == 32:
                page2 = tl.load(
                    block_ids_ptr
                    + batch * stride_block_ids_batch
                    + (page_base + 2) * stride_block_ids_page,
                    mask=kv_tile * BLOCK_N + 2 * PAGE_SIZE < kv_len,
                    other=0,
                )
                page3 = tl.load(
                    block_ids_ptr
                    + batch * stride_block_ids_batch
                    + (page_base + 3) * stride_block_ids_page,
                    mask=kv_tile * BLOCK_N + 3 * PAGE_SIZE < kv_len,
                    other=0,
                )
                physical_pages = tl.where(
                    page_slot == 0,
                    page0,
                    tl.where(page_slot == 1, page1, tl.where(page_slot == 2, page2, page3)),
                )
            else:
                physical_pages = tl.where(page_slot == 0, page0, page1)
            tokens_in_page = kv_tokens % PAGE_SIZE
            if KV_LAYOUT == 0:
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * (num_head_kv * HEAD_DIM)
                    + kv_head * HEAD_DIM
                    + offs_d[None, :] * stride_k_dim
                )
            elif KV_LAYOUT == 1:
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * HEAD_DIM
                    + kv_head * (PAGE_SIZE * HEAD_DIM)
                    + offs_d[None, :] * stride_k_dim
                )
            else:
                k_offsets = (
                    physical_pages[:, None] * stride_k_page
                    + tokens_in_page[:, None] * stride_k_token
                    + kv_head * stride_k_head
                    + offs_d[None, :] * stride_k_dim
                )
            if kv_tile_full:
                k = tl.load(k_ptr + k_offsets)
            else:
                k = tl.load(k_ptr + k_offsets, mask=kv_valid[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
            if K_SCALE_PER_TOKEN:
                ks_offsets = (
                    physical_pages * stride_ks_page
                    + (tokens_in_page // 32) * stride_ks_group
                    + kv_head * stride_ks_head
                    + (tokens_in_page % 32) * stride_ks_token
                )
                k_scale = tl.load(kscale_ptr + ks_offsets, mask=kv_valid, other=0.0).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale[None, :]
            else:
                k_scale = tl.load(kscale_ptr).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale
            if MAX_KV_TOKENS > BLOCK_N:
                causal_full = (
                    kv_tile * BLOCK_N + (BLOCK_N - 1)
                    <= q_start_in_kv + q_local_start
                )
                if causal_full:
                    score_valid = q_valid[:, None] & kv_valid[None, :]
                else:
                    q_positions = q_start_in_kv + q_local
                    score_valid = q_valid[:, None] & kv_valid[None, :] & (
                        kv_tokens[None, :] <= q_positions[:, None]
                    )
            else:
                q_positions = q_start_in_kv + q_local
                score_valid = q_valid[:, None] & kv_valid[None, :] & (
                    kv_tokens[None, :] <= q_positions[:, None]
                )
            scores = tl.where(
                score_valid,
                scores * score_scale * SCALE_LOG2E_OVER_SQRT_D,
                -float("inf"),
            )
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(m_i, tile_max)
            alpha = tl.exp2(m_i - new_max)
            p = tl.exp2(scores - new_max[:, None])
            tile_sum = tl.sum(p, axis=1)
            l_i = l_i * alpha + tile_sum
            acc *= alpha[:, None]
            p_fp8 = (p * FP8_P_SCALE).to(tl.float8e4nv)
            if KV_LAYOUT == 0:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * (num_head_kv * HEAD_DIM)
                    + kv_head * HEAD_DIM
                    + offs_d[None, :] * stride_v_dim
                )
            elif KV_LAYOUT == 1:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * HEAD_DIM
                    + kv_head * (PAGE_SIZE * HEAD_DIM)
                    + offs_d[None, :] * stride_v_dim
                )
            else:
                v_offsets = (
                    physical_pages[:, None] * stride_v_page
                    + tokens_in_page[:, None] * stride_v_token
                    + kv_head * stride_v_head
                    + offs_d[None, :] * stride_v_dim
                )
            if kv_tile_full:
                v = tl.load(v_ptr + v_offsets)
            else:
                v = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0.0)
            acc = tl.dot(p_fp8, v, acc, out_dtype=tl.float32)
            m_i = new_max

    if K_SCALE_PER_TOKEN:
        v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
    else:
        v_scale = tl.load(vscale_ptr).to(tl.float32)
    output = acc * (v_scale / FP8_P_SCALE) / l_i[:, None]
    out_offsets = (
        (q_begin + q_local[:, None]) * stride_out_token
        + q_head_offsets[:, None] * stride_out_head
        + offs_d[None, :] * stride_out_dim
    )
    tl.store(out_ptr + out_offsets, output.to(tl.bfloat16), mask=q_valid[:, None])


@triton.jit
def _bsa_fp8_prefill_segment_reduce_kernel(
    partial_m_ptr,
    partial_l_ptr,
    partial_acc_ptr,
    cu_seqlens_q_ptr,
    vscale_ptr,
    out_ptr,
    stride_out_token,
    stride_out_head,
    stride_out_dim,
    num_head_q,
    num_head_kv,
    Q_TILES: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
):
    """Merge split-KV online-softmax states into the final output."""
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    q_local_start = q_tile * BLOCK_M
    q_valid = q_local_start + tl.arange(0, BLOCK_M) < q_len
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    state_base = (
        ((batch * num_head_q + q_head) * Q_TILES + q_tile) * NUM_SEGMENTS * BLOCK_M
    )
    merged_m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    for segment in range(0, NUM_SEGMENTS):
        segment_base = state_base + segment * BLOCK_M
        segment_m = tl.load(partial_m_ptr + segment_base + offs_m)
        merged_m = tl.maximum(merged_m, segment_m)

    merged_l = tl.zeros((BLOCK_M,), tl.float32)
    merged_acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for segment in range(0, NUM_SEGMENTS):
        segment_base = state_base + segment * BLOCK_M
        segment_m = tl.load(partial_m_ptr + segment_base + offs_m)
        segment_l = tl.load(partial_l_ptr + segment_base + offs_m)
        has_state = segment_m != -float("inf")
        weight = tl.where(has_state, tl.exp2(segment_m - merged_m), 0.0)
        merged_l += segment_l * weight
        acc_base = segment_base * HEAD_DIM
        acc_offsets = acc_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
        segment_acc = tl.load(partial_acc_ptr + acc_offsets)
        merged_acc += segment_acc * weight[:, None]

    kv_head = q_head // (num_head_q // num_head_kv)
    if K_SCALE_PER_TOKEN:
        v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
    else:
        v_scale = tl.load(vscale_ptr).to(tl.float32)
    output = merged_acc * (v_scale / FP8_P_SCALE) / merged_l[:, None]
    out_offsets = (
        (q_begin + (q_local_start + offs_m)[:, None]) * stride_out_token
        + q_head * stride_out_head
        + offs_d[None, :] * stride_out_dim
    )
    tl.store(out_ptr + out_offsets, output.to(tl.bfloat16), mask=q_valid[:, None])


def _quant_type_value(quant_type) -> int:
    return int(getattr(quant_type, "value", quant_type))


def _check_inputs(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
) -> None:
    tensors = {
        "q": q,
        "kcache": kcache,
        "vcache": vcache,
        "qscale": qscale,
        "kscale": kscale,
        "vscale": vscale,
        "cu_seqlens_q": cu_seqlens_q,
        "block_ids": block_ids,
        "seqlens_kvcache": seqlens_kvcache,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}")

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None or q.dtype != fp8_dtype or kcache.dtype != fp8_dtype or vcache.dtype != fp8_dtype:
        raise TypeError("q, kcache and vcache must use torch.float8_e4m3fn")
    if q.ndim != 3 or kcache.ndim != 4 or vcache.ndim != 4:
        raise ValueError("q must be rank 3 and kcache/vcache must be rank 4")
    if q.shape[2] != _HEAD_DIM or kcache.shape[3] != _HEAD_DIM or vcache.shape[3] != _HEAD_DIM:
        raise ValueError("only head_dim=128 is supported")
    if kcache.shape[:3] != vcache.shape[:3]:
        raise ValueError("kcache and vcache page/token/head dimensions must match")
    if q.shape[1] % kcache.shape[2] != 0:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    if kcache.shape[1] not in (32, 64) or _BLOCK_N % kcache.shape[1] != 0:
        raise ValueError("paged BSA currently supports page sizes 32 and 64")
    if cu_seqlens_q.dtype != torch.int32 or block_ids.dtype != torch.int32 or seqlens_kvcache.dtype != torch.int32:
        raise TypeError("cu_seqlens_q, block_ids and seqlens_kvcache must be int32")
    if cu_seqlens_q.ndim != 1 or seqlens_kvcache.ndim != 1 or block_ids.ndim != 2:
        raise ValueError("invalid sequence metadata rank")
    num_batch = cu_seqlens_q.numel() - 1
    if num_batch <= 0 or seqlens_kvcache.numel() != num_batch or block_ids.shape[0] != num_batch:
        raise ValueError("batch dimensions of sequence metadata do not match")
    if block_ids.shape[1] <= 0:
        raise ValueError("block_ids must contain at least one logical KV page")
    if qscale.shape[:2] != (num_batch, q.shape[1]) or qscale.ndim != 3:
        raise ValueError("qscale must have shape [batch, num_head_q, max_seq_q_pad]")
    if qscale.dtype != torch.float32 or vscale.dtype != torch.float32:
        raise TypeError("qscale and vscale must be float32")
    if quant_type not in (0, 1):
        raise ValueError("quant_type must be 0 (per-token K) or 1 (per-tensor K/V)")
    if quant_type == 1:
        if kscale.dtype != torch.float32 or kscale.numel() < 1 or vscale.numel() < 1:
            raise TypeError("per-tensor K/V mode requires float32 scalar scales")
    else:
        if kscale.element_size() not in (1, 4):
            raise TypeError("per-token K scale must be an FP32 tensor or its FP8 byte view")
        if vscale.numel() != kcache.shape[2]:
            raise ValueError("per-head vscale must contain num_head_kv elements")
    if max_seqlens_q <= 0:
        raise ValueError("max_seqlens_q must be positive")

    expected_q_tiles = triton.cdiv(max_seqlens_q, _BLOCK_M)
    if block_mask is not None:
        if block_mask.device != q.device or block_mask.dtype != torch.uint8 or not block_mask.is_contiguous():
            raise ValueError("block_mask must be a contiguous uint8 tensor on the Q device")
        if block_mask.ndim != 4 or block_mask.shape[:3] != (
            num_batch,
            q.shape[1],
            expected_q_tiles,
        ):
            raise ValueError("block_mask must have shape [batch, num_head_q, ceil(max_seqlens_q/128), Kb]")
        if block_mask.shape[3] <= 0:
            raise ValueError("block_mask Kb dimension must be positive")

    if output is not None:
        if output.device != q.device or output.dtype != torch.bfloat16:
            raise ValueError("output must be a bfloat16 tensor on the Q device")
        if output.shape != q.shape or not output.is_contiguous():
            raise ValueError("output must be contiguous with shape [total_seq_q, num_head_q, 128]")


@dataclass(frozen=True)
class _LaunchContext:
    output: torch.Tensor
    kscale: torch.Tensor
    num_batch: int
    num_head_q: int
    num_head_kv: int
    page_size: int
    max_kv_tokens: int
    q_len_bucket: int
    kv_len_bucket: int
    sparsity_bucket: int
    kv_layout: int
    has_mask: bool
    num_mask_kv_tiles: int
    mask_arg: torch.Tensor
    mask_strides: tuple[int, int, int, int]


def _prepare_launch_context(
    q: torch.Tensor,
    kcache: torch.Tensor,
    kscale: torch.Tensor,
    block_ids: torch.Tensor,
    max_seqlens_q: int,
    quant_type: int,
    block_mask: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    sparsity_bucket: Optional[int],
) -> _LaunchContext:
    if output is None:
        output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)

    kernel_kscale = kscale
    if quant_type == 0 and kscale.element_size() == 1:
        if kscale.shape[-1] % 4 != 0:
            raise ValueError("the FP8 K-scale byte view must have a last dimension divisible by 4")
        kernel_kscale = kscale.view(torch.float32)

    num_batch = block_ids.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = kcache.shape[2]
    page_size = kcache.shape[1]
    expected_page_stride = page_size * num_head_kv * _HEAD_DIM
    if kcache.stride() == (expected_page_stride, num_head_kv * _HEAD_DIM, _HEAD_DIM, 1):
        kv_layout = 0
    elif kcache.stride() == (expected_page_stride, _HEAD_DIM, page_size * _HEAD_DIM, 1):
        kv_layout = 1
    else:
        kv_layout = 2
    max_kv_tokens = block_ids.shape[1] * page_size
    has_mask = block_mask is not None
    return _LaunchContext(
        output=output,
        kscale=kernel_kscale,
        num_batch=num_batch,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        page_size=page_size,
        max_kv_tokens=max_kv_tokens,
        q_len_bucket=_next_power_of_2(max_seqlens_q),
        kv_len_bucket=_next_power_of_2(max_kv_tokens),
        sparsity_bucket=_normalize_sparsity_bucket(block_mask, sparsity_bucket),
        kv_layout=kv_layout,
        has_mask=has_mask,
        num_mask_kv_tiles=block_mask.shape[3] if has_mask else 0,
        mask_arg=block_mask if has_mask else q,
        mask_strides=block_mask.stride() if has_mask else (0, 0, 0, 0),
    )


def _select_prefill_path(
    *,
    max_seqlens_q: int,
    max_kv_tokens: int,
    num_head_q: int,
    num_head_kv: int,
    num_batch: int,
    kv_layout: int,
    has_mask: bool,
    sparsity_bucket: int,
    segment_tiles: int,
    split_workspace_limit: Optional[int],
) -> str:
    """Select the validated Triton prefill variant for the current workload.

    The selector intentionally uses only host-side metadata; it never reads a
    mask or tensor value from the device.  ``split`` is useful when a long KV
    cache would otherwise leave too few programs in flight (short Q).  The
    GQA reuse kernel is selected for the measured middle-Q regime where its
    K/V load reuse pays for the larger per-program accumulator.  The smallest
    validated GQA cache bucket is 16K tokens; larger head counts are kept on
    the base kernel except for the measured Q=256 crossover.  The batch and KV
    layout are part of the policy: GQA is currently auto-enabled only for
    validated NHD, batch=1 inputs.  After fixing the split-KV segment count,
    the long-cache crossover moved substantially: for unmasked KV >= 32K,
    split-KV wins through Q=32768 in the measured 8--32-head range when the
    workspace budget permits it.  GQA
    therefore remains automatic only for the 16K--<32K cache interval, where
    split-KV is intentionally disabled.  A host-side workspace budget keeps
    this policy safe for large batch/head combinations.

    ``HY3_PREFILL_PATH`` accepts ``auto`` (default), ``base``, ``split`` and
    ``gqa``.  The legacy ``HY3_PREFILL_SEGMENTED=1`` and
    ``HY3_PREFILL_GQA_REUSE=1`` switches still force their respective path
    when ``HY3_PREFILL_PATH`` is left at ``auto``.  Automatically selected
    masked split-KV is limited to caller-declared bucket-3 sparsity and
    Q<=2048; other masked workloads remain on the base kernel.
    """
    group_size = num_head_q // num_head_kv if num_head_kv else 0
    gqa_eligible = (
        not has_mask
        and group_size >= 2
        and group_size % 2 == 0
    )
    split_eligible = not has_mask and max_kv_tokens >= _AUTO_SPLIT_MIN_KV_TOKENS
    mask_split_eligible = (
        has_mask
        and sparsity_bucket == 3
        and max_kv_tokens >= _AUTO_SPLIT_MIN_KV_TOKENS
        and triton.cdiv(max_seqlens_q, _BLOCK_M) <= 16
        and num_head_q <= 32
    )
    split_dispatch_eligible = split_eligible or mask_split_eligible
    split_workspace_bytes = _split_workspace_bytes(
        max_seqlens_q,
        max_kv_tokens,
        num_head_q,
        num_batch,
        segment_tiles,
    )
    split_workspace_safe = (
        split_workspace_limit is None
        or split_workspace_bytes <= split_workspace_limit
    )
    forced_mask_split = os.environ.get("HY3_PREFILL_MASK_SPLIT", "0") == "1"
    auto_gqa_eligible = (
        gqa_eligible
        and max_kv_tokens >= _AUTO_GQA_MIN_KV_TOKENS
        # GQA has only been validated on the NHD, batch=1 path.  HND and
        # multi-batch workloads use split-KV/base until their own tuning is
        # complete; this avoids transferring a batch-1 crossover blindly.
        and kv_layout == 0
        and num_batch == 1
    )

    requested = os.environ.get("HY3_PREFILL_PATH", "auto").strip().lower()
    if requested not in {"auto", "base", "split", "gqa"}:
        raise ValueError("HY3_PREFILL_PATH must be one of: auto, base, split, gqa")

    if requested != "auto":
        selected = requested
    elif os.environ.get("HY3_PREFILL_GQA_REUSE", "0") == "1":
        selected = "gqa"
    elif os.environ.get("HY3_PREFILL_SEGMENTED", "0") == "1":
        selected = "split"
    else:
        # A long cache needs enough independent KV segments to hide traversal
        # latency.  With the corrected segment count, split-KV is faster than
        # base/GQA through 256 Q tiles (Q <= 32768) for the tested 8--32-head
        # NHD cases when the workspace budget permits it, and is also
        # preferred for HND and multi-batch inputs.
        # Only caller-declared high-sparsity masks enter the separate masked
        # split branch below; dense or unknown masks stay on base.
        q_tiles = triton.cdiv(max_seqlens_q, _BLOCK_M)
        base_programs = q_tiles * num_head_q
        if split_dispatch_eligible and split_workspace_safe and (
            (
                not has_mask
                and (
                    (q_tiles <= 2 and num_head_q <= 16 and base_programs <= 32)
                    or (q_tiles == 4 and num_head_q <= 8)
                    or (q_tiles <= 256 and num_head_q <= 32)
                    or (
                        (kv_layout == 1 or num_batch > 1)
                        and q_tiles <= 4
                        and num_head_q <= 32
                    )
                )
            )
            or (mask_split_eligible and q_tiles <= 16)
        ):
            selected = "split"
        elif auto_gqa_eligible and (not split_eligible or split_workspace_safe) and (
            (q_tiles <= 4 and num_head_q <= 16)
            or (q_tiles <= 8 and num_head_q <= 8)
            or (q_tiles == 2 and num_head_q <= 32)
            or (
                q_tiles == 1
                and num_head_q <= 32
                and max_kv_tokens < _AUTO_SPLIT_MIN_KV_TOKENS
            )
        ):
            # Q=256..512 (<=16 heads), Q=1024 (<=8 heads), the
            # Q=256/<=32-head crossover, and the short-Q/16K-KV crossover
            # are validated GQA bands.  Q<=128 remains on split-KV when the
            # KV cache is long enough.
            selected = "gqa"
        elif split_eligible and split_workspace_safe and q_tiles <= 1 and base_programs <= 64:
            # For wider attention, only the single-tile short-Q case has
            # enough evidence to enable split-KV automatically.
            selected = "split"
        else:
            selected = "base"

    if selected == "gqa" and not gqa_eligible:
        selected = "base"
    if selected == "split" and (
        max_kv_tokens < _AUTO_SPLIT_MIN_KV_TOKENS
        or (
            has_mask
            and not (forced_mask_split or mask_split_eligible)
        )
        or (
            not split_workspace_safe
            and os.environ.get("HY3_PREFILL_ALLOW_LARGE_SPLIT", "0") != "1"
        )
    ):
        selected = "base"

    if os.environ.get("HY3_PREFILL_DEBUG_PATH", "0") == "1":
        print(
            "[hy3 prefill] path=%s q=%d kv=%d batch=%d q_heads=%d kv_heads=%d group=%d layout=%d mask=%s split_bytes=%d split_limit=%s"
            % (
                selected,
                max_seqlens_q,
                max_kv_tokens,
                num_batch,
                num_head_q,
                num_head_kv,
                group_size,
                kv_layout,
                has_mask,
                split_workspace_bytes,
                "none" if split_workspace_limit is None else split_workspace_limit,
            )
        )
    return selected


def attention_with_kvcache_blocksparse_prefill_fp8_triton(
    q: torch.Tensor,
    kcache: torch.Tensor,
    vcache: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    block_ids: torch.Tensor,
    seqlens_kvcache: torch.Tensor,
    max_seqlens_q: int,
    quant_type=1,
    block_mask: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    sparsity_bucket: Optional[int] = None,
) -> torch.Tensor:
    """Run portable Triton FP8 paged block-sparse prefill attention."""
    quant_type_value = _quant_type_value(quant_type)
    _check_inputs(
        q,
        kcache,
        vcache,
        qscale,
        kscale,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        max_seqlens_q,
        quant_type_value,
        block_mask,
        output,
    )

    launch = _prepare_launch_context(
        q,
        kcache,
        kscale,
        block_ids,
        max_seqlens_q,
        quant_type_value,
        block_mask,
        output,
        sparsity_bucket,
    )
    output = launch.output
    kscale_kernel = launch.kscale
    num_batch = launch.num_batch
    num_head_q = launch.num_head_q
    num_head_kv = launch.num_head_kv
    page_size = launch.page_size
    max_kv_tokens = launch.max_kv_tokens
    q_len_bucket = launch.q_len_bucket
    kv_len_bucket = launch.kv_len_bucket
    workload_sparsity_bucket = launch.sparsity_bucket
    kv_layout = launch.kv_layout
    has_mask = launch.has_mask
    num_mask_kv_tiles = launch.num_mask_kv_tiles
    mask_arg = launch.mask_arg
    mask_strides = launch.mask_strides
    ks_strides = kscale_kernel.stride() if quant_type_value == 0 else (0, 0, 0, 0)

    q_tiles = triton.cdiv(max_seqlens_q, _BLOCK_M)
    group_size = num_head_q // num_head_kv
    segment_tiles = _segment_tile_config(
        max_seqlens_q,
        max_kv_tokens,
        num_head_q,
        num_batch,
    )
    split_workspace_limit = _split_workspace_limit()
    selected_path = _select_prefill_path(
        max_seqlens_q=max_seqlens_q,
        max_kv_tokens=max_kv_tokens,
        num_head_q=num_head_q,
        num_head_kv=num_head_kv,
        num_batch=num_batch,
        kv_layout=kv_layout,
        has_mask=has_mask,
        sparsity_bucket=workload_sparsity_bucket,
        segment_tiles=segment_tiles,
        split_workspace_limit=split_workspace_limit,
    )
    use_gqa_reuse = selected_path == "gqa"
    if use_gqa_reuse:
        # GROUP_M=2 keeps the per-program accumulator bounded while halving
        # the number of programs that load the same KV head.  Larger groups
        # are intentionally left to future tuning because their accumulator
        # footprint grows linearly with GROUP_M.
        requested_group_m = int(os.environ.get("HY3_PREFILL_GQA_GROUP_M", "2"))
        gqa_group_m = requested_group_m if requested_group_m in (2, 4) else 2
        if group_size % gqa_group_m:
            gqa_group_m = 2
        gqa_warps = int(os.environ.get("HY3_PREFILL_GQA_WARPS", "8"))
        if gqa_warps not in (4, 8):
            gqa_warps = 4
        gqa_stages = 2
        gqa_block_m = 64
        gqa_q_tiles = triton.cdiv(max_seqlens_q, gqa_block_m)
        gqa_grid = (gqa_q_tiles, num_head_q // gqa_group_m, num_batch)
        with torch.cuda.device(q.device):
            _bsa_fp8_prefill_gqa_reuse_kernel[gqa_grid](
                q,
                kcache,
                vcache,
                qscale,
                kscale_kernel,
                vscale,
                cu_seqlens_q,
                block_ids,
                seqlens_kvcache,
                output,
                *q.stride(),
                *kcache.stride(),
                *vcache.stride(),
                *qscale.stride(),
                *ks_strides,
                *block_ids.stride(),
                *output.stride(),
                num_head_q,
                num_head_kv,
                PAGE_SIZE=page_size,
                MAX_KV_TOKENS=max_kv_tokens,
                K_SCALE_PER_TOKEN=quant_type_value == 0,
                KV_LAYOUT=kv_layout,
                GROUP_SIZE=group_size,
                GROUP_M=gqa_group_m,
                GQA_ROWS=gqa_group_m * gqa_block_m,
                Q_LEN_BUCKET=q_len_bucket,
                KV_LEN_BUCKET=kv_len_bucket,
                SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
                FP8_P_SCALE=_FP8_P_SCALE,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                GQA_BLOCK_M=gqa_block_m,
                HEAD_DIM=_HEAD_DIM,
                LOOP_STAGES=1 if max_kv_tokens >= 16384 else 2,
                num_warps=gqa_warps,
                num_stages=gqa_stages,
            )
        return output

    # The selector restricts split-KV to unmasked long-cache workloads.  A
    # forced ``HY3_PREFILL_PATH=split`` is still useful for profiling new
    # shapes, while the default ``auto`` policy only selects it for low-Q
    # workloads where the extra reduction launch has measured benefit.
    use_segmented = selected_path == "split"
    # SEGMENT_TILES is measured in BLOCK_N-sized KV tiles, not individual
    # tokens.  Dividing max_kv_tokens here would create 128x too many mostly
    # empty partials for the default 128-tile segment (e.g. 256 segments for
    # a 32K-token cache instead of 2).
    max_kv_tiles = triton.cdiv(max_kv_tokens, _BLOCK_N)
    num_segments = triton.cdiv(max_kv_tiles, segment_tiles) if use_segmented else 1
    partial_shape = (num_batch, num_head_q, q_tiles, num_segments, _BLOCK_M)
    if use_segmented:
        partial_m = torch.empty(partial_shape, dtype=torch.float32, device=q.device)
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty(
            (*partial_shape, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
    else:
        # The non-segmented specialization eliminates these pointers entirely;
        # passing the output tensor keeps one launch ABI for both variants.
        partial_m = partial_l = partial_acc = output

    grid = (q_tiles, num_head_q, num_batch * num_segments)

    # On gfx936, long unmasked prefill is consistently faster with the
    # validated Wave64/256-thread configuration.  Bypass the autotuner only
    # for this algorithmic case; short tiles and sparse masks retain
    # per-key autotuning because their measured best warp count can differ.
    use_fixed_long_config = not has_mask and max_kv_tokens > _BLOCK_N
    kernel = (
        _bsa_fp8_prefill_kernel.fn
        if use_fixed_long_config
        else _bsa_fp8_prefill_kernel
    )
    loop_stages = 1 if use_fixed_long_config and max_kv_tokens >= 16384 else 2
    launch_options = (
        {"num_warps": 4, "num_stages": 2}
        if use_fixed_long_config
        else {}
    )

    with torch.cuda.device(q.device):
        kernel[grid](
            q,
            kcache,
            vcache,
            qscale,
            kscale_kernel,
            vscale,
            cu_seqlens_q,
            block_ids,
            seqlens_kvcache,
            mask_arg,
            output,
            partial_m,
            partial_l,
            partial_acc,
            *q.stride(),
            *kcache.stride(),
            *vcache.stride(),
            *qscale.stride(),
            *ks_strides,
            *block_ids.stride(),
            *mask_strides,
            *output.stride(),
            num_head_q,
            num_head_kv,
            num_mask_kv_tiles,
            PAGE_SIZE=page_size,
            MAX_KV_TOKENS=max_kv_tokens,
            HAS_BLOCK_MASK=has_mask,
            K_SCALE_PER_TOKEN=quant_type_value == 0,
            Q_LEN_BUCKET=q_len_bucket,
            KV_LEN_BUCKET=kv_len_bucket,
            SPARSITY_BUCKET=workload_sparsity_bucket,
            KV_LAYOUT=kv_layout,
            GROUP_SIZE=group_size,
            Q_TILES=q_tiles,
            SEGMENTED=use_segmented,
            SEGMENT_TILES=segment_tiles,
            NUM_SEGMENTS=num_segments,
            SEGMENT_ACC_BF16=use_segmented,
            SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
            FP8_P_SCALE=_FP8_P_SCALE,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            HEAD_DIM=_HEAD_DIM,
            LOOP_STAGES=loop_stages,
            **launch_options,
        )
        if use_segmented:
            _bsa_fp8_prefill_segment_reduce_kernel[(q_tiles, num_head_q, num_batch)](
                partial_m,
                partial_l,
                partial_acc,
                cu_seqlens_q,
                vscale,
                output,
                *output.stride(),
                num_head_q,
                num_head_kv,
                Q_TILES=q_tiles,
                NUM_SEGMENTS=num_segments,
                K_SCALE_PER_TOKEN=quant_type_value == 0,
                BLOCK_M=_BLOCK_M,
                HEAD_DIM=_HEAD_DIM,
                FP8_P_SCALE=_FP8_P_SCALE,
                num_warps=4,
                num_stages=2,
            )
    return output


# Keep historical provider names importable without retaining an unreachable
# Hopper/TLE dispatch path in the Hygon backend.
attention_with_kvcache_blocksparse_prefill_fp8_hopper = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)
attention_with_kvcache_blocksparse_prefill_fp8_tle = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)
attention_with_kvcache_blocksparse_prefill_fp8 = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)

__all__ = [
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]
