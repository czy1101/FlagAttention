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


import contextlib
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

# Revision marker printed by the sweep tooling; bump when the S60 kernel
# changes so a stale deployment is detectable from benchmark output.
_S60_KERNEL_REV = "s60-prefill-r2"

# S60 launch knobs, read once per process.  They exist so the on-device
# configuration sweep can vary them without editing source:
#   HPC_BSA_S60_WARPS        launch num_warps  (default 1)
#   HPC_BSA_S60_STAGES       launch num_stages (default 1)
#   HPC_BSA_S60_LOOP_STAGES  KV-loop pipeline depth; > 1 historically caused
#                             speculative pointer chasing (NaNs / illegal
#                             accesses) and is opt-in only
#   HPC_BSA_S60_KS           per-token K-scale path: packed | block | gather
_S60_WARPS = max(1, int(os.environ.get("HPC_BSA_S60_WARPS", "1")))
_S60_STAGES = max(1, int(os.environ.get("HPC_BSA_S60_STAGES", "1")))
_S60_LOOP_STAGES = max(1, int(os.environ.get("HPC_BSA_S60_LOOP_STAGES", "1")))
_S60_KS_MODE = os.environ.get("HPC_BSA_S60_KS", "packed").strip().lower()
if _S60_KS_MODE not in ("packed", "block", "gather"):
    raise ValueError("HPC_BSA_S60_KS must be one of packed, block, gather")


def _device_guard(device: torch.device):
    """Enter the active accelerator context without assuming CUDA."""
    device_api = getattr(torch, device.type, None)
    device_context = getattr(device_api, "device", None)
    if callable(device_context):
        return device_context(device)
    return contextlib.nullcontext()



def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
    except (ImportError, OSError, ValueError):
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


@triton.jit
def _decode_fp8e4m3fn(bits):
    """Decode raw E4M3FN bytes without requiring a backend FP8 type."""
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
def _roundtrip_fp8e4m3fn(value):
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
def _decode_fp8e4m3fn_s60(bits):
    """Decode raw E4M3FN bytes by assembling FP32 exponent/mantissa bits.

    Bit-identical to :func:`_decode_fp8e4m3fn` but replaces the per-element
    ``exp2`` chain with integer bit operations.  The S60 software-FP8 path
    decodes two 64x128 pages per KV iteration, so the SFU exp2 traffic is a
    first-order cost there.
    """
    b = bits.to(tl.int32)
    # e4m3 bias 7 -> fp32 bias 127; (e<<3)|m + 120<<3 reassembles the FP32
    # exponent/mantissa fields without carries because m < 8.
    normal_bits = (((b & 0x7F) + 960) << 20) | ((b & 0x80) << 24)
    normal = normal_bits.to(tl.float32, bitcast=True)
    sign = tl.where((b & 0x80) != 0, -1.0, 1.0)
    subnormal = (b & 0x7).to(tl.float32) * 0.001953125 * sign
    value = tl.where((b & 0x78) == 0, subnormal, normal)
    is_nan = ((b & 0x78) == 0x78) & ((b & 0x7) == 0x7)
    return tl.where(is_nan, float("nan"), value)


@triton.jit
def _roundtrip_fp8e4m3fn_s60(value):
    """Round-to-nearest-even to E4M3FN for non-negative finite values.

    The BSA P matrix is a softmax output times a power-of-two scale, so the
    general helper's log2/exp2/division chain collapses to FP32 bit-field
    rounding plus a subnormal lane.  Results are bit-identical.
    """
    x = tl.maximum(value, 0.0)
    bits = x.to(tl.int32, bitcast=True)
    # Keep mantissa bits 22..20, round at bit 19: adding (2^19 - 1) plus the
    # retained LSB implements RNE, and a mantissa carry rolls into the
    # exponent field exactly like a hardware conversion.
    rounded = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -1048576
    normal = rounded.to(tl.float32, bitcast=True)
    # Below the smallest normal (2^-6) the E4M3 grid is a multiple of 2^-9.
    subnormal = tl.floor(x * 512.0 + 0.5) * 0.001953125
    return tl.where(((bits >> 23) & 0xFF) >= 121, normal, subnormal)


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
        "SOFTWARE_FP8",
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
    SOFTWARE_FP8: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """One program computes one [BLOCK_M, HEAD_DIM] output tile."""
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 128)
    tl.static_assert(BLOCK_N % PAGE_SIZE == 0)
    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)

    q_local_start = q_tile * BLOCK_M
    program_valid = q_local_start < q_len
    kv_group_size = num_head_q // num_head_kv
    kv_head = q_head // kv_group_size
    q_start_in_kv = kv_len - q_len

    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_valid = offs_m < q_len

    q_offsets = (q_begin + offs_m[:, None]) * stride_q_token + q_head * stride_q_head + offs_d[None, :] * stride_q_dim
    if SOFTWARE_FP8:
        q_bits = tl.load(q_ptr + q_offsets, mask=q_valid[:, None], other=0)
        q = _decode_fp8e4m3fn(q_bits).to(tl.bfloat16)
    else:
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
    for kv_tile in tl.range(0, max_kv_tiles, 1, num_stages=2):
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
            logical_pages = kv_tokens // PAGE_SIZE
            tokens_in_page = kv_tokens % PAGE_SIZE
            page_offsets = batch * stride_block_ids_batch + logical_pages * stride_block_ids_page
            physical_pages = tl.load(
                block_ids_ptr + page_offsets,
                mask=kv_valid,
                other=0,
            )

            k_offsets = (
                physical_pages[:, None] * stride_k_page
                + tokens_in_page[:, None] * stride_k_token
                + kv_head * stride_k_head
                + offs_d[None, :] * stride_k_dim
            )
            if SOFTWARE_FP8:
                k_bits = tl.load(k_ptr + k_offsets, mask=kv_valid[:, None], other=0)
                k = _decode_fp8e4m3fn(k_bits).to(tl.bfloat16)
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
            if SOFTWARE_FP8:
                p_fp8 = _roundtrip_fp8e4m3fn(p * FP8_P_SCALE).to(tl.bfloat16)
            else:
                p_fp8 = (p * FP8_P_SCALE).to(tl.float8e4nv)

            v_offsets = (
                physical_pages[:, None] * stride_v_page
                + tokens_in_page[:, None] * stride_v_token
                + kv_head * stride_v_head
                + offs_d[None, :] * stride_v_dim
            )
            if SOFTWARE_FP8:
                v_bits = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0)
                v = _decode_fp8e4m3fn(v_bits).to(tl.bfloat16)
            else:
                v = tl.load(v_ptr + v_offsets, mask=kv_valid[:, None], other=0.0)
            acc = tl.dot(p_fp8, v, acc, out_dtype=tl.float32)
            m_i = new_max

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
def _bsa_fp8_prefill_s60_kernel(
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
    MAX_KV_TOKENS: tl.constexpr,
    HAS_BLOCK_MASK: tl.constexpr,
    K_SCALE_PER_TOKEN: tl.constexpr,
    KS_MODE: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    SCALE_LOG2E_OVER_SQRT_D: tl.constexpr,
    FP8_P_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """S60-specialized FP8 prefill with page-sized block-pointer loads.

    Dense fast path: when a KV page lies entirely below the causal boundary
    with all Q rows in range, the iteration drops every score mask and
    boundary check.  FP8 decode and P quantization use FP32 bit-field
    arithmetic, the per-token K scale is fetched as one contiguous page
    block instead of a per-element gather, V is loaded before the QK dot,
    and the accumulator rescale is skipped while the running maximum is
    unchanged.
    """
    tl.static_assert(BLOCK_M == 128)
    tl.static_assert(BLOCK_N == 64)
    tl.static_assert(HEAD_DIM == 128)
    # The loop folds the P scale into exp2 as a +log2(256)=+8 exponent shift.
    tl.static_assert(FP8_P_SCALE == 256.0)

    q_tile = tl.program_id(0)
    q_head = tl.program_id(1)
    batch = tl.program_id(2)

    q_begin = tl.load(cu_seqlens_q_ptr + batch)
    q_end = tl.load(cu_seqlens_q_ptr + batch + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(seqlens_kv_ptr + batch)
    q_local_start = q_tile * BLOCK_M
    program_valid = q_local_start < q_len
    kv_group_size = num_head_q // num_head_kv
    kv_head = q_head // kv_group_size
    q_start_in_kv = kv_len - q_len

    offs_m = q_local_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    q_valid = offs_m < q_len
    q_positions = q_start_in_kv + offs_m

    # Q is contiguous in its token/dimension plane.  Passing the raw uint8
    # storage keeps the GCU target independent of a native FP8 pointer type.
    q_block = tl.make_block_ptr(
        base=q_ptr + q_begin * stride_q_token + q_head * stride_q_head,
        shape=(q_len, HEAD_DIM),
        strides=(stride_q_token, stride_q_dim),
        offsets=(q_local_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    q_bits = tl.load(
        q_block,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    q = _decode_fp8e4m3fn_s60(q_bits).to(tl.bfloat16)

    qs_offsets = (
        batch * stride_qs_batch
        + q_head * stride_qs_head
        + offs_m * stride_qs_token
    )
    q_scale = tl.load(
        qscale_ptr + qs_offsets,
        mask=q_valid,
        other=0.0,
    ).to(tl.float32)
    # Fold the constant score factor (and, in per-tensor mode, the scalar K
    # scale) into the per-row Q scale once, outside the KV loop.
    q_scale = q_scale * SCALE_LOG2E_OVER_SQRT_D
    if not K_SCALE_PER_TOKEN:
        q_scale = q_scale * tl.load(kscale_ptr).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    q_tile_end = tl.minimum(q_local_start + BLOCK_M, q_len)
    causal_kv_end = q_start_in_kv + q_tile_end
    num_causal_blocks = (causal_kv_end + BLOCK_N - 1) // BLOCK_N
    max_kv_blocks = (MAX_KV_TOKENS + BLOCK_N - 1) // BLOCK_N
    # A dense fast path applies only when every Q row of the tile is real.
    rows_full = (q_local_start + BLOCK_M) <= q_len
    # Smallest causal position of this tile + 1: any KV page that ends at or
    # before this edge is causally complete for the whole tile.
    causal_edge = q_start_in_kv + q_local_start + 1

    # One iteration is exactly one S60 page.  The indirect page-table loop
    # depth is configurable; depth above one is opt-in because speculative
    # pointer chasing across sparse pages has historically produced NaNs.
    for kv_block in tl.range(0, max_kv_blocks, 1, num_stages=LOOP_STAGES):
        tile_active = program_valid & (kv_block < num_causal_blocks)
        if HAS_BLOCK_MASK:
            # The public mask remains one bit per 128-token KV tile.  Both
            # page-sized 64-token halves therefore share the same decision.
            mask_kv_tile = kv_block // 2
            mask_in_range = mask_kv_tile < num_mask_kv_tiles
            mask_offset = (
                batch * stride_mask_batch
                + q_head * stride_mask_head
                + q_tile * stride_mask_qtile
                + mask_kv_tile * stride_mask_kvtile
            )
            selected = (
                tl.load(
                    block_mask_ptr + mask_offset,
                    mask=mask_in_range,
                    other=0,
                )
                != 0
            )
            selected = selected | (mask_kv_tile == num_mask_kv_tiles)
            tile_active = tile_active & selected

        # Program-uniform sparse branch: an inactive mask tile performs no
        # page-table lookup, K/V load, scale load, or matrix multiplication.
        if tile_active:
            logical_start = kv_block * BLOCK_N
            logical_page = logical_start // BLOCK_N
            physical_page = tl.load(
                block_ids_ptr
                + batch * stride_block_ids_batch
                + logical_page * stride_block_ids_page,
                mask=logical_start < kv_len,
                other=0,
            )
            kv_tokens = logical_start + offs_n
            kv_valid = kv_tokens < kv_len
            # Dense fast path: the page is fully in range and fully below the
            # causal edge, so no score mask and no boundary check are needed.
            tile_full = (
                rows_full
                & ((logical_start + BLOCK_N) <= kv_len)
                & ((logical_start + BLOCK_N) <= causal_edge)
            )

            k_block = tl.make_block_ptr(
                base=(
                    k_ptr
                    + physical_page * stride_k_page
                    + kv_head * stride_k_head
                ),
                shape=(BLOCK_N, HEAD_DIM),
                strides=(stride_k_token, stride_k_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
            )
            if tile_full:
                k_bits = tl.load(k_block)
            else:
                k_bits = tl.load(
                    k_block,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            k = _decode_fp8e4m3fn_s60(k_bits).to(tl.bfloat16)

            # V never depends on the scores: issue its page DMA before the
            # QK dot so the transfer overlaps QK and the softmax update.
            v_block = tl.make_block_ptr(
                base=(
                    v_ptr
                    + physical_page * stride_v_page
                    + kv_head * stride_v_head
                ),
                shape=(BLOCK_N, HEAD_DIM),
                strides=(stride_v_token, stride_v_dim),
                offsets=(0, 0),
                block_shape=(BLOCK_N, HEAD_DIM),
                order=(1, 0),
            )
            if tile_full:
                v_bits = tl.load(v_block)
            else:
                v_bits = tl.load(
                    v_block,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            v = _decode_fp8e4m3fn_s60(v_bits).to(tl.bfloat16)

            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

            if K_SCALE_PER_TOKEN:
                if KS_MODE == 2:
                    # Launcher-repacked [page, 1, kv_head, 64] scales: the
                    # canonical [page, group, kv_head, 32] layout interleaves
                    # other heads between the two 32-float groups, which no
                    # in-kernel addressing can turn into one coalesced access
                    # on S60.  After the repack each page is a single
                    # contiguous 256-byte load.
                    k_scale = tl.load(
                        kscale_ptr
                        + physical_page * stride_ks_page
                        + kv_head * stride_ks_head
                        + offs_n * stride_ks_token
                    )
                elif KS_MODE == 1:
                    # (2, 32) page tile of the canonical layout; kept for
                    # ablation only.
                    ks_block = tl.make_block_ptr(
                        base=(
                            kscale_ptr
                            + physical_page * stride_ks_page
                            + kv_head * stride_ks_head
                        ),
                        shape=(BLOCK_N // 32, 32),
                        strides=(stride_ks_group, stride_ks_token),
                        offsets=(0, 0),
                        block_shape=(BLOCK_N // 32, 32),
                        order=(1, 0),
                    )
                    k_scale = tl.reshape(tl.load(ks_block), (BLOCK_N,))
                else:
                    ks_offsets = (
                        physical_page * stride_ks_page
                        + (offs_n // 32) * stride_ks_group
                        + kv_head * stride_ks_head
                        + (offs_n % 32) * stride_ks_token
                    )
                    k_scale = tl.load(
                        kscale_ptr + ks_offsets,
                        mask=kv_valid,
                        other=0.0,
                    ).to(tl.float32)
                score_scale = q_scale[:, None] * k_scale[None, :]
                scores = scores * score_scale
            else:
                scores = scores * q_scale[:, None]

            if tile_full:
                new_max = tl.maximum(m_i, tl.max(scores, axis=1))
                p_scaled = tl.exp2(scores - new_max[:, None] + 8.0)
            else:
                causal_valid = kv_tokens[None, :] <= q_positions[:, None]
                score_valid = q_valid[:, None] & kv_valid[None, :] & causal_valid
                scores = tl.where(score_valid, scores, -float("inf"))
                new_max = tl.maximum(m_i, tl.max(scores, axis=1))
                p_scaled = tl.exp2(scores - new_max[:, None] + 8.0)

            alpha = tl.exp2(m_i - new_max)
            # p_scaled already carries the 2^8 FP8_P_SCALE factor; keep the
            # running sum in true-probability scale for the final normalize.
            l_i = l_i * alpha + tl.sum(p_scaled, axis=1) * 0.00390625
            # Rescale the accumulator only when some row's running maximum
            # actually moved; dense late tiles rarely do.  The integer test is
            # immune to the NaN that (-inf) - (-inf) produces for rows that
            # have never seen a valid score.
            row_moved = (new_max > m_i).to(tl.int32)
            if tl.max(row_moved, axis=0) > 0:
                acc = acc * alpha[:, None]
            p_fp8 = _roundtrip_fp8e4m3fn_s60(p_scaled).to(tl.bfloat16)
            acc = tl.dot(p_fp8, v, acc, out_dtype=tl.float32)
            m_i = new_max

    if K_SCALE_PER_TOKEN:
        v_scale = tl.load(vscale_ptr + kv_head).to(tl.float32)
    else:
        v_scale = tl.load(vscale_ptr).to(tl.float32)

    output = acc * (v_scale / FP8_P_SCALE) / l_i[:, None]
    out_block = tl.make_block_ptr(
        base=out_ptr + q_begin * stride_out_token + q_head * stride_out_head,
        shape=(q_len, HEAD_DIM),
        strides=(stride_out_token, stride_out_dim),
        offsets=(q_local_start, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    tl.store(
        out_block,
        output.to(tl.bfloat16),
        boundary_check=(0, 1),
    )


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
        if tensor.device.type not in ("cuda", "gcu"):
            raise ValueError(f"{name} must be a CUDA tensor or GCU tensor")
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
        kv_layout=int(kcache.stride(1) < kcache.stride(2)),
        has_mask=has_mask,
        num_mask_kv_tiles=block_mask.shape[3] if has_mask else 0,
        # The pointer is compile-time unused when HAS_BLOCK_MASK is false.  Use
        # integer metadata as its placeholder so backends without a native FP8
        # pointer type do not reject the kernel signature before specialization.
        mask_arg=block_mask if has_mask else block_ids,
        mask_strides=block_mask.stride() if has_mask else (0, 0, 0, 0),
    )


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
    """Run S60-specialized or portable Triton FP8 prefill attention."""
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
    software_fp8 = q.device.type == "gcu"
    s60_path = software_fp8 and page_size == 64
    ks_mode = 0
    if quant_type_value == 0 and s60_path:
        shape = kscale_kernel.shape
        if _S60_KS_MODE == "packed" and len(shape) == 4 and shape[1] * shape[3] == 64:
            # The canonical [page, group, kv_head, 32] layout interleaves the
            # other kv heads between the two 32-float scale groups, so no
            # in-kernel addressing can fetch one page's 64 scales in a single
            # coalesced access on S60.  Repack once per call into contiguous
            # [page, 1, kv_head, 64], so the kernel performs one 256-byte load.
            pages, groups, heads, width = shape
            kscale_kernel = (
                kscale_kernel.permute(0, 2, 1, 3)
                .contiguous()
                .view(pages, 1, heads, groups * width)
            )
            ks_mode = 2
        elif _S60_KS_MODE == "block":
            ks_mode = 1
    ks_strides = kscale_kernel.stride() if quant_type_value == 0 else (0, 0, 0, 0)
    q_kernel = q.view(torch.uint8) if software_fp8 else q
    k_kernel = kcache.view(torch.uint8) if software_fp8 else kcache
    v_kernel = vcache.view(torch.uint8) if software_fp8 else vcache

    grid = (triton.cdiv(max_seqlens_q, _BLOCK_M), num_head_q, num_batch)
    kernel_args = (
        q_kernel,
        k_kernel,
        v_kernel,
        qscale,
        kscale_kernel,
        vscale,
        cu_seqlens_q,
        block_ids,
        seqlens_kvcache,
        mask_arg,
        output,
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
    )
    with _device_guard(q.device):
        if s60_path:
            _bsa_fp8_prefill_s60_kernel[grid](
                *kernel_args,
                MAX_KV_TOKENS=max_kv_tokens,
                HAS_BLOCK_MASK=has_mask,
                K_SCALE_PER_TOKEN=quant_type_value == 0,
                KS_MODE=ks_mode,
                LOOP_STAGES=_S60_LOOP_STAGES,
                SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
                FP8_P_SCALE=_FP8_P_SCALE,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=64,
                HEAD_DIM=_HEAD_DIM,
                num_warps=_S60_WARPS,
                num_stages=_S60_STAGES,
            )
        else:
            _bsa_fp8_prefill_kernel[grid](
                *kernel_args,
                PAGE_SIZE=page_size,
                MAX_KV_TOKENS=max_kv_tokens,
                HAS_BLOCK_MASK=has_mask,
                K_SCALE_PER_TOKEN=quant_type_value == 0,
                Q_LEN_BUCKET=q_len_bucket,
                KV_LEN_BUCKET=kv_len_bucket,
                SPARSITY_BUCKET=workload_sparsity_bucket,
                KV_LAYOUT=kv_layout,
                SOFTWARE_FP8=software_fp8,
                SCALE_LOG2E_OVER_SQRT_D=_LOG2E / 11.313708498984761,
                FP8_P_SCALE=_FP8_P_SCALE,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                HEAD_DIM=_HEAD_DIM,
            )
    return output

# Preserve the package-level API while routing every public provider to the
# maintained Triton implementation.
attention_with_kvcache_blocksparse_prefill_fp8 = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)
attention_with_kvcache_blocksparse_prefill_fp8_hopper = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)
attention_with_kvcache_blocksparse_prefill_fp8_tle = (
    attention_with_kvcache_blocksparse_prefill_fp8_triton
)


__all__ = [
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]
