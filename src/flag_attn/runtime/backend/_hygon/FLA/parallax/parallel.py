# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Training-time Parallax attention implemented with Triton kernels."""

from __future__ import annotations

import copy

import torch
import triton
import triton.language as tl

from flag_attn.runtime.backend._hygon.FLA.index import prepare_chunk_indices
from flag_attn.runtime.backend._hygon.heuristics_config_utils import HEURISTICS_CONFIGS
from flag_attn.runtime.backend._hygon.device_info import (
    get_max_shared_memory,
    get_sm_count,
)
from flag_attn.runtime.backend._hygon.FLA.utils import input_guard


@triton.jit
def exp2(x):
    return tl.math.exp2(x.to(tl.float32))


def _block_size(head_dim: int, device_index: int) -> int:
    # Dense and variable-length paths share BT so chunk_indices stay consistent.
    del device_index
    config = HEURISTICS_CONFIGS.get("parallax", {}).get("default", {})
    required_shared_memory = int(config.get("forward_shared_memory_bytes", 232_448))
    if get_max_shared_memory() >= required_shared_memory:
        return 64 if head_dim <= 128 else 32
    return 64


def _make_parallax_configs(tile_pairs):
    """Build independent autotune spaces for each long-sequence pass."""
    configs = []
    for block_k, block_s in tile_pairs:
        if block_k == 64 and block_s == 16:
            # A narrow score tile is specifically useful when long-sequence
            # kernels hit the 255-register ceiling. One-stage variants keep
            # its shared-memory footprint low enough for more resident CTAs.
            launch_configs = ((2, 1), (2, 2), (4, 1), (4, 2))
        elif block_k == 64:
            launch_configs = ((2, 2), (4, 1), (4, 2), (8, 2))
        else:
            launch_configs = ((4, 1), (4, 2), (8, 1), (8, 2))
        for num_warps, num_stages in launch_configs:
            configs.append(
                triton.Config(
                    {"BK": block_k, "BS": block_s},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


# BT fixes the query-row grid. BS is tuned independently because FWD/DQR
# accumulate over query rows, while DKV carries two [BS, BK] fp32 accumulators.
# DQR additionally tries BS=16 for D=64 to relieve short-lived score-tile
# pressure. Forward retains BS>=32 to limit loop overhead because it keeps two
# output accumulators live.
_PARALLAX_DQR_TILES = (
    (16, 64),
    (16, 128),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
)

_PARALLAX_FWD_TILES = tuple(tile for tile in _PARALLAX_DQR_TILES if tile != (64, 16))

_PARALLAX_DKV_TILES = (
    (16, 64),
    (16, 128),
    (32, 32),
    (32, 64),
    (32, 128),
    (64, 32),
    (64, 64),
    (64, 128),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128),
    (256, 16),
    (256, 32),
    (256, 64),
    (256, 128),
)

PARALLAX_FWD_CONFIGS = _make_parallax_configs(_PARALLAX_FWD_TILES)
PARALLAX_DQR_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
PARALLAX_DQR_SPLIT_CONFIGS = _make_parallax_configs(_PARALLAX_DQR_TILES)
PARALLAX_DKV_CONFIGS = _make_parallax_configs(_PARALLAX_DKV_TILES)
PARALLAX_PREPROCESS_CONFIGS = [
    triton.Config(
        {"BP": block_p},
        num_warps=num_warps,
        num_stages=num_stages,
    )
    for block_p in (32, 64, 128, 256)
    for num_warps in (1, 2)
    for num_stages in (1, 2)
]

def _autotune_arg(named_args, kwargs, name):
    value = named_args.get(name)
    # FlagTree may keep constexpr names in `named_args` with a None value and
    # pass their actual launch value through kwargs.  `dict.get(key, default)`
    # does not use the default when such a placeholder exists.
    return kwargs.get(name) if value is None else value


def _copy_autotune_configs(configs):
    """Protect config templates from mutation by FlagTree's autotuner."""
    return copy.deepcopy(list(configs))


def parallax_prune_configs(configs, named_args, **kwargs):
    """Keep the matching head tile and preserve one fixed varlen config."""
    head_dim = _autotune_arg(named_args, kwargs, "K")
    if head_dim is None:
        q = _autotune_arg(named_args, kwargs, "q")
        if q is not None and hasattr(q, "shape"):
            head_dim = q.shape[-1]
    if head_dim is None:
        return _copy_autotune_configs(configs)

    expected_bk = triton.next_power_of_2(int(head_dim))
    valid = [cfg for cfg in configs if cfg.kwargs["BK"] == expected_bk]

    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(valid)

    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [cfg for cfg in valid if cfg.kwargs["BS"] == block_t]
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 8 and cfg.num_stages == 2]
    return _copy_autotune_configs(preferred or tile_matches)


def parallax_prune_preprocess_configs(configs, named_args, **kwargs):
    """Varlen chunk indices require preprocess BP to remain equal to BT."""
    cu_seqlens = _autotune_arg(named_args, kwargs, "cu_seqlens")
    if cu_seqlens is None:
        return _copy_autotune_configs(configs)
    block_t = _autotune_arg(named_args, kwargs, "BT")
    tile_matches = [cfg for cfg in configs if cfg.kwargs["BP"] == block_t]
    preferred = [cfg for cfg in tile_matches if cfg.num_warps == 4 and cfg.num_stages == 2]
    return _copy_autotune_configs(preferred or tile_matches)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_short(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    STORE_AUX: tl.constexpr,
):
    """One-tile dense forward used when the complete sequence fits in SRAM."""
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_r = tl.make_block_ptr(
        r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

    offs_q = tl.arange(0, BT)
    offs_kv = tl.arange(0, BT)
    row_mask = offs_q[:, None] < T
    mask = (offs_q[:, None] >= offs_kv[None, :]) & row_mask & (offs_kv[None, :] < T)
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_kv[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)

    scale_log2 = scale * 1.4426950216
    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))
    m_acc = tl.max(qk, axis=1, keep_dims=True)
    safe_m = tl.where(m_acc == -float("inf"), 0.0, m_acc)
    w = exp2(qk - safe_m)
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    wr = w * rk

    d1_acc = tl.sum(w, axis=1, keep_dims=True)
    d2_acc = tl.sum(wr, axis=1, keep_dims=True)
    barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32)
    rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32)
    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - rv_acc * inv_d1

    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_barv = tl.make_block_ptr(
        barv + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_d1 = tl.make_block_ptr(
        d1 + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_bart = tl.make_block_ptr(
        bart + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_m = tl.make_block_ptr(
        m + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if STORE_AUX:
        tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_d1, d1_acc, boundary_check=(0, 1))
        tl.store(p_bart, b_bart, boundary_check=(0, 1))
        tl.store(p_m, m_acc, boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
    STORE_AUX: tl.constexpr,
):
    """Dense forward for a small number of query tiles."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K
    q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)

    m_acc = tl.full((BT, 1), -float("inf"), dtype=tl.float32)
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
        col_mask = offs_s[:, None] < T
        kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        wr = w * tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)

        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        barv_acc = alpha * barv_acc
        rv_acc = alpha * rv_acc
        barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
        rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=rv_acc)
        m_acc = m_new

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - rv_acc * inv_d1

    scalar_offsets = (bos + offs_t[:, None]) * HQ + i_hq
    tl.store(o + q_offsets, b_o.to(o.dtype.element_ty), mask=row_mask & head_mask)
    if STORE_AUX:
        tl.store(barv + q_offsets, b_barv.to(barv.dtype.element_ty), mask=row_mask & head_mask)
        tl.store(d1 + scalar_offsets, d1_acc, mask=row_mask)
        tl.store(bart + scalar_offsets, b_bart, mask=row_mask)
        tl.store(m + scalar_offsets, m_acc, mask=row_mask)


@triton.autotune(
    configs=PARALLAX_FWD_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_fwd_kernel(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_AUX: tl.constexpr,
    FUSE_PV_ROWS: tl.constexpr,
    CTA_ORDER_GQA: tl.constexpr,
    KV_CACHE_CG: tl.constexpr,
):
    if CTA_ORDER_GQA:
        # Keep one query head per CTA; interleave only heads sharing KV.
        i_tg = tl.program_id(0).to(tl.int64)
        i_t, i_g = i_tg // G, i_tg % G
        i_bh = tl.program_id(1)
        i_b, i_h = i_bh // H, i_bh % H
        i_hq = i_h * G + i_g
    else:
        i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
        i_b, i_hq = i_bh // HQ, i_bh % HQ
        i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    # SWA col-block boundaries. WINDOW_SIZE_LEFT < 0 disables SWA.
    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    m_acc = tl.zeros((BT, 1), dtype=tl.float32) - float("inf")
    d1_acc = tl.zeros((BT, 1), dtype=tl.float32)
    d2_acc = tl.zeros((BT, 1), dtype=tl.float32)
    if FUSE_PV_ROWS:
        pv_acc = tl.zeros((2 * BT, BK), dtype=tl.float32)
    else:
        barv_acc = tl.zeros((BT, BK), dtype=tl.float32)
        Rv_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    # Phase 0: left-border blocks (SWA only). Window mask only.
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        if KV_CACHE_CG:
            b_k = tl.load(p_k, mask=m_kv, other=0.0, cache_modifier=".cg")
            b_v = tl.load(p_v, mask=m_kv, other=0.0, cache_modifier=".cg")
        else:
            b_k = tl.load(p_k, mask=m_kv, other=0.0)
            b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        if FUSE_PV_ROWS:
            alpha_rows = tl.reshape(
                tl.permute(tl.join(alpha, alpha), (0, 2, 1)), (2 * BT, 1)
            )
            pv_weights = tl.reshape(
                tl.permute(tl.join(w, wr), (0, 2, 1)), (2 * BT, BS)
            ).to(b_v.dtype)
            pv_acc = alpha_rows * pv_acc
            pv_acc = tl.dot(pv_weights, b_v, out_dtype=tl.float32, acc=pv_acc)
        else:
            barv_acc = alpha * barv_acc
            barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
            Rv_acc = alpha * Rv_acc
            Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase A: safe blocks (no mask).
    for _safe in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _safe.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        if KV_CACHE_CG:
            b_k = tl.load(p_k, mask=m_kv, other=0.0, cache_modifier=".cg")
            b_v = tl.load(p_v, mask=m_kv, other=0.0, cache_modifier=".cg")
        else:
            b_k = tl.load(p_k, mask=m_kv, other=0.0)
            b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        if FUSE_PV_ROWS:
            alpha_rows = tl.reshape(
                tl.permute(tl.join(alpha, alpha), (0, 2, 1)), (2 * BT, 1)
            )
            pv_weights = tl.reshape(
                tl.permute(tl.join(w, wr), (0, 2, 1)), (2 * BT, BS)
            ).to(b_v.dtype)
            pv_acc = alpha_rows * pv_acc
            pv_acc = tl.dot(pv_weights, b_v, out_dtype=tl.float32, acc=pv_acc)
        else:
            barv_acc = alpha * barv_acc
            barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
            Rv_acc = alpha * Rv_acc
            Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        if KV_CACHE_CG:
            b_k = tl.load(p_k, mask=m_kv, other=0.0, cache_modifier=".cg")
            b_v = tl.load(p_v, mask=m_kv, other=0.0, cache_modifier=".cg")
        else:
            b_k = tl.load(p_k, mask=m_kv, other=0.0)
            b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        qk = tl.where(mask, qk, -float("inf"))
        m_new = tl.maximum(m_acc, tl.max(qk, axis=1, keep_dims=True))
        safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = exp2(m_acc - safe_m)
        w = exp2(qk - safe_m)
        d1_acc = alpha * d1_acc + tl.sum(w, axis=1, keep_dims=True)
        wr = w * rk
        d2_acc = alpha * d2_acc + tl.sum(wr, axis=1, keep_dims=True)
        if FUSE_PV_ROWS:
            alpha_rows = tl.reshape(
                tl.permute(tl.join(alpha, alpha), (0, 2, 1)), (2 * BT, 1)
            )
            pv_weights = tl.reshape(
                tl.permute(tl.join(w, wr), (0, 2, 1)), (2 * BT, BS)
            ).to(b_v.dtype)
            pv_acc = alpha_rows * pv_acc
            pv_acc = tl.dot(pv_weights, b_v, out_dtype=tl.float32, acc=pv_acc)
        else:
            barv_acc = alpha * barv_acc
            barv_acc = tl.dot(w.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=barv_acc)
            Rv_acc = alpha * Rv_acc
            Rv_acc = tl.dot(wr.to(b_v.dtype), b_v, out_dtype=tl.float32, acc=Rv_acc)
        m_acc = m_new

    if FUSE_PV_ROWS:
        pv_pair = tl.reshape(pv_acc, (BT, 2, BK))
        barv_acc, Rv_acc = tl.split(tl.permute(pv_pair, (0, 2, 1)))

    inv_d1 = tl.where(row_mask, 1.0 / d1_acc, 0.0)
    b_barv = barv_acc * inv_d1
    b_bart = d2_acc * inv_d1
    b_o = b_barv + b_bart * b_barv - Rv_acc * inv_d1

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_qk)
    if STORE_AUX:
        tl.store(p_barv, b_barv.to(p_barv.dtype.element_ty), mask=m_qk)
        tl.store(p_d1, d1_acc, mask=row_mask)
        tl.store(p_bart, b_bart, mask=row_mask)
        tl.store(p_m, m_acc, mask=row_mask)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_short(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_q,
    grad_r,
    grad_k_buf,
    grad_v_buf,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
):
    """Fuse preprocess, dQ/dR, and dK/dV for a one-tile sequence."""
    i_bh = tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_r = tl.make_block_ptr(
        r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * K,
        (T, K),
        (H * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_barv = tl.make_block_ptr(
        barv + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_o = tl.make_block_ptr(
        grad_o + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_d1 = tl.make_block_ptr(
        d1 + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_bart = tl.make_block_ptr(
        bart + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )
    p_m = tl.make_block_ptr(
        m + bos * HQ + i_hq,
        (T, 1),
        (HQ, 1),
        (0, 0),
        (BT, 1),
        (1, 0),
    )

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_r = tl.load(p_r, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_barv = tl.load(p_barv, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_grad_o = tl.load(p_grad_o, boundary_check=(0, 1), padding_option="zero")
    b_d1 = tl.load(p_d1, boundary_check=(0, 1), padding_option="zero")
    b_bart = tl.load(p_bart, boundary_check=(0, 1), padding_option="zero")
    b_m = tl.load(p_m, boundary_check=(0, 1), padding_option="zero")

    delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)

    offs_q = tl.arange(0, BT)
    offs_kv = tl.arange(0, BT)
    row_mask = offs_q[:, None] < T
    mask = (offs_q[:, None] >= offs_kv[None, :]) & row_mask & (offs_kv[None, :] < T)
    if WINDOW_SIZE_LEFT >= 0:
        mask = mask & (offs_kv[None, :] >= offs_q[:, None] - WINDOW_SIZE_LEFT + 1)

    scale_log2 = scale * 1.4426950216
    qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
    qk = tl.where(mask, qk, -float("inf"))
    w = exp2(qk - b_m)
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    p = w * inv_d1

    a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
    rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
    delta = a - delta_b
    bart_minus_rk = b_bart - rk
    gl = p * (a - delta_t + bart_minus_rk * delta)
    gu = -p * delta

    grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32) * scale
    grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32)
    grad_k_acc = tl.dot(tl.trans(gl * scale).to(b_q.dtype), b_q, out_dtype=tl.float32)
    grad_k_acc += tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32)
    weights = p * (1.0 + bart_minus_rk)
    grad_v_acc = tl.dot(tl.trans(weights).to(b_grad_o.dtype), b_grad_o, out_dtype=tl.float32)

    p_grad_q = tl.make_block_ptr(
        grad_q + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_r = tl.make_block_ptr(
        grad_r + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_k = tl.make_block_ptr(
        grad_k_buf + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    p_grad_v = tl.make_block_ptr(
        grad_v_buf + (bos * HQ + i_hq) * K,
        (T, K),
        (HQ * K, 1),
        (0, 0),
        (BT, BK),
        (1, 0),
    )
    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), boundary_check=(0, 1))


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_q,
    grad_r,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    """Compute dQ/dR for a small fixed number of dense tiles."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_t = i_t * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    row_mask = offs_t[:, None] < T
    head_mask = offs_k[None, :] < K
    q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]

    b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
    b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
    b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
    scalar_offsets = (bos + offs_t) * HQ + i_hq
    b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
    b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
    b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

    delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
    delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)
    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_s in range(0, tl.minimum(i_t + 1, NT)):
        offs_s = (i_s * BT + tl.arange(0, BT)).to(tl.int64)
        col_mask = offs_s[:, None] < T
        kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
        b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
        b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)
        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_q_acc = tl.dot(gl.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)
        grad_r_acc = tl.dot(gu.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_r_acc)

    tl.store(grad_q + q_offsets, grad_q_acc.to(grad_q.dtype.element_ty), mask=row_mask & head_mask)
    tl.store(grad_r + q_offsets, grad_r_acc.to(grad_r.dtype.element_ty), mask=row_mask & head_mask)


@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv_short_multi(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    grad_o,
    grad_k_buf,
    grad_v_buf,
    scale,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    NT: tl.constexpr,
):
    """Compute dK/dV without atomics for a small fixed number of dense tiles."""
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = i_b * T

    offs_s = i_s * BT + tl.arange(0, BT).to(tl.int64)
    offs_k = tl.arange(0, BK).to(tl.int64)
    col_mask = offs_s[:, None] < T
    head_mask = offs_k[None, :] < K
    kv_offsets = ((bos + offs_s[:, None]) * H + i_h) * K + offs_k[None, :]
    b_k = tl.load(k + kv_offsets, mask=col_mask & head_mask, other=0.0)
    b_v = tl.load(v + kv_offsets, mask=col_mask & head_mask, other=0.0)

    grad_k_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * 1.4426950216

    for i_t in range(i_s, NT):
        offs_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
        row_mask = offs_t[:, None] < T
        q_offsets = ((bos + offs_t[:, None]) * HQ + i_hq) * K + offs_k[None, :]
        b_q = tl.load(q + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_r = tl.load(r + q_offsets, mask=row_mask & head_mask, other=0.0)
        b_o = tl.load(o + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_barv = tl.load(barv + q_offsets, mask=row_mask & head_mask, other=0.0).to(tl.float32)
        b_grad_o = tl.load(grad_o + q_offsets, mask=row_mask & head_mask, other=0.0)
        scalar_offsets = (bos + offs_t) * HQ + i_hq
        b_d1 = tl.load(d1 + scalar_offsets, mask=offs_t < T, other=1.0)[:, None]
        b_bart = tl.load(bart + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]
        b_m = tl.load(m + scalar_offsets, mask=offs_t < T, other=0.0)[:, None]

        mask = (offs_t[:, None] >= offs_s[None, :]) & row_mask & (offs_s[None, :] < T)
        if WINDOW_SIZE_LEFT >= 0:
            mask = mask & (offs_s[None, :] >= offs_t[:, None] - WINDOW_SIZE_LEFT + 1)

        delta_t = tl.sum(b_grad_o.to(tl.float32) * b_o, axis=1, keep_dims=True)
        delta_b = tl.sum(b_grad_o.to(tl.float32) * b_barv, axis=1, keep_dims=True)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * tl.where(row_mask, 1.0 / b_d1, 0.0)
        a = tl.dot(b_grad_o, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - delta_b
        bart_minus_rk = b_bart - tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        gl = p * (a - delta_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1.0 + bart_minus_rk)
        grad_v_acc = tl.dot(
            tl.trans(weights).to(b_grad_o.dtype),
            b_grad_o,
            out_dtype=tl.float32,
            acc=grad_v_acc,
        )

    grad_offsets = ((bos + offs_s[:, None]) * HQ + i_hq) * K + offs_k[None, :]
    tl.store(
        grad_k_buf + grad_offsets,
        grad_k_acc.to(grad_k_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )
    tl.store(
        grad_v_buf + grad_offsets,
        grad_v_acc.to(grad_v_buf.dtype.element_ty),
        mask=col_mask & head_mask,
    )


@triton.autotune(
    configs=PARALLAX_PREPROCESS_CONFIGS,
    key=["T", "K", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_preprocess_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_preprocess(
    grad_o,
    o,
    barv,
    delta_t,
    delta_b,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)

    row_offset = i_t * BP
    o_t = row_offset + tl.arange(0, BP)
    o_k = tl.arange(0, BK)
    m_tk = (o_t[:, None] < T) & (o_k[None, :] < K)
    m_t = o_t[:, None] < T
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_o = o + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_barv = barv + (bos * HQ + i_hq) * K + o_t[:, None] * (HQ * K) + o_k[None, :]
    p_t = delta_t + bos * HQ + i_hq + o_t[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + o_t[:, None] * HQ

    b_grad_o = tl.load(p_grad_o, mask=m_tk, other=0.0).to(tl.float32)
    b_o = tl.load(p_o, mask=m_tk, other=0.0).to(tl.float32)
    b_barv = tl.load(p_barv, mask=m_tk, other=0.0).to(tl.float32)

    b_t = tl.sum(b_grad_o * b_o, axis=1, keep_dims=True)
    b_b = tl.sum(b_grad_o * b_barv, axis=1, keep_dims=True)

    tl.store(p_t, b_t, mask=m_t)
    tl.store(p_b, b_b, mask=m_t)


@triton.jit(do_not_specialize=["T"])
def _parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    FUSED_PREPROCESS: tl.constexpr,
    CTA_ORDER_GQA: tl.constexpr,
):
    physical_i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    if CTA_ORDER_GQA:
        # Interleave query-head CTAs that scan the same shared KV head. This
        # changes launch order only; each CTA still computes one head/tile.
        i_t = physical_i_t // G
        group_rank = physical_i_t % G
        i_b, i_h = i_bh // H, i_bh % H
        i_hq = i_h * G + group_rank
    else:
        i_t = physical_i_t
        i_b, i_hq = i_bh // HQ, i_bh % HQ
        i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    row_offset = i_t * BT
    row_indices = row_offset + tl.arange(0, BT)
    row_mask = row_indices[:, None] < T
    NUM_TOTAL_BLOCKS = tl.cdiv(tl.minimum(T, row_offset + BT), BS)
    NUM_SAFE_BLOCKS = tl.minimum(row_offset, T) // BS

    if WINDOW_SIZE_LEFT >= 0:
        leftmost_valid = tl.maximum(0, row_offset - WINDOW_SIZE_LEFT + 1)
        FIRST_COL_BLOCK = leftmost_valid // BS
        # Phase A is unmasked, so the safe zone must clear the window's left edge for
        # the tile's LAST row (row_offset + BT - 1), not its first.
        safe_left_valid = tl.maximum(0, row_offset + BT - WINDOW_SIZE_LEFT)
        SAFE_LEFT_START = (safe_left_valid + BS - 1) // BS
    else:
        FIRST_COL_BLOCK = 0
        SAFE_LEFT_START = 0
    LEFT_BORDER_END = tl.minimum(SAFE_LEFT_START, NUM_SAFE_BLOCKS)
    SAFE_MIDDLE_START = tl.maximum(FIRST_COL_BLOCK, SAFE_LEFT_START)
    RIGHT_BORDER_START = tl.maximum(FIRST_COL_BLOCK, NUM_SAFE_BLOCKS)

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_qk = row_mask & m_k[None, :]
    p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
    p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_q = grad_q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_r = grad_r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]

    b_q = tl.load(p_q, mask=m_qk, other=0.0)
    b_r = tl.load(p_r, mask=m_qk, other=0.0)
    b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
    b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
    b_m = tl.load(p_m, mask=row_mask, other=0.0)
    grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)
    if FUSED_PREPROCESS:
        p_o = o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_barv = barv + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        grad_o_tile_fp32 = grad_o_tile.to(tl.float32)
        b_o = tl.load(p_o, mask=m_qk, other=0.0).to(tl.float32)
        b_barv = tl.load(p_barv, mask=m_qk, other=0.0).to(tl.float32)
        b_t = tl.sum(grad_o_tile_fp32 * b_o, axis=1, keep_dims=True)
        b_b = tl.sum(grad_o_tile_fp32 * b_barv, axis=1, keep_dims=True)
        tl.store(p_t, b_t, mask=row_mask)
        tl.store(p_b, b_b, mask=row_mask)
    else:
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
    grad_q_acc = tl.zeros((BT, BK), dtype=tl.float32)
    grad_r_acc = tl.zeros((BT, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)

    # Phase 0: left-border blocks (SWA only).
    for col_block_id in range(FIRST_COL_BLOCK, LEFT_BORDER_END):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        # Finish dR before materializing rk/gl. This shortens the live ranges
        # of several [BT, BS] matrices and substantially reduces register
        # pressure in the long-sequence kernel.
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase A: safe blocks (no mask).
    for _ in range(SAFE_MIDDLE_START, NUM_SAFE_BLOCKS):
        o_kv = _.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (o_kv[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + o_kv[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    # Phase B: right-border blocks (causal + boundary + window mask).
    for col_block_id in range(RIGHT_BORDER_START, NUM_TOTAL_BLOCKS):
        col_indices = col_block_id.to(tl.int64) * BS + tl.arange(0, BS)
        m_kv = (col_indices[:, None] < T) & m_k[None, :]
        p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_kv, other=0.0)
        b_v = tl.load(p_v, mask=m_kv, other=0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        qk = tl.where(mask, qk, -float("inf"))
        p = exp2(qk - b_m) * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        grad_r_acc = tl.dot(
            (-p * delta).to(b_k.dtype),
            b_k,
            out_dtype=tl.float32,
            acc=grad_r_acc,
        )
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        a = p * (a - b_t + (b_bart - rk) * delta)
        grad_q_acc = tl.dot(a.to(b_k.dtype), b_k, out_dtype=tl.float32, acc=grad_q_acc)

    grad_q_acc = scale * grad_q_acc

    tl.store(p_grad_q, grad_q_acc.to(p_grad_q.dtype.element_ty), mask=m_qk)
    tl.store(p_grad_r, grad_r_acc.to(p_grad_r.dtype.element_ty), mask=m_qk)






@triton.autotune(
    configs=PARALLAX_DQR_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr_fused(
    q,
    r,
    k,
    v,
    o,
    barv,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CTA_ORDER_GQA: tl.constexpr,
):
    _parallel_parallax_bwd_kernel_dqr(
        q,
        r,
        k,
        v,
        o,
        barv,
        d1,
        bart,
        m,
        delta_t,
        delta_b,
        grad_o,
        grad_q,
        grad_r,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        HQ,
        H,
        G,
        K,
        BK,
        WINDOW_SIZE_LEFT,
        BT,
        BS,
        IS_VARLEN,
        True,
        CTA_ORDER_GQA,
    )


@triton.autotune(
    configs=PARALLAX_DQR_SPLIT_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    reset_to_zero=["grad_q", "grad_r"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dqr(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_q,
    grad_r,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CTA_ORDER_GQA: tl.constexpr,
):
    _parallel_parallax_bwd_kernel_dqr(
        q,
        r,
        k,
        v,
        q,
        q,
        d1,
        bart,
        m,
        delta_t,
        delta_b,
        grad_o,
        grad_q,
        grad_r,
        scale,
        cu_seqlens,
        chunk_indices,
        T,
        HQ,
        H,
        G,
        K,
        BK,
        WINDOW_SIZE_LEFT,
        BT,
        BS,
        IS_VARLEN,
        False,
        CTA_ORDER_GQA,
    )




@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    CTA_ORDER_GQA: tl.constexpr,
):
    physical_i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    if CTA_ORDER_GQA:
        # Make query heads sharing one KV head adjacent in the x dimension.
        # This preserves one CTA per query head while bringing repeated K/V
        # tile loads closer together in the device's block issue order.
        n_kv_tiles = tl.cdiv(T, BS)
        i_g = physical_i_t
        i_bht = i_bh
        i_t = i_bht % n_kv_tiles
        i_bh = i_bht // n_kv_tiles
        i_b, i_h = i_bh // H, i_bh % H
        i_hq = i_h * G + i_g
    else:
        i_t = physical_i_t
        i_b, i_hq = i_bh // HQ, i_bh % HQ
        i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)

    start_row_block = col_offset // BT

    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_k = grad_k + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * HQ + i_hq) * K + col_indices[:, None] * (HQ * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)

    # Phase A: causal-border row blocks.
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    for row_block_id in range(start_row_block, causal_end):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        if WINDOW_SIZE_LEFT >= 0:
            mask = (
                (row_indices[:, None] >= col_indices[None, :])
                & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
        else:
            mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    # Phase B: safe row blocks (no causal/col/window mask).
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    # Phase C: window-border row blocks (SWA only).
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)
    for row_block_id in range(window_border_start, num_row_blocks):
        row_offset = row_block_id.to(tl.int64) * BT
        row_indices = row_offset + tl.arange(0, BT)
        row_mask = row_indices[:, None] < T
        m_qk = row_mask & m_k[None, :]
        p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
        p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        b_r = tl.load(p_r, mask=m_qk, other=0.0)
        b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
        b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
        b_m = tl.load(p_m, mask=row_mask, other=0.0)
        b_t = tl.load(p_t, mask=row_mask, other=0.0)
        b_b = tl.load(p_b, mask=row_mask, other=0.0)
        grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

        qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
        rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
        inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
        mask = (
            (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
            & row_mask
            & (col_indices[None, :] < T)
        )
        qk = tl.where(mask, qk, -float("inf"))
        w = exp2(qk - b_m)
        p = w * inv_d1
        a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
        delta = a - b_b
        bart_minus_rk = b_bart - rk
        gl = p * (a - b_t + bart_minus_rk * delta) * scale
        gu = -p * delta
        grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
        grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
        weights = p * (1 + bart_minus_rk)
        grad_v_acc = tl.dot(tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc)

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.autotune(
    configs=PARALLAX_DKV_CONFIGS,
    key=["T", "K", "G", "WINDOW_SIZE_LEFT", "IS_VARLEN"],
    prune_configs_by={"early_config_prune": parallax_prune_configs},
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_parallax_bwd_kernel_dkv_grouped(
    q,
    r,
    k,
    v,
    d1,
    bart,
    m,
    delta_t,
    delta_b,
    grad_o,
    grad_k,
    grad_v,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Accumulate all query heads in a GQA group directly into one KV head.

    The decode kernel packs 2/4/8 query heads that share K/V into one CTA. For
    training, packing the forward rows would create prohibitively large score
    tiles, but dK/dV already has exactly one destination per KV head. Owning
    that destination here removes the [B, T, HQ, K] staging tensors and their
    separate reduction while loading each K/V tile only once per group.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = (i_b * T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    col_offset = i_t * BS
    col_indices = col_offset + tl.arange(0, BS)
    start_row_block = col_offset // BT
    num_row_blocks_qbound = tl.cdiv(T, BT)
    if WINDOW_SIZE_LEFT >= 0:
        last_row_window = tl.cdiv(col_offset + BS + WINDOW_SIZE_LEFT - 1, BT)
        num_row_blocks = tl.minimum(num_row_blocks_qbound, last_row_window)
        WINDOW_SAFE_END = (col_offset + WINDOW_SIZE_LEFT) // BT
    else:
        num_row_blocks = num_row_blocks_qbound
        WINDOW_SAFE_END = num_row_blocks

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    m_ck = (col_indices[:, None] < T) & m_k[None, :]
    p_k = k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_v = v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_k = grad_k + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]
    p_grad_v = grad_v + (bos * H + i_h) * K + col_indices[:, None] * (H * K) + o_k[None, :]

    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    b_v = tl.load(p_v, mask=m_ck, other=0.0)
    grad_k_acc = tl.zeros((BS, BK), dtype=tl.float32)
    grad_v_acc = tl.zeros((BS, BK), dtype=tl.float32)
    scale_log2 = scale * RCP_LN2

    first_safe_row_block = tl.cdiv(col_offset + BS, BT)
    SAFE_MIDDLE_END = tl.minimum(WINDOW_SAFE_END, num_row_blocks)
    WINDOW_BORDER_START = tl.maximum(first_safe_row_block, WINDOW_SAFE_END)
    causal_end = tl.minimum(first_safe_row_block, num_row_blocks)
    safe_b_start = tl.maximum(first_safe_row_block, start_row_block)
    window_border_start = tl.maximum(WINDOW_BORDER_START, start_row_block)

    for i_g in range(0, G):
        i_hq = i_h * G + i_g

        # Phase A: causal-border row blocks.
        for row_block_id in range(start_row_block, causal_end):
            row_offset = row_block_id.to(tl.int64) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            if WINDOW_SIZE_LEFT >= 0:
                mask = (
                    (row_indices[:, None] >= col_indices[None, :])
                    & (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                    & row_mask
                    & (col_indices[None, :] < T)
                )
            else:
                mask = (row_indices[:, None] >= col_indices[None, :]) & row_mask & (col_indices[None, :] < T)
            qk = tl.where(mask, qk, -float("inf"))
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

        # Phase B: safe row blocks (no causal/col/window mask).
        for row_block_id in range(safe_b_start, SAFE_MIDDLE_END):
            row_offset = row_block_id.to(tl.int64) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

        # Phase C: window-border row blocks (SWA only).
        for row_block_id in range(window_border_start, num_row_blocks):
            row_offset = row_block_id.to(tl.int64) * BT
            row_indices = row_offset + tl.arange(0, BT)
            row_mask = row_indices[:, None] < T
            m_qk = row_mask & m_k[None, :]
            p_q = q + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_r = r + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            p_d1 = d1 + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_bart = bart + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_m = m + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_t = delta_t + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_b = delta_b + bos * HQ + i_hq + row_indices[:, None] * HQ
            p_grad_o = grad_o + (bos * HQ + i_hq) * K + row_indices[:, None] * (HQ * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_qk, other=0.0)
            b_r = tl.load(p_r, mask=m_qk, other=0.0)
            b_d1 = tl.load(p_d1, mask=row_mask, other=0.0)
            b_bart = tl.load(p_bart, mask=row_mask, other=0.0)
            b_m = tl.load(p_m, mask=row_mask, other=0.0)
            b_t = tl.load(p_t, mask=row_mask, other=0.0)
            b_b = tl.load(p_b, mask=row_mask, other=0.0)
            grad_o_tile = tl.load(p_grad_o, mask=m_qk, other=0.0)

            qk = tl.dot(b_q, tl.trans(b_k), out_dtype=tl.float32) * scale_log2
            rk = tl.dot(b_r, tl.trans(b_k), out_dtype=tl.float32)
            inv_d1 = tl.where(row_mask, 1.0 / b_d1, 0.0)
            mask = (
                (col_indices[None, :] >= row_indices[:, None] - WINDOW_SIZE_LEFT + 1)
                & row_mask
                & (col_indices[None, :] < T)
            )
            qk = tl.where(mask, qk, -float("inf"))
            p = exp2(qk - b_m) * inv_d1
            a = tl.dot(grad_o_tile, tl.trans(b_v), out_dtype=tl.float32)
            delta = a - b_b
            bart_minus_rk = b_bart - rk
            gl = p * (a - b_t + bart_minus_rk * delta) * scale
            gu = -p * delta
            grad_k_acc = tl.dot(tl.trans(gl).to(b_q.dtype), b_q, out_dtype=tl.float32, acc=grad_k_acc)
            grad_k_acc = tl.dot(tl.trans(gu).to(b_r.dtype), b_r, out_dtype=tl.float32, acc=grad_k_acc)
            weights = p * (1 + bart_minus_rk)
            grad_v_acc = tl.dot(
                tl.trans(weights).to(grad_o_tile.dtype), grad_o_tile, out_dtype=tl.float32, acc=grad_v_acc
            )

    tl.store(p_grad_k, grad_k_acc.to(p_grad_k.dtype.element_ty), mask=m_ck)
    tl.store(p_grad_v, grad_v_acc.to(p_grad_v.dtype.element_ty), mask=m_ck)


@triton.jit(do_not_specialize=["N"])
def parallel_parallax_bwd_kernel_reduce_gqa(
    grad_k_buf,
    grad_v_buf,
    grad_k,
    grad_v,
    N,
    HQ: tl.constexpr,
    H: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold per-query-head dK/dV into their shared GQA heads in one pass."""
    offsets = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    mask = offsets < N

    head_dim_offset = offsets % K
    row_head = offsets // K
    kv_head = row_head % H
    batch_row = row_head // H
    input_offsets = (batch_row * HQ + kv_head * G) * K + head_dim_offset

    grad_k_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    grad_v_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for i_g in range(0, G):
        group_offsets = input_offsets + i_g * K
        grad_k_acc += tl.load(grad_k_buf + group_offsets, mask=mask, other=0.0)
        grad_v_acc += tl.load(grad_v_buf + group_offsets, mask=mask, other=0.0)

    tl.store(grad_k + offsets, grad_k_acc, mask=mask)
    tl.store(grad_v + offsets, grad_v_acc, mask=mask)


def parallel_parallax_fwd(
    q, r, k, v, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1, return_aux=True
):
    """Parallax forward (Triton). `(B, T, HQ, D)` / packed `(1, T_total, HQ, D)` inputs.

    Returns `(o, barv, d1, bart, m)` by default.  For inference-only callers,
    ``return_aux=False`` returns just ``o`` and avoids allocating and writing
    the backward-only auxiliary tensors.
    """
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    use_fused_pv_rows = (
        cu_seqlens is None
        and q.dtype == torch.bfloat16
        and T >= 4096
        and K == 128
        and window_size_left < 0
    )
    use_gqa_cta_order = (
        cu_seqlens is None
        and q.dtype == torch.bfloat16
        and T >= 4096
        and G > 1
        and K in (64, 128)
        and window_size_left < 0
    )
    # L1-bypass K/V loads showed a small long-sequence benefit on the tested
    # dense GQA shapes. Keep the cache policy narrower than CTA ordering until
    # shorter sequences and other layouts have independent latency evidence.
    use_gqa_kv_cache_cg = use_gqa_cta_order and T >= 8192
    o = torch.empty_like(q)
    if return_aux:
        barv = torch.empty_like(q)
        stats = torch.empty((3, B, T, HQ), device=q.device, dtype=torch.float32)
        d1, bart, m = stats.unbind(0)
    else:
        # The auxiliary pointers are compile-time dead when STORE_AUX=False.
        # Reuse the output allocation as a valid pointer to keep Triton's
        # argument binder happy without allocating any auxiliary workspace.
        barv = d1 = bart = m = o

    if cu_seqlens is None:
        # BT only partitions query rows.  The long kernel autotunes BS
        # independently for the key/value columns it scans.
        NQ = triton.cdiv(T, BT)
        query_grid = (
            (NQ * G, B * H) if use_gqa_cta_order else (NQ, B * HQ)
        )
        if T <= BT:
            parallel_parallax_fwd_kernel_short[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                STORE_AUX=return_aux,
                num_warps=4,
                num_stages=1,
            )
        elif NQ <= 8:
            parallel_parallax_fwd_kernel_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                STORE_AUX=return_aux,
                num_warps=(4 if BT <= 64 else 8),
                num_stages=2,
            )
        else:
            parallel_parallax_fwd_kernel[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                STORE_AUX=return_aux,
                FUSE_PV_ROWS=use_fused_pv_rows,
                CTA_ORDER_GQA=use_gqa_cta_order,
                KV_CACHE_CG=use_gqa_kv_cache_cg,
            )
    else:
        # Packed chunks are created with BT, so the autotune pruner keeps
        # BS == BT for this path.  Dense inputs above remain fully decoupled.
        chunk_grid = (len(chunk_indices), B * HQ)
        parallel_parallax_fwd_kernel[chunk_grid](
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            scale,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            WINDOW_SIZE_LEFT=window_size_left,
            BT=BT,
            STORE_AUX=return_aux,
            FUSE_PV_ROWS=False,
            CTA_ORDER_GQA=False,
            KV_CACHE_CG=False,
        )
    if return_aux:
        return o, barv, d1, bart, m
    return o


def parallel_parallax_bwd(
    q, r, k, v, o, barv, d1, bart, m, grad_o, scale, cu_seqlens=None, chunk_indices=None, window_size_left=-1
):
    """Parallax backward (Triton). Returns grads matching `q, r, k, v`."""
    B, T, HQ, K = q.shape
    H = k.shape[2]
    G = HQ // H
    BK = triton.next_power_of_2(K)
    BT = _block_size(K, q.device.index)
    NQ = triton.cdiv(T, BT)
    dkv_programs = (len(chunk_indices) if cu_seqlens is not None else NQ) * B * H
    num_sms = get_sm_count()
    group_dkv_heads = G in (4, 8) or (G == 2 and (cu_seqlens is not None or T > 8192))
    use_grouped_dkv = (
        group_dkv_heads
        and (cu_seqlens is not None or NQ > 16)
        and num_sms > 0
        and dkv_programs >= num_sms
    )
    grad_q = torch.empty_like(q)
    grad_r = torch.empty_like(r)
    if use_grouped_dkv or G == 1:
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_k_buf, grad_v_buf = grad_k, grad_v
    else:
        # Short sequences retain one CTA per query head for occupancy, then
        # fold their private dK/dV tiles back to the shared KV-head axis.
        grad_k = grad_v = None
        grad_k_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
        grad_v_buf = torch.empty((B, T, HQ, K), device=q.device, dtype=q.dtype)
    if cu_seqlens is None:
        query_grid = (NQ, B * HQ)
        use_gqa_dqr_cta_order = (
            q.dtype == torch.bfloat16
            and T >= 4096
            and NQ > 16
            and G in (2, 4)
            and K in (64, 128)
            and window_size_left < 0
        )
        use_gqa_dkv_cta_order = (
            q.dtype == torch.bfloat16
            and T >= 4096
            and NQ > 16
            and G == 2
            and K == 128
            and window_size_left < 0
            and not use_grouped_dkv
        )
        dqr_grid = (NQ * G, B * H) if use_gqa_dqr_cta_order else query_grid
        if T <= BT:
            parallel_parallax_bwd_kernel_short[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_q,
                grad_r,
                grad_k_buf,
                grad_v_buf,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                num_warps=4,
                num_stages=1,
            )
        elif NQ <= 16:
            parallel_parallax_bwd_kernel_dqr_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_q,
                grad_r,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                num_warps=(4 if BT <= 64 else 8),
                num_stages=2,
            )
            parallel_parallax_bwd_kernel_dkv_short_multi[query_grid](
                q,
                r,
                k,
                v,
                o,
                barv,
                d1,
                bart,
                m,
                grad_o,
                grad_k_buf,
                grad_v_buf,
                scale,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                BK=BK,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                NT=NQ,
                num_warps=(4 if BT <= 64 else 8),
                num_stages=2,
            )
        else:
            deltas = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32)
            delta_t, delta_b = deltas.unbind(0)

            # dQ/dR owns BT query rows, whereas dK/dV owns BS key/value
            # columns.  Its grid must therefore be derived from tuned BS.
            def dkv_grid(meta):
                return triton.cdiv(T, meta["BS"]), B * HQ

            def grouped_dkv_grid(meta):
                return triton.cdiv(T, meta["BS"]), B * H

            def gqa_dkv_cta_order_grid(meta):
                return G, B * H * triton.cdiv(T, meta["BS"])

            # Small heads can keep the two scalar reductions in the DQR
            # program without excessive register pressure.  Larger heads
            # retain the separately autotuned preprocess pass.
            use_split_dqr = K >= 128
            if use_split_dqr:
                def preprocess_grid(meta):
                    return triton.cdiv(T, meta["BP"]), B * HQ

                parallel_parallax_bwd_kernel_preprocess[preprocess_grid](
                    grad_o, o, barv, delta_t, delta_b,
                    cu_seqlens, chunk_indices, T,
                    HQ=HQ, K=K, BK=BK, BT=BT,
                )
                parallel_parallax_bwd_kernel_dqr[dqr_grid](
                    q, r, k, v, d1, bart, m, delta_t, delta_b,
                    grad_o, grad_q, grad_r, scale,
                    cu_seqlens, chunk_indices, T,
                    HQ=HQ, H=H, G=G, K=K,
                    WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    CTA_ORDER_GQA=use_gqa_dqr_cta_order,
                )
            else:
                parallel_parallax_bwd_kernel_dqr_fused[dqr_grid](
                    q, r, k, v, o, barv, d1, bart, m,
                    delta_t, delta_b, grad_o, grad_q, grad_r, scale,
                    cu_seqlens, chunk_indices, T,
                    HQ=HQ, H=H, G=G, K=K,
                    WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    CTA_ORDER_GQA=use_gqa_dqr_cta_order,
                )
            if use_grouped_dkv:
                parallel_parallax_bwd_kernel_dkv_grouped[grouped_dkv_grid](
                    q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o,
                    grad_k, grad_v, scale, cu_seqlens, chunk_indices, T,
                    HQ=HQ, H=H, G=G, K=K,
                    WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                )
            else:
                dkv_launch_grid = (
                    gqa_dkv_cta_order_grid if use_gqa_dkv_cta_order else dkv_grid
                )
                parallel_parallax_bwd_kernel_dkv[dkv_launch_grid](
                    q, r, k, v, d1, bart, m, delta_t, delta_b, grad_o,
                    grad_k_buf, grad_v_buf, scale, cu_seqlens, chunk_indices, T,
                    HQ=HQ, H=H, G=G, K=K,
                    WINDOW_SIZE_LEFT=window_size_left, BT=BT,
                    CTA_ORDER_GQA=use_gqa_dkv_cta_order,
                )

    else:
        # chunk_indices is indexed in BT-sized units; the autotune pruners
        # intentionally pin BP and BS to BT for packed variable lengths.
        chunk_grid = (len(chunk_indices), B * HQ)
        grouped_chunk_grid = (len(chunk_indices), B * H)
        deltas = torch.empty((2, B, T, HQ), device=q.device, dtype=torch.float32)
        delta_t, delta_b = deltas.unbind(0)
        parallel_parallax_bwd_kernel_preprocess[chunk_grid](
            grad_o,
            o,
            barv,
            delta_t,
            delta_b,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            K=K,
            BK=BK,
            BT=BT,
        )
        parallel_parallax_bwd_kernel_dqr[chunk_grid](
            q,
            r,
            k,
            v,
            d1,
            bart,
            m,
            delta_t,
            delta_b,
            grad_o,
            grad_q,
            grad_r,
            scale,
            cu_seqlens,
            chunk_indices,
            T,
            HQ=HQ,
            H=H,
            G=G,
            K=K,
            WINDOW_SIZE_LEFT=window_size_left,
            BT=BT,
            CTA_ORDER_GQA=False,
        )
        if use_grouped_dkv:
            parallel_parallax_bwd_kernel_dkv_grouped[grouped_chunk_grid](
                q,
                r,
                k,
                v,
                d1,
                bart,
                m,
                delta_t,
                delta_b,
                grad_o,
                grad_k,
                grad_v,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
            )
        else:
            parallel_parallax_bwd_kernel_dkv[chunk_grid](
                q,
                r,
                k,
                v,
                d1,
                bart,
                m,
                delta_t,
                delta_b,
                grad_o,
                grad_k_buf,
                grad_v_buf,
                scale,
                cu_seqlens,
                chunk_indices,
                T,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
                WINDOW_SIZE_LEFT=window_size_left,
                BT=BT,
                CTA_ORDER_GQA=False,
            )
    if G > 1 and not use_grouped_dkv:
        if grad_k is None:
            grad_k = torch.empty_like(k)
            grad_v = torch.empty_like(v)
        num_grad_k_elements = grad_k.numel()
        reduction_block = 256
        reduction_grid = (triton.cdiv(num_grad_k_elements, reduction_block),)
        parallel_parallax_bwd_kernel_reduce_gqa[reduction_grid](
                grad_k_buf,
                grad_v_buf,
                grad_k,
                grad_v,
                num_grad_k_elements,
                HQ=HQ,
                H=H,
                G=G,
                K=K,
            BLOCK=reduction_block,
            num_warps=4,
        )
    return grad_q, grad_r, grad_k, grad_v


def _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens):
    tensors = {"q": q, "r": r, "k": k, "v": v}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [B, T, H, D], got {tuple(tensor.shape)}")
        if tensor.device.type != "cuda":
            raise ValueError(f"{name} must be a CUDA tensor, got device={tensor.device}")

    if r.shape != q.shape:
        raise ValueError(f"r shape {tuple(r.shape)} must match q shape {tuple(q.shape)}")
    if v.shape != k.shape:
        raise ValueError(f"v shape {tuple(v.shape)} must match k shape {tuple(k.shape)}")
    if any(tensor.device != q.device for tensor in (r, k, v)):
        raise ValueError("q, r, k, and v must be on the same CUDA device")
    if any(tensor.dtype != q.dtype for tensor in (r, k, v)):
        raise TypeError("q, r, k, and v must have the same dtype")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"parallel_parallax requires bf16 or fp16 inputs, got q.dtype={q.dtype}")

    B, T, HQ, K = q.shape
    if k.shape[0] != B or k.shape[1] != T or k.shape[3] != K:
        raise ValueError(
            f"k/v shape {tuple(k.shape)} is incompatible with q/r shape {tuple(q.shape)}; B, T, and D must match"
        )
    H = k.shape[2]
    if T <= 0 or H <= 0 or HQ <= 0:
        raise ValueError("sequence length and query/KV head counts must be positive")
    if HQ % H != 0:
        raise ValueError(f"GQA requires HQ % H == 0, got HQ={HQ}, H={H}")
    if not 16 <= K <= 256:
        raise ValueError(f"head dimension must be in [16, 256], got D={K}")
    if window_size is not None and (not isinstance(window_size, int) or window_size <= 0):
        raise ValueError(f"window_size must be a positive integer or None, got {window_size!r}")

    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor or None")
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using cu_seqlens. "
                "Flatten variable-length inputs before processing."
            )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be one-dimensional with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same CUDA device as q")


class ParallaxFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, q, r, k, v, scale, window_size_left, cu_seqlens):
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, _block_size(q.shape[-1], q.device.index))
            if cu_seqlens is not None
            else None
        )
        needs_backward = any(ctx.needs_input_grad[:4])
        if not needs_backward:
            return parallel_parallax_fwd(
                q,
                r,
                k,
                v,
                scale,
                cu_seqlens,
                chunk_indices,
                window_size_left,
                return_aux=False,
            )

        o, barv, d1, bart, m = parallel_parallax_fwd(
            q, r, k, v, scale, cu_seqlens, chunk_indices, window_size_left
        )
        ctx.save_for_backward(q, r, k, v, o, barv, d1, bart, m)
        ctx.scale = scale
        ctx.window_size_left = window_size_left
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        return o

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, r, k, v, o, barv, d1, bart, m = ctx.saved_tensors
        gq, gr, gk, gv = parallel_parallax_bwd(
            q,
            r,
            k,
            v,
            o,
            barv,
            d1,
            bart,
            m,
            do,
            ctx.scale,
            ctx.cu_seqlens,
            ctx.chunk_indices,
            ctx.window_size_left,
        )
        return gq.to(q), gr.to(r), gk.to(k), gv.to(v), None, None, None


def parallel_parallax(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Causal Parallax (parameterized local linear attention) with autograd,
    backed by Triton kernels. See `fla.ops.parallax.naive.naive_parallax` for
    the reference math.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, HQ, D]`.
        r (torch.Tensor):
            secondary queries of shape `[B, T, HQ, D]` (same shape as `q`). NOTE:
            `r` is *not* scaled by `scale`; pass it un-pre-scaled.
        k (torch.Tensor):
            keys of shape `[B, T, H, D]`. GQA is applied when `HQ` is divisible by `H`.
        v (torch.Tensor):
            values of shape `[B, T, H, D]`.
        scale (float, Optional):
            Scale applied to the `q @ k^T` logits only. If `None`, defaults to `1 / sqrt(D)`.
            Default: `None`.
        window_size (int, Optional):
            Sliding-window length. If provided, each query at position `i` only attends to
            keys in `[i - window_size + 1, i]`. If `None`, full causal attention is used.
            The value `-1` is also accepted for compatibility with FLA's low-level API
            and is treated the same as `None`.
            Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length training
            (FlashAttention convention). The batch size must be 1 when packing. Default: `None`.

    Returns:
        o (torch.Tensor):
            output of shape `[B, T, HQ, D]`.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"parallel_parallax got unexpected keyword argument(s): {unexpected}")
    # FLA's low-level kernels use -1 as the unbounded-window sentinel, while
    # this public API uses None.  Accept both conventions at the boundary.
    if isinstance(window_size, int) and window_size == -1:
        window_size = None
    _validate_parallel_parallax_inputs(q, r, k, v, window_size, cu_seqlens)
    if scale is None:
        scale = k.shape[-1] ** -0.5
    # The kernel keeps cols [i - W + 1, i] (W keys total, diagonal included),
    # matching FLA's `window_size=W` semantics exactly (no off-by-one).
    window_size_left = -1 if window_size is None else window_size
    return ParallaxFunction.apply(q, r, k, v, float(scale), window_size_left, cu_seqlens)
