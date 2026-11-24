# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Upstream: https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/wall_attn
# The general implementation is preserved from the accepted 13d0e0f snapshot.
# Wall attention was contributed upstream by Tilde Research (Timor Averbuch, Dhruv Pai).
"""Hygon Wall Attention forward kernels, gate scans, and compiled launchers."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from flagtree import _flagprism
from triton.runtime import _async_compile
from triton.runtime.jit import DistributedRtContext

from flag_attn.FLA.cumsum import chunk_global_cumsum_vector_kernel

from .parallel import RCP_LN2, exp2, log2, parallel_wall_attn_fwd_kernel


@triton.jit
def gate_block_sums(Gate, Sums, T, NC: tl.constexpr):
    block, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = block * 256 + tl.arange(0, 256)
    d = tile * 16 + tl.arange(0, 16)
    offsets = (t[:, None] * 8 + head) * 64 + d[None, :]
    g = tl.load(Gate + offsets, t[:, None] < T, other=0).to(tl.float32)
    tl.store(Sums + (head * NC + block) * 64 + d, tl.sum(g, 0))


@triton.jit
def gate_prefix(Gate, Sums, Prefix, T, NC: tl.constexpr, NP: tl.constexpr,
                SCALE: tl.constexpr):
    block, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = block * 256 + tl.arange(0, 256)
    d = tile * 32 + tl.arange(0, 32)
    earlier = tl.arange(0, NP)
    block_sums = tl.load(
        Sums + (head * NC + earlier[:, None]) * 64 + d[None, :],
        earlier[:, None] < block, other=0,
    )
    carry = tl.sum(block_sums, 0)
    offsets = (t[:, None] * 8 + head) * 64 + d[None, :]
    g = tl.load(Gate + offsets, t[:, None] < T, other=0).to(tl.float32)
    p = (tl.cumsum(g, 0) + carry[None, :]) * SCALE
    tl.store(Prefix + offsets, p, t[:, None] < T)


@lru_cache(maxsize=128)
def _scan_launchers(device, t, dtype):
    if triton.runtime.driver.active.get_current_target().arch != "gfx936":
        return None
    nc = triton.cdiv(t, 256)
    g = torch.empty((1, t, 8, 64), device=device, dtype=dtype)
    sums = torch.empty((8, nc, 64), device=device, dtype=torch.float32)
    prefix = torch.empty_like(g, dtype=torch.float32)
    grid_sums, grid_prefix = (nc, 8, 4), (nc, 8, 2)
    args_prefix = (t, nc, triton.next_power_of_2(nc), RCP_LN2)
    gather = gate_block_sums.run(g, sums, t, nc, grid=grid_sums, warmup=True, num_warps=4)
    scan = gate_prefix.run(
        g, sums, prefix, *args_prefix, grid=grid_prefix, warmup=True, num_warps=4, num_stages=1,
    )
    return gather[grid_sums], scan[grid_prefix]


def _can_reuse_launchers(g):
    if (
        triton.__version__ != "3.6.0"
        or type(g) is not torch.Tensor
        or not g.is_cuda
        or torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.is_autocast_enabled()
    ):
        return False
    try:
        from flagtree import _flagprism
        from triton.runtime import _async_compile
        from triton.runtime.jit import DistributedRtContext
    except ImportError:
        return False

    runtime = triton.knobs.runtime
    return not (
        runtime.interpret
        or runtime.debug
        or runtime.override_arch
        or runtime.jit_cache_hook
        or runtime.jit_post_compile_hook
        or triton.knobs.compilation.instrumentation_mode
        or gate_block_sums.pre_run_hooks
        or gate_prefix.pre_run_hooks
        or _async_compile.active_mode.get() is not None
        or _flagprism._registered_components()
        or DistributedRtContext().is_lite_mode
        or torch.cuda.is_current_stream_capturing()
        or g.is_neg()
        or g.data_ptr() % 16
        or g.untyped_storage().nbytes() > 2**31 - 1
        or g.device.index != torch.cuda.current_device()
    )


def build_gate_prefix(g):
    t = g.shape[1]
    nc = triton.cdiv(t, 256)
    sums = torch.empty((8, nc, 64), device=g.device, dtype=torch.float32)
    prefix = torch.empty_like(g, dtype=torch.float32)
    args_prefix = (t, nc, triton.next_power_of_2(nc), RCP_LN2)
    launchers = _scan_launchers(g.device, t, g.dtype) if _can_reuse_launchers(g) else None
    if launchers is not None:
        gather, scan = launchers
        stream = torch.cuda.current_stream(g.device).cuda_stream
        gather(g, sums, t, nc, stream=stream)
        scan(g, sums, prefix, *args_prefix, stream=stream)
    else:
        gate_block_sums[(nc, 8, 4)](g, sums, t, nc, num_warps=4)
        gate_prefix[(nc, 8, 2)](g, sums, prefix, *args_prefix, num_warps=4, num_stages=1)
    return prefix


@triton.jit
def prepare_history(k, prefix, cached_k, cached_valid, T, HQ: tl.constexpr, H: tl.constexpr,
                    D: tl.constexpr, C: tl.constexpr, CHUNK: tl.constexpr):
    block, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // HQ, bh % HQ
    t = block * CHUNK + tl.arange(0, CHUNK)
    d = tl.arange(0, D)
    end = tl.minimum((t // C + 1) * C, T) - 1
    pref = (batch * T * HQ + head) * D
    p = tl.load(prefix + pref + t[:, None] * HQ * D + d[None, :],
                t[:, None] < T, other=0)
    r = tl.load(prefix + pref + end[:, None] * HQ * D + d[None, :])
    idx = ((batch * T + t[:, None]) * H + head // (HQ // H)) * D + d[None, :]
    x = tl.load(k + idx, t[:, None] < T, other=0).to(tl.float32)
    good = ((tl.abs(x) <= 8.) & (tl.abs(p) < float('inf'))
            & (tl.abs(r) < float('inf')) & (r <= p)) | (t[:, None] >= T)
    tl.store(cached_valid + bh * tl.cdiv(T, C) + block,
             tl.min(tl.min(good.to(tl.int32), 0), 0))
    xt = x * tl.exp2(r - p)
    tl.store(cached_k + (bh * T + t[:, None]) * D + d[None, :],
             xt, t[:, None] < T)


@triton.heuristics({
    'USE_SINK_BIAS': lambda args: args['sink_bias'] is not None,
    'USE_WINDOW': lambda args: args['W'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_SCALAR_G': lambda args: args['g_scalar_cumsum'] is not None,
})
@triton.jit
def history_fwd_kernel(
    q,
    cached_k,
    cached_valid,
    k,
    v,
    o,
    g_cumsum,
    g_scalar_cumsum,
    sink_bias,
    lse,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    C: tl.constexpr,
    W: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
    HISTORY_BS: tl.constexpr = 64,
    HEAVY_FIRST: tl.constexpr = False,
):
    if HEAVY_FIRST:
        # Give long-history query tiles the earliest workgroup IDs, with
        # all heads for a query tile adjacent in the flattened grid.
        pid = tl.program_id(0)
        i_v, i_t, i_bh = 0, tl.cdiv(T, BT) - 1 - pid // (B * HQ), pid % (B * HQ)
    else:
        i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    logical_start = i_t * BT // 128 * 128
    p_q = tl.make_block_ptr(q + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * HQ + i_hq) * V, (T, V), (HQ*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    p_pq = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_t * BT, 0), (BT, BK), (1, 0))
    p_R = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (logical_start, 0), (1, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_pq = tl.load(p_pq, boundary_check=(0, 1)).to(tl.float32)
    b_R = tl.load(p_R, boundary_check=(0, 1)).to(tl.float32)
    # b_R is at i_t*BT; the same value is used in both off-diag and diag loops,
    # since |b_pq - b_R| within a BT-chunk is bounded by BT*|g_max|*RCP_LN2.
    # Off-diagonal: P_q - R <= 0 and R - P_k <= 0 -> exp2 <= 1 -> bf16 safe.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    # Keep PV in [value, query] layout until the output store.
    b_o = tl.zeros([BV, BT], dtype=tl.float32)
    b_m = tl.full([BT], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
        b_cq = tl.load(p_cq, boundary_check=(0,)).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    o_q = i_t * BT + tl.arange(0, BT)

    b_R_t = tl.trans(b_R)


    # Fully historical chunks only. Their tail is no later than any query row.
    active_q = o_q < T
    p_max = tl.max(tl.where(active_q[:, None], b_pq, float('-inf')), 0)
    q_valid = (~active_q[:, None]) | ((tl.abs(b_q.to(tl.float32)) <= 8.)
                 & (b_pq <= b_R) & (tl.abs(b_pq) < float('inf')))
    q_safe = tl.min(tl.min(q_valid.to(tl.int32), 0), 0) != 0
    hist_end = logical_start // C * C
    for chunk_start in range(0, hist_end, C):
        p_anchor = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1),
            (chunk_start + C - 1, 0), (1, BK), (1, 0))
        anchor = tl.load(p_anchor, boundary_check=(0, 1))
        good_k = tl.load(cached_valid + i_bh * tl.cdiv(T, C) + chunk_start // C)
        use_cached = q_safe & (good_k != 0) & (
            tl.min((p_max <= tl.reshape(anchor, (BK,))).to(tl.int32), 0) != 0)
        q_chunk = (b_q.to(tl.float32) * exp2(b_pq - anchor)).to(b_q.dtype)
        # Fewer online softmax updates for full history. The tail and diagonal keep BS.
        for offset in range(0, C, HISTORY_BS):
            i_s = chunk_start + offset
            p_kc = tl.make_block_ptr(cached_k + (i_bh * T) * K,
                (K, T), (1, K), (0, i_s), (BK, HISTORY_BS), (0, 1))
            p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V),
                (H*V, 1), (i_s, i_v * BV), (HISTORY_BS, BV), (1, 0))
            if use_cached:
                b_kc = tl.load(p_kc, boundary_check=(0, 1))
                q_dot = q_chunk
            else:
                p_kr = tl.make_block_ptr(k + (bos * H + i_h) * K,
                    (K, T), (1, H*K), (0, i_s), (BK, HISTORY_BS), (0, 1))
                p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K,
                    (K, T), (1, HQ*K), (0, i_s), (BK, HISTORY_BS), (0, 1))
                kr = tl.load(p_kr, boundary_check=(0, 1))
                pk = tl.load(p_pk, boundary_check=(0, 1))
                b_kc = (kr.to(tl.float32) * exp2(b_R_t - pk)).to(kr.dtype)
                q_dot = b_q_til
            b_v = tl.load(p_v, boundary_check=(0, 1))
            b_s = tl.dot(q_dot, b_kc) * scale * RCP_LN2
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_mw = tl.where(b_m == float('-inf'), 0., b_m)
            b_r = exp2(b_mp - b_mw)
            b_p = exp2(b_s - b_mw[:, None])
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            b_o = tl.dot(
                tl.trans(b_v), tl.trans(b_p.to(b_v.dtype)), b_o * b_r[None, :])
            b_mp = b_m

    for i_s in range(hist_end, logical_start, BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = tl.dot(
                tl.trans(b_v), tl.trans(b_p.to(b_v.dtype)), b_o * b_r[None, :])
        b_mp = b_m

    for i_s in range(logical_start, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        p_pk = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ*K), (0, i_s), (BK, BS), (0, 1))

        # Per-sub-block local reference to prevent exp2 overflow.
        p_R_local = tl.make_block_ptr(g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ*K, 1), (i_s, 0), (1, BK), (1, 0))
        b_R_local = tl.load(p_R_local, boundary_check=(0, 1)).to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_pk = tl.load(p_pk, boundary_check=(0, 1)).to(tl.float32)

        # k_til in local frame: exp2 bounded by BS positions <= 110 (fp32-safe with BK dot).
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame; clamp exp for Q before sub-block (masked by causality).
        b_exp_q = tl.minimum(b_pq - b_R_local_bc, tl.zeros([BT, BK], dtype=tl.float32))
        b_q_til_local = b_q.to(tl.float32) * exp2(b_exp_q)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float('-inf'))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float('-inf'), 0., b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = tl.dot(
                tl.trans(b_v), tl.trans(b_p.to(b_v.dtype)), b_o * b_r[None, :])
        b_mp = b_m

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float('-inf'), 0., b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = tl.trans(b_o / b_acc[None, :])
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))




def parallel_wall_attn_fwd(
    q, k, v, g_cumsum, sink_bias, scale,
    g_scalar_cumsum=None, window_size=None, cu_seqlens=None, chunk_indices=None,
):
    # The caller restricts this kernel to the validated dense inference route.
    b, t, h, d = k.shape
    hq = q.shape[2]
    chunk, bt = 256, 64
    n_chunks = triton.cdiv(t, chunk)
    cached_k = torch.empty((b * hq, t, d), dtype=k.dtype, device=k.device)
    cached_valid = torch.empty((b * hq, n_chunks), dtype=torch.int32, device=k.device)
    o = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    prepare_history[(n_chunks, b * hq)](
        k, g_cumsum, cached_k, cached_valid, t, hq, h, d, chunk, chunk, num_warps=4,
    )
    history_fwd_kernel[(triton.cdiv(t, bt) * b * hq,)](
        q=q, cached_k=cached_k, cached_valid=cached_valid, k=k, v=v, o=o,
        g_cumsum=g_cumsum, g_scalar_cumsum=None, sink_bias=None, lse=lse,
        scale=scale, cu_seqlens=None, chunk_indices=None, T=t, C=chunk, W=None,
        B=b, H=h, HQ=hq, G=hq // h, K=d, V=d, BT=bt, BS=32,
        BK=d, BV=d, HISTORY_BS=64, HEAVY_FIRST=True, num_warps=4, num_stages=1, waves_per_eu=1,
    )
    return o, lse


def _dense_buffers(q, k):
    t = q.shape[1]
    n_chunks = t // 256
    sums = torch.empty((8, n_chunks, 64), device=q.device, dtype=torch.float32)
    prefix = torch.empty_like(q, dtype=torch.float32)
    cached_k = torch.empty((8, t, 64), device=q.device, dtype=k.dtype)
    cached_valid = torch.empty((8, n_chunks), device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    return sums, prefix, cached_k, cached_valid, out, lse


def _dense_history_args(q, k, v, prefix, cached_k, cached_valid, out, lse):
    h = k.shape[2]
    return (q, cached_k, cached_valid, k, v, out, prefix, None, None, lse,
            0.125, None, None, q.shape[1], 256, None, 1, h, 8, 8 // h, 64, 64,
            64, 32, 64, 64)


@lru_cache(maxsize=32)
def _dense_launchers(device, t, h, dtype):
    q = torch.empty((1, t, 8, 64), device=device, dtype=dtype)
    k = torch.empty((1, t, h, 64), device=device, dtype=dtype)
    _, prefix, cached_k, cached_valid, out, lse = _dense_buffers(q, k)
    prepare_grid = (t // 256, 8, 1)
    history_grid = (t // 64 * 8, 1, 1)
    prepare = prepare_history.run(
        k, prefix, cached_k, cached_valid, t, 8, h, 64, 256, 256,
        grid=prepare_grid, warmup=True, num_warps=4,
    )
    history = history_fwd_kernel.run(
        *_dense_history_args(q, k, k, prefix, cached_k, cached_valid, out, lse),
        grid=history_grid, warmup=True, HISTORY_BS=64, HEAVY_FIRST=True,
        num_warps=4, num_stages=1, waves_per_eu=1,
    )
    return prepare[prepare_grid], history[history_grid]


def _launch_dense_history(q, k, v, prefix, cached_k, cached_valid, out, lse, stream):
    _, t, h, _ = k.shape
    prepare, history = _dense_launchers(q.device, t, h, q.dtype)
    args = _dense_history_args(q, k, v, prefix, cached_k, cached_valid, out, lse)
    prepare(k, prefix, cached_k, cached_valid, t, 8, h, 64, 256, 256, stream=stream)
    history(*args, False, False, False, False, 64, True, stream=stream)


def try_dense_wall_fwd(q, k, v, g, scale):
    if type(q) is not torch.Tensor or q.ndim != 4:
        return None
    t = q.shape[1]
    if t not in (2048, 4096):
        return None
    if type(k) is not torch.Tensor or k.ndim != 4 or k.shape[2] not in (2, 8):
        return None
    if q.dtype not in (torch.bfloat16, torch.float16):
        return None
    # Compiled launchers require an exact built-in float for this specialization.
    if scale is not None and (type(scale) is not float or scale != 0.125):  # noqa: E721
        return None

    device, dtype, h = q.device, q.dtype, k.shape[2]
    grad_enabled = torch.is_grad_enabled()
    for tensor, heads in ((q, 8), (k, h), (v, h), (g, 8)):
        if (
            type(tensor) is not torch.Tensor
            or tensor.shape != (1, t, heads, 64)
            or tensor.device != device
            or tensor.dtype != dtype
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or (grad_enabled and tensor.requires_grad)
            or tensor.is_neg()
            or tensor.data_ptr() % 16
            or tensor.untyped_storage().nbytes() > 2**31 - 1
        ):
            return None
    if (not _can_reuse_launchers(g)
            or prepare_history.pre_run_hooks or history_fwd_kernel.fn.pre_run_hooks):
        return None
    scan_launchers = _scan_launchers(device, t, dtype)
    if scan_launchers is None:
        return None

    gather, scan = scan_launchers
    sums, prefix, cached_k, cached_valid, out, lse = _dense_buffers(q, k)
    stream = torch.cuda.current_stream(device).cuda_stream
    n_chunks = t // 256
    gather(g, sums, t, n_chunks, stream=stream)
    scan(g, sums, prefix, t, n_chunks, n_chunks, RCP_LN2, stream=stream)
    _launch_dense_history(q, k, v, prefix, cached_k, cached_valid, out, lse, stream)
    return out, lse


_scan = chunk_global_cumsum_vector_kernel.fn.fn
_attention = parallel_wall_attn_fwd_kernel.fn.fn
_runtime = triton.knobs.runtime
_compilation = triton.knobs.compilation


@triton.jit
def short_gate_scan(Gate, Prefix, T: tl.constexpr, BD: tl.constexpr, SCALE: tl.constexpr):
    d = tl.program_id(0) * BD + tl.arange(0, BD)
    head = tl.program_id(1)
    t = tl.arange(0, T)
    offset = (t[:, None] * 8 + head) * 64 + d[None, :]
    gate = tl.load(Gate + offset).to(tl.float32)
    tl.store(Prefix + offset, tl.cumsum(gate, 0) * SCALE)


@lru_cache(maxsize=128)
def _launchers(device, t, h, dtype):
    if triton.runtime.driver.active.get_current_target().arch != "gfx936":
        return None

    q = torch.empty((1, t, 8, 64), device=device, dtype=dtype)
    k = torch.empty((1, t, h, 64), device=device, dtype=dtype)
    p = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty(q.shape[:-1], device=device, dtype=torch.float32)
    scan_grid = (2, 8, 1)
    # Positional suffixes match the JIT signatures. Only tensor arguments
    # before these suffixes are supplied again by each invocation.
    scan_args = (RCP_LN2, None, t, 1, 8, 64, 128, 32, False, True, False, False)
    if t in (512, 1024):
        bd = 8 if t == 512 else 4
        scan_grid, scan_args = (64 // bd, 8, 1), (t, bd, RCP_LN2)
        scan = short_gate_scan.run(
            q, p, *scan_args, grid=scan_grid, warmup=True, num_warps=8, num_stages=1,
        )
    else:
        scan = _scan.run(
            q, p, *scan_args, grid=scan_grid, warmup=True,
            num_warps=8, num_stages=3 if dtype == torch.bfloat16 else 2,
        )

    # The short grid favors more query tiles at 512 tokens and below.
    bt, warps, stages = (64, 4, 3) if t <= 512 else (128, 8, 2)
    if t == 1024:
        bt, warps, stages = (64, 4, 2)
    heavy_first = t == 1024
    query_tiles = triton.cdiv(t, bt)
    attn_grid = (query_tiles * 8, 1, 1) if heavy_first else (1, query_tiles, 8)
    attn_args = (
        0.125, None, None, t, None, 1, h, 8, 8 // h, 64, 64,
        bt, 32, 64, 64, False, False, False, False, heavy_first,
    )
    attention = _attention.run(
        q, k, k, q, p, None, None, lse, *attn_args, grid=attn_grid, warmup=True,
        num_warps=warps, num_stages=stages,
    )
    return scan[scan_grid], scan_args, attention[attn_grid], attn_args


def try_short_wall_fwd(q, k, v, g, scale):
    if type(q) is not torch.Tensor or q.ndim != 4 or not q.is_cuda:
        return None
    b, t, hq, d = q.shape
    if b != 1 or hq != 8 or d != 64 or not 256 <= t <= 1024:
        return None
    if torch.compiler.is_compiling() or torch._C._are_functorch_transforms_active():
        return None
    if q.dtype not in (torch.bfloat16, torch.float16) or q.device.index != torch.cuda.current_device():
        return None
    # Compiled launchers require an exact built-in float for this specialization.
    if scale is not None and (type(scale) is not float or scale != 0.125):  # noqa: E721
        return None
    if type(k) is not torch.Tensor or k.ndim != 4 or k.shape[2] not in (2, 8):
        return None
    h = k.shape[2]
    if k.shape != (1, t, h, 64):
        return None
    grad_enabled = torch.is_grad_enabled()
    device, dtype = q.device, q.dtype
    # These checks match the alignment and 32-bit range assumptions of the
    # Triton specializations compiled with fresh PyTorch tensors.
    for tensor, shape in ((q, q.shape), (k, k.shape), (v, k.shape), (g, q.shape)):
        if (
            type(tensor) is not torch.Tensor
            or tensor.shape != shape
            or tensor.device != device
            or tensor.dtype != dtype
            or tensor.layout != torch.strided
            or not tensor.is_contiguous()
            or tensor.is_neg()
            or (grad_enabled and tensor.requires_grad)
            or tensor.data_ptr() % 16
            or tensor.untyped_storage().nbytes() > 2**31 - 1
        ):
            return None
    if (
        torch.is_autocast_enabled()
        or _runtime.interpret
        or _runtime.debug
        or _runtime.override_arch
        or _runtime.jit_cache_hook
        or _runtime.jit_post_compile_hook
        or _compilation.instrumentation_mode
        or _scan.pre_run_hooks
        or short_gate_scan.pre_run_hooks
        or _attention.pre_run_hooks
        or _async_compile.active_mode.get() is not None
        or _flagprism._registered_components()
        or DistributedRtContext().is_lite_mode
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    launchers = _launchers(device, t, h, dtype)
    if launchers is None:
        return None

    scan, scan_args, attention, attn_args = launchers
    p = torch.empty_like(g, dtype=torch.float32)
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream(device).cuda_stream
    scan(g, p, *scan_args, stream=stream)
    attention(q, k, v, out, p, None, None, lse, *attn_args, stream=stream)
    return out
