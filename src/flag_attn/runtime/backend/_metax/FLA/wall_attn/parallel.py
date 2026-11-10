# SPDX-License-Identifier: MIT AND Apache-2.0
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright 2026 FlagOS Contributors
#
# Adapted helpers: Apache-2.0; see repository LICENSE.
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

import os
from contextvars import ContextVar
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Real

import torch
import triton
import triton.language as tl

from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit
def _copy_scale(
    x,
    y,
    N: tl.constexpr,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    SCALE: tl.constexpr,
    ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if ZERO:
        value = tl.full((BLOCK,), 0, tl.float32)
    else:
        rem = i
        offset = tl.full((BLOCK,), 0, tl.int64)
        for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
            offset += (rem % SHAPE[axis]) * STRIDES[axis]
            rem = rem // SHAPE[axis]
        value = tl.load(x + offset, i < N, 0)
        if SCALE != 1.0:
            value = value.to(tl.float32) * SCALE
    tl.store(y + i, value, i < N)


def copy_tensor(x, dtype=None, scale=1.0):
    dtype = x.dtype if dtype is None else dtype
    if x.is_contiguous() and x.dtype == dtype and scale == 1.0:
        return x
    result = torch.empty(x.shape, device=x.device, dtype=dtype)
    if x.numel():
        _copy_scale[(triton.cdiv(x.numel(), 256),)](
            x,
            result,
            x.numel(),
            tuple(x.shape),
            tuple(x.stride()),
            scale,
            False,
            256,
        )
    return result


def zero_flag(device):
    result = torch.empty((), device=device, dtype=torch.int32)
    _copy_scale[(1,)](result, result, 1, (), (), 1.0, True, 1, num_warps=1)
    return result


def input_guard(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        tensor = next((v for v in (*args, *kwargs.values()) if isinstance(v, torch.Tensor)), None)
        with torch.cuda.device(
            tensor.device if tensor is not None else torch.cuda.current_device()
        ):
            args = tuple(copy_tensor(v) if isinstance(v, torch.Tensor) else v for v in args)
            kwargs = {
                k: copy_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
            }
            return fn(*args, **kwargs)

    return wrapped


def resource_pruner(kernel_getter):
    """Reject compiled configurations exceeding C550 launch resource limits."""

    def prune(configs, named_args, **kwargs):
        from triton.runtime.errors import OutOfResources

        valid = []
        arguments = {**named_args, **kwargs}
        arguments.pop("grid", None)
        arguments.pop("warmup", None)
        for config in configs:
            try:
                compiled = kernel_getter().fn.run(
                    **{**arguments, **config.all_kwargs()},
                    grid=(1, 1, 1),
                    warmup=True,
                )
                # Loading metadata does not execute the kernel. The MACA driver
                # returns LOCAL_SIZE_BYTES / 4 as n_spills; C550 permits 4 KiB
                # private storage per thread in the supported runtime.
                compiled._init_handles()
                if compiled.n_spills * 4 <= 4096:
                    valid.append(config)
            except OutOfResources:
                continue
        return valid

    return prune


@triton.jit
def _add(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x + i, tl.load(x + i, i < N, 0) + tl.load(y + i, i < N, 0), i < N)


def add_inplace(x, y):
    _add[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)


@triton.jit
def _group_reduce(x, y, N: tl.constexpr, D: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = (i // D) * G * D + i % D
    value = tl.full((BLOCK,), 0, tl.float32)
    for group in range(G):
        value += tl.load(x + base + group * D, i < N, 0).to(tl.float32)
    tl.store(y + i, value, i < N)


def group_reduce(x, groups, dtype):
    if groups == 1:
        return copy_tensor(x, dtype)
    b, t, hq, d = x.shape
    result = torch.empty((b, t, hq // groups, d), device=x.device, dtype=dtype)
    _group_reduce[(triton.cdiv(result.numel(), 256),)](x, result, result.numel(), d, groups, 256)
    return result


@triton.jit
def _sink_grad(sink, lse, delta, out, N: tl.constexpr, HQ: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    s = tl.load(sink + head).to(tl.float32)
    total = tl.full((BLOCK,), 0, tl.float32)
    for start in range(0, N, BLOCK):
        i = start + off
        l = tl.load(lse + i * HQ + head, i < N, 0).to(tl.float32)
        d = tl.load(delta + i * HQ + head, i < N, 0).to(tl.float32)
        total -= tl.exp2(s - l) * d
    tl.store(out + head, tl.sum(total, 0))


def sink_gradient(sink, lse, delta):
    result = torch.empty_like(sink)
    _sink_grad[(sink.numel(),)](
        sink, lse, delta, result, lse.numel() // sink.numel(), sink.numel(), 256
    )
    return result


@triton.jit
def parallel_attn_bwd_kernel_preprocess(
    o,
    do,
    delta,
    B: tl.constexpr,
    V: tl.constexpr,
):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def parallel_attn_bwd_preprocess(o, do):
    """Compute FP32 ``delta = sum(o * do, dim=-1)`` for attention backward."""
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float)
    parallel_attn_bwd_kernel_preprocess[(delta.numel(),)](
        o=o,
        do=do,
        delta=delta,
        B=triton.next_power_of_2(V),
        V=V,
    )
    return delta


from flag_attn import runtime

RCP_LN2 = 1.4426950216


@triton.jit
def exp2(x):
    return tl.exp2(x)


@triton.jit
def log2(x):
    return tl.log2(x)


_DEBUG_ASSERTS = os.environ.get("WALL_ATTN_DEBUG", "0") == "1"

# Set WALL_ATTN_DKV_DIAG_BF16=0 to force fp32 tensor cores in the diagonal loop
# of parallel_wall_attn_bwd_kernel_dkv (precision over speed).
_DKV_DIAG_BF16 = os.environ.get("WALL_ATTN_DKV_DIAG_BF16", "1") == "1"

WALL_FWD_AUTOTUNE_CONFIGS = [
    # This MACA two-stage/eight-warp schedule miscomputes tail rows at D32.
    # Preserve the tile and precision using standard Triton pipelining.
    triton.Config(
        {"BT": BT, "BS": BS, **({"pipeline": ""} if (BT, BS, nw, ns) == (64, 64, 8, 2) else {})},
        num_warps=nw,
        num_stages=ns,
    )
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 3)
] + [
    # The original multi-stage candidates exceed C550 resources at D128.
    triton.Config({"BT": bt, "BS": 32}, num_warps=nw, num_stages=1)
    for bt in (32, 64)
    for nw in (2, 4, 8)
]

WALL_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({"BT": BT, "BS": BS}, num_warps=nw, num_stages=ns)
    for BT in (64, 128)
    for BS in (32, 64)
    for nw in (2, 4, 8)
    for ns in (1, 2, 3)
]

# MACA's optimized three-stage dQ pipeline corrupts results at D64. Keep
# the candidate and matrix operations, but use standard Triton pipelining.
WALL_DQ_AUTOTUNE_CONFIGS = [
    triton.Config(
        {**config.kwargs, **({"pipeline": ""} if config.num_stages == 3 else {})},
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    for config in WALL_BWD_AUTOTUNE_CONFIGS
]


# Full-V BF16 D128 inference can reuse QK and softmax across all value
# channels. Smaller KV tiles bound the live operands of that wider PV dot.
WALL_FWD_AUTOTUNE_CONFIGS += [
    triton.Config({"BT": 32, "BS": 16}, num_warps=nw, num_stages=1) for nw in (2, 4, 8)
]


def _wall_fwd_config_prune(configs, named_args, **kwargs):
    arguments = {**named_args, **kwargs}
    if not arguments.get("REUSE_PV_ACC", False):
        configs = [config for config in configs if config.kwargs["BS"] != 16]
    return resource_pruner(lambda: parallel_wall_attn_fwd_kernel)(configs, named_args, **kwargs)


@triton.jit
def _wall_qk_dot_split32(a, b):
    # Bound each QK operand fragment while retaining one FP32 accumulator.
    M: tl.constexpr = a.shape[0]
    D: tl.constexpr = a.shape[1]
    N: tl.constexpr = b.shape[1]
    a0, a1 = tl.split(tl.reshape(a, (M, 2, D // 2)).trans(0, 2, 1))
    b0, b1 = tl.split(tl.reshape(b, (2, D // 2, N)).trans(1, 2, 0))
    if D == 128:
        a00, a01 = tl.split(tl.reshape(a0, (M, 2, 32)).trans(0, 2, 1))
        a10, a11 = tl.split(tl.reshape(a1, (M, 2, 32)).trans(0, 2, 1))
        b00, b01 = tl.split(tl.reshape(b0, (2, 32, N)).trans(1, 2, 0))
        b10, b11 = tl.split(tl.reshape(b1, (2, 32, N)).trans(1, 2, 0))
        acc = tl.dot(a00, b00)
        acc = tl.dot(a01, b01, acc)
        acc = tl.dot(a10, b10, acc)
        return tl.dot(a11, b11, acc)
    else:
        return tl.dot(a1, b1, tl.dot(a0, b0))


@triton.autotune(
    configs=WALL_FWD_AUTOTUNE_CONFIGS,
    key=["T", "K", "V", "HQ", "H", "REUSE_PV_ACC"],
    prune_configs_by={"early_config_prune": _wall_fwd_config_prune},
)
@triton.heuristics(
    {
        "USE_SINK_BIAS": lambda args: args["sink_bias"] is not None,
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_SCALAR_G": lambda args: args["g_scalar_cumsum"] is not None,
        # MACA's multi-stage optimized pipeline miscomputes fragmented QK.
        # Keep its original dot; validated single-stage tiles use smaller fragments.
        "SPLIT_QK": lambda args: args.get("num_stages", 1) == 1
        and (
            args.get("REUSE_PV_ACC", False)
            or (args["K"] == args["V"] == 64 and args["q"].dtype in (torch.bfloat16, torch.float16))
        ),
    }
)
@triton.jit
def parallel_wall_attn_fwd_kernel(
    q,
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
    REUSE_PV_ACC: tl.constexpr = False,
    SPLIT_QK: tl.constexpr = False,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(
            tl.int64
        )
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_o = tl.make_block_ptr(
        o + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    p_pq = tl.make_block_ptr(
        g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_pq = tl.load(p_pq, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # Vector loads avoid unsupported MACA block-pointer layouts for one-row anchors.
    b_R = tl.load(
        g_cumsum + ((bos + i_t * BT) * HQ + i_hq) * K + tl.arange(0, BK),
        (tl.arange(0, BK) < K) & (i_t * BT < T),
        0,
    )[None, :].to(tl.float32)
    # For nonpositive gates and earlier KV blocks, both anchor-relative
    # exponents are nonpositive. Diagonal blocks use local anchors below.
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    if REUSE_PV_ACC:
        b_o = tl.zeros([BV, BT], dtype=tl.float32)
    else:
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(
            g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_cq = tl.load(p_cq, boundary_check=(0,), padding_option="zero").to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32)
    else:
        b_sink_bias = None

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0

    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )
        p_pk = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1)
        )

        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if not REUSE_PV_ACC:
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_pk = tl.load(p_pk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        if SPLIT_QK:
            if REUSE_PV_ACC and K == 128 and V == 128:
                # Orient off-diagonal QK for the transposed PV operand layout.
                # Keep diagonal FP32 QK unchanged: its layout has different costs.
                b_s = (
                    tl.trans(_wall_qk_dot_split32(tl.trans(b_k_til), tl.trans(b_q_til)))
                    * scale
                    * RCP_LN2
                )
            else:
                b_s = _wall_qk_dot_split32(b_q_til, b_k_til) * scale * RCP_LN2
        else:
            b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        if REUSE_PV_ACC:
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        if REUSE_PV_ACC:
            b_o = tl.dot(tl.trans(b_v), tl.trans(b_p.to(b_v.dtype)), b_o * b_r[None, :])
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )
        p_pk = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1)
        )

        # Per-sub-block local reference to prevent exp2 overflow.

        b_R_local = tl.load(
            g_cumsum + ((bos + i_s) * HQ + i_hq) * K + tl.arange(0, BK),
            (tl.arange(0, BK) < K) & (i_s < T),
            0,
        )[None, :].to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        if not REUSE_PV_ACC:
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_pk = tl.load(p_pk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        # Cap the local K exponent at 110 to leave FP32 accumulation headroom.
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame; clamp exp for Q before sub-block (masked by causality).
        b_exp_q = tl.minimum(b_pq - b_R_local_bc, tl.zeros([BT, BK], dtype=tl.float32))
        b_q_til_local = b_q.to(tl.float32) * exp2(b_exp_q)
        if SPLIT_QK:
            b_s = _wall_qk_dot_split32(b_q_til_local, b_k_til) * scale * RCP_LN2
        else:
            b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if REUSE_PV_ACC:
            b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )
            b_s += b_cq[:, None] - b_ck[None, :]

        m_s = (o_q[:, None] >= o_k[None, :]) & m_k[None, :]
        if USE_WINDOW:
            m_s = m_s & (o_q[:, None] - o_k[None, :] < W)
        b_s = tl.where(m_s, b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        if REUSE_PV_ACC:
            b_o = tl.dot(tl.trans(b_v), tl.trans(b_p.to(b_v.dtype)), b_o * b_r[None, :])
        else:
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)
        b_mp = b_m

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_acc += exp2(b_sink_bias - b_m)

    if REUSE_PV_ACC:
        b_o = tl.trans(b_o / b_acc[None, :])
    else:
        b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=WALL_DQ_AUTOTUNE_CONFIGS,
    key=["T", "K", "V", "HQ", "H"],
    prune_configs_by={
        "early_config_prune": resource_pruner(lambda: parallel_wall_attn_bwd_kernel_dq)
    },
)
@triton.heuristics(
    {
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_SCALAR_G": lambda args: args["g_scalar_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_wall_attn_bwd_kernel_dq(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dq,
    dg_cumsum,
    dg_scalar_cumsum,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
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
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(
            tl.int64
        )
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_q = tl.make_block_ptr(
        q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_pq = tl.make_block_ptr(
        g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )

    p_dq = tl.make_block_ptr(
        dq + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_dg = tl.make_block_ptr(
        dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_do = tl.make_block_ptr(
        do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))
    p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_pq = tl.load(p_pq, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_R = tl.load(
        g_cumsum + ((bos + i_t * BT) * HQ + i_hq) * K + tl.arange(0, BK),
        (tl.arange(0, BK) < K) & (i_t * BT < T),
        0,
    )[None, :].to(tl.float32)
    # Off-diagonal q_til (bf16 tensor cores, exp2 <= 1).
    b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
    b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
    b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero")

    b_dq_til = tl.zeros([BT, BK], dtype=tl.float32)

    if USE_SCALAR_G:
        p_cq = tl.make_block_ptr(
            g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_cq = tl.load(p_cq, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    o_q = i_t * BT + tl.arange(0, BT)
    i_start = tl.maximum((i_t * BT - W + 1) // BS * BS, 0) if USE_WINDOW else 0
    b_R_t = tl.trans(b_R)

    for i_s in range(i_start, i_t * BT, BS):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
        )
        p_pk = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1)
        )

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_pk = tl.load(p_pk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_k_til = (b_k.to(tl.float32) * exp2(b_R_t - b_pk)).to(b_k.dtype)
        b_s = tl.dot(b_q_til, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )
            b_s += b_cq[:, None] - b_ck[None, :]
        if USE_WINDOW:
            b_s = tl.where((o_q[:, None] - o_k[None, :] < W) & m_k[None, :], b_s, float("-inf"))
        b_p = exp2(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        b_dq_til += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k_til))
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Diagonal: use per-sub-block local reference to prevent exp2 overflow.
    # Accumulate dq_diag directly in output space to avoid large frame corrections.
    b_dq_diag = tl.zeros([BT, BK], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):
        p_k = tl.make_block_ptr(
            k + (bos * H + i_h) * K, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1)
        )
        p_v = tl.make_block_ptr(
            v + (bos * H + i_h) * V, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1)
        )
        p_pk = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (K, T), (1, HQ * K), (0, i_s), (BK, BS), (0, 1)
        )

        b_R_local = tl.load(
            g_cumsum + ((bos + i_s) * HQ + i_hq) * K + tl.arange(0, BK),
            (tl.arange(0, BK) < K) & (i_s < T),
            0,
        )[None, :].to(tl.float32)
        b_R_local_bc = tl.broadcast_to(b_R_local, (BT, BK))
        b_R_local_t = tl.trans(b_R_local)

        o_k = i_s + tl.arange(0, BS)
        m_k = o_k < T
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_pk = tl.load(p_pk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")

        # Cap the local K exponent at 110 to leave FP32 accumulation headroom.
        b_exp_k = b_R_local_t - b_pk
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_k_til = b_k.to(tl.float32) * exp2(b_exp_k)

        # q_til in local frame for computing scores.
        b_q_til_local = b_q.to(tl.float32) * exp2(b_pq - b_R_local_bc)
        b_s = tl.dot(b_q_til_local, b_k_til) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(g_scalar_cumsum + (bos + o_k) * HQ + i_hq, mask=m_k, other=0).to(
                tl.float32
            )
            b_s += b_cq[:, None] - b_ck[None, :]

        if USE_WINDOW:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & (o_q[:, None] - o_k[None, :] < W) & m_k[None, :],
                exp2(b_s - b_lse[:, None]),
                0,
            )
        else:
            b_p = tl.where(
                (o_q[:, None] >= o_k[None, :]) & m_k[None, :], exp2(b_s - b_lse[:, None]), 0
            )

        b_dp = tl.dot(b_do, b_v)
        b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
        # dq_sub in local frame; convert to output space: dq += dq_sub * scale * exp2(Pq - R_local).
        # For Q < sub-block start: causal mask makes ds=0, but mask explicitly to kill fp noise.
        b_dq_sub = tl.dot(b_ds.to(tl.float32), tl.trans(b_k_til))
        m_q_causal = (o_q >= i_s)[:, None]
        b_dq_diag += tl.where(m_q_causal, b_dq_sub * exp2(b_pq - b_R_local_bc), 0.0)
        if USE_SCALAR_G:
            b_dc += tl.sum(b_ds, 1)

    # Off-diagonal (chunk-ref) + diagonal (direct output space).
    b_dq = (b_dq_til * scale) * exp2(b_pq - b_R) + b_dq_diag * scale

    # Structural: b_dg derived from b_dq (saves [BT,BK] fp32 accumulator).
    b_dg = LN2 * b_q.to(tl.float32) * b_dq
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(
            dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=WALL_BWD_AUTOTUNE_CONFIGS,
    key=["T", "K", "V", "HQ", "H"],
    prune_configs_by={
        "early_config_prune": resource_pruner(lambda: parallel_wall_attn_bwd_kernel_dkv)
    },
)
@triton.heuristics(
    {
        "USE_WINDOW": lambda args: args["W"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_SCALAR_G": lambda args: args["g_scalar_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def parallel_wall_attn_bwd_kernel_dkv(
    q,
    k,
    v,
    g_cumsum,
    g_scalar_cumsum,
    lse,
    delta,
    do,
    dk,
    dv,
    dg_cumsum,
    dg_scalar_cumsum,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
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
    USE_WINDOW: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
    DIAG_BF16: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(
            tl.int64
        )
        T = (eos - bos).to(tl.int32)
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)
    RCP_LN2: tl.constexpr = 1.4426950216
    LN2: tl.constexpr = 0.6931471805599453

    p_k = tl.make_block_ptr(
        k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_pk = tl.make_block_ptr(
        g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_v = tl.make_block_ptr(
        v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_dk = tl.make_block_ptr(
        dk + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
    )
    p_dg = tl.make_block_ptr(
        dg_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)
    )

    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_pk = tl.load(p_pk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    o_k = i_t * BT + tl.arange(0, BT)

    if USE_SCALAR_G:
        p_ck = tl.make_block_ptr(
            g_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        b_ck = tl.load(p_ck, boundary_check=(0,), padding_option="zero").to(tl.float32)
        b_dc = tl.zeros([BT], dtype=tl.float32)

    for i_s in range(i_t * BT, min((i_t + 1) * BT, T), BS):

        p_q = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0)
        )
        p_pq = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(
            g_cumsum + ((bos + i_s) * HQ + i_hq) * K + tl.arange(0, BK),
            (tl.arange(0, BK) < K) & (i_s < T),
            0,
        )[None, :].to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_pq = tl.load(p_pq, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_exp_k = b_Rq_bc_k - b_pk
        # Non-causal keys (key_pos > last query in sub-block) can have
        # exp_k >> 110, overflowing the gradient headroom.  The causal mask
        # zeros their contribution in b_p, but inf*0 = NaN poisons b_dk/b_dg.
        # Clamp to 110 (fp32-safe with BK/BS dot accumulation) so products
        # in dg accumulation stay well within fp32.
        b_exp_k = tl.where(b_exp_k > 110.0, tl.zeros_like(b_exp_k) + 110.0, b_exp_k)
        b_exp_k_val = exp2(b_exp_k)  # CSE: reused for k_til and final dk scaling
        if DIAG_BF16:
            # Fast path: bf16 tensor cores, fp32 accumulation.
            b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
            b_k_til = (b_k.to(tl.float32) * b_exp_k_val).to(b_k.dtype)
        else:
            # Precise path: fp32 tensor cores.
            b_q_til = b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)
            b_k_til = b_k.to(tl.float32) * b_exp_k_val
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
        b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero")
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(
                tl.float32
            )
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & (o_q[None, :] - o_k[:, None] < W) & m_q[None, :],
                exp2(b_s - b_lse[None, :]),
                0,
            )
        else:
            b_p = tl.where(
                (o_k[:, None] <= o_q[None, :]) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0
            )
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        if DIAG_BF16:
            b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        else:
            b_dk_til = tl.dot(b_ds.to(tl.float32), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    i_end = min(tl.cdiv(T, BS) * BS, (i_t + 1) * BT + W - 1) if USE_WINDOW else tl.cdiv(T, BS) * BS

    for i_s in range((i_t + 1) * BT, i_end, BS):

        p_q = tl.make_block_ptr(
            q + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0)
        )
        p_pq = tl.make_block_ptr(
            g_cumsum + (bos * HQ + i_hq) * K, (T, K), (HQ * K, 1), (i_s, 0), (BS, BK), (1, 0)
        )
        p_do = tl.make_block_ptr(
            do + (bos * HQ + i_hq) * V, (T, V), (HQ * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0)
        )
        p_lse = tl.make_block_ptr(lse + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))
        p_delta = tl.make_block_ptr(delta + bos * HQ + i_hq, (T,), (HQ,), (i_s,), (BS,), (0,))

        o_q = i_s + tl.arange(0, BS)
        m_q = o_q < T
        b_Rq = tl.load(
            g_cumsum + ((bos + i_s) * HQ + i_hq) * K + tl.arange(0, BK),
            (tl.arange(0, BK) < K) & (i_s < T),
            0,
        )[None, :].to(tl.float32)
        b_Rq_bc_k = tl.broadcast_to(b_Rq, (BT, BK))
        b_Rq_bc_q = tl.broadcast_to(b_Rq, (BS, BK))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_pq = tl.load(p_pq, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_Rq_bc_q)).to(b_q.dtype)
        b_exp_k_off = b_Rq_bc_k - b_pk
        b_exp_k_off_val = exp2(b_exp_k_off)  # CSE: reused for k_til and final dk scaling
        b_k_til = (b_k.to(tl.float32) * b_exp_k_off_val).to(b_k.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_lse = tl.load(p_lse, boundary_check=(0,), padding_option="zero")
        b_delta = tl.load(p_delta, boundary_check=(0,), padding_option="zero")
        b_s = tl.dot(b_k_til, tl.trans(b_q_til)) * scale * RCP_LN2
        if USE_SCALAR_G:
            b_cq = tl.load(g_scalar_cumsum + (bos + o_q) * HQ + i_hq, mask=m_q, other=0).to(
                tl.float32
            )
            b_s += b_cq[None, :] - b_ck[:, None]
        if USE_WINDOW:
            b_p = tl.where(
                (o_q[None, :] - o_k[:, None] < W) & m_q[None, :], exp2(b_s - b_lse[None, :]), 0
            )
        else:
            b_p = tl.where(m_q[None, :], exp2(b_s - b_lse[None, :]), 0)
        b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
        b_dp = tl.dot(b_v, tl.trans(b_do))
        b_ds = b_p * (b_dp - b_delta[None, :])
        b_dk_til = tl.dot(b_ds.to(b_q.dtype), b_q_til)
        b_dk += (b_dk_til * scale) * b_exp_k_off_val
        if USE_SCALAR_G:
            b_dc -= tl.sum(b_ds, 1)

    # Structural: b_dg derived from b_dk (saves [BT,BK] fp32 accumulator).
    b_dg = -LN2 * b_k.to(tl.float32) * b_dk
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    if USE_SCALAR_G:
        p_dc = tl.make_block_ptr(
            dg_scalar_cumsum + bos * HQ + i_hq, (T,), (HQ,), (i_t * BT,), (BT,), (0,)
        )
        tl.store(p_dc, b_dc.to(p_dc.dtype.element_ty), boundary_check=(0,))


def _wall_varlen_config(T, K, V):
    """Use the same chunk size in forward, backward and the packed chunk map."""
    head_dim = max(triton.next_power_of_2(K), triton.next_power_of_2(V))
    if head_dim in (64, 128):
        # Bound the query-channel tile to 4096 elements on C550.
        return 4096 // head_dim, 16, 4, 1
    return 128, min(32, max(16, triton.next_power_of_2(T))), 2, 2


def parallel_wall_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_cumsum: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    reuse_pv_acc: bool = True,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError(
            f"`g_cumsum` must be [B, T, HQ, K] matching `q`; got {g_cumsum.shape} vs q {q.shape}"
        )
    if g_cumsum.stride(-1) != 1:
        raise ValueError("`g_cumsum` must be contiguous in the last (K) dimension")
    g_cumsum = copy_tensor(g_cumsum)

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BK = min(256, max(16, triton.next_power_of_2(K)))
    # Keep the C550 value-channel tile at most 64 wide.
    BV = min(64, max(16, triton.next_power_of_2(V)))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    if cu_seqlens is not None:
        # Keep the chunk map tied to the deterministic resource-safe schedule.
        BT, BS, num_warps, num_stages = _wall_varlen_config(T, K, V)
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
        lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        parallel_wall_attn_fwd_kernel.fn[(NV, NT, B * HQ)](
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=g_cumsum,
            g_scalar_cumsum=g_scalar_cumsum,
            sink_bias=sink_bias,
            lse=lse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            B=B,
            T=T,
            W=window_size,
            H=H,
            HQ=HQ,
            G=G,
            K=K,
            V=V,
            BT=BT,
            BS=BS,
            BK=BK,
            BV=BV,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return o, lse

    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)

    # Validated for ordinary BF16 D128 inference. Keep training and optional
    # features on the original accumulation path and tune the paths separately.
    reuse_pv_acc = (
        reuse_pv_acc
        and K == 128
        and V == 128
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and sink_bias is None
        and g_scalar_cumsum is None
        and window_size is None
    )

    if reuse_pv_acc:
        # One CTA owns the full output width: no duplicated QK/softmax and
        # no global probability workspace. Training/varlen keep their tiles.
        BV = triton.next_power_of_2(V)
        NV = 1

    def grid(meta):
        return (NV, triton.cdiv(T, meta["BT"]), B * HQ)

    parallel_wall_attn_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        sink_bias=sink_bias,
        lse=lse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        W=window_size,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        REUSE_PV_ACC=reuse_pv_acc,
    )
    return o, lse


def parallel_wall_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    g_cumsum: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    g_scalar_cumsum: torch.Tensor | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    if g_cumsum.dim() != 4 or g_cumsum.shape != (*q.shape[:-1], q.shape[-1]):
        raise ValueError("`g_cumsum` must be [B, T, HQ, K] matching `q`")
    g_cumsum = copy_tensor(g_cumsum)

    B, T, H, K, V = *k.shape, v.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    # dq/dk are reduced over the full value dim in one program (no cross-program accumulation),
    # so BV must span all of V (NV == 1). Don't cap it here -- the forward can, the backward can't.
    BK = max(triton.next_power_of_2(K), 16)
    BV = max(triton.next_power_of_2(V), 16)

    NV = triton.cdiv(V, BV)

    # Match the forward schedule so chunk indices always describe the same tiles.
    is_varlen = cu_seqlens is not None
    extra = {}
    if is_varlen:
        BT, BS, num_warps, num_stages = _wall_varlen_config(T, K, V)
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
        grid = (NV, NT, B * HQ)
        extra = dict(BT=BT, BS=BS, num_warps=num_warps, num_stages=num_stages)
    else:

        def grid(meta):
            return (NV, triton.cdiv(T, meta["BT"]), B * HQ)

    delta = parallel_attn_bwd_preprocess(o, do)

    dq = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dk = torch.empty(B, T, HQ, K, dtype=k.dtype if H == HQ else torch.float, device=q.device)
    dv = torch.empty(B, T, HQ, V, dtype=v.dtype if H == HQ else torch.float, device=q.device)
    dq_kernel = (
        parallel_wall_attn_bwd_kernel_dq.fn if is_varlen else parallel_wall_attn_bwd_kernel_dq
    )
    dkv_kernel = (
        parallel_wall_attn_bwd_kernel_dkv.fn if is_varlen else parallel_wall_attn_bwd_kernel_dkv
    )

    dg_cumsum = torch.empty(B, T, HQ, K, dtype=torch.float, device=q.device)
    dg_cumsum_k = torch.empty_like(dg_cumsum)

    if g_scalar_cumsum is not None:
        dg_scalar_cumsum = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
        dg_scalar_cumsum_k = torch.empty_like(dg_scalar_cumsum)
    else:
        dg_scalar_cumsum = None
        dg_scalar_cumsum_k = None

    dq_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dq=dq,
        dg_cumsum=dg_cumsum,
        dg_scalar_cumsum=dg_scalar_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        **extra,
    )
    dkv_kernel[grid](
        q=q,
        k=k,
        v=v,
        g_cumsum=g_cumsum,
        g_scalar_cumsum=g_scalar_cumsum,
        lse=lse,
        delta=delta,
        do=do,
        dk=dk,
        dv=dv,
        dg_cumsum=dg_cumsum_k,
        dg_scalar_cumsum=dg_scalar_cumsum_k,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        W=window_size,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        DIAG_BF16=_DKV_DIAG_BF16,
        **extra,
    )
    dk = group_reduce(dk, G, k.dtype)
    dv = group_reduce(dv, G, v.dtype)
    add_inplace(dg_cumsum, dg_cumsum_k)

    if g_scalar_cumsum is not None:
        add_inplace(dg_scalar_cumsum, dg_scalar_cumsum_k)

    dsink_bias = None
    if sink_bias is not None:
        dsink_bias = sink_gradient(sink_bias, lse, delta)

    return dq, dk, dv, dg_cumsum, dsink_bias, dg_scalar_cumsum


class WallParallelAttentionFunction(torch.autograd.Function):
    r"""Wall Attention: per-channel prefix :math:`P` (log_2); scores use
    :math:`\sum_n q_{in} k_{jn} \exp_2(P_{in}-P_{jn})` via per-Q-block reference
    :math:`R_n=P_{i_t\cdot BT,n}` inside :func:`parallel_wall_attn_fwd_kernel`."""

    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        sink_bias,
        scale,
        window_size,
        cu_seqlens,
        g_scalar=None,
        chunk_indices=None,
    ):
        if g.shape[-1] != q.shape[-1] or g.shape[2] != q.shape[2]:
            raise ValueError(
                f"`g` must be [B, T, HQ, K] with same HQ and K as `q`; got {g.shape} vs q {q.shape}"
            )
        if q.shape[2] % k.shape[2] != 0:
            raise ValueError("HQ must be divisible by `k.shape[2]` for GQA")

        if _DEBUG_ASSERTS:
            assert torch.isfinite(g).all(), f"[wall-attn fwd] NaN/Inf in g: {g.abs().max()}"
            assert torch.isfinite(q).all(), f"[wall-attn fwd] NaN/Inf in q: {q.abs().max()}"
            assert torch.isfinite(k).all(), f"[wall-attn fwd] NaN/Inf in k: {k.abs().max()}"
            assert torch.isfinite(v).all(), f"[wall-attn fwd] NaN/Inf in v: {v.abs().max()}"

        if cu_seqlens is not None and chunk_indices is None:
            BT, _, _, _ = _wall_varlen_config(q.shape[1], q.shape[-1], v.shape[-1])
            chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        P = chunk_global_cumsum(g, cu_seqlens=cu_seqlens, scale=RCP_LN2)
        if _DEBUG_ASSERTS:
            assert torch.isfinite(
                P
            ).all(), f"[wall-attn fwd] NaN/Inf in P=cumsum(g): P abs max={P.abs().max()}, g abs max={g.abs().max()}"

        if g_scalar is not None:
            if _DEBUG_ASSERTS:
                assert torch.isfinite(
                    g_scalar
                ).all(), f"[wall-attn fwd] NaN/Inf in g_scalar: {g_scalar.abs().max()}"
            c = chunk_global_cumsum(g_scalar, cu_seqlens=cu_seqlens, scale=RCP_LN2)
            if _DEBUG_ASSERTS:
                assert torch.isfinite(
                    c
                ).all(), f"[wall-attn fwd] NaN/Inf in c=cumsum(g_scalar): c abs max={c.abs().max()}"
        else:
            c = None

        sink_bias_scaled = copy_tensor(sink_bias, scale=RCP_LN2) if sink_bias is not None else None
        o, lse = parallel_wall_attn_fwd(
            q=q,
            k=k,
            v=v,
            g_cumsum=P,
            sink_bias=sink_bias_scaled,
            scale=scale,
            g_scalar_cumsum=c,
            window_size=window_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            reuse_pv_acc=not any(ctx.needs_input_grad),
        )
        if _DEBUG_ASSERTS:
            assert torch.isfinite(o).all(), (
                f"[wall-attn fwd] NaN/Inf in output o: o abs max={o.abs().max()}, "
                f"lse finite={torch.isfinite(lse).all()}, P abs max={P.abs().max()}"
            )

        ctx.save_for_backward(
            q, k, v, o, P, lse, sink_bias_scaled, c if c is not None else torch.empty(0)
        )
        ctx.scale = scale
        ctx.window_size = window_size
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.has_scalar_g = g_scalar is not None
        return copy_tensor(o, q.dtype)

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do):
        q, k, v, o, P, lse, sink_bias_scaled, c_or_empty = ctx.saved_tensors
        c = c_or_empty if ctx.has_scalar_g else None

        if _DEBUG_ASSERTS:
            assert torch.isfinite(do).all(), f"[wall-attn bwd] NaN/Inf in do: {do.abs().max()}"

        dq, dk, dv, dP, dsink_bias, dc = parallel_wall_attn_bwd(
            q=q,
            k=k,
            v=v,
            o=o,
            g_cumsum=P,
            lse=lse,
            do=do,
            sink_bias=sink_bias_scaled,
            scale=ctx.scale,
            g_scalar_cumsum=c,
            window_size=ctx.window_size,
            cu_seqlens=ctx.cu_seqlens,
            chunk_indices=ctx.chunk_indices,
        )
        if _DEBUG_ASSERTS:
            assert torch.isfinite(dq).all(), f"[wall-attn bwd] NaN/Inf in dq: {dq.abs().max()}"
            assert torch.isfinite(dk).all(), f"[wall-attn bwd] NaN/Inf in dk: {dk.abs().max()}"
            assert torch.isfinite(dv).all(), f"[wall-attn bwd] NaN/Inf in dv: {dv.abs().max()}"
            assert torch.isfinite(dP).all(), (
                f"[wall-attn bwd] NaN/Inf in dP: dP abs max={dP.abs().max()}, "
                f"P abs max={P.abs().max()}, lse finite={torch.isfinite(lse).all()}"
            )

        # The kernel emits dP with an LN2 factor (b_dg = LN2 * q * dq); the forward set
        # P = cumsum(g) * RCP_LN2, so dL/dg = RCP_LN2 * reverse_cumsum(dP) (nets the LN2).
        dg = chunk_global_cumsum(dP, cu_seqlens=ctx.cu_seqlens, reverse=True, scale=RCP_LN2)
        if _DEBUG_ASSERTS:
            assert torch.isfinite(dg).all(), (
                f"[wall-attn bwd] NaN/Inf in dg (after rev cumsum): dg abs max={dg.abs().max()}, "
                f"dP abs max={dP.abs().max()}"
            )

        if dc is not None:
            dg_scalar = chunk_global_cumsum(dc, cu_seqlens=ctx.cu_seqlens, reverse=True)
        else:
            dg_scalar = None

        return (
            copy_tensor(dq, q.dtype),
            copy_tensor(dk, k.dtype),
            copy_tensor(dv, v.dtype),
            dg,
            dsink_bias,
            None,
            None,
            None,
            dg_scalar,
            None,
        )


def _parallel_wall_attn_default(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    r"""Wall parallel attention (training / prefill forward + backward).

    Scores use a per-channel multiplicative decay: with :math:`P = \mathrm{cumsum}(g)`
    in :math:`\log_2` space, the logit for pair :math:`(i, j)` is
    :math:`\mathrm{scale} \sum_n q_{in} k_{jn} \exp_2(P_{in} - P_{jn})`.

    Args:
        q (torch.Tensor):
            Queries of shape ``[B, T, HQ, K]``.
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]`` (``H`` may be < ``HQ`` for GQA).
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            Per-channel log-decay of shape ``[B, T, HQ, K]`` (cumsum'd internally).
        g_scalar (torch.Tensor, Optional):
            FoX-style additive scalar gate of shape ``[B, T, HQ]``. Default: `None`.
        sink_bias (torch.Tensor, Optional):
            Attention-sink logit of shape ``[HQ]``. Default: `None`.
        scale (float, Optional):
            Softmax scale. If `None`, defaults to ``K ** -0.5``. Default: `None`.
        window_size (int, Optional):
            Sliding-window width (causal). Default: `None`.
        cu_seqlens (torch.LongTensor, Optional):
            Cumulative seqlens of shape ``[N + 1]`` for varlen packing
            (requires ``B == 1``). Default: `None`.

    Returns:
        Attention output of shape ``[B, T, HQ, V]``.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if g_scalar is not None and g_scalar.shape != q.shape[:-1]:
        raise ValueError(
            f"`g_scalar` must be [B, T, HQ] matching q.shape[:-1]; got {g_scalar.shape}"
        )
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError("`cu_seqlens` (varlen) requires batch size 1")
    if sink_bias is not None and sink_bias.shape != (q.shape[2],):
        raise ValueError(f"`sink_bias` must be [HQ]; got {sink_bias.shape}")
    return WallParallelAttentionFunction.apply(
        q, k, v, g, sink_bias, scale, window_size, cu_seqlens, g_scalar, None
    )


_FAST_T = frozenset((512, 1024, 2048, 4096))
_FAST_H = frozenset((2, 8))
_FAST_BF16_D = frozenset((64, 128))
_LAST_ROUTE: ContextVar[str | None] = ContextVar("wall_attn_route", default=None)


def get_last_route() -> str | None:
    """Return the completed call's route in this context, for benchmark reporting."""
    return _LAST_ROUTE.get()


@lru_cache(maxsize=1)
def _load_fwd_module():
    """Load cached-operand kernels after metadata accepts an inference call."""
    module = import_module(f"{__package__}.parallel_fwd")
    return module


def _requires_backward(*tensors: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def select_route(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> str:
    """Select a candidate path from metadata; preprocessing checks numerical safety."""
    if any(value is not None for value in (g_scalar, sink_bias, window_size, cu_seqlens)):
        return "official_fallback"
    if scale is not None and not isinstance(scale, Real):
        return "official_fallback"
    if any(tensor.layout != torch.strided for tensor in (q, k, v, g)):
        return "official_fallback"
    if not all(tensor.is_cuda for tensor in (q, k, v, g)):
        return "official_fallback"
    if not (q.device == k.device == v.device == g.device):
        return "official_fallback"
    if not all(tensor.is_contiguous() for tensor in (q, k, v, g)):
        return "official_fallback"
    if any(tensor.ndim != 4 for tensor in (q, k, v, g)):
        return "official_fallback"

    if runtime.device.vendor_name != "metax":
        return "official_fallback"

    b, t, hq, d = q.shape
    if b != 1 or t not in _FAST_T or hq != 8:
        return "official_fallback"
    h = k.shape[2]
    if h not in _FAST_H or hq % h != 0:
        return "official_fallback"
    if k.shape != (b, t, h, d) or v.shape != (b, t, h, d) or g.shape != q.shape:
        return "official_fallback"
    if _requires_backward(q, k, v, g):
        return "official_fallback"
    # Numerical validation reads a device flag. Keep graph capture on the
    # general implementation, which does not require a device-to-host read.
    with torch.cuda.device(q.device):
        if torch.cuda.is_current_stream_capturing():
            return "official_fallback"

    if q.dtype == k.dtype == v.dtype == g.dtype == torch.bfloat16 and d in _FAST_BF16_D:
        return "metax_bf16"
    if q.dtype == k.dtype == v.dtype == g.dtype == torch.float16 and d == 64:
        return "metax_fp16"
    return "official_fallback"


def _parallel_wall_attn_fast_path(
    route: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    fwd = _load_fwd_module()
    b, t, hq, d = q.shape
    cache_dtype = q.dtype if route == "metax_bf16" else torch.bfloat16
    q_cache = torch.empty((b, hq, t, d), device=q.device, dtype=cache_dtype)
    k_cache = torch.empty_like(q_cache)
    anchor = torch.empty((b, hq, d), device=q.device, dtype=torch.float32)
    output = torch.empty((b, t, hq, d), device=q.device, dtype=v.dtype)
    lse = torch.empty((b, t, hq), device=q.device, dtype=torch.float32)
    unsafe = zero_flag(q.device)

    prefix = fwd.prepare_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe)

    # Do not consume overflowing or underflowing rescaled Q/K operands.
    if unsafe.item():
        return None, prefix

    fwd.parallel_wall_attn_fwd(q_cache, k_cache, v, output, lse, scale=scale)
    return output, None


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Run a numerically checked MetaX path or the full general implementation.

    Fast paths require nonpositive gates and finite, non-underflowing BF16 Q/K
    operands with gauge exponents within +/-125. Other data use the general
    implementation. Validation synchronizes with the host; graph capture uses
    the general implementation instead.
    """
    _LAST_ROUTE.set(None)
    route = select_route(
        q,
        k,
        v,
        g,
        g_scalar=g_scalar,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    if route != "official_fallback":
        with torch.cuda.device(q.device):
            output, prefix = _parallel_wall_attn_fast_path(
                route,
                q,
                k,
                v,
                g,
                scale=k.shape[-1] ** -0.5 if scale is None else float(scale),
            )
        if output is not None:
            _LAST_ROUTE.set(route)
            return output
        if prefix is not None:
            # Metadata has excluded backward, extra gates, windows and varlen.
            # The failed cache safety check already computed this exact log2
            # prefix. Reuse it in the general kernel without a second scan.
            with torch.cuda.device(q.device):
                output, _ = parallel_wall_attn_fwd(
                    q,
                    k,
                    v,
                    prefix,
                    sink_bias=None,
                    scale=k.shape[-1] ** -0.5 if scale is None else float(scale),
                )
            _LAST_ROUTE.set("official_fallback")
            return output
    output = _parallel_wall_attn_default(
        q,
        k,
        v,
        g,
        g_scalar=g_scalar,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    _LAST_ROUTE.set("official_fallback")
    return output
