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

# Upstream-derived portions retain the following MIT license.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
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
# Upstream: https://github.com/fla-org/flash-linear-attention/tree/3e61322b615df248e7579222d1a68260560f7c24/fla/ops/wall_attn
# Wall attention was contributed upstream by Tilde Research (Timor Averbuch, Dhruv Pai).
"""Wall single-step decode kernel and pre-rescaled KV-cache builder."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_attn import runtime

from .parallel import exp2, log2


@lru_cache(maxsize=None)
def has_large_shared_memory(device_index: int) -> bool:
    """Use the input device's shared-memory limit when selecting tile sizes."""
    from triton.runtime import driver

    properties = driver.active.utils.get_device_properties(device_index)
    return properties["max_shared_mem"] >= 166_000


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
def _cache(
    k,
    prefix,
    kt,
    anchor,
    T: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    NC: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    head = (i // D) % HQ
    token = (i // (D * HQ)) % T
    batch = i // (D * HQ * T)
    chunk = token // C
    reference = tl.load(prefix + ((batch * T + chunk * C) * HQ + head) * D + d, i < N, 0).to(
        tl.float32
    )
    p = tl.load(prefix + i, i < N, 0).to(tl.float32)
    key = tl.load(k + ((batch * T + token) * H + head // (HQ // H)) * D + d, i < N, 0).to(
        tl.float32
    )
    tl.store(kt + i, key * tl.exp2(reference - p), i < N)
    tl.store(
        anchor + ((batch * NC + chunk) * HQ + head) * D + d, reference, (i < N) & (token % C == 0)
    )


def build_cache(k, prefix, chunk_size, dtype):
    k, prefix = copy_tensor(k), copy_tensor(prefix)
    b, t, h, d = k.shape
    hq = prefix.shape[2]
    nc = triton.cdiv(t, chunk_size)
    kt = torch.empty((b, t, hq, d), device=k.device, dtype=dtype)
    anchors = torch.empty((b, nc, hq, d), device=k.device, dtype=prefix.dtype)
    _cache[(triton.cdiv(kt.numel(), 256),)](
        k, prefix, kt, anchors, t, h, hq, d, chunk_size, nc, kt.numel(), 256
    )
    return kt, anchors


WALL_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({"BT": BT}, num_warps=nw, num_stages=ns)
    for BT in (16, 32)
    for nw in (2, 4, 8)
    for ns in (1, 2, 3)
]


@triton.autotune(
    configs=WALL_DECODE_AUTOTUNE_CONFIGS,
    prune_configs_by={
        "early_config_prune": resource_pruner(lambda: parallel_wall_attn_decode_kernel)
    },
    key=["KV_CHUNKS_BUCKET", "K", "V", "HQ", "H", "C"],
    **runtime.autotune_cache_kwargs(),
)
@triton.heuristics(
    {
        "USE_SINK_BIAS": lambda args: args["sink_bias"] is not None,
        "USE_SCALAR_G": lambda args: args["g_scalar_cumsum"] is not None,
    }
)
@triton.jit(do_not_specialize=["T_kv", "NC", "KV_CHUNKS_BUCKET"])
def parallel_wall_attn_decode_kernel(
    q,  # [B, T_q, HQ, K]
    k_tilde,  # [B, T_kv, HQ, K]  pre-rescaled keys (per-Q-head)
    v,  # [B, T_kv, H,  V]
    o,  # [B, T_q, HQ, V]
    p_curr,  # [B, T_q, HQ, K]
    r_cache,  # [B, NC,  HQ, K]  per-chunk anchors
    g_scalar_cumsum,  # [B, T_kv, HQ] or None  (cache-side scalar cumsum; query-side at tail)
    sink_bias,  # [HQ] or None
    lse,  # [B, T_q, HQ]
    scale,
    T_q,
    T_kv,
    NC,
    KV_CHUNKS_BUCKET,
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
    C: tl.constexpr,
    USE_SINK_BIAS: tl.constexpr,
    USE_SCALAR_G: tl.constexpr,
):
    pid = tl.program_id(0)
    NV, NT = tl.cdiv(V, BV), tl.cdiv(T_q, BT)
    i_v, i_t, i_bh = pid % NV, ((pid // NV) % NT).to(tl.int64), (pid // (NV * NT)).to(tl.int64)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    RCP_LN2: tl.constexpr = 1.4426950216

    bos_q = (i_b * T_q).to(tl.int64)
    bos_kv = (i_b * T_kv).to(tl.int64)
    bos_nc = (i_b * NC).to(tl.int64)

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_q = o_q < T_q
    p_q = q + (bos_q * HQ + i_hq) * K + o_q[:, None] * (HQ * K) + o_d[None, :]
    p_pq = p_curr + (bos_q * HQ + i_hq) * K + o_q[:, None] * (HQ * K) + o_d[None, :]
    p_o = o + (bos_q * HQ + i_hq) * V + o_q[:, None] * (HQ * V) + o_v[None, :]
    p_lse = lse + bos_q * HQ + i_hq + o_q * HQ

    b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    b_pq = tl.load(p_pq, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0).to(tl.float32)

    if USE_SCALAR_G:
        o_q_global = (T_kv - T_q) + i_t * BT + tl.arange(0, BT)
        m_q = o_q_global < T_kv
        b_cq = tl.load(
            g_scalar_cumsum + (bos_kv + o_q_global) * HQ + i_hq,
            mask=m_q,
            other=0,
        ).to(tl.float32)

    if USE_SINK_BIAS:
        b_sink_bias = tl.load(sink_bias + i_hq).to(tl.float32) * RCP_LN2

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_m = tl.full([BT], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BT], dtype=tl.float32)

    # Iterate over the KV cache one chunk at a time: BS == C
    for i_s in range(0, T_kv, BS):
        i_c = i_s // C
        o_k = (i_s + tl.arange(0, BS)).to(tl.int64)
        m_k = o_k < T_kv
        # A one-dimensional anchor load avoids MACA's singleton shared layout.
        p_r = r_cache + ((bos_nc + i_c) * HQ + i_hq) * K + o_d
        b_R = tl.load(p_r, mask=(i_c < NC) & (o_d < K), other=0.0).to(tl.float32)[None, :]
        b_q_til = (b_q.to(tl.float32) * exp2(b_pq - b_R)).to(b_q.dtype)

        p_kt = k_tilde + (bos_kv * HQ + i_hq) * K + o_d[:, None] + o_k[None, :] * (HQ * K)
        p_v = v + (bos_kv * H + i_h) * V + o_k[:, None] * (H * V) + o_v[None, :]
        b_kt = tl.load(p_kt, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)

        b_s = tl.dot(b_q_til, b_kt) * scale * RCP_LN2

        if USE_SCALAR_G:
            b_ck = tl.load(
                g_scalar_cumsum + (bos_kv + o_k) * HQ + i_hq,
                mask=m_k,
                other=0,
            ).to(tl.float32)
            b_s += b_cq[:, None] - b_ck[None, :]

        b_s = tl.where(m_k[None, :], b_s, float("-inf"))

        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_mw = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_r = exp2(b_mp - b_mw)
        b_p = exp2(b_s - b_mw[:, None])
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_v.dtype), b_v)

    if USE_SINK_BIAS:
        b_m = tl.where(b_m == float("-inf"), 0.0, b_m)
        b_acc += exp2(b_sink_bias - b_m)

    b_o = b_o / b_acc[:, None]
    b_m += log2(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_q[:, None] & (o_v[None, :] < V))
    if i_v == 0:
        tl.store(p_lse, b_m.to(p_lse.dtype.element_ty), mask=m_q)


def parallel_wall_attn_decode(
    q: torch.Tensor,
    v: torch.Tensor,
    p_curr: torch.Tensor,
    k_tilde: torch.Tensor,
    r_cache: torch.Tensor,
    sink_bias: torch.Tensor | None,
    scale: float,
    cache_chunk_size: int,
    g_scalar_cumsum: torch.Tensor | None = None,
):
    r"""Wall decode with a pre-rescaled KV cache.

    Use this directly from modeling code; it does NOT share a signature with
    `parallel_wall_attn_fwd`. The caller is responsible for picking which to
    call (e.g. training vs. cached generation).

    Shapes:
        q        : ``[B, T_q,  HQ, K]``  current queries
        v        : ``[B, T_kv, H,  V]``  cached values (standard layout)
        p_curr   : ``[B, T_q,  HQ, K]``  per-channel prefix at the current rows
        k_tilde  : ``[B, T_kv, HQ, K]``  pre-rescaled keys (see `build_wall_kv_cache`)
        r_cache  : ``[B, NC,   HQ, K]``  per-chunk anchors, ``NC = ceil(T_kv / C)``
        g_scalar_cumsum : ``[B, T_kv, HQ]`` or ``None``  optional FoX-style scalar gate

    Returns ``(o, lse)`` with ``o.shape == [B, T_q, HQ, V]``.
    """
    C = cache_chunk_size
    B, T_q, HQ, K = q.shape
    _, T_kv, H, V = v.shape
    NC = r_cache.shape[1]
    G = HQ // H
    # Bucket T_kv for autotuning; retain its exact value as the kernel loop bound.
    KV_CHUNKS_BUCKET = triton.next_power_of_2(max(1, triton.cdiv(T_kv, C)))

    if k_tilde.shape != (B, T_kv, HQ, K):
        raise ValueError(f"k_tilde shape {k_tilde.shape} != expected {(B, T_kv, HQ, K)}")
    if r_cache.shape != (B, NC, HQ, K):
        raise ValueError(f"r_cache shape {r_cache.shape} != expected {(B, NC, HQ, K)}")
    if p_curr.shape != (B, T_q, HQ, K):
        raise ValueError(f"p_curr shape {p_curr.shape} != expected {(B, T_q, HQ, K)}")
    if T_kv > NC * C:
        raise ValueError(f"NC*C ({NC * C}) < T_kv ({T_kv}); cache anchors do not cover cache")
    if HQ % H != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by H ({H})")

    for t in (q, v, p_curr, k_tilde, r_cache):
        if t.stride(-1) != 1:
            raise ValueError("decode tensors must be contiguous in the last (K or V) dim")
    if g_scalar_cumsum is not None and g_scalar_cumsum.stride(-1) != 1:
        raise ValueError("g_scalar_cumsum must be contiguous in the last dim")

    # The API permits outer-dimension views; the kernel uses packed strides.
    q, v, p_curr, k_tilde, r_cache = (copy_tensor(t) for t in (q, v, p_curr, k_tilde, r_cache))
    if g_scalar_cumsum is not None:
        g_scalar_cumsum = copy_tensor(g_scalar_cumsum)
    if sink_bias is not None:
        sink_bias = copy_tensor(sink_bias)

    large_shared_memory = has_large_shared_memory(q.device.index)
    BK = min(256, max(16, triton.next_power_of_2(K)))
    BV = min(256 if large_shared_memory else 64, max(16, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "decode kernel requires K <= 256"

    o = torch.empty(B, T_q, HQ, V, dtype=v.dtype, device=q.device)
    lse = torch.empty(B, T_q, HQ, dtype=torch.float, device=q.device)

    def grid(meta):
        return (NV * triton.cdiv(T_q, meta["BT"]) * B * HQ,)

    parallel_wall_attn_decode_kernel[grid](
        q=q,
        k_tilde=k_tilde,
        v=v,
        o=o,
        p_curr=p_curr,
        r_cache=r_cache,
        g_scalar_cumsum=g_scalar_cumsum,
        sink_bias=sink_bias,
        lse=lse,
        scale=scale,
        T_q=T_q,
        T_kv=T_kv,
        NC=NC,
        KV_CHUNKS_BUCKET=KV_CHUNKS_BUCKET,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        BS=triton.next_power_of_2(C),
        BK=BK,
        BV=BV,
        C=C,
    )
    return o, lse


def build_wall_kv_cache(
    k: torch.Tensor,  # [B, T, H, K]
    g_cumsum: torch.Tensor,  # [B, T, HQ, K]   per-channel prefix P = cumsum(log_2 g)
    chunk_size: int,
    *,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build a pre-rescaled Wall KV cache.

    Returns ``(k_tilde, r_cache)`` where, for each chunk ``c`` of size ``C``:
        ``R_c          = P[chunk_start_c]``                     ``[B, HQ, K]``
        ``k_tilde[j]   = k[j] * exp2(R_{c(j)} - P[j])``         ``[B, T, HQ, K]``
        ``r_cache[c]   = R_c``                                  ``[B, NC, HQ, K]``

    ``k_tilde`` is per-Q-head (carries the gate). Values are unchanged and
    should be cached as-is. ``g_cumsum`` is the same prefix tensor passed to
    the training forward.
    """
    if k.dim() != 4:
        raise ValueError("k must be [B, T, H, K]")
    if g_cumsum.dim() != 4:
        raise ValueError("g_cumsum must be [B, T, HQ, K]")
    B, T, H, K = k.shape
    _, _, HQ, _ = g_cumsum.shape
    if g_cumsum.shape[:2] != (B, T) or g_cumsum.shape[-1] != K:
        raise ValueError(f"g_cumsum shape {g_cumsum.shape} incompatible with k {k.shape}")
    if HQ % H != 0:
        raise ValueError(f"HQ ({HQ}) must be divisible by H ({H})")
    out_dtype = out_dtype if out_dtype is not None else k.dtype

    return build_cache(k, g_cumsum, chunk_size, out_dtype)
