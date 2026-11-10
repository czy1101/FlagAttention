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

"""MetaX Wall-Attention forward kernels with cached Q/K operands."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_attn.FLA.cumsum import chunk_global_cumsum

RCP_LN2 = 1.4426950216


@triton.jit
def _wall_attn_pv_split32(a, b, acc):
    # Keep the corrected history in one FP32 accumulator across both slices.
    a0, a1 = tl.split(tl.permute(tl.reshape(a, (a.shape[0], 2, 32)), (0, 2, 1)))
    b0, b1 = tl.split(tl.permute(tl.reshape(b, (2, 32, b.shape[1])), (1, 2, 0)))
    acc = tl.dot(a0, b0, acc)
    return tl.dot(a1, b1, acc)


@triton.jit
def _wall_attn_consumer(
    q_cache,
    k_cache,
    v,
    output,
    lse,
    scale_log2,
    T: tl.constexpr,
    HQ: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BD: tl.constexpr,
):
    tile = tl.program_id(0)
    head = tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, BD)
    cols = tl.arange(0, BN)
    qv = tl.load(
        q_cache + (head * T + rows[:, None]) * D + dims[None, :],
        (rows[:, None] < T) & (dims[None, :] < D),
        0,
    )
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    # Keep the D128 BF16 PV accumulator in the orientation used by MMA.
    # Other operand types/dimensions retain their measured baseline layout.
    TRANSPOSE_QK: tl.constexpr = D == 64 and T == 4096 and v.dtype.element_ty == tl.float16
    TRANSPOSE_PV: tl.constexpr = D == 128 and v.dtype.element_ty == tl.bfloat16 or TRANSPOSE_QK
    # FP16 short sequences did not benefit from this combination.
    SPLIT_QK: tl.constexpr = D == 64 and (v.dtype.element_ty == tl.bfloat16 or T >= 2048)
    if TRANSPOSE_PV:
        acc = tl.zeros((BD, BM), tl.float32)
    else:
        acc = tl.zeros((BM, BD), tl.float32)
    for start in range(0, tl.minimum((tile + 1) * BM, T), BN):
        keys = start + cols
        kval = tl.load(
            k_cache + (head * T + keys[None, :]) * D + dims[:, None],
            (keys[None, :] < T) & (dims[:, None] < D),
            0,
        )
        value = tl.load(
            v + (keys[:, None] * H + head // (HQ // H)) * D + dims[None, :],
            (keys[:, None] < T) & (dims[None, :] < D),
            0,
        )
        if SPLIT_QK:
            # Chain two reduction slices into the same FP32 score accumulator.
            # This avoids the full-width Q/K shared-memory staging on C550.
            q0, q1 = tl.split(tl.permute(tl.reshape(qv, (BM, 2, 32)), (0, 2, 1)))
            k0, k1 = tl.split(tl.permute(tl.reshape(kval, (2, 32, BN)), (1, 2, 0)))
            if TRANSPOSE_QK:
                score_acc = tl.dot(tl.trans(k0), tl.trans(q0))
                score = tl.dot(tl.trans(k1), tl.trans(q1), score_acc) * scale_log2
            else:
                score_acc = tl.dot(q0, k0)
                score = tl.dot(q1, k1, score_acc) * scale_log2
        else:
            score = tl.dot(qv, kval) * scale_log2
        if TRANSPOSE_QK:
            score = tl.where(
                (keys[:, None] <= rows[None, :]) & (keys[:, None] < T), score, -float("inf")
            )
            new_max = tl.maximum(maximum, tl.max(score, 0))
            alpha = tl.exp2(maximum - new_max)
            probability = tl.exp2(score - new_max[None, :])
            denominator = denominator * alpha + tl.sum(probability, 0)
        else:
            score = tl.where(
                (keys[None, :] <= rows[:, None]) & (keys[None, :] < T), score, -float("inf")
            )
            new_max = tl.maximum(maximum, tl.max(score, 1))
            alpha = tl.exp2(maximum - new_max)
            probability = tl.exp2(score - new_max[:, None])
            denominator = denominator * alpha + tl.sum(probability, 1)
        if TRANSPOSE_QK:
            acc = _wall_attn_pv_split32(
                tl.trans(value), probability.to(value.dtype), acc * alpha[None, :]
            )
        elif TRANSPOSE_PV:
            acc = tl.dot(
                tl.trans(value), tl.trans(probability.to(value.dtype)), acc * alpha[None, :]
            )
        elif SPLIT_QK:
            acc = tl.dot(probability.to(value.dtype), value, acc * alpha[:, None])
        else:
            acc = acc * alpha[:, None] + tl.dot(probability.to(value.dtype), value)
        maximum = new_max
    if TRANSPOSE_PV:
        result = tl.trans(acc) / denominator[:, None]
    else:
        result = acc / denominator[:, None]
    tl.store(
        output + (rows[:, None] * HQ + head) * D + dims[None, :],
        result,
        (rows[:, None] < T) & (dims[None, :] < D),
    )
    tl.store(lse + rows * HQ + head, maximum + tl.log2(denominator), rows < T)


def parallel_wall_attn_fwd(q_cache, k_cache, v, output, lse, scale=None):
    _, hq, t, d = q_cache.shape
    if scale is None:
        scale = d**-0.5
    return _wall_attn_consumer[(triton.cdiv(t, 64), hq)](
        q_cache,
        k_cache,
        v,
        output,
        lse,
        scale * RCP_LN2,
        T=t,
        HQ=hq,
        H=v.shape[2],
        D=d,
        BM=64,
        BN=64,
        BD=triton.next_power_of_2(d),
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def parallel_wall_attn_fwd_kernel_preprocess(
    q,
    k,
    g,
    q_cache,
    k_cache,
    anchor,
    unsafe,
    T_VALUE: tl.constexpr,
    HQ_VALUE: tl.constexpr,
    H_VALUE: tl.constexpr,
    G_VALUE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE: tl.constexpr,
):
    batch = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < HQ_VALUE * D_VALUE

    total = tl.zeros((1, BLOCK_C), tl.float32)
    first = tl.zeros((1, BLOCK_C), tl.float32)
    for tile in range(0, tl.cdiv(T_VALUE, BLOCK_T)):
        tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        qg_offsets = (
            batch * T_VALUE * HQ_VALUE * D_VALUE
            + tokens[:, None] * HQ_VALUE * D_VALUE
            + channels[None, :]
        )
        mask = (tokens[:, None] < T_VALUE) & channel_mask[None, :]
        values = tl.load(g + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(values, axis=0)[None, :]
        if tile == 0:
            first += tl.sum(tl.where(tokens[:, None] == 0, values, 0.0), axis=0)[None, :]
    reference = 0.5 * (first + total) * SCALE
    tl.store(
        anchor + batch * HQ_VALUE * D_VALUE + channels[None, :],
        reference,
        mask=channel_mask[None, :],
    )

    carry = tl.zeros((1, BLOCK_C), tl.float32)
    invalid = tl.full((), False, tl.int1)
    query_heads = channels // D_VALUE
    dims = channels - query_heads * D_VALUE
    kv_heads = query_heads // G_VALUE
    for tile in range(0, tl.cdiv(T_VALUE, BLOCK_T)):
        tokens = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        qg_offsets = (
            batch * T_VALUE * HQ_VALUE * D_VALUE
            + tokens[:, None] * HQ_VALUE * D_VALUE
            + channels[None, :]
        )
        k_offsets = (
            batch * T_VALUE * H_VALUE * D_VALUE
            + tokens[:, None] * H_VALUE * D_VALUE
            + kv_heads[None, :] * D_VALUE
            + dims[None, :]
        )
        mask = (tokens[:, None] < T_VALUE) & channel_mask[None, :]
        values = tl.load(g + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        raw_prefix = tl.cumsum(values, axis=0) + carry
        prefix = raw_prefix * SCALE
        q_value = tl.load(q + qg_offsets, mask=mask, other=0.0).to(tl.float32)
        k_value = tl.load(k + k_offsets, mask=mask, other=0.0).to(tl.float32)
        q_operand = q_value * tl.exp2(prefix - reference)
        k_operand = k_value * tl.exp2(reference - prefix)
        q_stored = q_operand.to(q_cache.dtype.element_ty)
        k_stored = k_operand.to(k_cache.dtype.element_ty)
        # Global-gauge operands use BF16. Bound the FP32 exp2 argument and
        # reject overflow or nonzero values lost to cache underflow. Substitutes
        # and exponent clipping would change the accepted attention operation.
        valid = (
            (values <= 0.0)
            & (tl.abs(prefix - reference) <= 125.0)
            & (tl.abs(q_stored.to(tl.float32)) < float("inf"))
            & (tl.abs(k_stored.to(tl.float32)) < float("inf"))
            & ((q_value == 0.0) | (q_stored != 0.0))
            & ((k_value == 0.0) | (k_stored != 0.0))
        )
        invalid |= tl.sum(tl.sum((mask & ~valid).to(tl.int32), axis=0), axis=0) > 0
        cache_offsets = (
            (batch * HQ_VALUE + query_heads[None, :]) * T_VALUE * D_VALUE
            + tokens[:, None] * D_VALUE
            + dims[None, :]
        )
        tl.store(q_cache + cache_offsets, q_stored, mask=mask)
        tl.store(k_cache + cache_offsets, k_stored, mask=mask)
        carry += tl.sum(values, axis=0)[None, :]
    if invalid:
        tl.atomic_or(unsafe, 1, sem="relaxed")


def _prepare_fused_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe, bt=128, bc=8, warps=4):
    b, t, hq, d = q.shape
    h = k.shape[2]
    return parallel_wall_attn_fwd_kernel_preprocess[(b, triton.cdiv(hq * d, bc))](
        q,
        k,
        g,
        q_cache,
        k_cache,
        anchor,
        unsafe,
        T_VALUE=t,
        HQ_VALUE=hq,
        H_VALUE=h,
        G_VALUE=hq // h,
        D_VALUE=d,
        BLOCK_T=bt,
        BLOCK_C=bc,
        SCALE=RCP_LN2,
        num_warps=warps,
        num_stages=1,
    )


@triton.jit
def _wall_attn_anchor_bounds_kernel(
    Q,
    K,
    P,
    L,
    U,
    T: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    tile = tl.program_id(0)
    hd = tl.program_id(1)
    head = hd // tl.cdiv(D, BD)
    db = hd % tl.cdiv(D, BD)
    t = tile * BT + tl.arange(0, BT)
    d = db * BD + tl.arange(0, BD)
    mask = (t[:, None] < T) & (d[None, :] < D)
    q = tl.load(Q + (t[:, None] * HQ + head) * D + d[None, :], mask, other=0).to(tl.float32)
    k = tl.load(K + (t[:, None] * H + head // (HQ // H)) * D + d[None, :], mask, other=0).to(
        tl.float32
    )
    p = tl.load(P + (t[:, None] * HQ + head) * D + d[None, :], mask, other=0)
    ql = tl.log2(tl.abs(q))
    kl = tl.log2(tl.abs(k))
    lo = tl.maximum(
        p - 125.0,
        tl.maximum(
            tl.where(q != 0, ql + p - 126.0, -float("inf")),
            tl.where(k != 0, p - kl - 125.0, -float("inf")),
        ),
    )
    hi = tl.minimum(
        p + 125.0,
        tl.minimum(
            tl.where(q != 0, ql + p + 125.0, float("inf")),
            tl.where(k != 0, p - kl + 126.0, float("inf")),
        ),
    )
    lo = tl.max(tl.where(mask, lo, -float("inf")), axis=0)
    hi = tl.min(tl.where(mask, hi, float("inf")), axis=0)
    off = (tile * HQ + head) * D + d
    tl.store(L + off, lo, d < D)
    tl.store(U + off, hi, d < D)


@triton.jit
def _wall_attn_anchor_reduce_kernel(
    L, U, A, D: tl.constexpr, HQ: tl.constexpr, NT: tl.constexpr, BN: tl.constexpr
):
    hd = tl.program_id(0)
    n = tl.arange(0, BN)
    d = tl.arange(0, 32)
    ch = hd * 32 + d
    off = n[:, None] * HQ * D + ch[None, :]
    lo = tl.max(
        tl.load(L + off, (n[:, None] < NT) & (ch[None, :] < HQ * D), other=-float("inf")), axis=0
    )
    hi = tl.min(
        tl.load(U + off, (n[:, None] < NT) & (ch[None, :] < HQ * D), other=float("inf")), axis=0
    )
    # Bounds propose an anchor; the cache guard must still validate actual
    # representability, including when conservative bounds cross.
    tl.store(A + ch, (lo + hi) * 0.5, ch < HQ * D)


@triton.jit
def _wall_attn_anchor_kernel(
    prefix, anchor, T: tl.constexpr, HQ: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr
):
    bhq = tl.program_id(0)
    block_d = tl.program_id(1)
    batch = bhq // HQ
    head = bhq % HQ
    dims = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dims < D
    first = tl.load(
        prefix + ((batch * T) * HQ + head) * D + dims,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    last = tl.load(
        prefix + ((batch * T + T - 1) * HQ + head) * D + dims,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(anchor + bhq * D + dims, 0.5 * (first + last), mask=mask)


@triton.jit
def _wall_attn_cache_kernel(
    q,
    k,
    g,
    prefix,
    anchor,
    q_cache,
    k_cache,
    unsafe,
    T: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_t = tl.program_id(0)
    block_d = tl.program_id(1)
    bhq = tl.program_id(2)
    batch = bhq // HQ
    query_head = bhq % HQ
    kv_head = query_head // G

    tokens = block_t * BLOCK_T + tl.arange(0, BLOCK_T)
    dims = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = (tokens[:, None] < T) & (dims[None, :] < D)
    p = tl.load(
        prefix + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    reference = tl.load(
        anchor + bhq * D + dims,
        mask=dims < D,
        other=0.0,
    ).to(tl.float32)
    q_value = tl.load(
        q + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    k_value = tl.load(
        k + ((batch * T + tokens[:, None]) * H + kv_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        g + ((batch * T + tokens[:, None]) * HQ + query_head) * D + dims[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    q_operand = q_value * tl.exp2(p - reference[None, :])
    k_operand = k_value * tl.exp2(reference[None, :] - p)
    q_stored = q_operand.to(q_cache.dtype.element_ty)
    k_stored = k_operand.to(k_cache.dtype.element_ty)
    valid = (
        (gate <= 0.0)
        & (tl.abs(p - reference[None, :]) <= 125.0)
        & (tl.abs(q_stored.to(tl.float32)) < float("inf"))
        & (tl.abs(k_stored.to(tl.float32)) < float("inf"))
        & ((q_value == 0.0) | (q_stored != 0.0))
        & ((k_value == 0.0) | (k_stored != 0.0))
    )
    if tl.sum(tl.sum((mask & ~valid).to(tl.int32), axis=0), axis=0) > 0:
        tl.atomic_or(unsafe, 1, sem="relaxed")
    output_offset = ((bhq * T + tokens[:, None]) * D) + dims[None, :]
    tl.store(q_cache + output_offset, q_stored, mask=mask)
    tl.store(k_cache + output_offset, k_stored, mask=mask)


def _build_qk_cache(q, k, g, prefix, q_cache, k_cache, anchor, unsafe):
    b, t, hq, d = q.shape
    h = k.shape[2]
    # Magnitude-aware anchors avoid losing small K values in long D64 caches.
    # Bounds only propose an anchor: the cache kernel below must still validate
    # actual representability before the public API can use cached attention.
    if b == 1 and d == 64 and t > 2048:
        nt = triton.cdiv(t, 256)
        lower = torch.empty((nt, hq, d), device=q.device, dtype=torch.float32)
        upper = torch.empty_like(lower)
        _wall_attn_anchor_bounds_kernel[(nt, hq * triton.cdiv(d, 8))](
            q,
            k,
            prefix,
            lower,
            upper,
            t,
            h,
            hq,
            d,
            256,
            8,
            num_warps=4,
            num_stages=1,
        )
        _wall_attn_anchor_reduce_kernel[(triton.cdiv(hq * d, 32),)](
            lower,
            upper,
            anchor,
            d,
            hq,
            nt,
            triton.next_power_of_2(nt),
            num_warps=4,
            num_stages=1,
        )
    else:
        _wall_attn_anchor_kernel[(b * hq, triton.cdiv(d, 32))](
            prefix,
            anchor,
            T=t,
            HQ=hq,
            D=d,
            BLOCK_D=32,
            num_warps=1,
            num_stages=1,
        )
    _wall_attn_cache_kernel[(triton.cdiv(t, 64), triton.cdiv(d, 32), b * hq)](
        q,
        k,
        g,
        prefix,
        anchor,
        q_cache,
        k_cache,
        unsafe,
        T=t,
        H=h,
        HQ=hq,
        G=hq // h,
        D=d,
        BLOCK_T=64,
        BLOCK_D=32,
        num_warps=4,
        num_stages=2,
    )


def prepare_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe):
    """Construct operands, flag unsafe data, and return an existing log2 prefix."""
    if q.shape[1] <= 2048:
        _prepare_fused_qk_cache(q, k, g, q_cache, k_cache, anchor, unsafe, bt=128, bc=8, warps=4)
    else:
        prefix = chunk_global_cumsum(g, scale=RCP_LN2)
        _build_qk_cache(q, k, g, prefix, q_cache, k_cache, anchor, unsafe)
        return prefix
    return None
