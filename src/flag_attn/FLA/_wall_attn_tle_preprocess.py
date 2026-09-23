"""A35-L6: generalized fused scan/QK preprocessing for MHA and GQA."""

from __future__ import annotations

import triton
import triton.language as tl


RCP_LN2 = 1.4426950216


@triton.jit
def fused_scan_qk_gqa_kernel(
    q, k, g, q_cache, k_cache, anchor,
    T_VALUE: tl.constexpr, HQ_VALUE: tl.constexpr, H_VALUE: tl.constexpr,
    G_VALUE: tl.constexpr, D_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_C: tl.constexpr, SCALE: tl.constexpr,
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
            first += tl.sum(
                tl.where(tokens[:, None] == 0, values, 0.0), axis=0
            )[None, :]
    reference = 0.5 * (first + total) * SCALE
    tl.store(
        anchor + batch * HQ_VALUE * D_VALUE + channels[None, :],
        reference,
        mask=channel_mask[None, :],
    )

    carry = tl.zeros((1, BLOCK_C), tl.float32)
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
        cache_offsets = (
            (batch * HQ_VALUE + query_heads[None, :]) * T_VALUE * D_VALUE
            + tokens[:, None] * D_VALUE
            + dims[None, :]
        )
        tl.store(q_cache + cache_offsets, q_operand, mask=mask)
        tl.store(k_cache + cache_offsets, k_operand, mask=mask)
        carry += tl.sum(values, axis=0)[None, :]


def launch(q, k, g, q_cache, k_cache, anchor, bt=128, bc=8, warps=4):
    b, t, hq, d = q.shape
    h = k.shape[2]
    return fused_scan_qk_gqa_kernel[(b, triton.cdiv(hq * d, bc))](
        q, k, g, q_cache, k_cache, anchor,
        T_VALUE=t, HQ_VALUE=hq, H_VALUE=h, G_VALUE=hq // h, D_VALUE=d,
        BLOCK_T=bt, BLOCK_C=bc, SCALE=RCP_LN2,
        num_warps=warps, num_stages=1,
    )


__all__ = ["launch", "fused_scan_qk_gqa_kernel"]
