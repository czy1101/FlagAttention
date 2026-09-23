"""Global-gauge Q/K cache construction for the T=4096 optimized route."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def global_anchor_kernel(prefix, anchor, T: tl.constexpr, HQ: tl.constexpr,
                         D: tl.constexpr, BLOCK_D: tl.constexpr):
    bhq = tl.program_id(0)
    block_d = tl.program_id(1)
    batch = bhq // HQ
    head = bhq % HQ
    dims = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = dims < D
    first = tl.load(
        prefix + ((batch * T) * HQ + head) * D + dims,
        mask=mask, other=0.0,
    ).to(tl.float32)
    last = tl.load(
        prefix + ((batch * T + T - 1) * HQ + head) * D + dims,
        mask=mask, other=0.0,
    ).to(tl.float32)
    tl.store(anchor + bhq * D + dims, 0.5 * (first + last), mask=mask)


@triton.jit
def global_gauge_cache_kernel(
    q, k, prefix, anchor, q_cache, k_cache,
    T: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr,
    G: tl.constexpr, D: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
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
        prefix
        + ((batch * T + tokens[:, None]) * HQ + query_head) * D
        + dims[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    reference = tl.load(
        anchor + bhq * D + dims, mask=dims < D, other=0.0,
    ).to(tl.float32)
    q_value = tl.load(
        q
        + ((batch * T + tokens[:, None]) * HQ + query_head) * D
        + dims[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    k_value = tl.load(
        k
        + ((batch * T + tokens[:, None]) * H + kv_head) * D
        + dims[None, :],
        mask=mask, other=0.0,
    ).to(tl.float32)
    q_operand = q_value * tl.exp2(p - reference[None, :])
    k_operand = k_value * tl.exp2(reference[None, :] - p)
    output_offset = ((bhq * T + tokens[:, None]) * D) + dims[None, :]
    tl.store(q_cache + output_offset, q_operand, mask=mask)
    tl.store(k_cache + output_offset, k_operand, mask=mask)


def build(q, k, prefix, q_cache, k_cache, anchor):
    b, t, hq, d = q.shape
    h = k.shape[2]
    global_anchor_kernel[(b * hq, triton.cdiv(d, 32))](
        prefix, anchor, T=t, HQ=hq, D=d, BLOCK_D=32,
        num_warps=1, num_stages=1,
    )
    global_gauge_cache_kernel[
        (triton.cdiv(t, 64), triton.cdiv(d, 32), b * hq)
    ](
        q, k, prefix, anchor, q_cache, k_cache,
        T=t, H=h, HQ=hq, G=hq // h, D=d,
        BLOCK_T=64, BLOCK_D=32,
        num_warps=4, num_stages=2,
    )


__all__ = ["build"]
