"""Ascend Triton forward kernel for Parallax training attention."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_attn.FLA.index import prepare_chunk_indices


@triton.jit
def _exp2(x):
    return tl.math.exp2(x.to(tl.float32))


@triton.jit(do_not_specialize=["T"])
def _parallax_forward_kernel(
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
    D: tl.constexpr,
    BD: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_t = tl.program_id(0).to(tl.int64)
    pid_bh = tl.program_id(1).to(tl.int32)
    batch = pid_bh // HQ
    hq = pid_bh % HQ
    hk = hq // G

    if IS_VARLEN:
        seq_id = tl.load(chunk_indices + pid_t * 2).to(tl.int32)
        chunk_id = tl.load(chunk_indices + pid_t * 2 + 1).to(tl.int64)
        seq_start = tl.load(cu_seqlens + seq_id).to(tl.int64)
        seq_end = tl.load(cu_seqlens + seq_id + 1).to(tl.int64)
        seq_len = seq_end - seq_start
        row_start = chunk_id * BT
        global_start = seq_start
    else:
        seq_len = T
        row_start = pid_t * BT
        global_start = batch.to(tl.int64) * T

    rows = row_start + tl.arange(0, BT).to(tl.int64)
    d = tl.arange(0, BD)
    row_mask = rows < seq_len
    feature_mask = d < D
    q_offsets = ((global_start + rows[:, None]) * HQ + hq) * D + d[None, :]
    q_mask = row_mask[:, None] & feature_mask[None, :]
    bq = tl.load(q + q_offsets, q_mask, other=0.0)
    br = tl.load(r + q_offsets, q_mask, other=0.0)

    max_score = tl.full((BT, 1), -float("inf"), tl.float32)
    denom = tl.zeros((BT, 1), tl.float32)
    correction_sum = tl.zeros((BT, 1), tl.float32)
    value_sum = tl.zeros((BT, BD), tl.float32)
    correction_value_sum = tl.zeros((BT, BD), tl.float32)
    scale_log2 = scale * 1.4426950216
    end_col = tl.minimum(row_start + BT, seq_len)

    for col_start in range(0, end_col, BS):
        cols = col_start + tl.arange(0, BS).to(tl.int64)
        kv_offsets = ((global_start + cols[:, None]) * H + hk) * D + d[None, :]
        kv_mask = (cols[:, None] < seq_len) & feature_mask[None, :]
        bk = tl.load(k + kv_offsets, kv_mask, other=0.0)
        bv = tl.load(v + kv_offsets, kv_mask, other=0.0)

        attn_mask = (rows[:, None] >= cols[None, :]) & row_mask[:, None] & (cols[None, :] < seq_len)
        if WINDOW_SIZE_LEFT >= 0:
            attn_mask = attn_mask & (cols[None, :] >= rows[:, None] - WINDOW_SIZE_LEFT + 1)
        score = tl.dot(bq, tl.trans(bk), out_dtype=tl.float32) * scale_log2
        score = tl.where(attn_mask, score, -float("inf"))
        new_max = tl.maximum(max_score, tl.max(score, axis=1, keep_dims=True))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = _exp2(max_score - safe_max)
        weight = _exp2(score - safe_max)

        denom = alpha * denom + tl.sum(weight, axis=1, keep_dims=True)
        value_sum = alpha * value_sum
        value_sum = tl.dot(weight.to(bv.dtype), bv, out_dtype=tl.float32, acc=value_sum)
        rk = tl.dot(br, tl.trans(bk), out_dtype=tl.float32)
        weighted_rk = weight * rk
        correction_sum = alpha * correction_sum + tl.sum(weighted_rk, axis=1, keep_dims=True)
        correction_value_sum = alpha * correction_value_sum
        correction_value_sum = tl.dot(weighted_rk.to(bv.dtype), bv, out_dtype=tl.float32, acc=correction_value_sum)
        max_score = new_max

    inv_denom = tl.where(row_mask[:, None], 1.0 / denom, 0.0)
    mean_value = value_sum * inv_denom
    bart_value = correction_sum * inv_denom
    output = mean_value + bart_value * mean_value - correction_value_sum * inv_denom

    out_offsets = ((global_start + rows[:, None]) * HQ + hq) * D + d[None, :]
    scalar_offsets = (global_start + rows) * HQ + hq
    out_mask = row_mask[:, None] & feature_mask[None, :]
    tl.store(o + out_offsets, output, out_mask)
    tl.store(barv + out_offsets, mean_value, out_mask)
    tl.store(d1 + scalar_offsets, tl.sum(denom, axis=1), row_mask)
    tl.store(bart + scalar_offsets, tl.sum(bart_value, axis=1), row_mask)
    tl.store(m + scalar_offsets, tl.sum(max_score, axis=1), row_mask)


def parallax_forward(
    q: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    window_size_left: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Parallax attention and return its saved forward state.

    Dense inputs use ``[B, T, H, D]``. Packed inputs use ``B == 1`` and
    ``cu_seqlens`` to reset causal/window positions at each sequence boundary.
    """
    if q.ndim != 4 or r.shape != q.shape:
        raise ValueError("q and r must have matching [B, T, HQ, D] shapes")
    if k.ndim != 4 or v.shape != k.shape or k.shape[:2] != q.shape[:2] or k.shape[-1] != q.shape[-1]:
        raise ValueError("k and v must have matching [B, T, H, D] shapes compatible with q")
    batch, time, hq, dim = q.shape
    h = k.shape[2]
    if h <= 0 or hq % h:
        raise ValueError("query head count must be divisible by a positive KV head count")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(x.dtype != q.dtype for x in (r, k, v)):
        raise TypeError("q, r, k, and v must use the same float16 or bfloat16 dtype")
    if not all(x.device == q.device for x in (r, k, v)):
        raise ValueError("q, r, k, and v must be on the same device")
    if q.device.type != "npu":
        raise ValueError("parallax_forward requires NPU tensors")
    if time <= 0:
        raise ValueError("sequence length must be positive")
    if not 16 <= dim <= 256:
        raise ValueError(f"head dimension must be in [16, 256], got {dim}")
    if window_size_left == 0 or window_size_left < -1:
        raise ValueError("window_size_left must be -1 or greater than zero")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed variable-length inputs require batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.device != q.device or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must be a one-dimensional NPU tensor with at least two entries")
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise TypeError("cu_seqlens must use int32 or int64")

    # Kernels use flattened dense offsets, matching the upstream input_guard's
    # normalization of tensor arguments at the wrapper boundary.
    q, r, k, v = (tensor.contiguous() for tensor in (q, r, k, v))
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()

    group_size = hq // h
    block_d = triton.next_power_of_2(dim)
    block_t = 16
    block_s = 32
    output = torch.empty_like(q)
    barv = torch.empty_like(q)
    d1 = torch.empty((batch, time, hq), dtype=torch.float32, device=q.device)
    bart = torch.empty_like(d1)
    m = torch.empty_like(d1)

    if cu_seqlens is None:
        query_tiles = triton.cdiv(time, block_t)
        chunk_indices = None
    else:
        chunk_indices = prepare_chunk_indices(cu_seqlens, block_t)
        query_tiles = chunk_indices.shape[0]

    _parallax_forward_kernel[(query_tiles, batch * hq)](
        q,
        r,
        k,
        v,
        output,
        barv,
        d1,
        bart,
        m,
        float(scale),
        cu_seqlens,
        chunk_indices,
        time,
        HQ=hq,
        H=h,
        G=group_size,
        D=dim,
        BD=block_d,
        BT=block_t,
        BS=block_s,
        WINDOW_SIZE_LEFT=window_size_left,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
        num_stages=1,
    )
    return output, barv, d1, bart, m
