# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""T-Head launch policy for the shared Triton MoBA attention kernels."""

import torch
import triton

from flag_attn.FLA.moba.attention import (
    _fused_merge_softmax_kernel as _merge_autotuner,
    _routed_gqa_attn_fwd_kernel,
    _varlen_attn_bwd_dkdv_kernel,
    _varlen_attn_bwd_dq_kernel,
    _varlen_attn_fwd_kernel,
)
from flag_attn.FLA.moba.routing import _copy_triton, cdiv

# PPU uses a fixed merge tile instead of the NVIDIA autotuning key.
_fused_merge_softmax_kernel = _merge_autotuner.fn


def _attention_launch_config(
    head_dim: int,
    device: torch.device,
) -> tuple[int, int, int, int, int]:
    """Use ordinary dot tiles on PPU for forward and backward attention."""
    block_d = max(16, triton.next_power_of_2(head_dim))
    if head_dim <= 128:
        return 64, 64, block_d, 4, 1
    return 32, 32, block_d, 4, 1


def _triton_varlen_forward(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"q heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})")
    group_size = num_heads // num_kv_heads
    num_sequences = cu_seqlens_q.numel() - 1
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    lse = torch.empty((num_heads, total_q), device=q.device, dtype=torch.float32)
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    num_blocks = cdiv(max_seqlen_q, block_m)
    grid = (num_blocks * num_sequences * num_heads,)
    _varlen_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        num_heads=num_heads,
        GROUP_SIZE=group_size,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_K=max_seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, lse


def _triton_varlen_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
):
    _, num_heads, head_dim = q.shape
    num_sequences = cu_seqlens_q.numel() - 1
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    # Keep both backward kernels within the 64x64 attention tile policy.
    block_m = min(block_m, 64)
    block_n = min(block_n, 64)
    common = (
        q,
        k,
        v,
        out,
        dout,
        softmax_lse,
    )
    strides = (
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        dout.stride(0),
        dout.stride(1),
        dout.stride(2),
        softmax_lse.stride(0),
        softmax_lse.stride(1),
    )
    num_q_blocks = cdiv(max_seqlen_q, block_m)
    dq_grid = (num_q_blocks * num_sequences * num_heads,)
    _varlen_attn_bwd_dq_kernel[dq_grid](
        *common,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        *strides,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        num_heads=num_heads,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_K=max_seqlen_k,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_q_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    num_k_blocks = cdiv(max_seqlen_k, block_n)
    dkdv_grid = (num_k_blocks * num_sequences * num_heads,)
    _varlen_attn_bwd_dkdv_kernel[dkdv_grid](
        *common,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        *strides,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        num_heads=num_heads,
        softmax_scale=softmax_scale,
        HEAD_DIM=head_dim,
        MAX_SEQLEN_Q=max_seqlen_q,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        NUM_BLOCKS=num_k_blocks,
        CAUSAL=causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )


class _TritonVarlenAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, causal):
        scale = q.shape[-1] ** -0.5
        out, lse = _triton_varlen_forward(q, k, v, cu_q, cu_k, max_q, max_k, scale, causal)
        ctx.save_for_backward(q, k, v, out, lse, cu_q, cu_k)
        ctx.max_q, ctx.max_k, ctx.scale, ctx.causal = max_q, max_k, scale, causal
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse, cu_q, cu_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _triton_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu_q,
            cu_k,
            ctx.max_q,
            ctx.max_k,
            ctx.scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None, None, None, None


def _triton_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal):
    return _TritonVarlenAttention.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
    )


def _routed_sparse_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_indices: torch.Tensor,
    expert_offsets: torch.Tensor,
    descriptor_experts: torch.Tensor,
    descriptor_starts: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_chunks_per_sequence: int,
    chunk_size: int,
):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    head_dim = q.shape[2]
    block_m, block_n, block_d, num_warps, num_stages = _attention_launch_config(head_dim, q.device)
    route_out = torch.empty((query_indices.numel(), head_dim), device=q.device, dtype=q.dtype)
    route_lse = torch.empty((query_indices.numel(),), device=q.device, dtype=torch.float32)
    _routed_gqa_attn_fwd_kernel[(descriptor_experts.numel(),)](
        q,
        k,
        v,
        query_indices,
        expert_offsets,
        descriptor_experts,
        descriptor_starts,
        cu_seqlens,
        route_out,
        route_lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        route_out.stride(0),
        route_out.stride(1),
        q.shape[-1] ** -0.5,
        NUM_Q_HEADS=num_q_heads,
        GROUP_SIZE=group_size,
        MAX_CHUNKS_PER_SEQUENCE=max_chunks_per_sequence,
        CHUNK_SIZE=chunk_size,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return route_out, route_lse


def _fused_merge_softmax_triton(
    self_out: torch.Tensor,
    self_lse: torch.Tensor,
    moba_out: torch.Tensor,
    moba_lse: torch.Tensor,
    route_positions: torch.Tensor,
    final_out: torch.Tensor,
    final_lse: torch.Tensor,
):
    """Wrapper for the fused merge kernel"""
    num_rows, head_dim = self_out.shape
    topk = route_positions.shape[0]

    if topk == 0 or moba_out.numel() == 0:
        _copy_triton(self_out, final_out)
        _copy_triton(self_lse, final_lse)
        return

    block_topk = triton.next_power_of_2(topk)

    grid = (num_rows,)

    _fused_merge_softmax_kernel[grid](
        self_out,
        self_lse,
        moba_out,
        moba_lse,
        route_positions,
        final_out,
        final_lse,
        num_rows,
        head_dim,
        self_out.stride(0),
        self_out.stride(1),
        self_lse.stride(0),
        self_lse.stride(1),
        moba_out.stride(0),
        moba_out.stride(1),
        route_positions.stride(0),
        route_positions.stride(1),
        final_out.stride(0),
        final_out.stride(1),
        final_lse.stride(0),
        final_lse.stride(1),
        num_heads=self_lse.shape[0],
        TOPK=block_topk,
        NUM_ROUTES=topk,
        ROW_BUCKET=0,
        GPU_ARCH=0,
        BLOCK_D=64,
        num_warps=4,
        num_stages=1,
    )
