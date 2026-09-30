# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Public continuous-layout APIs for InfLLM-V2 prefill, decode, and backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import InfLLMV2Config
from .kernels import compress_k, pool_scores, select_blocks, sparse_attention, stage1


@dataclass
class _InfLLMV2DebugInfo:
    """Intermediate tensors exposed only through the private test/debug API."""

    path: str
    k1: torch.Tensor | None = None
    k2: torch.Tensor | None = None
    cu_seqlens_k1: torch.Tensor | None = None
    cu_seqlens_k2: torch.Tensor | None = None
    token_score: torch.Tensor | None = None
    block_score: torch.Tensor | None = None
    selected_blocks: torch.Tensor | None = None


def _validate_packed_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must have packed [total_tokens, heads, head_dim] layout")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("K/V shapes must match and all head dimensions must match")
    if q.shape[1] <= 0 or k.shape[1] <= 0:
        raise ValueError("head counts must be positive")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("Q/K/V must have the same dtype")
    if q.shape[1] % k.shape[1]:
        raise ValueError("number of query heads must be divisible by KV heads")
    if q.shape[1] // k.shape[1] > 32:
        raise ValueError("the selector supports at most 32 query heads per KV head")
    if not 16 <= q.shape[-1] <= 256:
        raise ValueError("head dimension must be between 16 and 256")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise TypeError("cu_seqlens_q and cu_seqlens_k must be int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_k.ndim != 1:
        raise ValueError("cumulative lengths must be one-dimensional")
    if cu_seqlens_q.numel() < 2 or cu_seqlens_k.numel() < 2:
        raise ValueError("at least one nonempty sequence is required")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("Q and KV must have the same batch size")
    if not (q.device == k.device == v.device == cu_seqlens_q.device == cu_seqlens_k.device):
        raise ValueError("all inputs must be on the same device")
    if int(cu_seqlens_q[-1].item()) != q.shape[0] or int(cu_seqlens_k[-1].item()) != k.shape[0]:
        raise ValueError("the final cumulative length must equal the packed token count")
    if max_seqlen_q <= 0 or max_seqlen_k <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    q_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    if int(q_lengths.max().item()) != max_seqlen_q or int(k_lengths.max().item()) != max_seqlen_k:
        raise ValueError("max_seqlen_q/max_seqlen_k must match the packed lengths")
    if int(cu_seqlens_q[0].item()) != 0 or int(cu_seqlens_k[0].item()) != 0:
        raise ValueError("cumulative lengths must start at zero")
    if bool(torch.any(q_lengths <= 0).item()) or bool(torch.any(k_lengths <= 0).item()):
        raise ValueError("each packed Q/KV sequence must be nonempty")
    if bool(torch.any(q_lengths > k_lengths).item()):
        raise ValueError("this continuous-cache API requires every Q length to be <= its KV length")


def _dense_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float | None,
) -> torch.Tensor:
    """Reuse FlagAttention FlashAttention for each continuous sequence."""
    from flag_attn.flash import attention as flash_attention

    chunks: list[torch.Tensor] = []
    q_cu = cu_seqlens_q.detach().cpu().tolist()
    k_cu = cu_seqlens_k.detach().cpu().tolist()
    for qs, qe, ks, ke in zip(q_cu[:-1], q_cu[1:], k_cu[:-1], k_cu[1:]):
        q_seq = q[qs:qe].transpose(0, 1).unsqueeze(0)
        k_seq = k[ks:ke].transpose(0, 1).unsqueeze(0)
        v_seq = v[ks:ke].transpose(0, 1).unsqueeze(0)
        out = flash_attention(q_seq, k_seq, v_seq, causal=True, sm_scale=softmax_scale)
        chunks.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(chunks, dim=0)


def _run_sparse_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
    collect_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, _InfLLMV2DebugInfo]:
    # Top-k selection is discrete. Gradients flow through Stage-2 Q/K/V only.
    with torch.no_grad():
        k1, cu_k1 = compress_k(k, cu_seqlens_k, config.k1_kernel_size, config.k1_stride)
        k2, cu_k2 = compress_k(k, cu_seqlens_k, config.k2_kernel_size, config.k2_stride)
        token_score = stage1(
            q, k1, k2, cu_seqlens_q, cu_k1, cu_k2, max_seqlen_q,
            config.k1_stride, config.k2_stride, softmax_scale, config.causal, cu_seqlens_k,
        )
        block_score = pool_scores(
            token_score, cu_seqlens_q, cu_k1, max_seqlen_q,
            config.k1_kernel_size, config.k1_stride, config.block_size,
            config.init_blocks, config.local_blocks, cu_seqlens_k, max_seqlen_k,
            _valid_only=not collect_debug,
        )
        selected = select_blocks(
            block_score, cu_seqlens_q, config.topk, config.block_size, cu_seqlens_k
        )
    out = sparse_attention(
        q, k, v, selected, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
        config.block_size, softmax_scale, config.causal,
    )
    if not collect_debug:
        return out
    return out, _InfLLMV2DebugInfo("sparse", k1, k2, cu_k1, cu_k2, token_score, block_score, selected)


def _dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
    collect_debug: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, _InfLLMV2DebugInfo]:
    # Batch-wide decision from the LONGEST per-sequence KV, never sum(KV).
    # Match the model's strict boundary: KV < dense_len is dense; equality sparse.
    if max_seqlen_k < config.dense_len:
        out = _dense_packed(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
        return (out, _InfLLMV2DebugInfo("dense")) if collect_debug else out
    return _run_sparse_pipeline(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        config, softmax_scale, collect_debug,
    )


def _dispatch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
) -> tuple[torch.Tensor, _InfLLMV2DebugInfo]:
    """Slow PyTorch oracle used only by the private test/debug entry points."""
    from .reference import (
        compress_k_ref,
        dense_attention_ref,
        pool_scores_ref,
        select_blocks_ref,
        sparse_attention_ref,
        stage1_ref,
    )

    if max_seqlen_k < config.dense_len:
        out = dense_attention_ref(
            q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, config.causal
        )
        return out, _InfLLMV2DebugInfo("dense")
    with torch.no_grad():
        k1, cu_k1 = compress_k_ref(k, cu_seqlens_k, config.k1_kernel_size, config.k1_stride)
        k2, cu_k2 = compress_k_ref(k, cu_seqlens_k, config.k2_kernel_size, config.k2_stride)
        token_score = stage1_ref(
            q, k1, k2, cu_seqlens_q, cu_k1, cu_k2,
            config.k1_stride, config.k2_stride, softmax_scale, config.causal, cu_seqlens_k,
        )
        block_score = pool_scores_ref(
            token_score, cu_seqlens_q, cu_k1, max_seqlen_q,
            config.k1_kernel_size, config.k1_stride, config.block_size,
            config.init_blocks, config.local_blocks, cu_seqlens_k, max_seqlen_k,
        )
        selected = select_blocks_ref(
            block_score, cu_seqlens_q, config.topk, config.block_size, cu_seqlens_k
        )
    out = sparse_attention_ref(
        q, k, v, selected, cu_seqlens_q, cu_seqlens_k,
        config.block_size, softmax_scale, config.causal,
    )
    return out, _InfLLMV2DebugInfo(
        "sparse", k1, k2, cu_k1, cu_k2, token_score, block_score, selected
    )


def infllmv2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Continuous packed full, prefix-cache, or chunked causal prefill.

    Each Q sequence is the suffix of its KV sequence: query i has absolute
    position ``Lkv - Lq + i``. KV must include the cached prefix AND the current
    query chunk. Pass only the current Q chunk; previously computed outputs
    are not recomputed. The caller manages cache append/storage across calls.
    Unequal per-sample lengths and gradients through all supplied Q/K/V are
    supported. No paged cache or in-place cache updates are performed here.
    """
    config = config or InfLLMV2Config()
    config.validate()
    _validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    return _dispatch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        config, softmax_scale,
    )


def infllmv2_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """One-token decode over a continuous packed KV cache.

    ``q`` is ``[batch, Hq, D]``. ``k_cache`` and ``v_cache`` are packed
    ``[total_cached_tokens, Hkv, D]`` and already include the current token.
    This API intentionally excludes paged KV and in-place cache updates.
    """
    config = config or InfLLMV2Config()
    config.validate()
    batch = cu_seqlens_k.numel() - 1
    if q.ndim != 3 or q.shape[0] != batch:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    _validate_packed_inputs(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k)
    q, k_cache, v_cache = q.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    return _dispatch(
        q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k,
        config, softmax_scale,
    )


def _infllmv2_attention_debug(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
    backend: str = "triton",
) -> tuple[torch.Tensor, _InfLLMV2DebugInfo]:
    """Private correctness/debug entry point; never used by production timing."""
    config = config or InfLLMV2Config()
    config.validate()
    _validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    if backend == "reference":
        return _dispatch_reference(
            q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            config, softmax_scale,
        )
    if backend != "triton":
        raise ValueError("debug backend must be 'triton' or 'reference'")
    return _dispatch(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        config, softmax_scale, collect_debug=True,
    )


def _infllmv2_decode_debug(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_k: int,
    *,
    config: InfLLMV2Config | None = None,
    softmax_scale: float | None = None,
    backend: str = "triton",
) -> tuple[torch.Tensor, _InfLLMV2DebugInfo]:
    """Private decode counterpart of :func:`_infllmv2_attention_debug`."""
    config = config or InfLLMV2Config()
    config.validate()
    batch = cu_seqlens_k.numel() - 1
    if q.ndim != 3 or q.shape[0] != batch:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    _validate_packed_inputs(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k)
    q, k_cache, v_cache = q.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    cu_seqlens_q, cu_seqlens_k = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    if backend == "reference":
        return _dispatch_reference(
            q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k,
            config, softmax_scale,
        )
    if backend != "triton":
        raise ValueError("debug backend must be 'triton' or 'reference'")
    return _dispatch(
        q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, 1, max_seqlen_k,
        config, softmax_scale, collect_debug=True,
    )
