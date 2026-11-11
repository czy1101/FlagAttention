# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch

from flag_attn.infllmv2.operator import InfLLMV2Config, _validate_packed_inputs

from .forward import _select_attention_blocks, dense_attention, sparse_attention


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
) -> torch.Tensor:
    if max_seqlen_k < config.dense_len:
        return dense_attention(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            softmax_scale,
            max_seqlen_k,
        )

    with torch.no_grad():
        selected = _select_attention_blocks(
            q,
            k,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            config,
            softmax_scale,
        )
    return sparse_attention(
        q,
        k,
        v,
        selected,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        config.block_size,
        softmax_scale,
        config.causal,
        _max_seqlen_k=max_seqlen_k,
        _c550_local_blocks=(config.init_blocks, config.local_blocks),
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
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        config,
        softmax_scale,
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
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        1,
        max_seqlen_k,
        config,
        softmax_scale,
    )


__all__ = ["InfLLMV2Config", "infllmv2_attention", "infllmv2_decode"]
