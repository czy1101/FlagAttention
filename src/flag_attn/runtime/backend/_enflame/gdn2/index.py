# Enflame GDN2 Triton implementation.
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license.

import torch
import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    chunk_counts = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    chunk_offsets = torch.cat([cu_seqlens.new_tensor([0]), chunk_counts]).cumsum(
        -1,
        dtype=cu_seqlens.dtype,
    )
    chunk_arange = torch.arange(
        chunk_offsets[-1], device=cu_seqlens.device, dtype=cu_seqlens.dtype
    )
    seq_ids = torch.repeat_interleave(
        torch.arange(
            chunk_counts.numel(),
            device=cu_seqlens.device,
            dtype=cu_seqlens.dtype,
        ),
        chunk_counts,
    )
    chunk_ids = chunk_arange - torch.repeat_interleave(
        chunk_offsets[:-1], chunk_counts
    )
    return torch.stack([seq_ids, chunk_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(prepare_lens(cu_seqlens), chunk_size),
        ]
    ).cumsum(-1, dtype=cu_seqlens.dtype)
