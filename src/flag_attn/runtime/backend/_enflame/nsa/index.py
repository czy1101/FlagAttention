import torch

from flag_attn.parallel_nsa.index import (
    prepare_chunk_indices as _generic_prepare_chunk_indices,
)
from flag_attn.parallel_nsa.index import (
    prepare_chunk_offsets as _generic_prepare_chunk_offsets,
)
from flag_attn.parallel_nsa.index import (
    prepare_lens as _generic_prepare_lens,
)
from flag_attn.parallel_nsa.index import (
    prepare_token_indices as _generic_prepare_token_indices,
)


def _to_int32(value):
    if (
        isinstance(value, torch.Tensor)
        and value.dtype != torch.int32
    ):
        return value.to(dtype=torch.int32)

    return value


def prepare_lens_enflame(
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    result = _generic_prepare_lens(
        _to_int32(cu_seqlens)
    )
    return _to_int32(result)


def prepare_token_indices_enflame(
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    result = _generic_prepare_token_indices(
        _to_int32(cu_seqlens)
    )
    return _to_int32(result)


def prepare_chunk_offsets_enflame(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    result = _generic_prepare_chunk_offsets(
        _to_int32(cu_seqlens),
        chunk_size,
    )
    return _to_int32(result)


def prepare_chunk_indices_enflame(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    result = _generic_prepare_chunk_indices(
        _to_int32(cu_seqlens),
        chunk_size,
    )
    return _to_int32(result)


__all__ = [
    "prepare_lens_enflame",
    "prepare_token_indices_enflame",
    "prepare_chunk_offsets_enflame",
    "prepare_chunk_indices_enflame",
]
