"""Unchanged bounded scalar-threshold cache; no attention fallback."""
from functools import lru_cache
import torch

@lru_cache(maxsize=128)
def _get_cached_scalar_adaptive_threshold(
    device_index: int,
    B: int,
    H: int,
    value: float,
):
    """Create one reusable broadcast threshold view.

    v5.7a: cached scalar adaptive threshold.

    The returned tensor has shape (B, H), dtype FP32 and zero
    strides, matching torch.as_tensor(value).broadcast_to((B, H)).
    """
    scalar = torch.tensor(
        value,
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )

    return torch.broadcast_to(
        scalar,
        (B, H),
    )
