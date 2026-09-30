# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

"""Small runtime metadata helpers shared by Triton kernels and test references."""

from __future__ import annotations

import torch


def compressed_lengths(cu_seqlens: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    counts = torch.where(lengths >= kernel_size, (lengths - kernel_size) // stride + 1, 0)
    out = torch.zeros_like(cu_seqlens)
    out[1:] = torch.cumsum(counts, dim=0)
    return out
