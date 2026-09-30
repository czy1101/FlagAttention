# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InfLLMV2Config:
    """Configuration for continuous-packed causal InfLLM-V2 attention."""

    k1_kernel_size: int = 32
    k1_stride: int = 16
    k2_kernel_size: int = 128
    k2_stride: int = 64
    block_size: int = 64
    topk: int = 64
    init_blocks: int = 1
    local_blocks: int = 32
    dense_len: int = 8192  # Exclusive dense boundary; 0 means always sparse.
    causal: bool = True

    def validate(self) -> None:
        if self.k1_kernel_size <= 0 or self.k2_kernel_size <= 0:
            raise ValueError("compression kernel sizes must be positive")
        if self.k1_stride <= 0 or self.k2_stride <= 0 or self.block_size <= 0:
            raise ValueError("strides and block_size must be positive")
        if self.block_size % self.k1_stride:
            raise ValueError("block_size must be divisible by k1_stride")
        if self.topk <= 0:
            raise ValueError("topk must be positive")
        if self.topk > 2048 or self.topk & (self.topk - 1):
            raise ValueError("topk must be a power of two no greater than 2048")
        if self.init_blocks < 0 or self.local_blocks < 0:
            raise ValueError("init_blocks and local_blocks must be non-negative")
        if self.topk < self.init_blocks + self.local_blocks + 1:
            raise ValueError("topk must fit all forced init and local blocks")
        if self.dense_len < 0:
            raise ValueError("dense_len must be non-negative")
        if not self.causal:
            raise NotImplementedError("InfLLM-V2 attention currently supports causal mode only")
