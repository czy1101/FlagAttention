# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Shared utilities for the S60 MSA backend."""

from flag_attn.minimax_sparse_attention.utils import (
    current_platform,
    has_triton_tle,
    round_up,
)

__all__ = [
    "current_platform",
    "has_triton_tle",
    "round_up",
]
