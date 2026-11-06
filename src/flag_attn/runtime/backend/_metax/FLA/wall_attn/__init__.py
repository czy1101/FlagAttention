# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""MetaX Wall-Attention implementation."""

from .parallel import parallel_wall_attn, select_route, get_last_route
from .decode import parallel_wall_attn_decode, build_wall_kv_cache

__all__ = [
    "parallel_wall_attn",
    "select_route",
    "get_last_route",
    "parallel_wall_attn_decode",
    "build_wall_kv_cache",
]
