# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wall-Attention training and prefill implementations."""

from .parallel import get_last_route, parallel_wall_attn, select_route

__all__ = ["parallel_wall_attn", "select_route", "get_last_route"]
