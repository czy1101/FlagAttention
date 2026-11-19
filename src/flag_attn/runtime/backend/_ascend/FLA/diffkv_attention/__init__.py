# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for paged DiffKV decode attention."""

from .api import diffkv_attention

__all__ = ["diffkv_attention"]
