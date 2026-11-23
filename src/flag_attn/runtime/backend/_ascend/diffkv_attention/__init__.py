# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Ascend backend for paged DiffKV decode attention."""

__all__ = ["diffkv_attention"]


def __getattr__(name):
    if name != "diffkv_attention":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .api import diffkv_attention

    globals()[name] = diffkv_attention
    return diffkv_attention
