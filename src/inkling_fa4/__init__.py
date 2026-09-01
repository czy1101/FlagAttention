"""Inkling FA4 relative-attention implementations."""

from __future__ import annotations

from typing import Any

__all__ = [
    "inkling_fa4_rel_attention",
    "inkling_fa4_rel_attention_triton",
    "inkling_fa4_rel_attention_tle",
]


def __getattr__(name: str) -> Any:
    if name == "inkling_fa4_rel_attention":
        from .backend import inkling_fa4_rel_attention

        return inkling_fa4_rel_attention

    if name == "inkling_fa4_rel_attention_triton":
        from .triton_kernel import inkling_fa4_rel_attention_triton

        return inkling_fa4_rel_attention_triton

    if name == "inkling_fa4_rel_attention_tle":
        from .triton_tle_kernel import inkling_fa4_rel_attention_tle

        return inkling_fa4_rel_attention_tle

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")