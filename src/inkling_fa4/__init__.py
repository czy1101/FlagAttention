"""Inkling FA4 relative-attention implementations.

Triton and Hopper TLE share a single module, ``inkling_fa4.triton_kernel``.
``inkling_fa4_rel_attention`` is the unified entry point: it prefers TLE and
falls back to Triton when TLE is unavailable or unsupported for the shape.
"""

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

    if name in ("inkling_fa4_rel_attention_triton", "inkling_fa4_rel_attention_tle"):
        from .triton_kernel import (
            inkling_fa4_rel_attention_tle,
            inkling_fa4_rel_attention_triton,
        )

        return {
            "inkling_fa4_rel_attention_triton": inkling_fa4_rel_attention_triton,
            "inkling_fa4_rel_attention_tle": inkling_fa4_rel_attention_tle,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
