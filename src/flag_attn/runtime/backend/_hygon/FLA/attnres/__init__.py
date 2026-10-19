"""Hygon Attention Residuals backend with selectable Triton/PyTorch paths."""

from .api import (
    IMPLEMENTATION,
    OP_NAME,
    SUPPORTS_TRITON,
    backend_info,
    fused_attnres,
    fused_torch_attnres,
    fused_triton_attnres,
)

__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TRITON",
    "backend_info",
    "fused_attnres",
    "fused_torch_attnres",
    "fused_triton_attnres",
]
