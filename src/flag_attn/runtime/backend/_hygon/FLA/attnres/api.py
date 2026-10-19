"""Public Hygon Attention Residuals API with selectable implementations."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from typing import Any

import torch

from .attnres import fused_attnres as fused_torch_attnres

try:
    from .fused import (
        HAS_TRITON_ATTNRES,
        fused_attnres as fused_triton_attnres,
    )
except Exception:  # Keep the PyTorch path importable if Triton is absent.
    HAS_TRITON_ATTNRES = False
    fused_triton_attnres = None


OP_NAME = "hygon_attnres"
# The Hygon public entry uses the regular Triton/HIP implementation by default.
IMPLEMENTATION = "triton"
SUPPORTS_TRITON = bool(HAS_TRITON_ATTNRES)
_TRITON_AUTO_FAILED_KEYS: set[tuple[Any, ...]] = set()


def _requested_implementation() -> str:
    value = os.environ.get("FLAG_ATTN_ATTNRES_IMPL", "triton").strip().lower()
    if value not in {"auto", "triton", "torch"}:
        raise ValueError(
            "FLAG_ATTN_ATTNRES_IMPL must be one of: auto, triton, torch"
        )
    return value


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    output_rms_bias: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Dispatch Attention Residuals to Triton first, with explicit fallback.

    ``triton`` (the default) uses the Hygon Triton/HIP kernel;
    ``FLAG_ATTN_ATTNRES_IMPL=torch`` forces the portable implementation;
    ``auto`` tries Triton once and falls back to PyTorch after a
    lowering/runtime error. ``output_rms_bias`` and ``post_scale`` are
    applied in the same output tile as the optional output RMSNorm.
    """
    kwargs = dict(
        output_rms_weight=output_rms_weight,
        rms_eps=rms_eps,
        scale=scale,
        return_weights=return_weights,
        output_rms_bias=output_rms_bias,
        post_scale=post_scale,
    )
    requested = _requested_implementation()
    if requested == "torch":
        return fused_torch_attnres(query, residuals, rms_weight, **kwargs)
    if requested == "triton":
        if not SUPPORTS_TRITON or fused_triton_attnres is None:
            raise RuntimeError(
                "Hygon AttnRes Triton path requires the Triton/HIP backend"
            )
        return fused_triton_attnres(query, residuals, rms_weight, **kwargs)

    auto_key = (
        tuple(int(x) for x in query.shape),
        str(query.dtype),
        len(residuals),
        output_rms_weight is not None,
    )
    if (
        auto_key not in _TRITON_AUTO_FAILED_KEYS
        and SUPPORTS_TRITON
        and fused_triton_attnres is not None
    ):
        try:
            return fused_triton_attnres(query, residuals, rms_weight, **kwargs)
        except RuntimeError as exc:
            _TRITON_AUTO_FAILED_KEYS.add(auto_key)
            warnings.warn(
                f"Hygon AttnRes Triton lowering failed; falling back to PyTorch: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return fused_torch_attnres(query, residuals, rms_weight, **kwargs)


def backend_info() -> dict[str, Any]:
    """Return capability metadata used by backend registries and logs."""
    return {
        "op": OP_NAME,
        "implementation": IMPLEMENTATION,
        "requested_implementation": _requested_implementation(),
        "supports_triton": SUPPORTS_TRITON,
        "supports_triton_entry": fused_triton_attnres is not None,
        "supports_return_weights": True,
        "supports_output_rms_bias": True,
        "supports_post_scale": True,
        "max_residual_sources": 16,
        "max_hidden_size": 8192,
    }


__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TRITON",
    "backend_info",
    "fused_attnres",
    "fused_torch_attnres",
    "fused_triton_attnres",
]
