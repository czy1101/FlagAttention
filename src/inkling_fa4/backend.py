"""Backend selection for Inkling FA4 relative attention.

Triton and Hopper TLE live in a single module, :mod:`inkling_fa4.triton_kernel`,
which owns the ``if/else`` backend choice. This module is a thin compatibility
layer: it resolves a backend name to the matching callable and forwards the
public operator to the shared dispatcher.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Callable

_KERNEL_MODULE = "inkling_fa4.triton_kernel"

_ALIASES = {
    "auto": "auto",
    "tle": "tle",
    "triton_tle": "tle",
    "triton-tle": "tle",
    "triton": "triton",
    "base": "triton",
}


def _kernel() -> Any:
    return import_module(_KERNEL_MODULE)


def tle_available() -> tuple[bool, str]:
    """Return whether the Hopper TLE path can be imported, plus a reason."""
    module = _kernel()
    if module.TLE_AVAILABLE:
        return True, ""
    return False, f"{type(module.TLE_IMPORT_ERROR).__name__}: {module.TLE_IMPORT_ERROR}"


def _load_tle() -> Callable[..., Any]:
    module = _kernel()
    if not module.TLE_AVAILABLE:
        raise ImportError(
            f"triton.experimental.tle unavailable: {module.TLE_IMPORT_ERROR}"
        )
    return module.inkling_fa4_rel_attention_tle


def _load_triton() -> Callable[..., Any]:
    return _kernel().inkling_fa4_rel_attention_triton


def get_backend(name: str | None = None) -> Callable[..., Any]:
    """Return an implementation without launching the operator.

    Raises ``ImportError`` when ``tle`` is requested but
    ``triton.experimental.tle`` cannot be imported, so callers can fall back or
    skip. ``auto`` resolves to TLE when available and Triton otherwise.
    """
    requested = (name or os.getenv("INKLING_FA4_BACKEND", "auto")).lower()
    try:
        selected = _ALIASES[requested]
    except KeyError as exc:
        choices = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"unknown Inkling FA4 backend={requested!r}; choose from {choices}"
        ) from exc

    if selected == "tle":
        return _load_tle()
    if selected == "triton":
        return _load_triton()

    try:
        return _load_tle()
    except ImportError:
        return _load_triton()


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to the TLE or Triton implementation.

    Select with ``backend=...`` or ``INKLING_FA4_BACKEND``. The keyword is
    consumed here and is not forwarded to the kernel wrapper.
    """
    return _kernel().inkling_fa4_rel_attention(*args, backend=backend, **kwargs)


__all__ = [
    "get_backend",
    "inkling_fa4_rel_attention",
    "tle_available",
]
