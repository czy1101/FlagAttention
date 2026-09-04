"""Compatibility helpers shared by the MetaX Triton kernels."""

from __future__ import annotations

import inspect
import re

import triton
import triton.language as tl


def _triton_version() -> tuple[int, int, int]:
    parts = re.findall(r"\d+", str(getattr(triton, "__version__", "0.0.0")))
    values = [int(part) for part in parts[:3]]
    return tuple((values + [0, 0, 0])[:3])


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    """Return whether this Triton build provides the requested TLE extension."""

    if _triton_version() < (major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return True


@triton.jit
def exp2(x):
    """Base-2 exponential with fp32 computation."""

    return tl.math.exp2(x.to(tl.float32))


try:
    _SUPPORTS_AUTOTUNE_CACHE = (
        "cache_results" in inspect.signature(triton.autotune).parameters
    )
except Exception:
    _SUPPORTS_AUTOTUNE_CACHE = False


autotune_cache_kwargs = (
    {"cache_results": True} if _SUPPORTS_AUTOTUNE_CACHE else {}
)
