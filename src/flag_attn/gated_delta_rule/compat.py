"""Small compatibility helpers for the vendored GDN Triton kernels."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import triton


F = TypeVar("F", bound=Callable[..., Any])


def libentry() -> Callable[[F], F]:
    """Keep FlagGems kernel declarations usable without its runtime wrapper."""

    def decorator(fn: F) -> F:
        return fn

    return decorator


def libtuner(
    *,
    configs: Sequence[triton.Config],
    key: Sequence[str],
    use_cuda_graph: bool = False,
    **kwargs: Any,
):
    """Map FlagGems autotuning metadata to Triton's native autotuner."""

    del use_cuda_graph
    return triton.autotune(configs=list(configs), key=list(key), **kwargs)


def _triton_version() -> tuple[int, int, int]:
    parts = re.findall(r"\d+", str(getattr(triton, "__version__", "0.0.0")))
    values = [int(part) for part in parts[:3]]
    return tuple((values + [0, 0, 0])[:3])


def has_triton_tle(major: int = 0, minor: int = 0, patch: int = 0) -> bool:
    if _triton_version() < (major, minor, patch):
        return False
    try:
        import triton.experimental.tle.language  # noqa: F401
    except ImportError:
        return False
    return True
