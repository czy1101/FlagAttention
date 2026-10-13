"""Small hardware-compatibility layer for the Hygon Parallax port.

The upstream Parallax implementation contains a few CUDA/Hopper capability
queries.  Keeping those queries here avoids importing the generic FLA package
and, importantly, makes all NVIDIA/TLE-only fast paths opt-in on Hygon rather
than accidentally selected by a device-property fallback.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton


@lru_cache(maxsize=None)
def get_num_sms(device_index: int | None = None) -> int:
    """Return the Hygon CU/SM count when the driver exposes it."""
    if not torch.cuda.is_available():
        return 0
    try:
        props = torch.cuda.get_device_properties(device_index)
        return int(
            getattr(props, "multi_processor_count", None)
            or getattr(props, "multiProcessorCount", None)
            or 0
        )
    except (AssertionError, RuntimeError, TypeError, AttributeError):
        return 0


@lru_cache(maxsize=None)
def has_shared_memory(required_bytes: int, device_index: int | None = None) -> bool:
    """Check the active backend's reported LDS/shared-memory capacity."""
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(device_index)
        value = props.get("max_shared_mem") if hasattr(props, "get") else None
        return value is not None and int(value) >= int(required_bytes)
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
        return False


def is_cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except (AssertionError, RuntimeError):
        return False


def is_nvidia_hopper_device(device_index: int | None = None) -> bool:
    """Hygon is deliberately never treated as an NVIDIA Hopper device."""
    return False


def is_nvidia_blackwell_device(device_index: int | None = None) -> bool:
    """Hygon is deliberately never treated as an NVIDIA Blackwell device."""
    return False
