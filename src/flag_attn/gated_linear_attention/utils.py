# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Runtime-independent helpers for the vendored GLA kernels."""

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Cache the eight most recent calls by tensor identity."""
    cache_entries: list[tuple[tuple, dict, Any]] = []
    cache_size = 8

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for index, (last_args, last_kwargs, last_result) in enumerate(cache_entries):
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(arg is last_arg for arg, last_arg in zip(args, last_args))
                and all(
                    key in last_kwargs and value is last_kwargs[key]
                    for key, value in kwargs.items()
                )
            ):
                cache_entries = (
                    cache_entries[:index]
                    + cache_entries[index + 1 :]
                    + [(args, kwargs, last_result)]
                )
                return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper


def input_guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Make tensor inputs contiguous and select their CUDA device."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        contiguous_args = tuple(
            value.contiguous() if isinstance(value, torch.Tensor) else value
            for value in args
        )
        contiguous_kwargs = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }

        tensor = next(
            (
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        context = (
            torch.cuda.device(tensor.device)
            if tensor is not None and tensor.is_cuda
            else contextlib.nullcontext()
        )
        with context:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    """Return whether the active CUDA device supports the larger GLA tiles."""
    del arch, tensor_idx
    if not torch.cuda.is_available():
        return False
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        return False

    # PR #43 fixed the primary PyTorch property name. The remaining names keep
    # compatibility with alternative runtimes and older device wrappers.
    for name in (
        "shared_memory_per_multiprocessor",
        "max_shared_mem",
        "max_shared_memory_per_multiprocessor",
        "max_shared_memory",
    ):
        max_shared = getattr(properties, name, None)
        if max_shared is not None:
            return max_shared >= 166_000
    return False
