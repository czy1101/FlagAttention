# Enflame GDN2 Triton implementation.
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tensor and device helpers shared by the GDN2 Triton pipeline."""

import contextlib
import functools
from collections.abc import Callable
from typing import Any

import torch


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
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
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
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
        if tensor is not None and tensor.device.type == "gcu" and hasattr(torch, "gcu"):
            context = torch.gcu.device(tensor.device)
        else:
            context = contextlib.nullcontext()
        with context:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


def check_shared_mem(arch: str = "none", tensor_idx: int = 0) -> bool:
    del arch, tensor_idx
    if not hasattr(torch, "gcu") or not torch.gcu.is_available():
        return False
    try:
        properties = torch.gcu.get_device_properties(torch.gcu.current_device())
    except (AttributeError, RuntimeError):
        return False
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
