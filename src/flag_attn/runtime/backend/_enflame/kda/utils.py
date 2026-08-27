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

"""Small runtime-independent helpers for KDA."""

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
