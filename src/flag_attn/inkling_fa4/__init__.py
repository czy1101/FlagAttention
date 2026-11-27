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

"""Inkling FA4 paged relative attention with Triton and optional Hopper TLE."""

from flag_attn.runtime import device as _device

if _device.vendor_name == "metax":
    from flag_attn.runtime.backend import resolve_operator as _resolve_operator

    def __getattr__(name: str):
        if name not in __all__:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        operator = f"inkling_fa4_{name}" if name in ("get_backend", "tle_available") else name
        value = _resolve_operator(operator, "flag_attn.inkling_fa4.relative_attention", name)
        globals()[name] = value
        return value

else:
    from .relative_attention import (
        get_backend,
        inkling_fa4_rel_attention,
        inkling_fa4_rel_attention_tle,
        inkling_fa4_rel_attention_triton,
        tle_available,
    )

__all__ = [
    "get_backend",
    "inkling_fa4_rel_attention",
    "inkling_fa4_rel_attention_tle",
    "inkling_fa4_rel_attention_triton",
    "tle_available",
]
