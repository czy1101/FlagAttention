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

"""Parameterized local linear attention implemented with Triton kernels."""

from typing import Literal

from flag_attn.runtime.backend import resolve_operator

from .decode import HAS_TLE, parallax_attn_with_kvcache, parallax_decode
from .parallel import (
    ParallaxFunction,
    parallel_parallax_bwd,
    parallel_parallax_fwd,
)


def parallel_parallax(*args, stage: Literal["prefill", "decode", "kvcache"] = "prefill", **kwargs):
    """Run Parallax attention through a single public entry.

    Arguments and return values follow parallel.parallel_parallax for
    prefill (the default), parallax_decode for decode, and
    parallax_attn_with_kvcache for KV-cache decode. Each interface retains
    its original scale, window and output-buffer conventions.
    """
    if stage == "prefill":
        return resolve_operator("parallel_parallax", "flag_attn.FLA.parallax.parallel")(*args, **kwargs)
    if stage == "decode":
        return resolve_operator("parallax_decode", "flag_attn.FLA.parallax.decode")(*args, **kwargs)
    if stage == "kvcache":
        return resolve_operator("parallax_attn_with_kvcache", "flag_attn.FLA.parallax.decode")(*args, **kwargs)
    raise ValueError(f"Unsupported Parallax stage: {stage!r}; expected 'prefill', 'decode' or 'kvcache'")


__all__ = [
    "HAS_TLE",
    "ParallaxFunction",
    "parallax_attn_with_kvcache",
    "parallax_decode",
    "parallel_parallax",
    "parallel_parallax_bwd",
    "parallel_parallax_fwd",
]
