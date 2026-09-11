# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained DiffKV Attention implementations.

``triton`` contains the complete Triton/TLE operator and is the default
runtime implementation.  ``FA3`` contains the optional FA3 CUDA-extension
adapter used by the benchmark for an apples-to-apples reference.  Keeping the
two adapters in this package means callers and tests only import
``flag_attn.diffkv_attention``; they do not depend on vLLM source modules.
"""

from . import FA3, triton
from .FA3 import (
    build_fa3_runner,
    fa3_op_available,
    load_fa3_provider,
)
from .triton import *  # noqa: F401,F403
from .triton import __all__ as _TRITON_ALL

__all__ = [
    *_TRITON_ALL,
    "FA3",
    "triton",
    "build_fa3_runner",
    "fa3_op_available",
    "load_fa3_provider",
]
