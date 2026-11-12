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

from __future__ import annotations

import os
from typing import Any

import torch

from .forward import inkling_fa4_rel_attention_triton

TLE_AVAILABLE = False
TLE_IMPORT_ERROR = ImportError("Hopper TLE is unavailable on MetaX")

_TRITON_ALIASES = {"triton", "base"}
_TLE_ALIASES = {"tle", "triton_tle", "triton-tle"}


def inkling_fa4_rel_attention_tle(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Placeholder used when ``triton.experimental.tle`` is unavailable."""
    raise RuntimeError(f"TLE backend unavailable: {type(TLE_IMPORT_ERROR).__name__}: {TLE_IMPORT_ERROR}")


def tle_available() -> tuple[bool, str]:
    """Return TLE import availability and an optional diagnostic reason."""
    if TLE_AVAILABLE:
        return True, ""
    return False, f"{type(TLE_IMPORT_ERROR).__name__}: {TLE_IMPORT_ERROR}"


def get_backend(name: str | None = None):
    """Resolve an implementation without launching it or checking its input layout.

    Auto selects TLE when importable. Use ``inkling_fa4_rel_attention`` for
    layout-aware fallback. An explicitly requested unavailable TLE raises
    ImportError so benchmark callers can report it as unavailable.
    """
    selected = (name or os.getenv("INKLING_FA4_BACKEND", "auto")).strip().lower()
    if selected in _TRITON_ALIASES:
        return inkling_fa4_rel_attention_triton
    if selected in _TLE_ALIASES:
        if not TLE_AVAILABLE:
            raise ImportError(f"triton.experimental.tle unavailable: {TLE_IMPORT_ERROR}")
        return inkling_fa4_rel_attention_tle
    if selected == "auto":
        return inkling_fa4_rel_attention_tle if TLE_AVAILABLE else inkling_fa4_rel_attention_triton
    choices = ", ".join(sorted(_TRITON_ALIASES | _TLE_ALIASES | {"auto"}))
    raise ValueError(f"unknown Inkling FA4 backend={selected!r}; choose from {choices}")


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run Inkling FA4 relative attention on the best available backend.

    ``backend`` accepts ``"auto"`` (default), ``"triton"``, or ``"tle"``, and
    can also be set through the ``INKLING_FA4_BACKEND`` environment variable.
    Auto prefers TLE and falls back to Triton when TLE is unavailable. The TLE
    implementation handles short query shapes with masked WGMMA tiles; it
    raises ``NotImplementedError`` only for unsupported tensor layouts.
    """
    selected = (backend or os.getenv("INKLING_FA4_BACKEND", "auto")).strip().lower()

    if selected in _TRITON_ALIASES:
        return inkling_fa4_rel_attention_triton(*args, **kwargs)

    if selected in _TLE_ALIASES:
        return inkling_fa4_rel_attention_tle(*args, **kwargs)

    if selected != "auto":
        choices = ", ".join(sorted(_TRITON_ALIASES | _TLE_ALIASES | {"auto"}))
        raise ValueError(f"unknown Inkling FA4 backend={selected!r}; choose from {choices}")

    if not TLE_AVAILABLE:
        return inkling_fa4_rel_attention_triton(*args, **kwargs)

    try:
        return inkling_fa4_rel_attention_tle(*args, **kwargs)
    except NotImplementedError:
        # TLE only supports block_size=16 with 128-wide Q/K/V heads. Fall back
        # instead of failing the call for unsupported tensor layouts.
        return inkling_fa4_rel_attention_triton(*args, **kwargs)


__all__ = [
    "get_backend",
    "inkling_fa4_rel_attention",
    "inkling_fa4_rel_attention_tle",
    "inkling_fa4_rel_attention_triton",
    "tle_available",
]
