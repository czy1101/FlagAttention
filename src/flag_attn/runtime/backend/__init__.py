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

"""Runtime backend detection shared by backend-specific implementations."""

from __future__ import annotations


def current_backend_name() -> str:
    """Return the active Triton backend using FlagAttention vendor names."""
    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        backend = str(getattr(target, "backend", "")).lower()
        if backend in {"maca", "metax"}:
            return "metax"
        if backend:
            return backend
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            device_name = str(torch.cuda.get_device_name()).lower()
            if "metax" in device_name or "曦云" in device_name:
                return "metax"
    except (AssertionError, RuntimeError):
        pass
    return "unknown"


def is_metax_backend() -> bool:
    """Return whether the active accelerator uses the MetaX/MACA backend."""
    return current_backend_name() == "metax"


__all__ = ["current_backend_name", "is_metax_backend"]
