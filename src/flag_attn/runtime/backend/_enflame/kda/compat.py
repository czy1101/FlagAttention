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

"""Triton feature checks used by the Enflame KDA kernels."""

import re

import triton


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
