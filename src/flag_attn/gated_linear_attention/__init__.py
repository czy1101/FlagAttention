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

"""Chunked gated linear attention implemented with Triton kernels."""

import torch

if "metax" in str(torch.__version__).lower():
    from flag_attn.runtime.backend._metax.gla import chunk_gla
else:
    from .chunk_gla import chunk_gla

__all__ = ["chunk_gla"]
