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

"""MetaX Wall-Attention implementation."""

from .parallel import parallel_wall_attn, select_route, get_last_route
from .decode import parallel_wall_attn_decode, build_wall_kv_cache

__all__ = [
    "parallel_wall_attn",
    "select_route",
    "get_last_route",
    "parallel_wall_attn_decode",
    "build_wall_kv_cache",
]
