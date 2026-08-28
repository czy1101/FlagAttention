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

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "0.0.0"
    version_tuple = (0, 0, 0)


from flag_attn.piecewise import attention as piecewise_attention # noqa: F401
from flag_attn.flash import attention as flash_attention # noqa: F401
from flag_attn.split_kv import attention as flash_attention_split_kv # noqa: F401
from flag_attn.paged import attention as paged_attention # noqa: F401
from flag_attn.gated_delta_rule import chunk_gated_delta_rule # noqa: F401
from flag_attn.gated_linear_attention import chunk_gla as chunk_gla
from flag_attn.minimax_sparse_attention import (
    minimax_m3_index_decode as minimax_m3_index_decode,
    minimax_m3_index_decode_score as minimax_m3_index_decode_score,
    minimax_m3_index_score as minimax_m3_index_score,
    minimax_m3_index_topk as minimax_m3_index_topk,
    minimax_m3_sparse_attn as minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode as minimax_m3_sparse_attn_decode,
)

from flag_attn.runtime.backend._metax.kda import chunk_kda as chunk_kda
from flag_attn import testing # noqa: F401
