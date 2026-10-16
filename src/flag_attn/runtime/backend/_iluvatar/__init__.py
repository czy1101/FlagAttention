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

"""Iluvatar attention backend."""

from __future__ import annotations

import importlib

_PACKAGE = "flag_attn.runtime.backend._iluvatar"

_OPERATOR_EXPORTS = {
    "SPARSE_BLOCK_SIZE": (f"{_PACKAGE}.minimax_sparse_attention.index_topk", "SPARSE_BLOCK_SIZE"),
    "chunk_gdn2": (f"{_PACKAGE}.FLA.gdn2.chunk_gdn2", "chunk_gdn2"),
    "diffkv_attention": (f"{_PACKAGE}.diffkv_attention.api", "diffkv_attention"),
    "minimax_m3_index_decode": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_index_decode"),
    "minimax_m3_index_decode_score": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_index_decode_score"),
    "minimax_m3_index_score": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_index_score"),
    "minimax_m3_index_topk": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (f"{_PACKAGE}.minimax_sparse_attention", "minimax_m3_sparse_attn_decode"),
    "sage_attention": (f"{_PACKAGE}.sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (f"{_PACKAGE}.sage_attention.quant_per_block", "per_block_int8"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
