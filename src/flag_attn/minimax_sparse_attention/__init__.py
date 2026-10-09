# Copyright 2026 FlagOS Contributors
# Copyright contributors to the vLLM project
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

"""MiniMax M3 Sparse Attention with a paged KV cache.

The cache layout is compatible with vLLM:
  kv_cache: [num_blocks, num_kv_heads, 128, 2*head_dim]  K=[..., :head_dim] V=[..., head_dim:]
  index_kv_cache: [num_blocks, 128, head_dim]
  block_table: [batch, max_blocks]
"""

from typing import Literal

from flag_attn.runtime.backend import resolve_operator

_OPERATOR_EXPORTS = {
    "SPARSE_BLOCK_SIZE": ("SPARSE_BLOCK_SIZE", "flag_attn.minimax_sparse_attention.index_topk", "SPARSE_BLOCK_SIZE"),
    "minimax_m3_index_decode": (
        "minimax_m3_index_decode",
        "flag_attn.minimax_sparse_attention.index_topk",
        "minimax_m3_index_decode",
    ),
    "minimax_m3_index_decode_score": (
        "minimax_m3_index_decode_score",
        "flag_attn.minimax_sparse_attention.index_topk",
        "minimax_m3_index_decode_score",
    ),
    "minimax_m3_index_score": (
        "minimax_m3_index_score",
        "flag_attn.minimax_sparse_attention.index_topk",
        "minimax_m3_index_score",
    ),
    "minimax_m3_index_topk": (
        "minimax_m3_index_topk",
        "flag_attn.minimax_sparse_attention.index_topk",
        "minimax_m3_index_topk",
    ),
    "minimax_m3_sparse_attn_decode": (
        "minimax_m3_sparse_attn_decode",
        "flag_attn.minimax_sparse_attention.sparse_attn",
        "minimax_m3_sparse_attn_decode",
    ),
    "_minimax_m3_sparse_attn_prefill": (
        "minimax_m3_sparse_attn",
        "flag_attn.minimax_sparse_attention.sparse_attn",
        "minimax_m3_sparse_attn",
    ),
}


def __getattr__(name: str):
    try:
        operator, module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = resolve_operator(operator, module, symbol)
    globals()[name] = value
    return value


def minimax_m3_sparse_attn(*args, stage: Literal["prefill", "decode"] = "prefill", **kwargs):
    """Run MiniMax M3 sparse attention for the explicitly selected stage.

    The default preserves the existing prefill call. Arguments and return
    values follow sparse_attn.minimax_m3_sparse_attn for prefill and
    minimax_m3_sparse_attn_decode for decode, using the active backend.
    No cache preparation or tensor conversion is performed here.
    """
    if stage == "prefill":
        implementation = globals().get("_minimax_m3_sparse_attn_prefill") or __getattr__(
            "_minimax_m3_sparse_attn_prefill"
        )
        return implementation(*args, **kwargs)
    if stage == "decode":
        implementation = globals().get("minimax_m3_sparse_attn_decode") or __getattr__("minimax_m3_sparse_attn_decode")
        return implementation(*args, **kwargs)
    raise ValueError(f"Unsupported MiniMax stage: {stage!r}; expected 'prefill' or 'decode'")


__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_index_decode",
    "minimax_m3_index_decode_score",
    "minimax_m3_index_score",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn_decode",
    "minimax_m3_sparse_attn",
]
