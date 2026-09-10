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

import importlib

import torch


_BACKEND_MODULES = {
    "enflame": "flag_attn.runtime.backend._enflame",
    "hygon": "flag_attn.runtime.backend._hygon",
    "nvidia": "flag_attn.runtime.backend._nvidia",
}


def _cuda_backend(device: torch.device) -> str:
    if getattr(torch.version, "hip", None):
        return "hygon"

    index = 0 if device.index is None else device.index
    try:
        device_name = torch.cuda.get_device_name(index).lower()
    except (AttributeError, RuntimeError):
        device_name = ""

    if "hygon" in device_name or "dcu" in device_name:
        return "hygon"
    return "nvidia"


def resolve_backend(tensor: torch.Tensor) -> str:
    """Return the FlagAttention backend selected for an input tensor."""

    if tensor.device.type == "gcu":
        return "enflame"
    if tensor.device.type == "cuda":
        return _cuda_backend(tensor.device)
    return "generic"


def resolve_gdn2_backend(q: torch.Tensor) -> str:
    """Backward-compatible alias for GDN2 backend inspection."""

    return resolve_backend(q)


def _load_operator(backend: str, name: str, fallback_module: str | None):
    module_name = _BACKEND_MODULES.get(backend)
    if module_name is not None:
        module = importlib.import_module(module_name)
        implementation = getattr(module, name, None)
        if implementation is not None:
            return implementation

    if fallback_module is not None:
        module = importlib.import_module(fallback_module)
        implementation = getattr(module, name, None)
        if implementation is not None:
            return implementation

    raise NotImplementedError(
        f"{name} is not implemented for the selected {backend!r} backend"
    )


def _dispatch(
    name: str,
    tensor: torch.Tensor,
    args: tuple,
    kwargs: dict,
    fallback_module: str | None,
):
    backend = resolve_backend(tensor)
    implementation = _load_operator(backend, name, fallback_module)
    return implementation(tensor, *args, **kwargs)


def chunk_gdn2(q: torch.Tensor, *args, **kwargs):
    return _dispatch("chunk_gdn2", q, args, kwargs, "flag_attn.gdn2")


def chunk_gdn2_native(q: torch.Tensor, *args, **kwargs):
    return _dispatch("chunk_gdn2_native", q, args, kwargs, "flag_attn.gdn2")


def chunk_gla(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "chunk_gla",
        q,
        args,
        kwargs,
        "flag_attn.gated_linear_attention",
    )


def chunk_kda(q: torch.Tensor, *args, **kwargs):
    return _dispatch("chunk_kda", q, args, kwargs, None)


def parallel_nsa(q: torch.Tensor, *args, **kwargs):
    return _dispatch("parallel_nsa", q, args, kwargs, "flag_attn.parallel_nsa")


def parallel_nsa_compression(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "parallel_nsa_compression",
        q,
        args,
        kwargs,
        "flag_attn.parallel_nsa",
    )


def minimax_m3_index_decode(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_index_decode",
        q,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def minimax_m3_index_score(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_index_score",
        q,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def minimax_m3_index_score_topk(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_index_score_topk",
        q,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def minimax_m3_index_topk(scores: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_index_topk",
        scores,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def minimax_m3_sparse_attn(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_sparse_attn",
        q,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def minimax_m3_sparse_attn_decode(q: torch.Tensor, *args, **kwargs):
    return _dispatch(
        "minimax_m3_sparse_attn_decode",
        q,
        args,
        kwargs,
        "flag_attn.minimax_sparse_attention",
    )


def sage_attention(q: torch.Tensor, *args, **kwargs):
    return _dispatch("sage_attention_forward", q, args, kwargs, None)


__all__ = [
    "chunk_gla",
    "chunk_gdn2",
    "chunk_gdn2_native",
    "chunk_kda",
    "minimax_m3_index_decode",
    "minimax_m3_index_score",
    "minimax_m3_index_score_topk",
    "minimax_m3_index_topk",
    "minimax_m3_sparse_attn",
    "minimax_m3_sparse_attn_decode",
    "parallel_nsa",
    "parallel_nsa_compression",
    "resolve_backend",
    "resolve_gdn2_backend",
    "sage_attention",
]
