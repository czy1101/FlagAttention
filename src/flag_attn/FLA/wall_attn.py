"""Wall Attention provider with guarded H100 TLE fast paths.

The public signature is compatible with the upstream FLA provider. A small
kernel family serves the frozen BF16 and FP16 inference shape families; every
unsupported architecture, dtype, shape, feature, or autograd use delegates to
the bundled official implementation.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from numbers import Real

import torch
import triton

from flag_attn.FLA.cumsum import chunk_global_cumsum
from flag_attn.utils import has_triton_tle

from .wall_attn_official_fallback import parallel_wall_attn as _official_wall_attn

RCP_LN2 = 1.4426950216
_FAST_T = frozenset((512, 1024, 2048, 4096))
_FAST_H = frozenset((2, 8))
_FAST_BF16_D = frozenset((64, 128))


@lru_cache(maxsize=1)
def _load_tle_modules():
    """Load H100-only TLE code only after dispatch accepts a fast-path call."""
    package = __package__
    attention_bf16 = import_module(f"{package}._wall_attn_tle_attention")
    attention_fp16 = import_module(f"{package}._wall_attn_tle_attention_fp16")
    cache = import_module(f"{package}._wall_attn_tle_cache")
    preprocess = import_module(f"{package}._wall_attn_tle_preprocess")
    triton.set_allocator(attention_bf16.allocator)
    return attention_bf16, attention_fp16, cache, preprocess


@lru_cache(maxsize=None)
def _is_sm90(device_index: int) -> bool:
    """Return whether ``device_index`` is an NVIDIA Hopper SM90 device."""
    return torch.cuda.get_device_capability(device_index) == (9, 0)


def _requires_backward(*tensors: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(tensor is not None and tensor.requires_grad for tensor in tensors)


def select_route(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> str:
    """Select ``tle_bf16``, ``tle_fp16``, or ``official_fallback``."""
    if any(value is not None for value in (g_scalar, sink_bias, window_size, cu_seqlens)):
        return "official_fallback"
    if scale is not None and not isinstance(scale, Real):
        return "official_fallback"
    if any(tensor.layout != torch.strided for tensor in (q, k, v, g)):
        return "official_fallback"
    if not all(tensor.is_cuda for tensor in (q, k, v, g)):
        return "official_fallback"
    if not (q.device == k.device == v.device == g.device):
        return "official_fallback"
    if not all(tensor.is_contiguous() for tensor in (q, k, v, g)):
        return "official_fallback"
    if any(tensor.ndim != 4 for tensor in (q, k, v, g)):
        return "official_fallback"

    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if not _is_sm90(device_index):
        return "official_fallback"
    if not has_triton_tle():
        return "official_fallback"

    b, t, hq, d = q.shape
    if b != 1 or t not in _FAST_T or hq != 8:
        return "official_fallback"
    h = k.shape[2]
    if h not in _FAST_H or hq % h != 0:
        return "official_fallback"
    if k.shape != (b, t, h, d) or v.shape != (b, t, h, d) or g.shape != q.shape:
        return "official_fallback"
    if _requires_backward(q, k, v, g):
        return "official_fallback"

    if q.dtype == k.dtype == v.dtype == g.dtype == torch.bfloat16 and d in _FAST_BF16_D:
        return "tle_bf16"
    if q.dtype == k.dtype == v.dtype == g.dtype == torch.float16 and d == 64:
        return "tle_fp16"
    return "official_fallback"


def _parallel_wall_attn_tle(
    route: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    attention_bf16, attention_fp16, cache, preprocess = _load_tle_modules()
    b, t, hq, d = q.shape
    cache_dtype = q.dtype if route == "tle_bf16" else torch.bfloat16
    q_cache = torch.empty((b, hq, t, d), device=q.device, dtype=cache_dtype)
    k_cache = torch.empty_like(q_cache)
    anchor = torch.empty((b, hq, d), device=q.device, dtype=torch.float32)
    output = torch.empty((b, t, hq, d), device=q.device, dtype=v.dtype)
    lse = torch.empty((b, t, hq), device=q.device, dtype=torch.float32)

    if t <= 2048:
        preprocess.launch(q, k, g, q_cache, k_cache, anchor, bt=128, bc=8, warps=4)
    else:
        prefix = chunk_global_cumsum(g, scale=RCP_LN2)
        cache.build(q, k, prefix, q_cache, k_cache, anchor)

    attention = attention_bf16 if route == "tle_bf16" else attention_fp16
    attention.launch(q_cache, k_cache, v, output, lse, capacity=2, scale=scale)
    return output


def parallel_wall_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    *,
    g_scalar: torch.Tensor | None = None,
    sink_bias: torch.Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Run the accepted H100 fast path or the full official implementation."""
    route = select_route(
        q,
        k,
        v,
        g,
        g_scalar=g_scalar,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )
    if route != "official_fallback":
        return _parallel_wall_attn_tle(
            route,
            q,
            k,
            v,
            g,
            scale=k.shape[-1] ** -0.5 if scale is None else float(scale),
        )
    return _official_wall_attn(
        q,
        k,
        v,
        g,
        g_scalar=g_scalar,
        sink_bias=sink_bias,
        scale=scale,
        window_size=window_size,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["parallel_wall_attn", "select_route"]
