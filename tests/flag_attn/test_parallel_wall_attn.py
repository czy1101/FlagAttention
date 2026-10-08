# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness and routing tests for the public Wall-Attention provider."""

from __future__ import annotations

import pytest
import torch

from flag_attn import parallel_wall_attn
from flag_attn.FLA import wall_attn as provider
from flag_attn.utils import has_triton_tle

SEQUENCE_LENGTHS = (512, 1024, 2048, 4096)
SHAPE_FAMILIES = (("mha", 8), ("gqa", 2))
DTYPES = (("bf16", torch.bfloat16), ("fp16", torch.float16))
SEEDS = (0, 1, 7)


def _has_h100_tle() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0) and has_triton_tle()


def _fla_reference():
    try:
        from fla.ops.wall_attn import parallel_wall_attn as reference
    except Exception as exc:
        pytest.skip(f"official FLA Wall-Attention is unavailable: {exc}")
    return reference


def _make_inputs(t: int, h: int, d: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    q = torch.randn(1, t, 8, d, device="cuda", dtype=dtype)
    k = torch.randn(1, t, h, d, device="cuda", dtype=dtype)
    v = torch.randn(1, t, h, d, device="cuda", dtype=dtype)
    g = (-torch.randn(1, t, 8, d, device="cuda").abs() * 0.05).to(dtype)
    return q, k, v, g


@pytest.mark.parallel_wall_attn
def test_parallel_wall_attn_is_public_api():
    assert parallel_wall_attn is provider.parallel_wall_attn


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("family,h", SHAPE_FAMILIES, ids=lambda value: str(value))
@pytest.mark.parametrize("t", SEQUENCE_LENGTHS, ids=lambda value: f"t{value}")
@pytest.mark.parametrize("dtype_name,dtype", DTYPES, ids=lambda value: str(value))
@pytest.mark.parametrize("seed", SEEDS, ids=lambda value: f"seed{value}")
@pytest.mark.skipif(not _has_h100_tle(), reason="fast-path matrix requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_fast_matrix_matches_fla(family, h, t, dtype_name, dtype, seed):
    del family
    inputs = _make_inputs(t, h, 64, dtype, seed)
    expected_route = "tle_bf16" if dtype is torch.bfloat16 else "tle_fp16"
    assert provider.select_route(*inputs, scale=64**-0.5) == expected_route

    expected = _fla_reference()(*inputs, scale=64**-0.5).float()
    actual = parallel_wall_attn(*inputs, scale=64**-0.5).float()
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    if dtype_name == "fp16":
        assert (actual - expected).abs().max().item() <= 0.02


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("h", (2, 8), ids=("gqa", "mha"))
@pytest.mark.skipif(not _has_h100_tle(), reason="D128 fast path requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_bf16_d128_smoke(h):
    inputs = _make_inputs(512, h, 128, torch.bfloat16, seed=0)
    assert provider.select_route(*inputs) == "tle_bf16"
    expected = _fla_reference()(*inputs).float()
    actual = parallel_wall_attn(*inputs).float()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize(
    "change",
    (
        "fp16_d128",
        "unsupported_t",
        "noncontiguous",
        "requires_grad",
        "g_scalar",
        "sink_bias",
        "window_size",
        "cu_seqlens",
    ),
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="routing test requires CUDA tensors")
def test_parallel_wall_attn_unsupported_contract_routes_to_fallback(change, monkeypatch):
    q, k, v, g = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    kwargs = {}
    if change == "fp16_d128":
        q, k, v, g = _make_inputs(512, 8, 128, torch.float16, seed=0)
    elif change == "unsupported_t":
        q, k, v, g = _make_inputs(256, 8, 64, torch.bfloat16, seed=0)
    elif change == "noncontiguous":
        q = q.transpose(1, 2)
    elif change == "requires_grad":
        q.requires_grad_(True)
    elif change == "g_scalar":
        kwargs[change] = torch.zeros(1, 512, 8, device="cuda", dtype=q.dtype)
    elif change == "sink_bias":
        kwargs[change] = torch.zeros(8, device="cuda", dtype=q.dtype)
    elif change == "window_size":
        kwargs[change] = 128
    elif change == "cu_seqlens":
        kwargs[change] = torch.tensor([0, 512], device="cuda", dtype=torch.long)

    assert provider.select_route(q, k, v, g, **kwargs) == "official_fallback"
    sentinel = torch.empty(0, device="cuda")
    monkeypatch.setattr(provider, "_official_wall_attn", lambda *args, **kw: sentinel)
    assert provider.parallel_wall_attn(q, k, v, g, **kwargs) is sentinel


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not torch.cuda.is_available(), reason="architecture routing test requires CUDA")
def test_parallel_wall_attn_non_sm90_routes_to_fallback(monkeypatch):
    inputs = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    monkeypatch.setattr(provider, "_is_sm90", lambda device_index: False)
    assert provider.select_route(*inputs) == "official_fallback"
