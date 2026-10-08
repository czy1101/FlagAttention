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
        pytest.fail(f"official FLA Wall-Attention is required for acceptance: {exc}", pytrace=False)
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
    assert provider.get_last_route() in (expected_route, "official_fallback")
    torch.cuda.synchronize()

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    if dtype_name == "fp16":
        assert (actual - expected).abs().max().item() <= 0.02


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("h", (2, 8), ids=("gqa", "mha"))
@pytest.mark.parametrize("t", SEQUENCE_LENGTHS)
@pytest.mark.skipif(not _has_h100_tle(), reason="D128 fast path requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_bf16_d128_matches_fla(h, t):
    inputs = _make_inputs(t, h, 128, torch.bfloat16, seed=0)
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


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("t,gate", ((4096, -0.05), (2048, -0.10)))
@pytest.mark.skipif(not _has_h100_tle(), reason="numerical guard test requires H100/SM90 with TLE")
@torch.inference_mode()
def test_parallel_wall_attn_constant_gate_uses_safe_fallback(dtype, t, gate):
    q = torch.ones(1, t, 8, 64, device="cuda", dtype=dtype)
    k, v = torch.ones_like(q), torch.ones_like(q)
    g = torch.full_like(q, gate)
    actual = parallel_wall_attn(q, k, v, g)
    assert provider.get_last_route() == "official_fallback"
    assert torch.isfinite(actual).all()
    # All value rows are one, so any finite normalized causal attention is one.
    torch.testing.assert_close(actual, torch.ones_like(actual), atol=0.01, rtol=0.01)


@pytest.mark.parallel_wall_attn
@pytest.mark.parametrize("change", ("tail", "noncontiguous", "window_size", "sink_bias", "g_scalar"))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="real fallback test requires CUDA")
@torch.inference_mode()
def test_parallel_wall_attn_real_fallback_matches_fla(change):
    t = 257 if change == "tail" else 256
    q, k, v, g = _make_inputs(t, 2, 64, torch.bfloat16, seed=1)
    kwargs = {}
    if change == "noncontiguous":
        q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        assert not q.is_contiguous()
    elif change == "window_size":
        kwargs[change] = 128
    elif change == "sink_bias":
        kwargs[change] = torch.zeros(8, device=q.device, dtype=q.dtype)
    elif change == "g_scalar":
        kwargs[change] = torch.zeros(1, t, 8, device=q.device, dtype=q.dtype)
    expected = _fla_reference()(q, k, v, g, **kwargs)
    actual = parallel_wall_attn(q, k, v, g, **kwargs)
    assert provider.get_last_route() == "official_fallback"
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not torch.cuda.is_available(), reason="gradient fallback test requires CUDA")
def test_parallel_wall_attn_real_fallback_gradients_match_fla():
    inputs = _make_inputs(128, 2, 64, torch.bfloat16, seed=7)
    reference_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    candidate_inputs = tuple(x.detach().clone().requires_grad_(True) for x in inputs)
    expected = _fla_reference()(*reference_inputs)
    actual = parallel_wall_attn(*candidate_inputs)
    assert provider.get_last_route() == "official_fallback"
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad)
    actual_grads = torch.autograd.grad(actual, candidate_inputs, grad)
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=0.05, rtol=0.05)


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="device context regression requires two CUDA GPUs")
@torch.inference_mode()
def test_parallel_wall_attn_runs_on_input_device():
    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        with torch.cuda.device(1):
            inputs = _make_inputs(512, 2, 64, torch.bfloat16, seed=0)
            expected = _fla_reference()(*inputs)
        actual = parallel_wall_attn(*inputs)
        assert actual.device == inputs[0].device
        assert torch.cuda.current_device() == 0
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.05)
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.parallel_wall_attn
@pytest.mark.skipif(not _has_h100_tle(), reason="capture policy test requires H100/SM90 with TLE")
def test_parallel_wall_attn_capture_selects_fallback(monkeypatch):
    inputs = _make_inputs(512, 8, 64, torch.bfloat16, seed=0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert provider.select_route(*inputs) == "official_fallback"
