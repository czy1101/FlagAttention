import os
from contextlib import contextmanager

import pytest
import torch

from flag_attn import chunk_gated_delta_rule
from flag_attn.gated_delta_rule import chunk_gated_delta_rule_fwd
from flag_attn.gated_delta_rule.compat import has_triton_tle

ASSERT_RATIO = 0.01
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"

GDN_RECOMPUTE_TEST_SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    (8, 2048, 32, 256, 256),
]

GDN_FUSED_FWD_TEST_SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
]


def _cuda_tle_available() -> bool:
    return torch.cuda.is_available() and has_triton_tle(3, 6, 0)


@contextmanager
def _set_gdn_tle(*, full_tle: bool, recompute_tle: bool):
    old_full = os.environ.get(FULL_TLE_ENV)
    old_recompute = os.environ.get(RECOMPUTE_TLE_ENV)
    os.environ[FULL_TLE_ENV] = "1" if full_tle else "0"
    os.environ[RECOMPUTE_TLE_ENV] = "1" if recompute_tle else "0"
    try:
        yield
    finally:
        if old_full is None:
            os.environ.pop(FULL_TLE_ENV, None)
        else:
            os.environ[FULL_TLE_ENV] = old_full
        if old_recompute is None:
            os.environ.pop(RECOMPUTE_TLE_ENV, None)
        else:
            os.environ[RECOMPUTE_TLE_ENV] = old_recompute


def _make_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    *,
    use_initial_state: bool,
):
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    initial_state = (
        torch.zeros(B, H, K, V, device=device, dtype=dtype)
        if use_initial_state
        else None
    )
    return q, k, v, g, beta, K**-0.5, initial_state, True, None


def _call_fwd(args, *, full_tle: bool, recompute_tle: bool):
    with _set_gdn_tle(full_tle=full_tle, recompute_tle=recompute_tle):
        return chunk_gated_delta_rule_fwd(*args)


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    err = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    base = expected.float().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs().max().item()
    if abs_err <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    ratio = _err_ratio(expected, actual)
    assert ratio < ASSERT_RATIO, (
        f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} " f"limit={ASSERT_RATIO}"
    )


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN recompute TLE tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_RECOMPUTE_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_recompute_tle_matches_native(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=True)

    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    actual = _call_fwd(args, full_tle=False, recompute_tle=True)

    _assert_close("g", actual[0], baseline[0])
    _assert_close("o", actual[1], baseline[1])
    _assert_close("A", actual[2], baseline[2])
    _assert_close("final_state", actual[3], baseline[3])


@pytest.mark.skipif(
    not _cuda_tle_available(), reason="GDN fused TLE tests require CUDA/TLE"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN_FUSED_FWD_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gated_delta_rule_fwd_full_tle_matches_native(dtype, shape):
    torch.manual_seed(42)
    args = _make_inputs(*shape, dtype=dtype, use_initial_state=False)

    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    actual = _call_fwd(args, full_tle=True, recompute_tle=True)

    _assert_close("g", actual[0], baseline[0])
    _assert_close("o", actual[1], baseline[1])
    _assert_close("A", actual[2], baseline[2])
    _assert_close("final_state", actual[3], baseline[3])


@pytest.mark.skipif(not _cuda_tle_available(), reason="GDN public API test requires CUDA/TLE")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_chunk_gated_delta_rule_public_api_matches_native(dtype):
    torch.manual_seed(42)
    args = _make_inputs(4, 2048, 16, 128, 128, dtype, use_initial_state=False)
    baseline = _call_fwd(args, full_tle=False, recompute_tle=False)
    q, k, v, g, beta, scale, _, _, _ = args

    with _set_gdn_tle(full_tle=True, recompute_tle=True):
        o, final_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            beta,
            g,
            head_first=False,
            scale=scale,
            output_final_state=True,
        )

    _assert_close("o", o, baseline[1])
    _assert_close("final_state", final_state, baseline[3])
