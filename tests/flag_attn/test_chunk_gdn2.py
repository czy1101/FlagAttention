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

import importlib
import math

import pytest
import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from flag_attn.runtime.backend._mthreads.gdn2 import chunk_gdn2
from flag_attn.runtime.backend._mthreads.gdn2.gdn2_native.chunk_fwd import (
    chunk_gdn2_fwd,
)

ASSERT_RATIO = 0.01
TLE_ASSERT_RATIO = 1.5

GDN2_TEST_SHAPES = [
    (2, 512, 8, 64, 64),
    (4, 1024, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (1, 8192, 96, 128, 128),
    # MUSA illegal-memory cases reported by the source compatibility run.
    # (2, 2048, 16, 256, 512),
    # (2, 16384, 16, 128, 128),
    (4, 1024, 8, 256, 512),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
    (8, 1024, 8, 64, 64),
    # (8, 2048, 32, 256, 256),
]


def _musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


pytestmark = [
    pytest.mark.chunk_gdn2,
    pytest.mark.gdn2,
    pytest.mark.skipif(not _musa_available(), reason="chunk_gdn2 tests require MUSA"),
]


def _set_public_gdn2_native_for_call(force_native: bool):
    module = importlib.import_module("flag_attn.runtime.backend._mthreads.gdn2.chunk_gdn2")
    old = module.HAS_TLE_GDN2
    if force_native:
        module.HAS_TLE_GDN2 = False
    return module, old


def _native_gdn2_reference(
    q,
    k,
    v,
    g,
    b,
    w,
    *,
    scale,
    initial_state,
    output_final_state,
    use_gate_in_kernel=False,
    safe_gate=False,
    lower_bound=None,
    A_log=None,
    dt_bias=None,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_size=16,
):
    (
        o,
        final_state,
        _g,
        _Aqk,
        _Akk,
        _w_wy,
        _u_wy,
        _qg,
        _kg,
        _v_new,
        _h,
        _initial_state,
    ) = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        disable_recompute=True,
        state_v_first=state_v_first,
    )
    return o, final_state


def _make_inputs(*, B, T, H, K, V, dtype, state_v_first):
    device = torch.device("musa")
    scale = K**-0.5

    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / math.sqrt(K)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    b = torch.rand(B, T, H, K, device=device, dtype=dtype)
    w = torch.rand(B, T, H, V, device=device, dtype=dtype)
    initial_state = None
    g = (-torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.1).to(dtype)

    A_log = None
    dt_bias = None
    lower_bound = None

    kwargs = {
        "scale": scale,
        "initial_state": initial_state,
        "output_final_state": True,
        "use_gate_in_kernel": False,
        "safe_gate": False,
        "lower_bound": lower_bound,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state_v_first": state_v_first,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
        "chunk_size": 64,
    }
    return (q, k, v, g, b, w), kwargs


def _err_ratio(expected: torch.Tensor, actual: torch.Tensor) -> float:
    err = (expected.float() - actual.float()).flatten().square().mean().sqrt().item()
    base = expected.float().flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    assert_ratio: float = ASSERT_RATIO,
) -> None:
    actual = actual.float()
    expected = expected.float()
    abs_err = (actual - expected).abs().max().item()
    ratio = _err_ratio(expected, actual)
    if abs_err <= 1e-6:
        return
    assert not torch.isnan(actual).any(), f"{name}: NaN detected in actual"
    assert not torch.isnan(expected).any(), f"{name}: NaN detected in baseline"
    assert ratio < assert_ratio, f"{name} diff: abs={abs_err:.6f} ratio={ratio:.6f} limit={assert_ratio}"


@pytest.mark.parametrize(
    "impl",
    [pytest.param("tle", id="tle"), pytest.param("native", id="native")],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", GDN2_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gdn2_matches_native_triton(impl, dtype, shape):
    module = importlib.import_module("flag_attn.runtime.backend._mthreads.gdn2.chunk_gdn2")
    if impl == "tle" and not module.HAS_TLE_GDN2:
        pytest.skip("TLE GDN2 path is unavailable in this environment")

    torch.manual_seed(42)
    B, T, H, K, V = shape
    args, kwargs = _make_inputs(
        B=B,
        T=T,
        H=H,
        K=K,
        V=V,
        dtype=dtype,
        state_v_first=False,
    )

    force_native = impl == "native"
    actual_kwargs = dict(kwargs)
    if impl == "tle":
        # Public TLE inference requires BT=16; retain the source benchmark's
        # native Triton BT=64 reference.
        actual_kwargs["chunk_size"] = 16

    expected, expected_final = _native_gdn2_reference(*args, **kwargs)

    module, old = _set_public_gdn2_native_for_call(force_native)
    try:
        actual, actual_final = chunk_gdn2(*args, **actual_kwargs)
    finally:
        module.HAS_TLE_GDN2 = old

    ratio_limit = TLE_ASSERT_RATIO if impl == "tle" else ASSERT_RATIO
    _assert_close("o", actual, expected, assert_ratio=ratio_limit)
    _assert_close("ht", actual_final, expected_final, assert_ratio=ratio_limit)
