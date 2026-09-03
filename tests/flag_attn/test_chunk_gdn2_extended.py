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

import flag_attn
from flag_attn.runtime.backend import is_metax_backend


chunk_gdn2 = getattr(flag_attn, "chunk_gdn2", None)


ASSERT_RATIO = 0.01
UPSTREAM_FLAGGEMS_COMMIT = "f771e65aba3bba8f9683e409b5e6355e14213371"

GDN2_TEST_SHAPES = [
    pytest.param((2, 512, 8, 64, 64), id="shape00_b2_t512_h8_k64_v64"),
    pytest.param((4, 1024, 8, 64, 64), id="shape01_b4_t1024_h8_k64_v64"),
    pytest.param((1, 2048, 8, 64, 64), id="shape02_b1_t2048_h8_k64_v64"),
    pytest.param((1, 4096, 16, 64, 64), id="shape03_b1_t4096_h16_k64_v64"),
    pytest.param((1, 8192, 96, 128, 128), id="shape04_b1_t8192_h96_k128_v128"),
    pytest.param((2, 2048, 16, 256, 512), id="shape05_b2_t2048_h16_k256_v512"),
    pytest.param((2, 16384, 16, 128, 128), id="shape06_b2_t16384_h16_k128_v128"),
    pytest.param((4, 1024, 8, 256, 512), id="shape07_b4_t1024_h8_k256_v512"),
    pytest.param((4, 2048, 16, 128, 128), id="shape08_b4_t2048_h16_k128_v128"),
    pytest.param((4, 4096, 64, 128, 128), id="shape09_b4_t4096_h64_k128_v128"),
    pytest.param((8, 1024, 8, 64, 64), id="shape10_b8_t1024_h8_k64_v64"),
    pytest.param((8, 2048, 32, 256, 256), id="shape11_b8_t2048_h32_k256_v256"),
]

GDN2_DTYPES = [
    pytest.param(torch.float16, id="dtype_fp16"),
    pytest.param(torch.bfloat16, id="dtype_bf16"),
]

GDN2_IMPLS = [
    pytest.param("tle", id="impl_tle"),
    pytest.param("native", id="impl_native"),
]

pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and is_metax_backend()
        and chunk_gdn2 is not None
    ),
    reason="extended chunk_gdn2 tests require a MetaX CUDA-compatible device",
)


def _make_inputs(*, batch, length, heads, key_dim, value_dim, dtype):
    device = "cuda"
    scale = key_dim**-0.5
    q = torch.randn(
        batch, length, heads, key_dim, device=device, dtype=dtype
    ) / math.sqrt(key_dim)
    k = torch.randn(
        batch, length, heads, key_dim, device=device, dtype=dtype
    ) / math.sqrt(key_dim)
    v = torch.randn(batch, length, heads, value_dim, device=device, dtype=dtype)
    erase = torch.rand(
        batch, length, heads, key_dim, device=device, dtype=dtype
    )
    write = torch.rand(
        batch, length, heads, value_dim, device=device, dtype=dtype
    )
    gate = (
        -torch.rand(
            batch,
            length,
            heads,
            key_dim,
            device=device,
            dtype=torch.float32,
        )
        * 0.1
    ).to(dtype)
    return (q, k, v, gate, erase, write), scale


def _native_reference(native, args, scale):
    q, k, v, gate, erase, write = args
    result = native(
        q=q,
        k=k,
        v=v,
        g=gate,
        b=erase,
        w_gate=write,
        scale=scale,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,
        cu_seqlens_cpu=None,
        chunk_size=64,
        safe_gate=False,
        lower_bound=None,
        use_gate_in_kernel=False,
        A_log=None,
        dt_bias=None,
        disable_recompute=True,
        state_v_first=False,
    )
    return result[0], result[1]


def _metrics(expected, actual):
    expected_fp32 = expected.float()
    actual_fp32 = actual.float()
    difference = expected_fp32 - actual_fp32
    rms = difference.flatten().square().mean().sqrt().item()
    reference_rms = expected_fp32.flatten().square().mean().sqrt().item()
    return {
        "max_abs": difference.abs().max().item(),
        "rms": rms,
        "rms_ratio": rms / (reference_rms + 1e-8),
    }


def _assert_close(name, actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all(), f"{name}: actual contains non-finite values"
    assert torch.isfinite(expected).all(), f"{name}: reference contains non-finite values"
    metrics = _metrics(expected, actual)
    print(
        f"{name}: max_abs={metrics['max_abs']:.8f} "
        f"rms={metrics['rms']:.8f} ratio={metrics['rms_ratio']:.8f}"
    )
    if metrics["max_abs"] > 1e-6:
        assert metrics["rms_ratio"] < ASSERT_RATIO, (
            name,
            metrics,
            ASSERT_RATIO,
        )


@pytest.mark.parametrize("impl", GDN2_IMPLS)
@pytest.mark.parametrize("dtype", GDN2_DTYPES)
@pytest.mark.parametrize("shape", GDN2_TEST_SHAPES)
@torch.inference_mode()
def test_chunk_gdn2_extended_matrix(shape, dtype, impl):
    torch.manual_seed(42)
    batch, length, heads, key_dim, value_dim = shape
    args, scale = _make_inputs(
        batch=batch,
        length=length,
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
    )
    inputs_before = tuple(tensor.clone() for tensor in args)

    module = importlib.import_module(
        "flag_attn.runtime.backend._metax.chunk_gdn2"
    )
    assert module.chunk_gdn2 is chunk_gdn2
    original_has_tle = module.HAS_TLE_GDN2
    original_tle = getattr(module, "chunk_gdn2_fwd_infer", None)
    original_native = module.chunk_gdn2_fwd

    if impl == "tle" and (not original_has_tle or original_tle is None):
        pytest.skip("TLE GDN2 path is unavailable in this environment")

    expected, expected_final = _native_reference(original_native, args, scale)
    torch.cuda.synchronize()

    route_counts = {"tle": 0, "native": 0}

    def wrapped_tle(*wrapper_args, **wrapper_kwargs):
        route_counts["tle"] += 1
        return original_tle(*wrapper_args, **wrapper_kwargs)

    def wrapped_native(*wrapper_args, **wrapper_kwargs):
        route_counts["native"] += 1
        return original_native(*wrapper_args, **wrapper_kwargs)

    if original_tle is not None:
        module.chunk_gdn2_fwd_infer = wrapped_tle
    module.chunk_gdn2_fwd = wrapped_native
    module.HAS_TLE_GDN2 = impl == "tle"
    try:
        actual, actual_final = chunk_gdn2(
            *args,
            scale=scale,
            initial_state=None,
            output_final_state=True,
            use_gate_in_kernel=False,
            safe_gate=False,
            lower_bound=None,
            A_log=None,
            dt_bias=None,
            state_v_first=False,
            cu_seqlens=None,
            cu_seqlens_cpu=None,
            chunk_size=16 if impl == "tle" else 64,
        )
        torch.cuda.synchronize()
    finally:
        module.HAS_TLE_GDN2 = original_has_tle
        if original_tle is not None:
            module.chunk_gdn2_fwd_infer = original_tle
        module.chunk_gdn2_fwd = original_native

    assert module.HAS_TLE_GDN2 == original_has_tle
    if original_tle is not None:
        assert module.chunk_gdn2_fwd_infer is original_tle
    assert module.chunk_gdn2_fwd is original_native
    expected_routes = (
        {"tle": 1, "native": 0}
        if impl == "tle"
        else {"tle": 0, "native": 1}
    )
    assert route_counts == expected_routes

    _assert_close("output", actual, expected)
    assert actual_final is not None
    assert expected_final is not None
    _assert_close("final_state", actual_final, expected_final)
    assert all(
        torch.equal(before, after)
        for before, after in zip(inputs_before, args)
    )
    print(f"route_counts={route_counts}")
    print("inputs_unchanged=PASS")
