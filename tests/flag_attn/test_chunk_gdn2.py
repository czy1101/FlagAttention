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
# Test semantics derived from:
# flagos-ai/FlagGems-vllm@f771e65aba3bba8f9683e409b5e6355e14213371

from __future__ import annotations

import importlib
import math

import pytest
import torch

try:
    from flag_attn import chunk_gdn2
except ImportError:
    chunk_gdn2 = None


ASSERT_RATIO = 0.01
REPEAT_ASSERT_RATIO = 0.005

pytestmark = pytest.mark.skipif(
    chunk_gdn2 is None or not torch.cuda.is_available(),
    reason="MetaX chunk_gdn2 tests require an available GPU backend",
)


CASES = [
    pytest.param(
        {
            "shape": (1, 15, 2, 64, 64),
            "dtype": torch.bfloat16,
            "route": "tle",
            "initial": False,
            "output_final_state": False,
            "state_v_first": False,
            "repeats": 1,
            "seed": 101,
        },
        id="tle_bf16_t15_no_state_no_final",
    ),
    pytest.param(
        {
            "shape": (1, 17, 2, 128, 128),
            "dtype": torch.bfloat16,
            "route": "tle",
            "initial": True,
            "output_final_state": True,
            "state_v_first": False,
            "repeats": 3,
            "seed": 102,
        },
        id="tle_bf16_t17_initial_kv_repeat",
    ),
    pytest.param(
        {
            "shape": (1, 16, 2, 64, 128),
            "dtype": torch.float16,
            "route": "tle",
            "initial": True,
            "output_final_state": True,
            "state_v_first": True,
            "repeats": 1,
            "seed": 103,
        },
        id="tle_fp16_t16_initial_vk",
    ),
    pytest.param(
        {
            "shape": (1, 17, 1, 256, 512),
            "dtype": torch.bfloat16,
            "route": "tle",
            "initial": True,
            "output_final_state": True,
            "state_v_first": False,
            "repeats": 1,
            "seed": 104,
        },
        id="tle_bf16_t17_k256_v512",
    ),
    pytest.param(
        {
            "shape": (2, 32, 4, 64, 64),
            "dtype": torch.float16,
            "route": "tle",
            "initial": False,
            "output_final_state": True,
            "state_v_first": False,
            "repeats": 1,
            "seed": 105,
        },
        id="tle_fp16_b2_t32_multi_chunk",
    ),
    pytest.param(
        {
            "shape": (1, 17, 2, 64, 64),
            "dtype": torch.bfloat16,
            "route": "fallback",
            "initial": True,
            "output_final_state": True,
            "state_v_first": False,
            "repeats": 2,
            "seed": 106,
        },
        id="fallback_bf16_t17_initial_kv",
    ),
]


def _make_inputs(case):
    batch, length, heads, key_dim, value_dim = case["shape"]
    dtype = case["dtype"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case["seed"])

    def randn(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    def rand(shape):
        return torch.rand(shape, generator=generator, dtype=torch.float32)

    q_shape = (batch, length, heads, key_dim)
    v_shape = (batch, length, heads, value_dim)

    q = (randn(q_shape) / math.sqrt(key_dim)).to(
        device="cuda",
        dtype=dtype,
    )
    k = (randn(q_shape) / math.sqrt(key_dim)).to(
        device="cuda",
        dtype=dtype,
    )
    v = randn(v_shape).to(device="cuda", dtype=dtype)
    g = (-rand(q_shape) * 0.1).to(device="cuda", dtype=dtype)
    b = rand(q_shape).to(device="cuda", dtype=dtype)
    w = rand(v_shape).to(device="cuda", dtype=dtype)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    b = b.contiguous()
    w = w.contiguous()

    initial_state = None
    if case["initial"]:
        canonical = randn((batch, heads, key_dim, value_dim)) * 0.05
        if case["state_v_first"]:
            canonical = canonical.transpose(-1, -2).contiguous()
        initial_state = canonical.to(
            device="cuda",
            dtype=torch.float32,
        ).contiguous()

    return (q, k, v, g, b, w), initial_state, key_dim**-0.5


@torch.inference_mode()
def _torch_reference(
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
    state_v_first,
):
    qf, kf, vf, gf, bf, wf = (
        tensor.detach().cpu().double()
        for tensor in (q, k, v, g, b, w)
    )

    batch, length, heads, key_dim = qf.shape
    value_dim = vf.shape[-1]

    if initial_state is None:
        state = torch.zeros(
            batch,
            heads,
            key_dim,
            value_dim,
            dtype=torch.float64,
        )
    else:
        state = initial_state.detach().cpu().double().clone()
        if state_v_first:
            state = state.transpose(-1, -2).contiguous()

    outputs = []
    for index in range(length):
        q_t = qf[:, index]
        k_t = kf[:, index]
        v_t = vf[:, index]
        g_t = gf[:, index]
        b_t = bf[:, index]
        w_t = wf[:, index]

        state = torch.exp(g_t).unsqueeze(-1) * state
        recalled = torch.einsum(
            "bhk,bhkv->bhv",
            b_t * k_t,
            state,
        )
        residual = w_t * v_t - recalled
        state = state + k_t.unsqueeze(-1) * residual.unsqueeze(-2)

        output = scale * torch.einsum(
            "bhk,bhkv->bhv",
            q_t,
            state,
        )
        outputs.append(output)

    output = torch.stack(outputs, dim=1)

    final_state = None
    if output_final_state:
        final_state = (
            state.transpose(-1, -2).contiguous()
            if state_v_first
            else state
        )

    return output, final_state


def _assert_close(name, actual, expected, ratio_limit):
    actual_cpu = actual.detach().cpu().double()
    expected_cpu = expected.detach().cpu().double()

    assert actual_cpu.shape == expected_cpu.shape
    assert torch.isfinite(actual_cpu).all(), f"{name}: non-finite actual"
    assert torch.isfinite(expected_cpu).all(), f"{name}: non-finite reference"

    difference = actual_cpu - expected_cpu
    max_abs = difference.abs().max().item()
    rms = difference.square().mean().sqrt().item()
    reference_rms = expected_cpu.square().mean().sqrt().item()
    ratio = rms / (reference_rms + 1e-12)

    print(
        f"{name}: max_abs={max_abs:.8f} "
        f"rms={rms:.8f} ratio={ratio:.8f}"
    )
    assert ratio < ratio_limit, (
        f"{name}: ratio={ratio:.8f}, limit={ratio_limit:.8f}, "
        f"max_abs={max_abs:.8f}"
    )


@pytest.mark.parametrize("case", CASES)
@torch.inference_mode()
def test_chunk_gdn2_matches_independent_torch(case):
    args, initial_state, scale = _make_inputs(case)
    q, k, v, g, b, w = args
    batch, length, heads, key_dim, value_dim = case["shape"]

    before = tuple(
        None if tensor is None else tensor.clone()
        for tensor in (*args, initial_state)
    )

    expected_output, expected_final = _torch_reference(
        *args,
        scale=scale,
        initial_state=initial_state,
        output_final_state=case["output_final_state"],
        state_v_first=case["state_v_first"],
    )

    module = importlib.import_module(
        "flag_attn.runtime.backend._metax.chunk_gdn2"
    )
    original_has_tle = module.HAS_TLE_GDN2
    original_tle = getattr(module, "chunk_gdn2_fwd_infer", None)
    original_native = module.chunk_gdn2_fwd
    route_counts = {"tle": 0, "fallback": 0}

    if case["route"] == "tle":
        if not original_has_tle or original_tle is None:
            pytest.skip("TLE GDN2 path is unavailable")
        chunk_size = 16
    else:
        chunk_size = 64

    def wrapped_tle(*wrapper_args, **wrapper_kwargs):
        route_counts["tle"] += 1
        return original_tle(*wrapper_args, **wrapper_kwargs)

    def wrapped_native(*wrapper_args, **wrapper_kwargs):
        route_counts["fallback"] += 1
        return original_native(*wrapper_args, **wrapper_kwargs)

    module.chunk_gdn2_fwd = wrapped_native
    if original_tle is not None:
        module.chunk_gdn2_fwd_infer = wrapped_tle
    module.HAS_TLE_GDN2 = case["route"] == "tle"

    results = []
    try:
        for _ in range(case["repeats"]):
            actual_output, actual_final = chunk_gdn2(
                *args,
                scale=scale,
                initial_state=initial_state,
                output_final_state=case["output_final_state"],
                state_v_first=case["state_v_first"],
                chunk_size=chunk_size,
            )
            torch.cuda.synchronize()
            results.append((actual_output, actual_final))
    finally:
        module.HAS_TLE_GDN2 = original_has_tle
        module.chunk_gdn2_fwd = original_native
        if original_tle is not None:
            module.chunk_gdn2_fwd_infer = original_tle

    assert module.HAS_TLE_GDN2 == original_has_tle
    assert module.chunk_gdn2_fwd is original_native
    if original_tle is not None:
        assert module.chunk_gdn2_fwd_infer is original_tle

    expected_tle_calls = (
        case["repeats"] if case["route"] == "tle" else 0
    )
    expected_fallback_calls = (
        case["repeats"] if case["route"] == "fallback" else 0
    )
    assert route_counts == {
        "tle": expected_tle_calls,
        "fallback": expected_fallback_calls,
    }

    expected_output_shape = (
        batch,
        length,
        heads,
        value_dim,
    )
    expected_state_shape = (
        (batch, heads, value_dim, key_dim)
        if case["state_v_first"]
        else (batch, heads, key_dim, value_dim)
    )

    first_output = None
    first_final = None

    for repeat_index, (actual_output, actual_final) in enumerate(results):
        assert actual_output.shape == expected_output_shape
        assert actual_output.dtype == case["dtype"]

        _assert_close(
            f"output[{repeat_index}]",
            actual_output,
            expected_output,
            ASSERT_RATIO,
        )

        if case["output_final_state"]:
            assert actual_final is not None
            assert expected_final is not None
            assert actual_final.shape == expected_state_shape
            assert actual_final.dtype == torch.float32
            _assert_close(
                f"final_state[{repeat_index}]",
                actual_final,
                expected_final,
                ASSERT_RATIO,
            )
        else:
            assert actual_final is None
            assert expected_final is None

        if repeat_index == 0:
            first_output = actual_output
            first_final = actual_final
        else:
            _assert_close(
                f"repeat_output[{repeat_index}]",
                actual_output,
                first_output,
                REPEAT_ASSERT_RATIO,
            )
            print(
                f"repeat_output_bitwise[{repeat_index}]="
                f"{torch.equal(actual_output, first_output)}"
            )
            if actual_final is not None:
                _assert_close(
                    f"repeat_state[{repeat_index}]",
                    actual_final,
                    first_final,
                    REPEAT_ASSERT_RATIO,
                )
                print(
                    f"repeat_state_bitwise[{repeat_index}]="
                    f"{torch.equal(actual_final, first_final)}"
                )

    current = (*args, initial_state)
    names = ("q", "k", "v", "g", "b", "w", "initial_state")
    for name, tensor, saved in zip(names, current, before, strict=True):
        if tensor is None:
            assert saved is None
        else:
            assert torch.equal(tensor, saved), f"{name} was modified"

    print("route_counts:", route_counts)
    print("inputs_unchanged=PASS")
