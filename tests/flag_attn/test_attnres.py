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

import pytest
import torch

from flag_attn.testing import backend as test_backend
import flag_attn


pytestmark = pytest.mark.fused_attnres


FLAG_ATTNRES_MODULE = importlib.import_module("flag_attn.FLA.attnres.fused")


def test_fused_attnres_public_export():
    from flag_attn import fused_attnres

    assert flag_attn.fused_attnres is fused_attnres
    backend_name = f"flag_attn.runtime.backend._{flag_attn.runtime.device.vendor_name}"
    try:
        backend = importlib.import_module(backend_name)
    except ModuleNotFoundError as exc:
        if exc.name != backend_name:
            raise
        backend = None

    exports = vars(backend).get("_OPERATOR_EXPORTS", {}) if backend is not None else {}
    generic_only = vars(backend).get("_GENERIC_ONLY_OPS", ()) if backend is not None else ()
    if "fused_attnres" in exports and "fused_attnres" not in generic_only:
        module_name, symbol = exports["fused_attnres"]
        expected = getattr(importlib.import_module(module_name, backend_name), symbol)
    else:
        expected = FLAG_ATTNRES_MODULE.fused_attnres
    assert fused_attnres is expected
    assert "fused_attnres" in flag_attn.__all__


@pytest.mark.parametrize(
    ("num_sources", "batch", "tokens", "hidden_size", "output_norm", "dtype"),
    [
        (1, 1, 3, 7168, False, torch.bfloat16),
        (5, 1, 3, 7168, False, torch.bfloat16),
        (9, 1, 3, 7168, True, torch.bfloat16),
        (5, 2, 7, 4096, True, torch.float16),
    ],
)
@torch.inference_mode()
def test_fused_attnres(
    num_sources: int,
    batch: int,
    tokens: int,
    hidden_size: int,
    output_norm: bool,
    dtype: torch.dtype,
):
    if not test_backend.supports_operator("fused_attnres", tle=True):
        pytest.skip("selected AttnRes implementation is unavailable")

    torch.manual_seed(42)
    residuals = [
        torch.randn(batch, tokens, hidden_size, device=test_backend.device, dtype=dtype) for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device=test_backend.device, dtype=dtype)
    rms_weight = torch.randn(hidden_size, device=test_backend.device, dtype=dtype)
    output_rms_weight = torch.randn(hidden_size, device=test_backend.device, dtype=dtype) if output_norm else None

    expected, expected_weights = flag_attn.testing.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )
    actual, actual_weights = flag_attn.fused_attnres(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        scale=hidden_size**-0.5,
        return_weights=True,
    )

    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(actual_weights, expected_weights, atol=5e-5, rtol=5e-5)


@torch.inference_mode()
def test_fused_attnres_without_weights():
    if not test_backend.supports_operator("fused_attnres", tle=True):
        pytest.skip("selected AttnRes implementation is unavailable")

    hidden_size = 7168
    residuals = [torch.randn(2, hidden_size, device=test_backend.device, dtype=torch.bfloat16) for _ in range(5)]
    query = torch.randn(hidden_size, device=test_backend.device, dtype=torch.bfloat16)
    rms_weight = torch.randn(hidden_size, device=test_backend.device, dtype=torch.bfloat16)

    output = flag_attn.fused_attnres(query, residuals, rms_weight, scale=hidden_size**-0.5)
    assert isinstance(output, torch.Tensor)
    assert output.shape == residuals[0].shape


def test_fused_attnres_rejects_empty_residuals():
    query = torch.empty(128)
    rms_weight = torch.empty(128)
    with pytest.raises(ValueError, match="at least one"):
        flag_attn.fused_attnres(query, [], rms_weight)
