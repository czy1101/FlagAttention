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

"""The Enflame Sage quantizer launch change preserves every output bit."""

import pytest
import torch
import triton

from flag_attn.runtime.backend import get_backend_name
from flag_attn.runtime.backend._enflame.sage_attention.quant_per_block import (
    per_block_int8,
    quant_per_block_int8_kernel,
)


pytestmark = pytest.mark.skipif(
    get_backend_name() != "enflame", reason="Enflame quantizer launch test"
)


def _legacy_quant(x, block, scale):
    batch, heads, length, dim = x.shape
    quantized = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(
        (batch, heads, triton.cdiv(length, block)),
        device=x.device,
        dtype=torch.float32,
    )
    quant_per_block_int8_kernel[(triton.cdiv(length, block), heads, batch)](
        x,
        quantized,
        scales,
        length,
        x.stride(0), x.stride(1), x.stride(2),
        quantized.stride(0), quantized.stride(1), quantized.stride(2),
        scales.stride(0), scales.stride(1),
        sm_scale=scale,
        C=dim,
        BLK=block,
        num_warps=4,
        num_stages=3,
    )
    return quantized, scales


@pytest.mark.parametrize("dim,q_len,k_len", [(64, 129, 257), (128, 300, 129)])
def test_enflame_sage_quant_launch_matches_legacy(dim, q_len, k_len):
    torch.manual_seed(2026)
    device = f"gcu:{torch.gcu.current_device()}"
    q = torch.randn(1, 2, q_len, dim, device=device, dtype=torch.float16)
    k = torch.randn(1, 2, k_len, dim, device=device, dtype=torch.float16)

    tuned_q, tuned_q_scale, tuned_k, tuned_k_scale = per_block_int8(q, k)
    old_q, old_q_scale = _legacy_quant(q, 128, dim**-0.5 * 1.44269504)
    old_k, old_k_scale = _legacy_quant(k, 64, 1.0)

    for tuned, old in (
        (tuned_q, old_q),
        (tuned_q_scale, old_q_scale),
        (tuned_k, old_k),
        (tuned_k_scale, old_k_scale),
    ):
        assert torch.equal(tuned, old)
