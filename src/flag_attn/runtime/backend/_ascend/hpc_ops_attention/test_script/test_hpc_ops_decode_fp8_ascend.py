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

"""Ascend correctness coverage for both NVIDIA HY3 FP8 decode formats."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch_npu  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_dynamic as qk_dynamic,
    fp8_qpertoken_perhead_kvpertensor_dynamic as qt_dynamic,
)
from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.static import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_static as qk_static,
    fp8_qpertoken_perhead_kvpertensor_static as qt_static,
)


BLOCK_SIZE = 64
HEAD_DIM = 128


@dataclass
class _Panel:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    reference_q: torch.Tensor
    reference_k: torch.Tensor
    reference_v: torch.Tensor


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.uint8)


def _as_hnd(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _make_panel(
    quant_type: str,
    mtp: int,
    kv_len: int,
    hkv: int,
    layout: str,
) -> _Panel:
    """Build E4M3 on CPU and transfer its byte representation to the NPU."""
    torch.manual_seed(41 + mtp + kv_len + hkv)
    hq = hkv * 8
    blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    q_bf16 = torch.randn(mtp, hq, HEAD_DIM) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.abs().amax(-1) / 10.0
    q_fp8 = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)
    reference_q = q_fp8.float() * q_scale[..., None]

    if quant_type == "qpertoken_perhead_kvpertensor":
        k_fp8 = (
            torch.randn(blocks, BLOCK_SIZE, hkv, HEAD_DIM)
            / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        v_fp8 = torch.randn(
            blocks, BLOCK_SIZE, hkv, HEAD_DIM,
        ).to(torch.float8_e4m3fn)
        k_scale = torch.tensor([0.7], dtype=torch.float32)
        v_scale = torch.tensor([0.4], dtype=torch.float32)
        k = _bits(k_fp8)
        v = _bits(v_fp8)
        reference_k = k_fp8.float() * k_scale
        reference_v = v_fp8.float() * v_scale
    else:
        raw_k = torch.randn(blocks, BLOCK_SIZE, hkv, HEAD_DIM)
        raw_v = torch.randn(blocks, BLOCK_SIZE, hkv, HEAD_DIM)
        token_scale = raw_k.abs().amax(-1) / 448.0
        k_fp8 = (raw_k / token_scale[..., None]).to(torch.float8_e4m3fn)
        packed_scale = (
            token_scale.permute(0, 2, 1)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(blocks, hkv, 2, HEAD_DIM)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k_storage = torch.empty(
            blocks, BLOCK_SIZE + 2, hkv, HEAD_DIM, dtype=torch.uint8,
        )
        k_storage[:, :BLOCK_SIZE] = _bits(k_fp8)
        k_storage[:, BLOCK_SIZE:] = _bits(packed_scale)
        head_scale = (
            raw_v.abs().permute(2, 0, 1, 3).reshape(hkv, -1).amax(-1)
            / 448.0
        )
        v_fp8 = (raw_v / head_scale[None, None, :, None]).to(
            torch.float8_e4m3fn
        )
        k = k_storage[:, :BLOCK_SIZE]
        v = _bits(v_fp8)
        k_scale = k_storage[:, BLOCK_SIZE:]
        v_scale = head_scale
        reference_k = k_fp8.float() * token_scale[..., None]
        reference_v = v_fp8.float() * head_scale[None, None, :, None]

    if layout == "HND":
        k = _as_hnd(k)
        v = _as_hnd(v)
        if quant_type == "qkpertoken_perhead_vperhead":
            k_scale = _as_hnd(k_scale)
    return _Panel(
        q=_bits(q_fp8).npu(),
        k=k.npu(),
        v=v.npu(),
        block_ids=torch.arange(blocks, dtype=torch.int32)[None].npu(),
        kv_lens=torch.tensor([kv_len], dtype=torch.int32).npu(),
        q_scale=q_scale.npu(),
        k_scale=k_scale.npu(),
        v_scale=v_scale.npu(),
        reference_q=reference_q,
        reference_k=reference_k,
        reference_v=reference_v,
    )


def _reference(panel: _Panel, mtp: int, kv_len: int, hkv: int) -> torch.Tensor:
    hq = hkv * 8
    k = (
        panel.reference_k.reshape(-1, hkv, HEAD_DIM)[:kv_len]
        .transpose(0, 1)
        .repeat_interleave(8, dim=0)
    )
    v = (
        panel.reference_v.reshape(-1, hkv, HEAD_DIM)[:kv_len]
        .transpose(0, 1)
        .repeat_interleave(8, dim=0)
    )
    scores = torch.einsum("mhd,hnd->hmn", panel.reference_q, k)
    scores /= math.sqrt(HEAD_DIM)
    query_position = kv_len - mtp + torch.arange(mtp)
    causal = torch.arange(kv_len)[None, :] <= query_position[:, None]
    scores.masked_fill_(~causal[None], -float("inf"))
    weights = torch.exp(scores - scores.amax(-1, keepdim=True))
    denominator = weights.sum(-1, keepdim=True)
    # Match the NVIDIA HY3 contract: probabilities are rounded through E4M3
    # after multiplication by 256, then rescaled after PV.
    weights = (weights * 256.0).to(torch.float8_e4m3fn).float()
    return (
        torch.einsum("hmn,hnd->mhd", weights, v)
        / denominator.transpose(0, 1)
        / 256.0
    ).reshape(mtp, hq, HEAD_DIM).to(torch.bfloat16)


_MODULES = {
    ("qpertoken_perhead_kvpertensor", "dynamic"): qt_dynamic,
    ("qpertoken_perhead_kvpertensor", "static"): qt_static,
    ("qkpertoken_perhead_vperhead", "dynamic"): qk_dynamic,
    ("qkpertoken_perhead_vperhead", "static"): qk_static,
}


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "quant_type,schedule,mtp,kv_len,hkv,layout",
    [
        ("qpertoken_perhead_kvpertensor", "dynamic", 1, 64, 1, "NHD"),
        ("qpertoken_perhead_kvpertensor", "static", 2, 130, 1, "HND"),
        ("qpertoken_perhead_kvpertensor", "dynamic", 4, 4096, 4, "HND"),
        ("qkpertoken_perhead_vperhead", "static", 1, 64, 1, "NHD"),
        ("qkpertoken_perhead_vperhead", "dynamic", 2, 4096, 1, "NHD"),
        ("qkpertoken_perhead_vperhead", "static", 4, 130, 4, "HND"),
    ],
)
@torch.no_grad()
def test_hy3_fp8_decode_ascend(
    quant_type: str,
    schedule: str,
    mtp: int,
    kv_len: int,
    hkv: int,
    layout: str,
) -> None:
    implementation: ModuleType = _MODULES[quant_type, schedule]
    panel = _make_panel(quant_type, mtp, kv_len, hkv, layout)
    inputs = implementation.FP8DecodeInputs(
        panel.q,
        panel.k,
        panel.v,
        panel.block_ids,
        panel.kv_lens,
        panel.q_scale,
        panel.k_scale,
        panel.v_scale,
    )
    workspace = implementation.prepare_decode_workspace(inputs)
    actual = implementation.attention_decode_fp8(inputs, workspace).clone()
    torch.npu.synchronize()
    expected = _reference(panel, mtp, kv_len, hkv)
    atol = 0.2 if quant_type == "qpertoken_perhead_kvpertensor" else 0.1
    torch.testing.assert_close(actual.cpu(), expected, atol=atol, rtol=1e-5)
    assert torch.isfinite(actual).all()
    assert implementation.workspace_is_reset(workspace)
