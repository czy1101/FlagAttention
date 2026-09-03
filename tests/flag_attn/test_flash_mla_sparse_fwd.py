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
from __future__ import annotations

import dataclasses
import random
from typing import List, Optional, Tuple

import pytest
import torch

from flag_attn.runtime.backend import is_metax_backend

SOURCE_REPOSITORY = "flagos-ai/FlagGems-vllm"
SOURCE_COMMIT = "f771e65aba3bba8f9683e409b5e6355e14213371"
SOURCE_PATH = "tests/test_flash_mla_sparse_fwd.py"


def _load_public_operator():
    from flag_attn import flash_mla_sparse_fwd as operator

    return operator


IS_METAX = torch.cuda.is_available() and is_metax_backend()
flash_mla_sparse_fwd = _load_public_operator() if IS_METAX else None

pytestmark = pytest.mark.skipif(
    not IS_METAX,
    reason="requires the MetaX backend and a CUDA-compatible MetaX device",
)


@dataclasses.dataclass
class FlashMlaSparseTestParam:
    s_q: int
    s_kv: int
    topk: int
    h_q: int = 128
    h_kv: int = 1
    d_qk: int = 512
    d_v: int = 512
    is_all_indices_invalid: bool = False
    num_warmup: int = 5
    num_runs: int = 10
    have_attn_sink: bool = False
    have_topk_length: bool = False
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device("cuda")


_flashmla_sparse_counter = 0

class FlashmlaSparseTestKit:
    @staticmethod
    def _merge_two_lse(
        lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
    ) -> torch.Tensor:
        if lse1 is None:
            return lse0

        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )

    @staticmethod
    def torch_flash_mla_sparse_fwd(
        s_q: int,
        s_kv: int,
        h_q: int,
        h_kv: int,
        d_qk: int,
        topk: int,
        q: torch.Tensor,  # [s_q, h_q, d_qk]
        kv: torch.Tensor,  # [s_q, 1, d_qk]
        indices: torch.Tensor,  # [s_q, 1, topk]
        sm_scale: float,
        d_v: int,
        attn_sink: Optional[torch.Tensor],  # [h_q]
        topk_length: Optional[torch.Tensor],  # [s_q]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        - o: [s_q, h_q, dv]
        - o_fp32: [s_q, h_q, dv]
        - max_logits: [s_q, h_q]
        - lse: [s_q, h_q]
        """
        indices = indices.clone().squeeze(1)
        if topk_length is not None:
            mask = torch.arange(topk, device=topk_length.device).unsqueeze(
                0
            ).broadcast_to(s_q, topk) >= topk_length.unsqueeze(1)
            indices[mask] = -1
        invalid_mask = (indices < 0) | (indices >= s_kv)
        indices[invalid_mask] = 0
        q = q.float()
        gathered_kv = (
            kv.index_select(dim=0, index=indices.flatten())
            .reshape(s_q, topk, d_qk)
            .float()
        )
        P = q @ gathered_kv.transpose(1, 2)
        P *= sm_scale
        P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

        orig_lse = torch.logsumexp(P, dim=-1)
        max_logits = P.max(dim=-1).values

        lse_for_o = FlashmlaSparseTestKit._merge_two_lse(orig_lse, attn_sink, s_q, h_q)
        if not torch.is_inference_mode_enabled():
            lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float("-inf")] = float(
            "+inf"
        )  # So that corresponding O will be 0
        s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
        out = s_for_o @ gathered_kv[..., :d_v]

        lonely_q_mask = orig_lse == float("-inf")
        orig_lse[lonely_q_mask] = float("+inf")
        return (out.to(torch.bfloat16), max_logits, orig_lse)

    @staticmethod
    def _init_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    @staticmethod
    def make_input(param: FlashMlaSparseTestParam):
        """Create input data for sparse MLA operator"""
        S = param.s_q
        H = param.h_q
        DQK = param.d_qk
        SKV = param.s_kv
        HKV = param.h_kv
        topk = param.topk
        dtype = param.dtype
        device = param.device
        requires_grad = False

        FlashmlaSparseTestKit._init_seed(42)

        q = torch.randn((S, H, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )
        kv = torch.randn((SKV, HKV, DQK), dtype=dtype, device=device).requires_grad_(
            requires_grad
        )

        indices = torch.full((S, HKV, topk), SKV, dtype=torch.int32, device=device)
        for t in range(S):
            for h in range(HKV):
                i_i = torch.randperm(max(1, t))[:topk]
                indices[t, h, : len(i_i)] = i_i

        return q, kv, indices

    @staticmethod
    def _randperm_batch(
        batch_size: int,
        perm_range: torch.Tensor,
        perm_size: int,
        paddings: List[int],
    ) -> torch.Tensor:
        """
        Generate random permutations in batch
        The return tensor, denoted as `res`, has a shape of [batch_size, perm_size]. `0 <= res[i, :] < perm_range[i]`
        holds.
        Values within each row are unique.
        If, for some `i`, `perm_range[i] < perm_size` holds, then `res[i, :]` contains values in `[0, perm_range[i])`
        as many as possible, and the rest are filled with `padding`.
        """
        assert not torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        perm_range_max = max(int(torch.max(perm_range).item()), perm_size)
        rand = torch.rand(batch_size, perm_range_max, dtype=torch.float32)
        rand[
            torch.arange(0, perm_range_max).broadcast_to(batch_size, perm_range_max)
            >= perm_range.view(batch_size, 1)
        ] = float("-inf")
        res = rand.topk(perm_size, dim=-1, sorted=True).indices.to(torch.int32)
        if len(paddings) == 1:
            res[res >= perm_range.view(batch_size, 1)] = paddings[0]
        else:
            fillers = torch.tensor(paddings, dtype=torch.int32).index_select(
                0,
                torch.randint(0, len(paddings), (res.numel(),), dtype=torch.int32),
            )
            res.masked_scatter_(res >= perm_range.view(batch_size, 1), fillers)
        torch.use_deterministic_algorithms(False)
        return res

    @staticmethod
    def make_input_flashmla(param: FlashMlaSparseTestParam):
        """Create input data for sparse MLA operator by referring to the FlashMLA examples"""
        s_q = param.s_q
        s_kv = param.s_kv
        h_q = param.h_q
        h_kv = param.h_kv
        d_qk = param.d_qk
        topk = param.topk
        have_attn_sink = param.have_attn_sink
        have_topk_length = param.have_topk_length
        is_all_indices_invalid = param.is_all_indices_invalid
        dtype = param.dtype
        device = param.device

        global _flashmla_sparse_counter
        FlashmlaSparseTestKit._init_seed(_flashmla_sparse_counter)
        _flashmla_sparse_counter = _flashmla_sparse_counter + 1

        q = (
            torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        kv = (
            torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device) / 10
            + (random.random() - 0.5) / 10
        )
        q = q.clamp_(-10, 10)
        kv = kv.clamp_(-10, 10)
        invalid_indices_candidate = [
            -2147483648,
            -123456,
            -1,
            s_kv,
            114514,
            1919810,
            2147480000,
            2147483647,
        ]
        indices = FlashmlaSparseTestKit._randperm_batch(
            s_q,
            torch.full((s_q,), s_kv, dtype=torch.int32),
            topk,
            invalid_indices_candidate,
        ).view(s_q, h_kv, topk)
        if is_all_indices_invalid:
            all_indices_invalid_mask = torch.randn(s_q, device="cpu") < -2
            indices[
                all_indices_invalid_mask[:, None, None].broadcast_to(indices.shape)
            ] = random.choice(invalid_indices_candidate)
        indices = indices.to(device)

        attn_sink = None
        if have_attn_sink:
            attn_sink = torch.randn((h_q,), dtype=torch.float32, device=device)
            mask = torch.randn((h_q,), dtype=torch.float32, device=device)
            attn_sink[mask < -0.5] = float("-inf")
            attn_sink[mask > +0.5] = float("+inf")

        topk_length = None
        if have_topk_length:
            topk_length = torch.randint(
                0, max(topk + 1, 64), (s_q,), dtype=torch.int32, device=device
            ).clamp_max(topk)
        return q, kv, indices, attn_sink, topk_length

@pytest.fixture(autouse=True)
def _restore_process_state():
    global _flashmla_sparse_counter

    precision = torch.get_float32_matmul_precision()
    deterministic = torch.are_deterministic_algorithms_enabled()
    python_random_state = random.getstate()
    cpu_rng_state = torch.random.get_rng_state()
    counter = _flashmla_sparse_counter

    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)

    try:
        yield
    finally:
        _flashmla_sparse_counter = counter
        torch.random.set_rng_state(cpu_rng_state)
        random.setstate(python_random_state)
        torch.use_deterministic_algorithms(deterministic)
        torch.set_float32_matmul_precision(precision)


BASIC_CASES = [
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=1,
            s_kv=95,
            topk=128,
            h_q=64,
            h_kv=1,
            d_qk=576,
            d_v=512,
        ),
        id="basic_sq1_skv95_topk128_h64_d576",
    ),
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=64,
            s_kv=1024,
            topk=128,
            h_q=64,
            h_kv=1,
            d_qk=576,
            d_v=512,
        ),
        id="basic_sq64_skv1024_topk128_h64_d576",
    ),
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=128,
            s_kv=2048,
            topk=256,
            h_q=128,
            h_kv=1,
            d_qk=576,
            d_v=512,
        ),
        id="basic_sq128_skv2048_topk256_h128_d576",
    ),
]

FLASHMLA_CASES = [
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=1,
            s_kv=592,
            topk=128,
            h_q=64,
            d_qk=512,
        ),
        id="flash_sq1_skv592_topk128_h64_d512",
    ),
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=16,
            s_kv=1521,
            topk=512,
            h_q=64,
            d_qk=576,
            have_attn_sink=True,
        ),
        id="flash_sq16_skv1521_topk512_sink",
    ),
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=62,
            s_kv=1840,
            topk=256,
            h_q=128,
            d_qk=576,
            have_attn_sink=True,
            have_topk_length=True,
        ),
        id="flash_sq62_skv1840_topk256_sink_length",
    ),
    pytest.param(
        FlashMlaSparseTestParam(
            s_q=16,
            s_kv=95,
            topk=128,
            h_q=64,
            d_qk=576,
            is_all_indices_invalid=True,
            have_attn_sink=True,
            have_topk_length=True,
        ),
        id="flash_sq16_all_invalid_sink_length",
    ),
]


def _snapshot_inputs(*tensors):
    return tuple(
        None if tensor is None else tensor.detach().clone()
        for tensor in tensors
    )


def _assert_inputs_unchanged(before, *after):
    assert len(before) == len(after)
    for index, (expected, actual) in enumerate(zip(before, after)):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected), f"input {index} was modified"


def _assert_output_contract(result, param, q):
    assert isinstance(result, tuple)
    assert len(result) == 3

    output, max_logits, lse = result
    assert output.shape == (param.s_q, param.h_q, param.d_v)
    assert max_logits.shape == (param.s_q, param.h_q)
    assert lse.shape == (param.s_q, param.h_q)

    assert output.dtype == param.dtype
    assert max_logits.dtype == torch.float32
    assert lse.dtype == torch.float32

    assert output.device == q.device
    assert max_logits.device == q.device
    assert lse.device == q.device

    assert torch.isfinite(output).all().item()
    assert not torch.isnan(max_logits).any().item()
    assert not torch.isnan(lse).any().item()
    return output, max_logits, lse


def _assert_special_value_masks(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.equal(torch.isposinf(actual), torch.isposinf(expected))
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def _assert_close(actual, expected, *, atol, rtol):
    torch.testing.assert_close(
        actual,
        expected.to(dtype=actual.dtype),
        atol=atol,
        rtol=rtol,
        equal_nan=False,
    )


@pytest.mark.parametrize("param", BASIC_CASES)
def test_flashmla_sparse(param):
    assert flash_mla_sparse_fwd is not None

    q, kv, indices = FlashmlaSparseTestKit.make_input(param)
    inputs_before = _snapshot_inputs(q, kv, indices)
    sm_scale = param.d_qk**-0.5

    reference = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        param.s_q,
        param.s_kv,
        param.h_q,
        param.h_kv,
        param.d_qk,
        param.topk,
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
        None,
        None,
    )
    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
    )

    output, max_logits, lse = _assert_output_contract(actual, param, q)
    ref_output, ref_max_logits, ref_lse = reference

    _assert_special_value_masks(max_logits, ref_max_logits)
    _assert_special_value_masks(lse, ref_lse)
    _assert_close(output, ref_output, atol=1e-2, rtol=0.016)
    _assert_close(max_logits, ref_max_logits, atol=1e-4, rtol=1.3e-6)
    _assert_close(lse, ref_lse, atol=1e-4, rtol=1.3e-6)
    _assert_inputs_unchanged(inputs_before, q, kv, indices)


@pytest.mark.parametrize("param", FLASHMLA_CASES)
def test_flash_mla_sparse_flashmla(param):
    assert flash_mla_sparse_fwd is not None

    q, kv, indices, attn_sink, topk_length = (
        FlashmlaSparseTestKit.make_input_flashmla(param)
    )

    if param.is_all_indices_invalid:
        assert topk_length is not None
        indices[0].fill_(-1)
        topk_length[0] = param.topk
        invalid = (indices[0] < 0) | (indices[0] >= param.s_kv)
        assert invalid.all().item()

    inputs_before = _snapshot_inputs(
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
    )
    sm_scale = 0.5

    reference = FlashmlaSparseTestKit.torch_flash_mla_sparse_fwd(
        param.s_q,
        param.s_kv,
        param.h_q,
        param.h_kv,
        param.d_qk,
        param.topk,
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
        attn_sink,
        topk_length,
    )
    actual = flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        param.d_v,
        attn_sink,
        topk_length,
    )

    output, max_logits, lse = _assert_output_contract(actual, param, q)
    ref_output, ref_max_logits, ref_lse = reference

    _assert_special_value_masks(max_logits, ref_max_logits)
    _assert_special_value_masks(lse, ref_lse)
    _assert_close(output, ref_output, atol=8e-4, rtol=3.01 / 128)
    _assert_close(
        max_logits,
        ref_max_logits,
        atol=1e-6,
        rtol=2.01 / 65536,
    )
    _assert_close(
        lse,
        ref_lse,
        atol=1e-6,
        rtol=2.01 / 65536,
    )
    _assert_inputs_unchanged(
        inputs_before,
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
    )
