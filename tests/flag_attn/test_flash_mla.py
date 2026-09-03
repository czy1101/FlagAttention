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
"""Correctness tests for the public MetaX FlashMLA decode operator.

Adapted from FlagGems-vllm at SOURCE_COMMIT. The Torch reference in this
version follows block_table explicitly, including non-identity page mappings.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch
import triton.knobs

from flag_attn.runtime.backend import is_metax_backend

SOURCE_REPOSITORY = "flagos-ai/FlagGems-vllm"
SOURCE_COMMIT = "f771e65aba3bba8f9683e409b5e6355e14213371"
SOURCE_PATH = "tests/test_flash_mla.py"

S_Q = 1
H_KV = 1
D_QK = 576
D_V = 512
BLOCK_SIZE = 64


def _load_public_operator():
    from flag_attn import flash_mla as operator

    return operator


IS_METAX = torch.cuda.is_available() and is_metax_backend()
flash_mla = _load_public_operator() if IS_METAX else None

pytestmark = pytest.mark.skipif(
    not IS_METAX,
    reason="requires the MetaX backend and a CUDA-compatible MetaX device",
)


@dataclasses.dataclass(frozen=True)
class FlashMlaCase:
    cache_seqlens: tuple[int, ...]
    h_q: int
    permute_pages: bool
    seed: int


CASES = [
    pytest.param(
        FlashMlaCase(
            cache_seqlens=(95,),
            h_q=64,
            permute_pages=False,
            seed=101,
        ),
        id="b1_s95_h64_tail",
    ),
    pytest.param(
        FlashMlaCase(
            cache_seqlens=(1024, 1026, 1028, 1030),
            h_q=128,
            permute_pages=True,
            seed=102,
        ),
        id="b4_s1024_mixed_permuted",
    ),
    pytest.param(
        FlashMlaCase(
            cache_seqlens=(2048, 2050),
            h_q=128,
            permute_pages=False,
            seed=103,
        ),
        id="b2_s2048_mixed",
    ),
    pytest.param(
        FlashMlaCase(
            cache_seqlens=(4096, 4098),
            h_q=128,
            permute_pages=True,
            seed=104,
        ),
        id="b2_s4096_mixed_permuted",
    ),
    pytest.param(
        FlashMlaCase(
            cache_seqlens=(8192,),
            h_q=128,
            permute_pages=False,
            seed=105,
        ),
        id="b1_s8192_long",
    ),
]


@pytest.fixture(autouse=True)
def _restore_process_state(monkeypatch):
    precision = torch.get_float32_matmul_precision()
    adjust_block_size = triton.knobs.autotuning.adjust_block_size

    monkeypatch.delenv(
        "FLAGGEMS_VLLM_FLASH_MLA_FORCE_TRITON",
        raising=False,
    )
    torch.set_float32_matmul_precision("highest")
    triton.knobs.autotuning.adjust_block_size = False

    try:
        yield
    finally:
        triton.knobs.autotuning.adjust_block_size = adjust_block_size
        torch.set_float32_matmul_precision(precision)


def _random_bfloat16(shape, generator):
    return torch.randn(
        shape,
        dtype=torch.float32,
        device="cpu",
        generator=generator,
    ).to(device="cuda", dtype=torch.bfloat16)


def _make_inputs(case):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    batch = len(case.cache_seqlens)
    max_seqlen = max(case.cache_seqlens)
    max_seqlen_pad = ((max_seqlen + 255) // 256) * 256
    pages_per_request = max_seqlen_pad // BLOCK_SIZE
    total_pages = batch * pages_per_request

    q = _random_bfloat16(
        (batch, S_Q, case.h_q, D_QK),
        generator,
    )
    blocked_k = _random_bfloat16(
        (total_pages, BLOCK_SIZE, H_KV, D_QK),
        generator,
    )

    page_ids = torch.arange(total_pages, dtype=torch.int32)
    if case.permute_pages:
        page_ids = torch.flip(page_ids, dims=(0,))
    block_table = page_ids.reshape(batch, pages_per_request).to("cuda")

    cache_seqlens = torch.tensor(
        case.cache_seqlens,
        dtype=torch.int32,
        device="cuda",
    )
    return (
        q,
        block_table.contiguous(),
        blocked_k,
        max_seqlen_pad,
        cache_seqlens,
    )


def _scaled_dot_product_attention(query, key, value, h_q, h_kv, causal):
    query = query.float()
    key = key.float().repeat_interleave(h_q // h_kv, dim=0)
    value = value.float().repeat_interleave(h_q // h_kv, dim=0)

    weights = query @ key.transpose(-2, -1)
    weights /= math.sqrt(query.size(-1))

    if causal:
        s_q = query.shape[-2]
        s_k = key.shape[-2]
        allowed = torch.ones(
            (s_q, s_k),
            dtype=torch.bool,
            device=query.device,
        ).tril(diagonal=s_k - s_q)
        weights = weights.masked_fill(
            allowed.logical_not(),
            float("-inf"),
        )

    probabilities = torch.softmax(
        weights,
        dim=-1,
        dtype=torch.float32,
    )
    return probabilities @ value


def _reference_flash_mla(
    q,
    block_table,
    blocked_k,
    cache_seqlens,
    block_size,
    h_q,
    h_kv,
    d_v,
    causal,
):
    outputs = []

    for batch_index in range(q.shape[0]):
        sequence_length = int(cache_seqlens[batch_index].item())
        page_count = (sequence_length + block_size - 1) // block_size
        physical_pages = block_table[
            batch_index,
            :page_count,
        ].to(torch.int64)

        kv = blocked_k.index_select(0, physical_pages)
        kv = kv.reshape(-1, h_kv, blocked_k.shape[-1])
        kv = kv[:sequence_length]

        output = _scaled_dot_product_attention(
            q[batch_index].transpose(0, 1),
            kv.transpose(0, 1),
            kv[..., :d_v].transpose(0, 1),
            h_q=h_q,
            h_kv=h_kv,
            causal=causal,
        )
        outputs.append(output.transpose(0, 1))

    return torch.stack(outputs, dim=0)


def _snapshot_inputs(*tensors):
    return tuple(tensor.detach().clone() for tensor in tensors)


def _assert_inputs_unchanged(before, *after):
    assert len(before) == len(after)
    for index, (expected, actual) in enumerate(zip(before, after)):
        assert torch.equal(actual, expected), f"input {index} was modified"


def _difference_metrics(actual, reference):
    actual64 = actual.double()
    reference64 = reference.double()

    delta = actual64 - reference64
    rms = delta.square().mean().sqrt().item()
    max_abs = delta.abs().max().item()
    denominator = max(
        (actual64.square() + reference64.square()).sum().item(),
        1e-12,
    )
    cosine_difference = (
        1.0
        - 2.0 * (actual64 * reference64).sum().item() / denominator
    )
    return {
        "rms": rms,
        "max_abs": max_abs,
        "cosine_difference": cosine_difference,
    }


@pytest.mark.parametrize("case", CASES)
def test_flash_mla(case):
    assert flash_mla is not None

    (
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        cache_seqlens,
    ) = _make_inputs(case)

    inputs_before = _snapshot_inputs(
        q,
        block_table,
        blocked_k,
        cache_seqlens,
    )

    reference = _reference_flash_mla(
        q,
        block_table,
        blocked_k,
        cache_seqlens,
        BLOCK_SIZE,
        case.h_q,
        H_KV,
        D_V,
        True,
    )

    actual = flash_mla(
        q,
        block_table,
        blocked_k,
        max_seqlen_pad,
        BLOCK_SIZE,
        len(case.cache_seqlens),
        S_Q,
        cache_seqlens,
        case.h_q,
        H_KV,
        D_QK,
        D_V,
        True,
    )

    assert isinstance(actual, torch.Tensor)
    assert actual.shape == reference.shape
    assert actual.shape == (
        len(case.cache_seqlens),
        S_Q,
        case.h_q,
        D_V,
    )
    assert actual.dtype == torch.bfloat16
    assert actual.device == q.device
    assert torch.isfinite(actual).all().item()

    metrics = _difference_metrics(actual, reference)
    print(f"case={case} metrics={metrics}")
    assert metrics["cosine_difference"] < 1e-5, metrics

    torch.testing.assert_close(
        actual.float(),
        reference,
        atol=1e-2,
        rtol=0.016,
    )
    _assert_inputs_unchanged(
        inputs_before,
        q,
        block_table,
        blocked_k,
        cache_seqlens,
    )
