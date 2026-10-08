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

"""Local Inkling FA4 correctness tests against an FP32 PyTorch reference.

Parametrized Triton/TLE cases cover FP16/BF16, paged KV, GQA, ragged prefill,
decode, sliding windows, non-causal attention and split-KV reduction.
Only unavailable hardware or the optional TLE dependency cause skips.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from flag_attn import inkling_fa4_rel_attention
from flag_attn.inkling_fa4 import tle_available
from flag_attn.testing.inkling_fa4 import ref_rel_attn

HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPES = (torch.float16, torch.bfloat16)

BACKENDS = ("triton", "tle")

pytestmark = [
    pytest.mark.inkling_fa4_rel_attention,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

CASES = [
    ([(64, 64)], 4, 4, 128, None),
    ([(64, 64), (33, 33), (17, 17)], 8, 2, 128, None),
    ([(200, 512), (50, 300), (1, 400)], 8, 2, 128, None),
    ([(64, 512)], 8, 2, 256, 255),
    ([(1, 50), (1, 7), (1, 200)], 8, 2, 128, None),
    ([(1, 1024)], 8, 2, 1024, None),
]


@pytest.fixture(params=BACKENDS)
def operator(request):
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        pytest.skip(f"Tensor Core tests require SM80+, got {capability}")
    if request.param == "tle":
        if capability[0] != 9:
            pytest.skip(f"Hopper TLE requires SM90, got {capability}")
        available, reason = tle_available()
        if not available:
            pytest.skip(f"TLE dependency unavailable: {reason}")
    return partial(inkling_fa4_rel_attention, backend=request.param)


def run_case(
    operator,
    seq_lens,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
    num_splits: int = 1,
    *,
    dtype: torch.dtype = torch.bfloat16,
    causal: bool = True,
    use_out: bool = True,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    q_lens = [ql for ql, _ in seq_lens]
    kv_lens = [kl for _, kl in seq_lens]
    total_q = sum(q_lens)
    num_seqs = len(seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, num_heads, HEAD_DIM, device=device, dtype=dtype)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(dtype)

    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(dtype)
    value_cache = torch.randn(num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=dtype)

    # Non-contiguous physical pages catch accidental assumptions about cache layout.
    block_table = torch.randperm(num_blocks - 1, device=device).reshape(num_seqs, max_blocks).to(torch.int32)

    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    rel_logits = torch.randn(total_q, num_heads, rel_extent, device=device, dtype=dtype)
    window_size = (-1, -1) if window_left is None else (window_left, 0)

    out = torch.empty_like(q) if use_out else None
    actual = operator(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(q_lens),
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        num_splits=num_splits,
        out=out,
    )

    assert actual.shape == q.shape
    assert actual.dtype == dtype
    if out is not None:
        assert actual.data_ptr() == out.data_ptr()
    expected = ref_rel_attn(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(actual.float(), expected.float(), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@torch.inference_mode()
def test_split_kv_decode(operator, dtype: torch.dtype, num_splits: int) -> None:
    """Exercise paged split-KV reduction, including uneven split boundaries."""
    run_case(operator, [(1, 4097), (1, 777)], 8, 2, 1024, None, num_splits, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "seq_lens,num_heads,num_kv_heads,rel_extent,window_left",
    CASES,
    ids=["prefill", "ragged_prefill", "chunked_prefill", "sliding_window", "ragged_decode", "decode_1k"],
)
@torch.inference_mode()
def test_relative_attention(
    operator,
    dtype: torch.dtype,
    seq_lens,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
) -> None:
    run_case(operator, seq_lens, num_heads, num_kv_heads, rel_extent, window_left, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@torch.inference_mode()
def test_noncausal_attention(operator, dtype: torch.dtype) -> None:
    """Check non-causal semantics and the operator-allocated output path."""
    run_case(operator, [(17, 33), (7, 19)], 8, 2, 128, None, dtype=dtype, causal=False, use_out=False)
