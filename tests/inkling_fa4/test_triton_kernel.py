"""Correctness tests for the Triton paged relative-attention kernels.

Every case runs against both backends through a single parametrized
implementation: ``triton`` (portable) and ``tle`` (Hopper warp-specialized).
A backend whose sources cannot be imported, or whose device requirement is not
met, is skipped instead of failing the suite.
"""

from __future__ import annotations

import pytest
import torch

from inkling_fa4 import backend as backend_dispatch
from inkling_fa4.reference import ref_rel_attn

HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16

BACKENDS = ("triton", "tle")

CASES = [
    ([(64, 64)], 4, 4, 128, None),
    ([(64, 64), (33, 33), (17, 17)], 8, 2, 128, None),
    ([(200, 512), (50, 300), (1, 400)], 8, 2, 128, None),
    ([(64, 512)], 8, 2, 256, 255),
    ([(1, 50), (1, 7), (1, 200)], 8, 2, 128, None),
    ([(1, 1024)], 8, 2, 1024, None),
]


def _load(name: str) -> tuple[object | None, str]:
    try:
        return backend_dispatch.get_backend(name), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


RESOLVED = {name: _load(name) for name in BACKENDS}


def _operator(name: str):
    operator, reason = RESOLVED[name]
    if operator is None:
        pytest.skip(f"{name} backend unavailable ({reason})")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        pytest.skip(f"BF16 Tensor Core path requires SM80+, got {capability}")
    if name == "tle" and capability < (9, 0):
        pytest.skip(f"TLE path requires SM90+, got {capability}")
    return operator


def run_case(
    operator,
    seq_lens,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
    num_splits: int = 1,
) -> None:
    torch.manual_seed(0)
    device = "cuda"
    q_lens = [ql for ql, _ in seq_lens]
    kv_lens = [kl for _, kl in seq_lens]
    total_q = sum(q_lens)
    num_seqs = len(seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)

    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seqs * max_blocks + 1
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(
        num_blocks, BLOCK_SIZE, num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )

    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    for seq in range(num_seqs):
        block_table[seq] = torch.arange(
            1 + seq * max_blocks,
            1 + (seq + 1) * max_blocks,
            dtype=torch.int32,
            device=device,
        )

    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    rel_logits = torch.randn(
        total_q, num_heads, rel_extent, device=device, dtype=DTYPE
    )
    window_size = (-1, -1) if window_left is None else (window_left, 0)

    out = torch.empty_like(q)
    actual = operator(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(q_lens),
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
        num_splits=num_splits,
        out=out,
    )

    assert actual.data_ptr() == out.data_ptr()
    expected = ref_rel_attn(
        q,
        key_cache,
        value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        rel_extent=rel_extent,
        rel_logits=rel_logits,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@torch.inference_mode()
def test_split_kv_decode(backend: str, num_splits: int) -> None:
    """Exercise the split-KV two-kernel reduction path."""
    operator = _operator(backend)
    run_case(operator, [(1, 4097), (1, 777)], 8, 2, 1024, None, num_splits)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "seq_lens,num_heads,num_kv_heads,rel_extent,window_left", CASES
)
@torch.inference_mode()
def test_relative_attention(
    backend: str,
    seq_lens,
    num_heads: int,
    num_kv_heads: int,
    rel_extent: int,
    window_left: int | None,
) -> None:
    operator = _operator(backend)
    run_case(
        operator, seq_lens, num_heads, num_kv_heads, rel_extent, window_left
    )
