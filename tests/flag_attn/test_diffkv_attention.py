# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

import pathlib
import sys

import pytest
import torch

# Keep the test runnable directly from a source checkout.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from flag_attn.diffkv_attention import diffkv_attention, unified_attention_diffkv


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="DiffKV Triton tests require CUDA"
)


def _reference(query, key_cache, value_cache, context_lens, block_tables,
               scale, window_size):
    batch, num_query_heads, _ = query.shape
    num_kv_heads = key_cache.shape[2]
    group = num_query_heads // num_kv_heads
    output = torch.empty(
        batch, num_query_heads, value_cache.shape[-1],
        device=query.device, dtype=torch.float32,
    )
    for b in range(batch):
        length = int(context_lens[b])
        blocks = block_tables[b, : (length + key_cache.shape[1] - 1) // key_cache.shape[1]]
        keys = key_cache[blocks].reshape(-1, num_kv_heads, key_cache.shape[-1])[:length]
        values = value_cache[blocks].reshape(-1, num_kv_heads, value_cache.shape[-1])[:length]
        positions = torch.arange(length, device=query.device)
        for h in range(num_query_heads):
            # Elementwise reduction avoids a cuBLAS GEMV dispatch for this
            # narrow (192-wide) vector and is numerically equivalent to the
            # reference dot product.
            scores = (
                keys[:, h // group, :].float() * query[b, h].float()
            ).sum(dim=-1) * scale
            if window_size > 0:
                scores = scores.masked_fill(
                    positions < length - window_size, float("-inf")
                )
            probs = torch.softmax(scores, dim=0)
            output[b, h] = torch.sum(probs[:, None] * values[:, h // group].float(), dim=0)
    return output.to(query.dtype)


def _make_case(batch, seq_len, dtype=torch.bfloat16):
    device = "cuda"
    num_query_heads, num_kv_heads = 64, 4
    head_size_qk, head_size_v, block_size = 192, 128, 16
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq
    query = torch.randn(
        batch, num_query_heads, head_size_qk, device=device, dtype=dtype
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size_qk,
        device=device, dtype=dtype,
    )
    value_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size_v,
        device=device, dtype=dtype,
    )
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(
        batch, blocks_per_seq
    )
    context_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    return query, key_cache, value_cache, context_lens, block_tables


def _run_baseline(query, key_cache, value_cache, context_lens, block_tables,
                  scale, window_size, path):
    """Invoke the standard Triton fallback with the production layout.

    The benchmark uses this path for shapes rejected by the TLE acceptance
    policy; keeping a direct correctness check here prevents a benchmark-only
    wrapper from hiding regressions in the fallback implementation.
    """
    batch, num_query_heads, _ = query.shape
    seq_len = int(context_lens.max().item())
    block_size = key_cache.shape[1]
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    out = torch.empty(
        batch, num_query_heads, value_cache.shape[-1],
        device=query.device, dtype=query.dtype,
    )
    use_3d = path == "3d"
    if use_3d:
        num_segments = 64 if batch == 1 and seq_len == 2048 else 16
        segm_output = torch.empty(
            batch, num_query_heads, num_segments,
            128, device=query.device, dtype=torch.float32,
        )
        segm_max = torch.empty(
            batch, num_query_heads, num_segments,
            device=query.device, dtype=torch.float32,
        )
        segm_expsum = torch.empty_like(segm_max)
        threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        threshold = None
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=context_lens,
        softmax_scale=scale,
        causal=True,
        window_size=(window_size - 1, 0) if window_size > 0 else (-1, -1),
        block_table=block_tables,
        softcap=0.0,
        max_seqlen_q=1,
        seq_threshold_3D=threshold,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        max_seqlen_k=seq_len,
        backend="triton",
    )
    return out


@pytest.mark.parametrize(
    "batch,seq_len", [(1, 512), (1, 2048), (1, 8192), (8, 2048), (32, 8192)]
)
@pytest.mark.parametrize("path", ["2d", "3d"])
@pytest.mark.parametrize("window_size", [-1, 128])
def test_diffkv_attention_matches_reference(batch, seq_len, path, window_size):
    torch.manual_seed(0)
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        batch, seq_len
    )
    scale = query.shape[-1] ** -0.5
    actual = diffkv_attention(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        attn_scale=scale,
        window_size=window_size,
        path=path,
    )
    expected = _reference(
        query,
        key_cache,
        value_cache,
        context_lens,
        block_tables,
        scale,
        window_size,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("batch,seq_len,path,window_size", [
    (1, 512, "3d", -1),
    (8, 2048, "3d", 128),
])
def test_diffkv_baseline_matches_reference(batch, seq_len, path, window_size):
    """Validate the standard Triton implementation used by benchmark fallback."""
    torch.manual_seed(0)
    query, key_cache, value_cache, context_lens, block_tables = _make_case(
        batch, seq_len
    )
    scale = query.shape[-1] ** -0.5
    actual = _run_baseline(
        query, key_cache, value_cache, context_lens, block_tables,
        scale, window_size, path,
    )
    expected = _reference(
        query, key_cache, value_cache, context_lens, block_tables,
        scale, window_size,
    )
    torch.testing.assert_close(actual.float(), expected.float(), atol=3e-2, rtol=3e-2)
