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
"""Independent Torch-reference tests for MetaX flash_mla_with_kvcache.

The case matrix and cache layouts are derived from FlagGems-vllm commit
f771e65aba3bba8f9683e409b5e6355e14213371.  No vLLM implementation is
imported: the expected output is computed by explicit Torch recurrences.
"""

import pytest
import torch

import flag_attn
from flag_attn.runtime.backend import is_metax_backend


SOURCE_REPOSITORY = "https://github.com/flagos-ai/FlagGems-vllm"
SOURCE_COMMIT = "f771e65aba3bba8f9683e409b5e6355e14213371"
SOURCE_PATH = "tests/test_flash_mla_with_kvcache.py"
SOURCE_SHA256 = "e87aef2710f29d939f964e77d30a8099535a282125bec2de1e937084339eb519"

DEVICE = "cuda"
FP8_MAX = 448.0

FlashMLASchedMeta = getattr(flag_attn, "FlashMLASchedMeta", None)
flash_mla_with_kvcache = getattr(flag_attn, "flash_mla_with_kvcache", None)
get_mla_metadata = getattr(flag_attn, "get_mla_metadata", None)
HAS_PUBLIC_API = all(
    item is not None
    for item in (FlashMLASchedMeta, flash_mla_with_kvcache, get_mla_metadata)
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_metax_backend() and HAS_PUBLIC_API),
    reason="MetaX C550 and the public FlashMLA-with-KV-cache API are required",
)


def _generate_v32_fp8_kv_cache(num_pages, page_block_size, seed):
    """Create the 656-byte V32 cache layout used by sparse decode."""
    torch.manual_seed(seed)
    total_tokens = num_pages * page_block_size
    nope = (
        torch.randn(
            total_tokens,
            1,
            512,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    groups = nope.reshape(total_tokens, 4, 128).float()
    scales = (groups.abs().amax(dim=-1) / FP8_MAX).clamp(min=1e-12)
    quantized = (groups / scales[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8 = quantized.reshape(total_tokens, 512).to(torch.float8_e4m3fn)
    rope = (
        torch.randn(
            total_tokens,
            1,
            64,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )

    cache = torch.zeros(
        num_pages,
        page_block_size,
        1,
        656,
        dtype=torch.uint8,
        device=DEVICE,
    )
    cache[..., :512] = fp8.view(torch.uint8).reshape(
        num_pages,
        page_block_size,
        1,
        512,
    )
    cache[..., 512:528] = (
        scales.reshape(num_pages, page_block_size, 1, 4)
        .to(torch.float32)
        .view(torch.uint8)
        .reshape(num_pages, page_block_size, 1, 16)
    )
    cache[..., 528:656] = rope.view(torch.uint8).reshape(
        num_pages,
        page_block_size,
        1,
        128,
    )
    return cache


def _decode_v32_fp8_kv_cache(cache):
    """Decode V32 bytes into independent Torch K and V tensors."""
    flat = cache.reshape(-1, 656)
    fp8 = flat[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    scales = (
        flat[:, 512:528]
        .contiguous()
        .view(torch.float32)
        .reshape(-1, 4)
    )
    nope = (
        fp8.reshape(-1, 4, 128) * scales[:, :, None]
    ).reshape(-1, 512).to(torch.bfloat16)
    rope = (
        flat[:, 528:656]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, 64)
    )
    key = torch.cat((nope, rope), dim=-1)
    return key, nope


def _generate_model1_fp8_kv_cache(num_pages, page_block_size, seed):
    """Create page-oriented MODEL1 data followed by E8M0 scale bytes."""
    torch.manual_seed(seed)
    total_tokens = num_pages * page_block_size
    nope = (
        torch.randn(
            total_tokens,
            448,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    groups = nope.reshape(total_tokens, 7, 64).float()
    block_max = groups.abs().amax(dim=-1).clamp(min=1e-4)
    exponent = torch.ceil(torch.log2(block_max / FP8_MAX))
    scales = torch.exp2(exponent)
    fp8 = (
        (groups / scales[:, :, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .reshape(total_tokens, 448)
        .to(torch.float8_e4m3fn)
    )
    rope = (
        torch.randn(
            total_tokens,
            64,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )

    token_data = torch.empty(
        total_tokens,
        576,
        dtype=torch.uint8,
        device=DEVICE,
    )
    token_data[:, :448] = fp8.view(torch.uint8).reshape(total_tokens, 448)
    token_data[:, 448:] = rope.view(torch.uint8).reshape(total_tokens, 128)
    encoded_scales = torch.zeros(
        total_tokens,
        8,
        dtype=torch.uint8,
        device=DEVICE,
    )
    encoded_scales[:, :7] = (exponent + 127.0).clamp(0, 255).to(torch.uint8)

    cache = torch.zeros(
        num_pages,
        page_block_size,
        1,
        584,
        dtype=torch.uint8,
        device=DEVICE,
    )
    page_flat = cache.view(num_pages, -1)
    page_flat[:, : page_block_size * 576] = token_data.reshape(
        num_pages,
        page_block_size * 576,
    )
    page_flat[:, page_block_size * 576 :] = encoded_scales.reshape(
        num_pages,
        page_block_size * 8,
    )
    return cache


def _decode_model1_fp8_kv_cache(cache):
    """Decode MODEL1 bytes; K and V are NoPE448 concatenated with RoPE64."""
    num_pages, page_block_size = cache.shape[:2]
    page_flat = cache.reshape(num_pages, -1)
    token_data = page_flat[:, : page_block_size * 576].reshape(-1, 576)
    encoded_scales = page_flat[:, page_block_size * 576 :].reshape(-1, 8)

    fp8 = token_data[:, :448].contiguous().view(torch.float8_e4m3fn).float()
    scale_bits = (encoded_scales[:, :7].to(torch.int32) << 23).contiguous()
    scales = scale_bits.view(torch.float32)
    nope = (
        fp8.reshape(-1, 7, 64) * scales[:, :, None]
    ).reshape(-1, 448).to(torch.bfloat16)
    rope = (
        token_data[:, 448:]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(-1, 64)
    )
    key_value = torch.cat((nope, rope), dim=-1)
    return key_value, key_value


def _snapshot_tensors(*values):
    return [
        (value, value.detach().clone())
        for value in values
        if isinstance(value, torch.Tensor)
    ]


def _assert_inputs_unchanged(snapshots):
    for current, original in snapshots:
        assert torch.equal(current, original)


def _new_metadata():
    metadata, aux = get_mla_metadata()
    assert aux is None
    assert isinstance(metadata, FlashMLASchedMeta)
    assert not metadata.have_initialized
    assert metadata.config is None
    return metadata


def _valid_selected(cache_key, cache_value, ids):
    ids = ids.to(torch.long)
    valid = (ids >= 0) & (ids < cache_key.shape[0])
    ids = ids[valid]
    return cache_key.index_select(0, ids), cache_value.index_select(0, ids)


def _sparse_torch_reference(
    q,
    key,
    value,
    indices,
    softmax_scale,
    topk_length=None,
    attn_sink=None,
    extra_key=None,
    extra_value=None,
    extra_indices=None,
    extra_topk_length=None,
):
    """Serial sparse attention reference, including sink and extra cache."""
    batch, seq_q, num_heads, _ = q.shape
    head_dim_v = value.shape[-1]
    output = torch.empty(
        batch,
        seq_q,
        num_heads,
        head_dim_v,
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        batch,
        num_heads,
        seq_q,
        dtype=torch.float32,
        device=q.device,
    )

    for batch_id in range(batch):
        main_length = (
            indices.shape[-1]
            if topk_length is None
            else int(topk_length[batch_id].item())
        )
        extra_length = 0
        if extra_indices is not None:
            extra_length = (
                extra_indices.shape[-1]
                if extra_topk_length is None
                else int(extra_topk_length[batch_id].item())
            )

        for query_id in range(seq_q):
            selected_key, selected_value = _valid_selected(
                key,
                value,
                indices[batch_id, query_id, :main_length],
            )
            if extra_indices is not None:
                selected_extra_key, selected_extra_value = _valid_selected(
                    extra_key,
                    extra_value,
                    extra_indices[batch_id, query_id, :extra_length],
                )
                selected_key = torch.cat((selected_key, selected_extra_key), dim=0)
                selected_value = torch.cat(
                    (selected_value, selected_extra_value),
                    dim=0,
                )

            if selected_key.shape[0] == 0:
                output[batch_id, query_id].zero_()
                lse[batch_id, :, query_id].fill_(float("inf"))
                continue

            scores = torch.einsum(
                "hd,nd->hn",
                q[batch_id, query_id].float(),
                selected_key.float(),
            ) * softmax_scale
            real_lse = torch.logsumexp(scores, dim=-1)
            normalizer = real_lse
            if attn_sink is not None:
                normalizer = torch.logaddexp(real_lse, attn_sink.float())
            probabilities = torch.exp(scores - normalizer[:, None])
            expected = torch.einsum(
                "hn,nd->hd",
                probabilities,
                selected_value.float(),
            )
            output[batch_id, query_id].copy_(expected.to(q.dtype))
            lse[batch_id, :, query_id].copy_(real_lse)
    return output, lse


def _dense_torch_reference(
    q,
    cache,
    block_table,
    cache_seqlens,
    head_dim_v,
    softmax_scale,
):
    """Gather paged dense KV and compute decode attention explicitly."""
    batch, seq_q, num_heads, _ = q.shape
    page_block_size = cache.shape[1]
    output = torch.empty(
        batch,
        seq_q,
        num_heads,
        head_dim_v,
        dtype=q.dtype,
        device=q.device,
    )
    lse = torch.empty(
        batch,
        num_heads,
        seq_q,
        dtype=torch.float32,
        device=q.device,
    )

    for batch_id in range(batch):
        sequence_length = int(cache_seqlens[batch_id].item())
        logical_ids = torch.arange(sequence_length, device=q.device)
        page_slots = torch.div(
            logical_ids,
            page_block_size,
            rounding_mode="floor",
        )
        offsets = logical_ids.remainder(page_block_size)
        physical_pages = block_table[batch_id].to(torch.long)[page_slots]
        kv = cache[physical_pages, offsets, 0]
        key = kv
        value = kv[:, :head_dim_v]

        for query_id in range(seq_q):
            scores = torch.einsum(
                "hd,nd->hn",
                q[batch_id, query_id].float(),
                key.float(),
            ) * softmax_scale
            current_lse = torch.logsumexp(scores, dim=-1)
            probabilities = torch.softmax(scores, dim=-1)
            expected = torch.einsum(
                "hn,nd->hd",
                probabilities,
                value.float(),
            )
            output[batch_id, query_id].copy_(expected.to(q.dtype))
            lse[batch_id, :, query_id].copy_(current_lse)
    return output, lse


def _assert_close(actual, expected, name, cosine_threshold):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    assert torch.isfinite(expected).all(), f"{name} reference is non-finite"
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    difference = actual_float - expected_float
    rms = torch.sqrt(torch.mean(difference.square())).item()
    reference_rms = torch.sqrt(torch.mean(expected_float.square())).item()
    rms_ratio = rms / max(reference_rms, 1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        actual_float,
        expected_float,
        dim=0,
    ).item()
    max_abs = difference.abs().max().item()
    print(
        f"{name}: max_abs={max_abs:.8f} rms_ratio={rms_ratio:.8f} "
        f"cosine={cosine:.8f}"
    )
    assert cosine > cosine_threshold
    assert rms_ratio < 0.08


@pytest.mark.parametrize(
    "batch,num_heads,topk,num_pages,use_sink",
    [
        pytest.param(1, 64, 64, 4, False, id="v32_h64_topk64"),
        pytest.param(2, 128, 128, 5, True, id="v32_h128_topk128_sink"),
    ],
)
def test_sparse_decode_v32_against_torch(
    batch,
    num_heads,
    topk,
    num_pages,
    use_sink,
):
    seq_q = 1
    page_block_size = 64
    softmax_scale = 576**-0.5
    torch.manual_seed(100 + num_heads)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        576,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    cache = _generate_v32_fp8_kv_cache(num_pages, page_block_size, 200)
    key, value = _decode_v32_fp8_kv_cache(cache)
    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    sink = None
    if use_sink:
        sink = torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    snapshots = _snapshot_tensors(q, cache, indices, sink)
    expected_out, expected_lse = _sparse_torch_reference(
        q,
        key,
        value,
        indices,
        softmax_scale,
        attn_sink=sink,
    )
    out_buffer = torch.empty_like(expected_out)
    metadata = _new_metadata()
    actual_out, actual_lse = flash_mla_with_kvcache(
        q,
        cache,
        None,
        None,
        512,
        metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=sink,
        out=out_buffer,
    )
    assert actual_out.data_ptr() == out_buffer.data_ptr()
    assert metadata.have_initialized
    assert metadata.config.b == batch
    _assert_close(actual_out, expected_out, "V32 output", 0.99)
    _assert_close(actual_lse, expected_lse, "V32 LSE", 0.99)
    _assert_inputs_unchanged(snapshots)


def test_sparse_decode_model1_against_torch():
    batch, seq_q, num_heads, topk = 2, 1, 64, 128
    num_pages, page_block_size = 5, 64
    softmax_scale = 512**-0.5
    torch.manual_seed(301)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        512,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    cache = _generate_model1_fp8_kv_cache(num_pages, page_block_size, 302)
    key, value = _decode_model1_fp8_kv_cache(cache)
    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.tensor([65, 127], dtype=torch.int32, device=DEVICE)
    sink = torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    snapshots = _snapshot_tensors(q, cache, indices, topk_length, sink)
    expected_out, expected_lse = _sparse_torch_reference(
        q,
        key,
        value,
        indices,
        softmax_scale,
        topk_length=topk_length,
        attn_sink=sink,
    )
    out_buffer = torch.empty_like(expected_out)
    metadata = _new_metadata()
    actual_out, actual_lse = flash_mla_with_kvcache(
        q,
        cache,
        None,
        None,
        512,
        metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=sink,
        out=out_buffer,
    )
    assert actual_out.data_ptr() == out_buffer.data_ptr()
    _assert_close(actual_out, expected_out, "MODEL1 output", 0.98)
    _assert_close(actual_lse, expected_lse, "MODEL1 LSE", 0.98)
    _assert_inputs_unchanged(snapshots)


def test_sparse_decode_model1_extra_kv_against_torch():
    batch, seq_q, num_heads = 1, 1, 64
    topk, extra_topk = 128, 512
    num_pages, page_block_size = 4, 64
    extra_num_pages, extra_page_block_size = 260, 2
    softmax_scale = 512**-0.5
    torch.manual_seed(401)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        512,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    cache = _generate_model1_fp8_kv_cache(num_pages, page_block_size, 402)
    extra_cache = _generate_model1_fp8_kv_cache(
        extra_num_pages,
        extra_page_block_size,
        403,
    )
    key, value = _decode_model1_fp8_kv_cache(cache)
    extra_key, extra_value = _decode_model1_fp8_kv_cache(extra_cache)
    indices = torch.randint(
        0,
        num_pages * page_block_size,
        (batch, seq_q, topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    extra_indices = torch.randint(
        0,
        extra_num_pages * extra_page_block_size,
        (batch, seq_q, extra_topk),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.tensor([97], dtype=torch.int32, device=DEVICE)
    extra_topk_length = torch.tensor([509], dtype=torch.int32, device=DEVICE)
    sink = torch.randn(num_heads, dtype=torch.float32, device=DEVICE)
    snapshots = _snapshot_tensors(
        q,
        cache,
        extra_cache,
        indices,
        extra_indices,
        topk_length,
        extra_topk_length,
        sink,
    )
    expected_out, expected_lse = _sparse_torch_reference(
        q,
        key,
        value,
        indices,
        softmax_scale,
        topk_length=topk_length,
        attn_sink=sink,
        extra_key=extra_key,
        extra_value=extra_value,
        extra_indices=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    metadata = _new_metadata()
    actual_out, actual_lse = flash_mla_with_kvcache(
        q,
        cache,
        None,
        None,
        512,
        metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=sink,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_length,
    )
    _assert_close(actual_out, expected_out, "MODEL1 extra output", 0.98)
    _assert_close(actual_lse, expected_lse, "MODEL1 extra LSE", 0.98)
    _assert_inputs_unchanged(snapshots)


def test_dense_decode_against_torch():
    batch, seq_q, num_heads = 2, 1, 128
    page_block_size = 64
    max_pages_per_sequence = 3
    total_pages = batch * max_pages_per_sequence
    softmax_scale = 576**-0.5
    torch.manual_seed(501)
    q = torch.randn(
        batch,
        seq_q,
        num_heads,
        576,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    cache = (
        torch.randn(
            total_pages,
            page_block_size,
            1,
            576,
            dtype=torch.bfloat16,
            device=DEVICE,
        )
        * 0.1
    )
    block_table = torch.arange(
        total_pages,
        dtype=torch.int32,
        device=DEVICE,
    ).reshape(batch, max_pages_per_sequence)
    cache_seqlens = torch.tensor([65, 129], dtype=torch.int32, device=DEVICE)
    snapshots = _snapshot_tensors(q, cache, block_table, cache_seqlens)
    expected_out, expected_lse = _dense_torch_reference(
        q,
        cache,
        block_table,
        cache_seqlens,
        512,
        softmax_scale,
    )
    metadata = _new_metadata()
    actual_out, actual_lse = flash_mla_with_kvcache(
        q,
        cache,
        block_table,
        cache_seqlens,
        512,
        metadata,
        num_splits=1,
        softmax_scale=softmax_scale,
        causal=True,
    )
    _assert_close(actual_out, expected_out, "dense output", 0.99)
    _assert_close(actual_lse, expected_lse, "dense LSE", 0.99)
    _assert_inputs_unchanged(snapshots)


def test_error_v32_rejects_topk_length():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    cache = _generate_v32_fp8_kv_cache(4, 64, 601)
    indices = torch.randint(
        0,
        256,
        (1, 1, 64),
        dtype=torch.int32,
        device=DEVICE,
    )
    topk_length = torch.ones(1, dtype=torch.int32, device=DEVICE)
    with pytest.raises(AssertionError, match="dynamic topk length"):
        flash_mla_with_kvcache(
            q,
            cache,
            None,
            None,
            512,
            _new_metadata(),
            is_fp8_kvcache=True,
            indices=indices,
            topk_length=topk_length,
        )


def test_error_model1_extra_cache_requires_extra_indices():
    q = torch.randn(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    cache = _generate_model1_fp8_kv_cache(4, 64, 701)
    extra_cache = _generate_model1_fp8_kv_cache(4, 64, 702)
    indices = torch.randint(
        0,
        256,
        (1, 1, 64),
        dtype=torch.int32,
        device=DEVICE,
    )
    with pytest.raises(AssertionError, match="extra_indices_in_kvcache"):
        flash_mla_with_kvcache(
            q,
            cache,
            None,
            None,
            512,
            _new_metadata(),
            is_fp8_kvcache=True,
            indices=indices,
            extra_k_cache=extra_cache,
        )


def test_error_dense_rejects_sparse_only_args():
    q = torch.randn(1, 1, 64, 576, dtype=torch.bfloat16, device=DEVICE)
    cache = torch.randn(
        4,
        64,
        1,
        576,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    block_table = torch.arange(4, dtype=torch.int32, device=DEVICE).reshape(1, 4)
    cache_seqlens = torch.full((1,), 64, dtype=torch.int32, device=DEVICE)
    sink = torch.randn(64, dtype=torch.float32, device=DEVICE)
    with pytest.raises(AssertionError, match="must be None when dense"):
        flash_mla_with_kvcache(
            q,
            cache,
            block_table,
            cache_seqlens,
            512,
            _new_metadata(),
            attn_sink=sink,
        )


def test_error_sched_meta_reuse_mismatch():
    q = torch.empty(1, 1, 64, 512, dtype=torch.bfloat16, device=DEVICE)
    cache = torch.empty(4, 64, 1, 584, dtype=torch.uint8, device=DEVICE)
    metadata = FlashMLASchedMeta(
        have_initialized=True,
        config=FlashMLASchedMeta.Config(
            b=2,
            s_q=1,
            h_q=64,
            page_block_size=64,
            h_k=1,
            causal=False,
            is_fp8_kvcache=True,
            topk=64,
            extra_page_block_size=None,
            extra_topk=None,
        ),
    )
    with pytest.raises(AssertionError, match="sched_meta.config.b"):
        flash_mla_with_kvcache(q, cache, None, None, 512, metadata)
