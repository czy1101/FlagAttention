import pytest
import torch

from inkling_fa4.triton_kernel import inkling_fa4_rel_attention


HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def reference_attention(
    q,
    key_cache,
    value_cache,
    rel_logits,
    *,
    q_lens,
    kv_lens,
    block_table,
    scale,
    rel_extent,
    window_left,
):
    num_kv_heads = key_cache.shape[2]
    num_heads = q.shape[1]
    heads_per_kv_head = num_heads // num_kv_heads
    host_block_table = block_table.cpu().numpy()
    output = torch.empty_like(q)

    query_start = 0
    for sequence_index, (query_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        query = q[query_start : query_start + query_len].float()
        relative = rel_logits[query_start : query_start + query_len].float()
        num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        physical_blocks = host_block_table[sequence_index, :num_blocks]
        key = key_cache[physical_blocks].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len].float()
        value = value_cache[physical_blocks].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len].float()
        key = key.repeat_interleave(heads_per_kv_head, dim=1)
        value = value.repeat_interleave(heads_per_kv_head, dim=1)

        scores = torch.matmul(
        query.permute(1, 0, 2).contiguous(),
        key.permute(1, 2, 0).contiguous(),
    ) * scale
        query_position = (
            torch.arange(query_len, device=q.device).view(query_len, 1)
            + kv_len
            - query_len
        )
        key_position = torch.arange(kv_len, device=q.device).view(1, kv_len)
        relative_distance = query_position - key_position
        relative_in_range = (
            (relative_distance >= 0) & (relative_distance < rel_extent)
        )
        relative_index = relative_distance.clamp(0, rel_extent - 1)
        relative_bias = relative.permute(1, 0, 2).gather(
            2,
            relative_index.unsqueeze(0).expand(num_heads, -1, -1),
        )
        scores += torch.where(
            relative_in_range.unsqueeze(0),
            relative_bias,
            torch.zeros_like(relative_bias),
        )

        mask = relative_distance < 0
        if window_left is not None:
            mask |= relative_distance > window_left
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        output[query_start : query_start + query_len] = torch.einsum(
            "hqk,khd->qhd",
            probabilities,
            value,
        ).to(q.dtype)
        query_start += query_len

    return output


def run_case(seq_lens, num_heads, num_kv_heads, rel_extent, window_left,
             num_splits=1):
    torch.manual_seed(0)
    q_lens = [query_len for query_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    total_q = sum(q_lens)
    num_sequences = len(seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, num_heads, HEAD_DIM, device="cuda", dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_sequences * max_blocks + 1
    key_cache = torch.randn(
        num_blocks,
        BLOCK_SIZE,
        num_kv_heads,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(
        num_sequences,
        max_blocks,
        device="cuda",
        dtype=torch.int32,
    )
    for sequence_index in range(num_sequences):
        block_table[sequence_index] = torch.arange(
            1 + sequence_index * max_blocks,
            1 + (sequence_index + 1) * max_blocks,
            device="cuda",
            dtype=torch.int32,
        )

    cu_seqlens_q = torch.tensor(
        [0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    cache_seqlens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    rel_logits = torch.randn(
        total_q,
        num_heads,
        rel_extent,
        device="cuda",
        dtype=DTYPE,
    )
    window_size = (-1, -1) if window_left is None else (window_left, 0)
    output_buffer = torch.empty_like(q)
    actual = inkling_fa4_rel_attention(
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
        out=output_buffer,
    )
    expected = reference_attention(
        q,
        key_cache,
        value_cache,
        rel_logits,
        q_lens=q_lens,
        kv_lens=kv_lens,
        block_table=block_table,
        scale=scale,
        rel_extent=rel_extent,
        window_left=window_left,
    )
    assert actual.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@torch.inference_mode()
def test_split_kv_decode(num_splits):
    """Exercise the formerly untested two-kernel reduction path."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("requires BF16 Tensor Cores")
    run_case([(1, 4097), (1, 777)], 8, 2, 1024, None, num_splits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "seq_lens,num_heads,num_kv_heads,rel_extent,window_left",
    [
        ([(64, 64)], 4, 4, 128, None),
        ([(64, 64), (33, 33), (17, 17)], 8, 2, 128, None),
        ([(200, 512), (50, 300), (1, 400)], 8, 2, 128, None),
        ([(64, 512)], 8, 2, 256, 255),
        ([(1, 50), (1, 7), (1, 200)], 8, 2, 128, None),
        ([(1, 1024)], 8, 2, 1024, None),
    ],
)
@torch.inference_mode()
def test_triton_relative_attention(
    seq_lens,
    num_heads,
    num_kv_heads,
    rel_extent,
    window_left,
):
    capability = torch.cuda.get_device_capability()
    if capability < (8, 0):
        pytest.skip(f"BF16 Tensor Core path requires SM80+, got {capability}")
    run_case(
        seq_lens,
        num_heads,
        num_kv_heads,
        rel_extent,
        window_left,
    )
