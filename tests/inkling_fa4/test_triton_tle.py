import pytest
import torch

from inkling_fa4.triton_tle_kernel import inkling_fa4_rel_attention_tle


HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


def reference_attention(q, key_cache, value_cache, rel_logits, *, q_lens,
                        kv_lens, block_table, scale, rel_extent, window_left):
    num_kv_heads = key_cache.shape[2]
    num_heads = q.shape[1]
    heads_per_kv_head = num_heads // num_kv_heads
    host_table = block_table.cpu().numpy()
    output = torch.empty_like(q)
    q_start = 0

    for seq, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        query = q[q_start:q_start + q_len].float()
        relative = rel_logits[q_start:q_start + q_len].float()
        nblocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        pages = host_table[seq, :nblocks]
        key = key_cache[pages].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len].float()
        value = value_cache[pages].reshape(-1, num_kv_heads, HEAD_DIM)[:kv_len].float()
        key = key.repeat_interleave(heads_per_kv_head, dim=1)
        value = value.repeat_interleave(heads_per_kv_head, dim=1)

        scores = torch.matmul(
            query.permute(1, 0, 2).contiguous(),
            key.permute(1, 2, 0).contiguous(),
        ) * scale
        q_pos = torch.arange(q_len, device=q.device).view(q_len, 1) + kv_len - q_len
        k_pos = torch.arange(kv_len, device=q.device).view(1, kv_len)
        distance = q_pos - k_pos
        in_range = (distance >= 0) & (distance < rel_extent)
        rel_index = distance.clamp(0, rel_extent - 1)
        rel_bias = relative.permute(1, 0, 2).gather(
            2, rel_index.unsqueeze(0).expand(num_heads, -1, -1)
        )
        scores += torch.where(in_range.unsqueeze(0), rel_bias,
                              torch.zeros_like(rel_bias))
        mask = distance < 0
        if window_left is not None:
            mask |= distance > window_left
        probabilities = torch.softmax(scores.masked_fill(mask.unsqueeze(0), -torch.inf), dim=-1)
        output[q_start:q_start + q_len] = torch.einsum(
            "hqk,khd->qhd", probabilities, value
        ).to(q.dtype)
        q_start += q_len
    return output


def run_case(seq_lens, num_heads, num_kv_heads, rel_extent, window_left,
             num_splits=1):
    torch.manual_seed(0)
    q_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    total_q = sum(q_lens)
    batch = len(seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, num_heads, HEAD_DIM, device="cuda", dtype=DTYPE)
    q = torch.nn.functional.normalize(q.float(), dim=-1).to(DTYPE)
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    key_cache = torch.randn(batch * max_blocks + 1, BLOCK_SIZE, num_kv_heads,
                            HEAD_DIM, device="cuda", dtype=DTYPE)
    key_cache = torch.nn.functional.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.zeros(batch, max_blocks, device="cuda", dtype=torch.int32)
    for seq in range(batch):
        block_table[seq] = torch.arange(
            1 + seq * max_blocks, 1 + (seq + 1) * max_blocks,
            device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, *torch.cumsum(torch.tensor(q_lens), 0).tolist()],
                        device="cuda", dtype=torch.int32)
    cache_lens = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    rel = torch.randn(total_q, num_heads, rel_extent, device="cuda", dtype=DTYPE)
    window = (-1, -1) if window_left is None else (window_left, 0)
    out = torch.empty_like(q)

    actual = inkling_fa4_rel_attention_tle(
        q, key_cache, value_cache, block_table=block_table,
        cache_seqlens=cache_lens, cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lens), softmax_scale=scale, causal=True,
        window_size=window, rel_extent=rel_extent, rel_logits=rel,
        num_splits=num_splits, out=out)
    expected = reference_attention(
        q, key_cache, value_cache, rel, q_lens=q_lens, kv_lens=kv_lens,
        block_table=block_table, scale=scale, rel_extent=rel_extent,
        window_left=window_left)
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@torch.inference_mode()
def test_tle_split_kv_decode(num_splits):
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("TLE path requires SM90+")
    run_case([(1, 4097), (1, 777)], 8, 2, 1024, None, num_splits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seq_lens,num_heads,num_kv_heads,rel_extent,window_left", [
    ([(64, 64)], 4, 4, 128, None),
    ([(64, 64), (33, 33), (17, 17)], 8, 2, 128, None),
    ([(200, 512), (50, 300), (1, 400)], 8, 2, 128, None),
    ([(64, 512)], 8, 2, 256, 255),
    ([(1, 50), (1, 7), (1, 200)], 8, 2, 128, None),
    ([(1, 1024)], 8, 2, 1024, None),
])
@torch.inference_mode()
def test_tle_relative_attention(seq_lens, num_heads, num_kv_heads,
                                rel_extent, window_left):
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("TLE path requires SM90+")
    run_case(seq_lens, num_heads, num_kv_heads, rel_extent, window_left)
