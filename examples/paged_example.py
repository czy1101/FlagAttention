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

import torch
from typing import Tuple

import flag_attn

MAX_SEQ_LEN = 4096
NUM_BLOCKS = 2000


def test_paged_attention(
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    device: int,
):
    torch.set_default_dtype(dtype)
    torch.set_default_device(device=device)

    torch.cuda.manual_seed(seed)

    num_query_heads, num_kv_heads = num_heads

    context_lens = torch.randint(1, MAX_SEQ_LEN, [num_seqs], dtype=torch.int32)
    max_context_len = context_lens.max().item()
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size

    attn_scale = head_size**-0.5
    q = torch.empty(num_seqs, num_query_heads, head_size)
    q.uniform_(-attn_scale, attn_scale)

    k_cache = torch.empty(NUM_BLOCKS, num_kv_heads, block_size, head_size)
    k_cache.uniform_(-attn_scale, attn_scale)
    v_cache = torch.empty_like(k_cache)
    v_cache.uniform_(-attn_scale, attn_scale)

    # (NUM_SEQS, MAX_NUM_BLOCKS_PER_SEQ)
    block_tables = torch.randint(0, NUM_BLOCKS, (num_seqs, max_num_blocks_per_seq))

    out = flag_attn.paged_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
    )

    ref_out = flag_attn.testing.paged_attention(
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        attn_scale,
    )
    print(torch.abs(out - ref_out).max())
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-5)


def main():
    test_paged_attention(
        num_seqs=32,
        num_heads=(64, 64),
        head_size=64,
        block_size=16,
        dtype=torch.float16,
        seed=1,
        device="cuda:0",
    )


if __name__ == "__main__":
    main()
