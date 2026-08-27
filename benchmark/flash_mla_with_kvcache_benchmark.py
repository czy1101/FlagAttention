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

"""Dense decode benchmark for MetaX FlashMLA with paged KV cache."""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.knobs

from flag_attn.runtime.backend._metax import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)


BATCH = 128
H_Q = 128
H_KV = 1
D_QK = 576
D_V = 512
PAGE_BLOCK_SIZE = 64
BASE_SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)


def _bench(fn, warmup: int, rep: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _make_inputs(base_seqlen: int):
    cache_seqlens = torch.tensor(
        [base_seqlen + 2 * i for i in range(BATCH)],
        device="cuda",
        dtype=torch.int32,
    )
    max_seqlen = base_seqlen + 2 * (BATCH - 1)
    pages_per_request = math.ceil(max_seqlen / PAGE_BLOCK_SIZE)
    total_pages = BATCH * pages_per_request
    q = torch.randn(
        (BATCH, 1, H_Q, D_QK), device="cuda", dtype=torch.bfloat16
    ).div_(10)
    k_cache = torch.randn(
        (total_pages, PAGE_BLOCK_SIZE, H_KV, D_QK),
        device="cuda",
        dtype=torch.bfloat16,
    ).div_(10)
    block_table = torch.arange(
        total_pages, device="cuda", dtype=torch.int32
    ).view(BATCH, pages_per_request)
    return q, k_cache, block_table, cache_seqlens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-compatible device.")
    triton.knobs.autotuning.adjust_block_size = False
    torch.manual_seed(0)
    sched_meta, _ = get_mla_metadata()

    print("| mean KV | FlagAttention MetaX latency (ms) |")
    print("| ---: | ---: |")
    for base_seqlen in BASE_SEQUENCE_LENGTHS:
        q, k_cache, block_table, cache_seqlens = _make_inputs(base_seqlen)
        fn = lambda: flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            D_V,
            sched_meta,
            causal=True,
        )
        latency = _bench(fn, args.warmup, args.rep)
        mean_kv = base_seqlen + BATCH - 1
        print(f"| {mean_kv} | {latency:.6f} |")
        del fn
        del q, k_cache, block_table, cache_seqlens
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
