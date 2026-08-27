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

"""Latency benchmark for the MetaX FlashMLA prefill/decode operator."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.knobs

from flag_attn.runtime.backend._metax import flash_mla


BATCHES = (32, 64, 128, 256, 512)
SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768)
BLOCK_SIZE = 64
S_Q = 1
H_Q = 128
H_KV = 1
D_QK = 576
D_V = 512


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


def _make_inputs(batch: int, s_kv: int):
    max_seqlen_pad = triton.cdiv(s_kv, 256) * 256
    pages_per_request = triton.cdiv(max_seqlen_pad, BLOCK_SIZE)
    q = torch.randn(
        (batch, S_Q, H_Q, D_QK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_table = torch.arange(
        batch * pages_per_request,
        device="cuda",
        dtype=torch.int32,
    ).view(batch, pages_per_request)
    blocked_k = torch.randn(
        (block_table.numel(), BLOCK_SIZE, H_KV, D_QK),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache_seqlens = torch.full(
        (batch,), s_kv, device="cuda", dtype=torch.int32
    )
    return q, block_table, blocked_k, max_seqlen_pad, cache_seqlens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-compatible device.")
    triton.knobs.autotuning.adjust_block_size = False
    torch.manual_seed(0)

    print("| batch | s_kv | FlagAttention MetaX latency (ms) |")
    print("| ---: | ---: | ---: |")
    for batch in BATCHES:
        for s_kv in SEQUENCE_LENGTHS:
            q, block_table, blocked_k, max_seqlen_pad, cache_seqlens = (
                _make_inputs(batch, s_kv)
            )
            fn = lambda: flash_mla(
                q,
                block_table,
                blocked_k,
                max_seqlen_pad,
                BLOCK_SIZE,
                batch,
                S_Q,
                cache_seqlens,
                H_Q,
                H_KV,
                D_QK,
                D_V,
                True,
            )
            latency = _bench(fn, args.warmup, args.rep)
            print(f"| {batch} | {s_kv} | {latency:.6f} |")
            del fn
            del q, block_table, blocked_k, cache_seqlens
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
