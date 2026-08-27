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

"""Latency benchmark for the MetaX sparse FlashMLA operator."""

from __future__ import annotations

import argparse

import torch
import triton
import triton.knobs

from flag_attn.runtime.backend._metax import flash_mla_sparse_fwd


D_QK = 576
D_V = 512
H_KV = 1
CASES = (
    (1, 64, 4096, 512),
    (16, 64, 4096, 512),
    (64, 64, 4096, 512),
    (256, 64, 4096, 512),
    (1, 64, 8192, 1024),
    (16, 64, 8192, 1024),
    (64, 64, 8192, 1024),
    (256, 64, 8192, 1024),
    (1, 64, 32768, 2048),
    (16, 64, 32768, 2048),
    (64, 64, 32768, 2048),
    (256, 64, 32768, 2048),
    (512, 64, 32768, 2048),
    (1, 128, 32768, 2048),
    (16, 128, 32768, 2048),
    (64, 128, 32768, 2048),
    (256, 128, 32768, 2048),
    (512, 128, 32768, 2048),
)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA-compatible device.")
    triton.knobs.autotuning.adjust_block_size = False
    torch.manual_seed(0)

    print("| SQ | HQ | SKV | TOPK | FlagAttention MetaX latency (ms) |")
    print("| ---: | ---: | ---: | ---: | ---: |")
    for s_q, h_q, s_kv, topk in CASES:
        q = torch.randn(
            (s_q, h_q, D_QK), device="cuda", dtype=torch.bfloat16
        ).div_(10)
        kv = torch.randn(
            (s_kv, H_KV, D_QK), device="cuda", dtype=torch.bfloat16
        ).div_(10)
        indices = torch.randint(
            0,
            s_kv,
            (s_q, H_KV, topk),
            device="cuda",
            dtype=torch.int32,
        )
        fn = lambda: flash_mla_sparse_fwd(
            q,
            kv,
            indices,
            0.5,
            D_V,
            None,
            None,
        )
        latency = _bench(fn, args.warmup, args.rep)
        print(f"| {s_q} | {h_q} | {s_kv} | {topk} | {latency:.6f} |")
        del fn
        del q, kv, indices
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
