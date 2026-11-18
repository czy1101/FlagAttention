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

from __future__ import annotations

import torch
import triton


def get_preset(*, head_dim_q: int, head_dim_v: int) -> dict[str, int]:
    wide = max(head_dim_q, head_dim_v) > 128
    return {
        "BLOCK_Q": 16,
        "BLOCK_K": 16 if wide else 32,
        "num_warps": 4 if wide else 2,
        "num_stages": 1,
    }


def select_num_splits(
    *,
    num_splits: int | None,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_k: int,
    query_blocks: int,
    batch_size: int,
    scheduler_heads: int,
    head_dim_q: int,
    head_dim_v: int,
    is_local: bool,
    device: torch.device,
) -> int:
    if num_splits is not None:
        return num_splits
    kv_blocks = triton.cdiv(max_seqlen_k, block_k)
    if is_local or kv_blocks < 8:
        return 1
    groups = query_blocks * batch_size * scheduler_heads
    resident_groups = 4 if max(head_dim_q, head_dim_v) <= 128 else 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    parallel_limit = max(1, resident_groups * num_sms // groups)
    minimum_blocks = 1 if max_seqlen_q == 1 else 2
    work_limit = max(1, kv_blocks // minimum_blocks)
    split_limit = min(parallel_limit, work_limit, 128)
    blocks_per_split = triton.cdiv(kv_blocks, split_limit)
    return triton.cdiv(kv_blocks, blocks_per_split)


def get_kernel_options(
    *,
    head_dim_q: int,
    head_dim_v: int,
    split_kv: bool,
    max_seqlen_q: int,
) -> dict[str, str]:
    if (
        (head_dim_q <= 128 or max(head_dim_q, head_dim_v) <= 256)
        and (split_kv or max_seqlen_q > 1)
    ):
        return {"pipeline": ""}
    return {}


def get_combine_config(
    *,
    output_rows: int,
    head_dim_v: int,
    num_splits: int,
) -> tuple[int, int]:
    padded_splits = triton.next_power_of_2(num_splits)
    dimension_limit = 16 if output_rows < 32 and num_splits >= 16 else 128
    element_limit = max(1, 16384 // padded_splits)
    block_d = min(dimension_limit, triton.next_power_of_2(head_dim_v), element_limit)
    elements = padded_splits * block_d
    warps = min(4, triton.next_power_of_2(triton.cdiv(elements, 4096)))
    return block_d, warps
