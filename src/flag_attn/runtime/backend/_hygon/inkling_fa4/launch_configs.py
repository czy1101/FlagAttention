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

"""Hygon gfx936 launch presets for Inkling FA4 relative attention."""

from __future__ import annotations

from functools import lru_cache

from flag_attn.runtime.backend._hygon.heuristics_config_utils import (
    get_heuristics_config,
)
from flag_attn.runtime.backend._hygon.utils import round_up


@lru_cache(maxsize=1)
def _get_tuned_kernel_config() -> dict[str, int]:
    """Load the curated FA4 launch config using the backend config loader."""
    try:
        from flag_attn.runtime import get_tuned_config

        configs = get_tuned_config("_fa4_rel_attn_paged_kernel")
    except (ImportError, OSError, TypeError, ValueError):
        configs = []
    if not configs:
        return {"BLOCK_Q": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}

    config = configs[0]
    return {
        **config.kwargs,
        "num_warps": config.num_warps or 4,
        "num_stages": config.num_stages or 2,
    }


def get_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int,
    head_dim_q: int,
    head_dim_v: int,
    q_heads_per_kv_head: int,
) -> dict[str, int]:
    """Conservative launch configuration for Hygon BW/gfx936."""
    del max_seqlen_k
    tuned = _get_tuned_kernel_config()
    scheduler_rows = max_seqlen_q * q_heads_per_kv_head
    wide = max(head_dim_q, head_dim_v) > 128
    block_q = 16 if scheduler_rows <= 16 or wide else tuned["BLOCK_Q"]
    return {
        "BLOCK_Q": block_q,
        "BLOCK_K": tuned["BLOCK_K"],
        "num_warps": tuned["num_warps"],
        "num_stages": tuned["num_stages"],
    }


def should_auto_split(
    *, max_seqlen_q: int, max_seqlen_k: int, block_k: int
) -> bool:
    """Apply the shared FA4 decode split heuristic for the active Hygon device."""
    heuristics = get_heuristics_config("inkling_fa4_rel_attention")
    if max_seqlen_q > heuristics.get("auto_split_max_seqlen_q", 4):
        return False
    min_k_blocks = heuristics.get("auto_split_min_k_blocks", 8)
    return block_k > 0 and round_up(max_seqlen_k, block_k) // block_k >= min_k_blocks


def get_auto_num_splits(
    requested: int,
    *,
    max_seqlen_k: int,
    block_k: int,
    base_programs: int,
    multiprocessors: int,
    min_blocks_per_split: int | None = None,
    max_splits: int | None = None,
) -> int:
    """Split long KV work when the unsplit CTA grid underfills the device.

    Keep at least ``min_blocks_per_split`` K tiles in each partition so split
    reduction overhead is amortized. Explicit caller values greater than one
    are preserved. The selected count is a power of two for regular partitioning.
    """
    heuristics = get_heuristics_config("inkling_fa4_rel_attention")
    if min_blocks_per_split is None:
        min_blocks_per_split = heuristics.get("auto_split_min_blocks_per_split", 32)
    if max_splits is None:
        max_splits = heuristics.get("auto_split_max_splits", 64)
    if requested != 1 or block_k <= 0 or base_programs <= 0 or multiprocessors <= 0:
        return requested
    num_k_blocks = round_up(max_seqlen_k, block_k) // block_k
    split_limit = min(max_splits, num_k_blocks // min_blocks_per_split)
    if split_limit < 2 or base_programs >= multiprocessors:
        return 1
    desired = (multiprocessors + base_programs - 1) // base_programs
    target = 1 << (desired - 1).bit_length()
    split_limit = 1 << (split_limit.bit_length() - 1)
    return min(target, split_limit)
