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

# Heuristics keyed by operation. Add per-device overrides only after measuring
# them on the corresponding Hygon device.
HEURISTICS_CONFIGS = {
    "fused_attnres": {
        "default": {"join_reductions_max_rows": 1024},
        "by_sm_count": {},
    },
    "inkling_fa4_rel_attention": {
        "default": {
            "auto_split_max_seqlen_q": 4,
            "auto_split_min_k_blocks": 8,
            "auto_split_min_blocks_per_split": 32,
            "auto_split_max_splits": 64,
        },
        "by_sm_count": {},
    },
}


def get_heuristics_config(op_name: str) -> dict:
    """Return the operation heuristics selected for the active Hygon device."""
    from flag_attn.runtime.backend._hygon.device_info import get_device_info

    op_heuristics = HEURISTICS_CONFIGS.get(op_name, {})
    defaults = op_heuristics.get("default", {})
    device_overrides = op_heuristics.get("by_sm_count", {}).get(
        get_device_info().sm_count, {}
    )
    return {**defaults, **device_overrides}
