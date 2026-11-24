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

# AttnRes policy keyed by operation. The default preserves the current Hygon
# threshold; add per-device overrides only after measuring them.
HEURISTICS_CONFIGS = {
    "fused_attnres": {
        "default": {"join_reductions_max_rows": 1024},
        "by_sm_count": {},
    },
    "forgetting_attention": {
        "default": {
            "fused_start_max_m": 128,
            "fused_binary_start_min_m": 129,
            "fused_binary_start_max_query_tiles": 16,
        },
        # Add CU-count overrides here after measuring on the target SKU.
        "by_sm_count": {},
    },
}


def get_heuristics_config(op_name: str) -> dict:
    """Return defaults merged with overrides for the active Hygon device."""
    op_config = HEURISTICS_CONFIGS.get(op_name, {})
    config = dict(op_config.get("default", {}))
    if not op_config:
        return config

    from flag_attn.runtime.backend._hygon.device_info import get_device_info

    sm_count = get_device_info().sm_count
    overrides = op_config.get("by_sm_count", {}).get(
        sm_count, op_config.get("by_sm_count", {}).get(str(sm_count), {})
    )
    return config | overrides
