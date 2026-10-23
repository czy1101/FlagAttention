# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend GDN v1 (chunked gated delta rule) forward implementation."""

from .api import chunk_gated_delta_rule
from .chunk import chunk_gated_delta_rule_fwd

__all__ = ["chunk_gated_delta_rule", "chunk_gated_delta_rule_fwd"]
