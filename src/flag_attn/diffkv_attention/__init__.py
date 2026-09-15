# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffKV attention implemented with Triton kernels."""

from .diffkv_attention import diffkv_attention

__all__ = ["diffkv_attention"]
