"""Compatibility names for the Hygon pure-Triton prefill API."""

from .triton_fp8 import (
    attention_with_kvcache_blocksparse_prefill_fp8,
    attention_with_kvcache_blocksparse_prefill_fp8_hopper,
    attention_with_kvcache_blocksparse_prefill_fp8_tle,
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)

__all__ = [
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]
