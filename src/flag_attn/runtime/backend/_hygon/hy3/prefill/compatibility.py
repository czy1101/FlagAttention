"""Stable import names for the Hygon Hy3 BF16/INT8 prefill APIs."""

from .core import (
    attention_with_kvcache_blocksparse_prefill_bf16,
    attention_with_kvcache_blocksparse_prefill_bf16_triton,
    attention_with_kvcache_blocksparse_prefill_int8,
    attention_with_kvcache_blocksparse_prefill_int8_triton,
)

__all__ = [
    "attention_with_kvcache_blocksparse_prefill_bf16",
    "attention_with_kvcache_blocksparse_prefill_bf16_triton",
    "attention_with_kvcache_blocksparse_prefill_int8",
    "attention_with_kvcache_blocksparse_prefill_int8_triton",
]
