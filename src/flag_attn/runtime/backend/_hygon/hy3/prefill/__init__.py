"""Public Hygon Hy3 BF16/INT8 prefill surface."""

from .. import OP_NAME
from .core import (
    attention_with_kvcache_blocksparse_prefill,
    attention_with_kvcache_blocksparse_prefill_bf16,
    attention_with_kvcache_blocksparse_prefill_bf16_triton,
    attention_with_kvcache_blocksparse_prefill_int8,
    attention_with_kvcache_blocksparse_prefill_int8_triton,
)

IMPLEMENTATION = "pure-triton"
SUPPORTS_TLE = False

__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TLE",
    "attention_with_kvcache_blocksparse_prefill_bf16",
    "attention_with_kvcache_blocksparse_prefill_bf16_triton",
    "attention_with_kvcache_blocksparse_prefill",
    "attention_with_kvcache_blocksparse_prefill_int8",
    "attention_with_kvcache_blocksparse_prefill_int8_triton",
]
