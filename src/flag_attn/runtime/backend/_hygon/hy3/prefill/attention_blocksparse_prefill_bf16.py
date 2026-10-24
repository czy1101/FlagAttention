"""Public module path for the Hygon Hy3 BF16 prefill API."""

from .. import OP_NAME
from .compatibility import (
    attention_with_kvcache_blocksparse_prefill_bf16,
    attention_with_kvcache_blocksparse_prefill_bf16_triton,
)

IMPLEMENTATION = "pure-triton"
SUPPORTS_TLE = False

__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TLE",
    "attention_with_kvcache_blocksparse_prefill_bf16",
    "attention_with_kvcache_blocksparse_prefill_bf16_triton",
]
