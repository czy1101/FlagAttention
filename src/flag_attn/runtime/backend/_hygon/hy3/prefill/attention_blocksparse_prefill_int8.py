"""Public module path for the Hygon Hy3 INT8 prefill API."""

from .. import OP_NAME
from .compatibility import (
    attention_with_kvcache_blocksparse_prefill_int8,
    attention_with_kvcache_blocksparse_prefill_int8_triton,
)

IMPLEMENTATION = "pure-triton"
SUPPORTS_TLE = False

__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TLE",
    "attention_with_kvcache_blocksparse_prefill_int8",
    "attention_with_kvcache_blocksparse_prefill_int8_triton",
]
