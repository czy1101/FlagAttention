"""Public Hygon hy3 prefill surface."""

from .triton_fp8 import attention_with_kvcache_blocksparse_prefill_fp8
from .compatibility import (
    attention_with_kvcache_blocksparse_prefill_fp8_hopper,
    attention_with_kvcache_blocksparse_prefill_fp8_tle,
    attention_with_kvcache_blocksparse_prefill_fp8_triton,
)

OP_NAME = "hygon_hy3_prefill_fp8"
IMPLEMENTATION = "pure-triton"
SUPPORTS_TLE = False

__all__ = [
    "IMPLEMENTATION",
    "OP_NAME",
    "SUPPORTS_TLE",
    "attention_with_kvcache_blocksparse_prefill_fp8",
    "attention_with_kvcache_blocksparse_prefill_fp8_hopper",
    "attention_with_kvcache_blocksparse_prefill_fp8_tle",
    "attention_with_kvcache_blocksparse_prefill_fp8_triton",
]
