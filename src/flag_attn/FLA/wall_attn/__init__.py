# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Wall-Attention prefill/training, KV-cache decode, and correctness reference."""

from flag_attn.runtime.backend import resolve_operator as _resolve_operator

from . import decode as _decode
from . import parallel as _parallel
from .parallel import (
    get_last_route as _generic_get_last_route,
    parallel_wall_attn as _generic_parallel_wall_attn,
    select_route as _generic_select_route,
)
from .decode import (
    build_wall_kv_cache as _generic_build_wall_kv_cache,
    parallel_wall_attn_decode as _generic_parallel_wall_attn_decode,
)
from .naive import naive_wall_attn

_OPERATOR_EXPORTS = {
    "parallel_wall_attn": ("parallel_wall_attn", __name__, "_generic_parallel_wall_attn"),
    "parallel_wall_attn_decode": (
        "parallel_wall_attn_decode",
        __name__,
        "_generic_parallel_wall_attn_decode",
    ),
    "build_wall_kv_cache": ("wall_attn_build_kv_cache", __name__, "_generic_build_wall_kv_cache"),
    "get_last_route": ("wall_attn_get_last_route", __name__, "_generic_get_last_route"),
    "select_route": ("wall_attn_select_route", __name__, "_generic_select_route"),
}


def __getattr__(name: str):
    try:
        operator, module, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = _resolve_operator(operator, module, symbol)
    globals()[name] = value
    return value


# Concrete-module imports must agree with the public exports. Keep the original
# functions above as fallbacks before replacing these compatibility entries.
_decode.parallel_wall_attn_decode = __getattr__("parallel_wall_attn_decode")
_decode.build_wall_kv_cache = __getattr__("build_wall_kv_cache")
_parallel.get_last_route = __getattr__("get_last_route")
del _decode, _parallel

__all__ = [
    "build_wall_kv_cache",
    "naive_wall_attn",
    "parallel_wall_attn",
    "parallel_wall_attn_decode",
    "select_route",
    "get_last_route",
]
