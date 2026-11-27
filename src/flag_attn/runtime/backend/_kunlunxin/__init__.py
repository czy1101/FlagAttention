# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Kunlunxin backend for FlagAttention."""

import importlib

_OPERATOR_EXPORTS = {
    "fused_attnres": (".FLA.attnres", "fused_attnres"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
