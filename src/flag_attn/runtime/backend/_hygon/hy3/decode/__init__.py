"""Public Hygon hy3 decode surface.

The package mirrors the upstream Hy3 layout: shared pure-Triton kernels stay
in :mod:`.core`, while this module owns the small Hygon compatibility layer
and dispatches the static/dynamic BF16 and FP8 variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from .core import (
    DecodeWorkload,
    ENABLE_TLE_LOAD,
    HAS_TLE,
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    USE_TLE,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    gpu_types,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
    tle,
)


OP_NAME = "hygon_hy3_decode"
IMPLEMENTATION = "pure-triton"
SUPPORTS_TLE = False


@dataclass(frozen=True)
class DecodeVariant:
    """Small immutable description of one decode implementation."""

    key: str
    mode: str
    quant_type: str | None
    module: str
    implementation: str
    prepare_workspace: str


# Keep the variant table beside the public decode entry point.  The upstream
# package uses the same direct static/dynamic module layout; no second registry
# module is needed for this backend.
DECODE_VARIANTS = {
    "bf16_static": DecodeVariant(
        "bf16_static", "static", "bf16", ".static",
        "attention_decode_bf16_static", "prepare_static_bf16_workspace",
    ),
    "bf16_dynamic": DecodeVariant(
        "bf16_dynamic", "dynamic", "bf16", ".dynamic",
        "attention_decode_bf16_dynamic", "prepare_dynamic_bf16_workspace",
    ),
    "fp8_qk_static": DecodeVariant(
        "fp8_qk_static", "static", "qkpertoken_perhead_vperhead",
        ".static",
        "attention_decode_fp8_qk_static", "prepare_decode_workspace_qk_static",
    ),
    "fp8_qk_dynamic": DecodeVariant(
        "fp8_qk_dynamic", "dynamic", "qkpertoken_perhead_vperhead",
        ".dynamic",
        "attention_decode_fp8_qk_dynamic", "prepare_decode_workspace_qk_dynamic",
    ),
    "fp8_kv_static": DecodeVariant(
        "fp8_kv_static", "static", "qpertoken_perhead_kvpertensor",
        ".static",
        "attention_decode_fp8_kv_static", "prepare_decode_workspace_kv_static",
    ),
    "fp8_kv_dynamic": DecodeVariant(
        "fp8_kv_dynamic", "dynamic", "qpertoken_perhead_kvpertensor",
        ".dynamic",
        "attention_decode_fp8_kv_dynamic", "prepare_decode_workspace_kv_dynamic",
    ),
}

_FP8_QUANT_VARIANTS: dict[str, dict[str, str]] = {}
for _spec in DECODE_VARIANTS.values():
    if _spec.quant_type not in (None, "bf16"):
        _FP8_QUANT_VARIANTS.setdefault(_spec.quant_type, {})[_spec.mode] = _spec.key

_FP8_QUANT_TYPES = frozenset(_FP8_QUANT_VARIANTS)
_WORKSPACE_ATTR = "_hygon_hy3_workspace"
_WORKSPACE_CACHE: dict[int, object] = {}


def get_decode_variant(key: str) -> DecodeVariant:
    try:
        return DECODE_VARIANTS[key]
    except KeyError as exc:
        raise ValueError(
            f"unsupported Hygon Hy3 decode variant {key!r}; "
            f"expected one of {tuple(DECODE_VARIANTS)}"
        ) from exc


def get_bf16_variant(mode: str) -> DecodeVariant:
    return get_decode_variant(f"bf16_{mode}")


def get_fp8_variant(mode: str, quant_type: str) -> DecodeVariant:
    try:
        return get_decode_variant(_FP8_QUANT_VARIANTS[quant_type][mode])
    except KeyError as exc:
        raise ValueError(
            f"unsupported Hygon Hy3 FP8 combination: mode={mode!r}, "
            f"quant_type={quant_type!r}"
        ) from exc


def backend_info() -> dict[str, Any]:
    """Return machine-readable capability metadata for logs and registries."""
    return {
        "op": OP_NAME,
        "implementation": IMPLEMENTATION,
        "supports_tle": SUPPORTS_TLE,
        "quant_types": tuple(sorted(_FP8_QUANT_TYPES)),
        "schedules": ("dynamic", "static"),
        "variants": tuple(DECODE_VARIANTS),
    }


def _load_variant(variant: str):
    return import_module(get_decode_variant(variant).module, package=__package__)


def _bf16_module(mode: str):
    return _load_variant(get_bf16_variant(mode).key)


def _fp8_module(mode: str, quant_type: str):
    return _load_variant(get_fp8_variant(mode, quant_type).key)


def prepare_workspace(inputs, *, mode: str = "dynamic", quant_type: str | None = None):
    """Validate inputs and allocate the selected pure-Triton workspace."""
    if hasattr(inputs, "q_scale"):
        if quant_type is None:
            raise ValueError("quant_type is required for FP8 decode")
        module = _fp8_module(mode, quant_type)
        return module.prepare_decode_workspace(inputs)
    module = _bf16_module(mode)
    prepare_name = (
        "prepare_dynamic_bf16_workspace"
        if mode == "dynamic"
        else "prepare_static_bf16_workspace"
    )
    return getattr(module, prepare_name)(inputs)


def _validate_device_inputs(inputs) -> None:
    """Apply the common device guard before workspace allocation or launch."""
    q = getattr(inputs, "q", None)
    if not isinstance(q, torch.Tensor):
        raise TypeError("Hy3 decode inputs must provide tensor attribute 'q'")
    if q.device.type != "cuda":
        raise ValueError(
            "Hygon Hy3 decode requires tensors on the CUDA-compatible Hygon device"
        )
    tensor_names = ("k_cache", "v_cache", "block_ids", "kv_lens")
    if hasattr(inputs, "q_scale"):
        tensor_names += ("q_scale", "k_scale", "v_scale")
    for name in tensor_names:
        value = getattr(inputs, name, None)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Hy3 decode inputs must provide tensor attribute {name!r}")
        if value.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")


def _is_hygon_workspace(workspace) -> bool:
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))


def _coerce_workspace(workspace, inputs, prepare):
    """Convert a generic workspace once and reuse it on later calls."""
    if workspace is None:
        return prepare(inputs)
    if _is_hygon_workspace(workspace):
        return workspace

    native = getattr(workspace, _WORKSPACE_ATTR, None)
    if native is None:
        native = _WORKSPACE_CACHE.get(id(workspace))
    if native is None or not _is_hygon_workspace(native):
        native = prepare(inputs)
        try:
            setattr(workspace, _WORKSPACE_ATTR, native)
        except (AttributeError, TypeError):
            _WORKSPACE_CACHE[id(workspace)] = native
    return native


def _run_variant(variant: str, inputs, workspace):
    _validate_device_inputs(inputs)
    spec = get_decode_variant(variant)
    module = _load_variant(variant)
    native = _coerce_workspace(
        workspace, inputs, getattr(module, spec.prepare_workspace)
    )
    try:
        return getattr(module, spec.implementation)(inputs, native)
    except (CompilationError, OutOfResources) as exc:
        raise RuntimeError(
            f"Hygon Hy3 {variant} pure-Triton launch failed during compilation "
            "or resource allocation. Check the shape and adjust HY3_* or "
            "FLAG_ATTN_HY3_* tuning variables; this backend has no TLE fallback."
        ) from exc


def attention_decode(
    inputs,
    workspace=None,
    *,
    mode: str = "dynamic",
    quant_type: str | None = None,
):
    """Run one decode call through the selected Hygon compatibility path."""
    if workspace is None:
        workspace = prepare_workspace(inputs, mode=mode, quant_type=quant_type)
    if hasattr(inputs, "q_scale"):
        if quant_type is None:
            raise ValueError("quant_type is required for FP8 decode")
        variant = get_fp8_variant(mode, quant_type).key
    else:
        variant = get_bf16_variant(mode).key
    return _run_variant(variant, inputs, workspace)


def attention_decode_bf16_static(inputs, workspace=None):
    return _run_variant("bf16_static", inputs, workspace)


def attention_decode_bf16_dynamic(inputs, workspace=None):
    return _run_variant("bf16_dynamic", inputs, workspace)


def attention_decode_fp8_qk_static(inputs, workspace=None):
    return _run_variant("fp8_qk_static", inputs, workspace)


def attention_decode_fp8_qk_dynamic(inputs, workspace=None):
    return _run_variant("fp8_qk_dynamic", inputs, workspace)


def attention_decode_fp8_kv_static(inputs, workspace=None):
    return _run_variant("fp8_kv_static", inputs, workspace)


def attention_decode_fp8_kv_dynamic(inputs, workspace=None):
    return _run_variant("fp8_kv_dynamic", inputs, workspace)


__all__ = [
    "DecodeVariant",
    "DECODE_VARIANTS",
    "DecodeWorkload",
    "ENABLE_TLE_LOAD",
    "HAS_TLE",
    "IMPLEMENTATION",
    "OP_NAME",
    "PureTritonMTP1Workspace",
    "PureTritonMTPWorkspace",
    "SUPPORTS_TLE",
    "USE_TLE",
    "attention_decode",
    "attention_decode_bf16_static",
    "attention_decode_bf16_dynamic",
    "attention_decode_fp8_qk_static",
    "attention_decode_fp8_qk_dynamic",
    "attention_decode_fp8_kv_static",
    "attention_decode_fp8_kv_dynamic",
    "attention_decode_pure_triton_mtp",
    "attention_decode_pure_triton_mtp1",
    "backend_info",
    "get_bf16_variant",
    "get_decode_variant",
    "get_fp8_variant",
    "gpu_types",
    "prepare_pure_triton_mtp_workspace",
    "prepare_pure_triton_mtp1_workspace",
    "prepare_workspace",
    "tle",
]
