"""Public Hygon hy3 decode surface.

The package mirrors the upstream Hy3 layout: shared pure-Triton kernels stay
in :mod:`.core`, while this module owns the small Hygon compatibility layer
and dispatches the static/dynamic BF16 and INT8 variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import OutOfResources

from .. import OP_NAME as HY3_OP_NAME
from .core import (
    DecodeWorkload,
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
)


OP_NAME = HY3_OP_NAME


@dataclass
class Int8DecodeInputs:
    """Paged decode inputs with signed INT8 Q/K/V and per-tensor scales."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str
    q_scale: float = 1.0
    k_scale: float = 1.0
    v_scale: float = 1.0

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


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
    "int8_static": DecodeVariant(
        "int8_static", "static", "int8", ".static",
        "attention_decode_int8_static", "prepare_static_int8_workspace",
    ),
    "int8_dynamic": DecodeVariant(
        "int8_dynamic", "dynamic", "int8", ".dynamic",
        "attention_decode_int8_dynamic", "prepare_dynamic_int8_workspace",
    ),
}


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


def backend_info() -> dict[str, Any]:
    """Return machine-readable capability metadata for logs and registries."""
    return {
        "op": OP_NAME,
        "implementation": IMPLEMENTATION,
        "supports_tle": SUPPORTS_TLE,
        "quant_types": ("bf16", "int8"),
        "schedules": ("dynamic", "static"),
        "variants": tuple(DECODE_VARIANTS),
    }


def _load_variant(variant: str):
    return import_module(get_decode_variant(variant).module, package=__package__)


def prepare_workspace(inputs, *, mode: str = "dynamic", quant_type: str | None = None):
    """Validate inputs and allocate the selected pure-Triton workspace."""
    if quant_type is None:
        quant_type = "int8" if inputs.q.dtype == torch.int8 else "bf16"
    if quant_type not in ("bf16", "int8"):
        raise ValueError("Hygon Hy3 decode supports BF16 and INT8")
    module = _load_variant(f"{quant_type}_{mode}")
    prepare_name = f"prepare_{mode}_{quant_type}_workspace"
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
            "FLAG_ATTN_HY3_* tuning variables; no alternate kernel implementation is available."
        ) from exc


def attention_decode(
    inputs,
    workspace=None,
    *,
    mode: str = "dynamic",
    quant_type: str | None = None,
):
    """Run one decode call through the selected Hygon compatibility path."""
    if quant_type is None:
        quant_type = "int8" if inputs.q.dtype == torch.int8 else "bf16"
    if quant_type not in ("bf16", "int8"):
        raise ValueError("Hygon Hy3 decode supports BF16 and INT8")
    if workspace is None:
        workspace = prepare_workspace(inputs, mode=mode, quant_type=quant_type)
    variant = get_decode_variant(f"{quant_type}_{mode}").key
    return _run_variant(variant, inputs, workspace)


def attention_decode_bf16_static(inputs, workspace=None):
    return _run_variant("bf16_static", inputs, workspace)


def attention_decode_bf16_dynamic(inputs, workspace=None):
    return _run_variant("bf16_dynamic", inputs, workspace)


def attention_decode_int8_static(inputs, workspace=None):
    return _run_variant("int8_static", inputs, workspace)


def attention_decode_int8_dynamic(inputs, workspace=None):
    return _run_variant("int8_dynamic", inputs, workspace)



__all__ = [
    "DecodeVariant",
    "DECODE_VARIANTS",
    "DecodeWorkload",
    "IMPLEMENTATION",
    "OP_NAME",
    "PureTritonMTP1Workspace",
    "PureTritonMTPWorkspace",
    "SUPPORTS_TLE",
    "attention_decode",
    "attention_decode_bf16_static",
    "attention_decode_bf16_dynamic",
    "attention_decode_int8_static",
    "attention_decode_int8_dynamic",
    "attention_decode_pure_triton_mtp",
    "attention_decode_pure_triton_mtp1",
    "backend_info",
    "get_bf16_variant",
    "Int8DecodeInputs",
    "get_decode_variant",
    "prepare_pure_triton_mtp_workspace",
    "prepare_pure_triton_mtp1_workspace",
    "prepare_workspace",
]
