"""Dynamic BF16/INT8 decode for Hygon using pure Triton.

The shared producer, reducer, and scheduler live in ``decode.core`` and
``decode.scheduler``.  This module validates dynamic inputs and selects its
workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..core import (
    DecodeWorkload,
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
)



BLOCK_SIZE = 64
HEAD_DIM = 128


@dataclass(frozen=True)
class DynamicBF16Policy:
    """Compatibility policy describing the pure-Triton route."""

    layout: str
    mtp: int
    workload: DecodeWorkload
    cluster_size: int = 1
    chunk_tokens: int = 512
    kernel: str = "pure-triton"
    narrow: bool = False
    compact_direct_narrow: bool = False
    finalizer_heads: int = 1
    pdl_finalizer: bool = False
    deterministic_tail_election: bool = False
    reduction_only: bool = False
    aligned_full_chunk: bool = False

    @property
    def label(self) -> str:
        return "pure-triton"


@dataclass
class DynamicBF16Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    layout: str

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)


@dataclass
class DynamicBF16Workspace:
    """Compatibility wrapper for the selected dynamic decode policy."""

    policy: DynamicBF16Policy


def _validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype not in (torch.bfloat16, torch.int8) or inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be BF16 or INT8 [batch * MTP, Hq, 128]")
    dtype = inputs.q.dtype
    if inputs.k_cache.dtype != dtype or inputs.v_cache.dtype != dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if dtype == torch.int8:
        for name in ("q_scale", "k_scale", "v_scale"):
            value = float(getattr(inputs, name, 1.0))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive scalar")
    mtp = inputs.mtp
    if mtp < 1:
        raise ValueError("MTP must be positive")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE or cache.shape[3] != HEAD_DIM:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32 or not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA int32 tensor")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if hkv <= 0 or hq <= 0 or hq % hkv:
        raise ValueError("query heads must be a positive multiple of KV heads")
    return mtp, hq, hkv


def select_dynamic_bf16_policy(inputs: DynamicBF16Inputs) -> DynamicBF16Policy:
    mtp, _, _ = _validate(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return DynamicBF16Policy(
        layout=inputs.layout,
        mtp=mtp,
        workload=DecodeWorkload.from_lengths(lengths),
    )


def prepare_dynamic_bf16_workspace(inputs: DynamicBF16Inputs):
    """Prepare the shared pure-Triton workspace for dynamic decode."""
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
    return prepare_pure_triton_mtp_workspace(inputs, "bf16")


def attention_decode_bf16_dynamic(inputs: DynamicBF16Inputs, workspace):
    """Run BF16 decode through the shared pure-Triton kernels."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    raise TypeError(f"unsupported BF16 Triton workspace: {type(workspace)!r}")


def prepare_dynamic_int8_workspace(inputs):
    """Prepare the shared pure-Triton workspace for INT8 decode."""
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "int8")
    return prepare_pure_triton_mtp_workspace(inputs, "int8")


def attention_decode_int8_dynamic(inputs, workspace):
    """Run INT8 Q/K/V decode, returning BF16 output."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    raise TypeError(f"unsupported INT8 Triton workspace: {type(workspace)!r}")


def refresh_dynamic_bf16_task_map(inputs: DynamicBF16Inputs, workspace) -> None:
    """Pure Triton rebuilds/launches its task map on each decode call."""
    if not isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace)):
        raise TypeError(f"unsupported BF16 Triton workspace: {type(workspace)!r}")


def bf16_dynamic_workspace_is_reset(workspace) -> bool:
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))



__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "DynamicBF16Inputs", "DynamicBF16Policy",
    "DynamicBF16Workspace", "attention_decode_bf16_dynamic", "bf16_dynamic_workspace_is_reset",
    "attention_decode_int8_dynamic", "prepare_dynamic_int8_workspace",
    "prepare_dynamic_bf16_workspace", "refresh_dynamic_bf16_task_map", "select_dynamic_bf16_policy",
]
