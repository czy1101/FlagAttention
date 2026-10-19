"""Portable FP8 decode API used by the Hygon hy3 backend.

This module is deliberately independent of TLE/TMA.  It keeps the public
decode input/policy/workspace names used by the old static and dynamic entry
points, while routing every workload through the shared pure-Triton split-K
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .core import (
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
SUPPORTED_MTP = (1, 2, 4)
OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}


@dataclass
class FallbackFP8DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def mtp(self) -> int:
        if self.batch <= 0 or self.q.shape[0] % self.batch:
            raise ValueError("q leading dimension must equal batch * MTP")
        return int(self.q.shape[0] // self.batch)

    @property
    def layout(self) -> str:
        if self.k_cache.ndim == 4 and self.k_cache.stride(1) < self.k_cache.stride(2):
            return "HND"
        return "NHD"


@dataclass(frozen=True)
class FallbackFP8DecodePolicy:
    quant_type: str
    schedule: str
    mtp: int
    workload: DecodeWorkload
    layout: str
    cluster_size: int = 1
    chunk_tokens: int = 64
    route: str = "pure-triton"

    @property
    def label(self) -> str:
        return "pure-triton"


@dataclass
class FallbackFP8DecodeWorkspace:
    """Compatibility marker; actual storage is a pure Triton workspace."""

    policy: FallbackFP8DecodePolicy


def validate_fallback_inputs(inputs: FallbackFP8DecodeInputs) -> None:
    if inputs.q.ndim != 3 or inputs.k_cache.ndim != 4 or inputs.v_cache.ndim != 4:
        raise ValueError("FP8 decode expects q rank-3 and paged KV rank-4 tensors")
    if inputs.block_ids.ndim != 2 or inputs.kv_lens.ndim != 1:
        raise ValueError("block_ids must be rank-2 and kv_lens rank-1")
    if inputs.k_cache.shape != inputs.v_cache.shape:
        raise ValueError("k_cache and v_cache must have identical shapes")
    if inputs.k_cache.shape[1] != BLOCK_SIZE or inputs.k_cache.shape[-1] != HEAD_DIM:
        raise ValueError("paged FP8 KV cache must use [blocks,64,heads,128]")
    if inputs.q.shape[1] <= 0 or inputs.k_cache.shape[2] <= 0:
        raise ValueError("FP8 decode head dimensions must be non-zero")
    if inputs.q.shape[1] % inputs.k_cache.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")
    if inputs.mtp not in SUPPORTED_MTP:
        raise ValueError(f"FP8 decode supports MTP values {SUPPORTED_MTP}")


def select_fallback_policy(
    inputs: FallbackFP8DecodeInputs,
    *,
    quant_type: str,
    schedule: str,
) -> FallbackFP8DecodePolicy:
    validate_fallback_inputs(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return FallbackFP8DecodePolicy(
        quant_type=quant_type,
        schedule=schedule,
        mtp=inputs.mtp,
        workload=DecodeWorkload.from_lengths(lengths),
        layout=inputs.layout,
    )


def prepare_fallback_workspace(inputs: FallbackFP8DecodeInputs, quant_type: str):
    validate_fallback_inputs(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, quant_type)
    return prepare_pure_triton_mtp_workspace(inputs, quant_type)


def run_fallback_decode(inputs: FallbackFP8DecodeInputs, workspace):
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace, use_tle_load=False)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace, use_tle_load=False)
    raise TypeError(f"unsupported FP8 fallback workspace: {type(workspace)!r}")


def fallback_workspace_is_reset(workspace) -> bool:
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))

