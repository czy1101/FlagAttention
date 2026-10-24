"""Static BF16/INT8 decode for Hygon using pure Triton."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..dynamic import (
    BLOCK_SIZE,
    HEAD_DIM,
    DynamicBF16Inputs,
    DynamicBF16Policy,
)
from ..core import (
    DecodeWorkload,
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
)



StaticBF16Inputs = DynamicBF16Inputs
StaticBF16Policy = DynamicBF16Policy


@dataclass
class StaticBF16Workspace:
    """Compatibility marker; pure-Triton preparation returns the shared workspace."""

    policy: StaticBF16Policy


def _validate(inputs: StaticBF16Inputs) -> tuple[int, int, int]:
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


def select_static_bf16_policy(inputs: StaticBF16Inputs) -> StaticBF16Policy:
    mtp, _, _ = _validate(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return StaticBF16Policy(
        layout=inputs.layout,
        mtp=mtp,
        workload=DecodeWorkload.from_lengths(lengths),
    )


def prepare_static_bf16_workspace(inputs: StaticBF16Inputs):
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
    return prepare_pure_triton_mtp_workspace(inputs, "bf16")


def attention_decode_bf16_static(inputs: StaticBF16Inputs, workspace):
    """Run static BF16 decode through the shared pure-Triton kernels."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    raise TypeError(f"unsupported BF16 Triton workspace: {type(workspace)!r}")



def prepare_static_int8_workspace(inputs):
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "int8")
    return prepare_pure_triton_mtp_workspace(inputs, "int8")


def attention_decode_int8_static(inputs, workspace):
    """Run INT8 Q/K/V decode, returning BF16 output."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace)
    raise TypeError(f"unsupported INT8 Triton workspace: {type(workspace)!r}")

__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "StaticBF16Inputs", "StaticBF16Policy",
    "StaticBF16Workspace", "attention_decode_bf16_static",
    "attention_decode_int8_static", "prepare_static_bf16_workspace",
    "prepare_static_int8_workspace",
]
