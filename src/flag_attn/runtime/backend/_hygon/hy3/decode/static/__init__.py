"""BF16 static decode for Hygon: pure Triton fallback only."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core import DecodeWorkload
from ..dynamic import (
    BLOCK_SIZE,
    HEAD_DIM,
    OFFICIAL_CASES,
    DynamicBF16Inputs,
    DynamicBF16Policy,
)
from ..core import (
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    attention_decode_pure_triton_mtp,
    attention_decode_pure_triton_mtp1,
    prepare_pure_triton_mtp_workspace,
    prepare_pure_triton_mtp1_workspace,
)


from ..triton_fp8 import (
    FallbackFP8DecodeInputs, FallbackFP8DecodePolicy, FallbackFP8DecodeWorkspace,
    OFFICIAL_CASES as FP8_OFFICIAL_CASES, SUPPORTED_MTP,
    BLOCK_SIZE as FP8_BLOCK_SIZE, HEAD_DIM as FP8_HEAD_DIM,
    fallback_workspace_is_reset, prepare_fallback_workspace,
    run_fallback_decode, select_fallback_policy,
)

StaticBF16Inputs = DynamicBF16Inputs
StaticBF16Policy = DynamicBF16Policy


@dataclass
class StaticBF16Workspace:
    """Compatibility marker; fallback preparation returns a pure workspace."""

    policy: StaticBF16Policy


def _validate(inputs: StaticBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("BF16 static decode supports MTP 1, 2, or 3")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
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
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("BF16 static decode requires official GQA8 heads")
    return inputs.mtp, hq, hkv


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
    """Compatibility entry point; implementation is pure Triton only."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace, use_tle_load=False)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace, use_tle_load=False)
    raise TypeError(f"unsupported BF16 fallback workspace: {type(workspace)!r}")


# Backward-compatible name retained for the shared static API.
attention_decode_bf16_tle = attention_decode_bf16_static

# FP8 variants share one portable adapter; only quantization metadata differs.
FP8DecodeInputs = FallbackFP8DecodeInputs
FP8DecodePolicy = FallbackFP8DecodePolicy
FP8DecodeWorkspace = FallbackFP8DecodeWorkspace
QK_QUANT_TYPE = "qkpertoken_perhead_vperhead"
QK_QUANT_TYPE_ID = 0
KV_QUANT_TYPE = "qpertoken_perhead_kvpertensor"
KV_QUANT_TYPE_ID = 1

def select_decode_policy_qk_static(inputs):
    return select_fallback_policy(inputs, quant_type=QK_QUANT_TYPE, schedule="static")
def prepare_decode_workspace_qk_static(inputs):
    return prepare_fallback_workspace(inputs, QK_QUANT_TYPE)
def attention_decode_fp8_qk_static(inputs, workspace):
    return run_fallback_decode(inputs, workspace)
def workspace_is_reset_qk_static(workspace):
    return fallback_workspace_is_reset(workspace)
def select_decode_policy_kv_static(inputs):
    return select_fallback_policy(inputs, quant_type=KV_QUANT_TYPE, schedule="static")
def prepare_decode_workspace_kv_static(inputs):
    return prepare_fallback_workspace(inputs, KV_QUANT_TYPE)
def attention_decode_fp8_kv_static(inputs, workspace):
    return run_fallback_decode(inputs, workspace)
def workspace_is_reset_kv_static(workspace):
    return fallback_workspace_is_reset(workspace)

__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "OFFICIAL_CASES", "StaticBF16Inputs", "StaticBF16Policy",
    "StaticBF16Workspace", "attention_decode_bf16_static", "attention_decode_bf16_tle",
    "prepare_static_bf16_workspace", "select_static_bf16_policy", "FP8_BLOCK_SIZE",
    "FP8_HEAD_DIM", "FP8_OFFICIAL_CASES", "SUPPORTED_MTP", "FP8DecodeInputs",
    "FP8DecodePolicy", "FP8DecodeWorkspace", "QK_QUANT_TYPE", "QK_QUANT_TYPE_ID",
    "KV_QUANT_TYPE", "KV_QUANT_TYPE_ID", "attention_decode_fp8_qk_static",
    "prepare_decode_workspace_qk_static", "select_decode_policy_qk_static", "workspace_is_reset_qk_static",
    "attention_decode_fp8_kv_static", "prepare_decode_workspace_kv_static",
    "select_decode_policy_kv_static", "workspace_is_reset_kv_static",
]
