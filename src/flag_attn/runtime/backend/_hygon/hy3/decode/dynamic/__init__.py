"""BF16 dynamic decode for Hygon: pure Triton fallback only.

The Hygon TLE/TMA implementation is intentionally not imported here.  The
portable producer/reducer kernels in ``decode.__init__`` are used for every
MTP and layout, with TLE loads explicitly disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

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


from ..triton_fp8 import (
    FallbackFP8DecodeInputs, FallbackFP8DecodePolicy, FallbackFP8DecodeWorkspace,
    OFFICIAL_CASES as FP8_OFFICIAL_CASES, SUPPORTED_MTP,
    BLOCK_SIZE as FP8_BLOCK_SIZE, HEAD_DIM as FP8_HEAD_DIM,
    fallback_workspace_is_reset, prepare_fallback_workspace,
    run_fallback_decode, select_fallback_policy,
)

BLOCK_SIZE = 64
HEAD_DIM = 128
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


@dataclass(frozen=True)
class DynamicBF16Policy:
    """Compatibility policy describing the fallback route."""

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
    """Compatibility marker; fallback preparation returns a pure workspace."""

    policy: DynamicBF16Policy


def _validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("BF16 dynamic decode supports MTP 1, 2, or 3")
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
        raise ValueError("BF16 dynamic decode requires official GQA8 heads")
    return inputs.mtp, hq, hkv


def select_dynamic_bf16_policy(inputs: DynamicBF16Inputs) -> DynamicBF16Policy:
    mtp, _, _ = _validate(inputs)
    lengths = tuple(inputs.kv_lens.detach().cpu().to(torch.int64).tolist())
    return DynamicBF16Policy(
        layout=inputs.layout,
        mtp=mtp,
        workload=DecodeWorkload.from_lengths(lengths),
    )


def prepare_dynamic_bf16_workspace(inputs: DynamicBF16Inputs):
    """Prepare only the portable Triton workspace; never dispatch TLE."""
    _validate(inputs)
    if inputs.mtp == 1:
        return prepare_pure_triton_mtp1_workspace(inputs, "bf16")
    return prepare_pure_triton_mtp_workspace(inputs, "bf16")


def attention_decode_bf16_dynamic(inputs: DynamicBF16Inputs, workspace):
    """Run BF16 decode through pure Triton kernels only."""
    if isinstance(workspace, PureTritonMTP1Workspace):
        return attention_decode_pure_triton_mtp1(inputs, workspace, use_tle_load=False)
    if isinstance(workspace, PureTritonMTPWorkspace):
        return attention_decode_pure_triton_mtp(inputs, workspace, use_tle_load=False)
    raise TypeError(f"unsupported BF16 fallback workspace: {type(workspace)!r}")


def refresh_dynamic_bf16_task_map(inputs: DynamicBF16Inputs, workspace) -> None:
    """Pure Triton rebuilds/launches its task map on each decode call."""
    if not isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace)):
        raise TypeError(f"unsupported BF16 fallback workspace: {type(workspace)!r}")


def bf16_dynamic_workspace_is_reset(workspace) -> bool:
    return isinstance(workspace, (PureTritonMTP1Workspace, PureTritonMTPWorkspace))


# FP8 variants share one portable adapter; only quantization metadata differs.
FP8DecodeInputs = FallbackFP8DecodeInputs
FP8DecodePolicy = FallbackFP8DecodePolicy
FP8DecodeWorkspace = FallbackFP8DecodeWorkspace
QK_QUANT_TYPE = "qkpertoken_perhead_vperhead"
QK_QUANT_TYPE_ID = 0
KV_QUANT_TYPE = "qpertoken_perhead_kvpertensor"
KV_QUANT_TYPE_ID = 1

def select_decode_policy_qk_dynamic(inputs):
    return select_fallback_policy(inputs, quant_type=QK_QUANT_TYPE, schedule="dynamic")
def prepare_decode_workspace_qk_dynamic(inputs):
    return prepare_fallback_workspace(inputs, QK_QUANT_TYPE)
def attention_decode_fp8_qk_dynamic(inputs, workspace):
    return run_fallback_decode(inputs, workspace)
def workspace_is_reset_qk_dynamic(workspace):
    return fallback_workspace_is_reset(workspace)
def select_decode_policy_kv_dynamic(inputs):
    return select_fallback_policy(inputs, quant_type=KV_QUANT_TYPE, schedule="dynamic")
def prepare_decode_workspace_kv_dynamic(inputs):
    return prepare_fallback_workspace(inputs, KV_QUANT_TYPE)
def attention_decode_fp8_kv_dynamic(inputs, workspace):
    return run_fallback_decode(inputs, workspace)
def workspace_is_reset_kv_dynamic(workspace):
    return fallback_workspace_is_reset(workspace)

__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "OFFICIAL_CASES", "DynamicBF16Inputs", "DynamicBF16Policy",
    "DynamicBF16Workspace", "attention_decode_bf16_dynamic", "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace", "refresh_dynamic_bf16_task_map", "select_dynamic_bf16_policy",
    "FP8_BLOCK_SIZE", "FP8_HEAD_DIM", "FP8_OFFICIAL_CASES", "SUPPORTED_MTP",
    "FP8DecodeInputs", "FP8DecodePolicy", "FP8DecodeWorkspace", "QK_QUANT_TYPE", "QK_QUANT_TYPE_ID",
    "KV_QUANT_TYPE", "KV_QUANT_TYPE_ID", "attention_decode_fp8_qk_dynamic",
    "prepare_decode_workspace_qk_dynamic", "select_decode_policy_qk_dynamic", "workspace_is_reset_qk_dynamic",
    "attention_decode_fp8_kv_dynamic", "prepare_decode_workspace_kv_dynamic",
    "select_decode_policy_kv_dynamic", "workspace_is_reset_kv_dynamic",
]
