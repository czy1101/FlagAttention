# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass

import torch

from .. import (
    DecodeWorkload,
    PureTritonMTP1Workspace,
    PureTritonMTPWorkspace,
    coerce_pure_triton_workspace,
    prepare_pure_triton_mtp1_workspace,
    prepare_pure_triton_mtp_workspace,
    run_pure_triton_decode,
)

HEAD_DIM = 128
BLOCK_SIZE = 64

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
class StaticBF16Inputs:
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
            raise ValueError('q leading dimension must equal batch * MTP')
        return int(self.q.shape[0] // self.batch)


@dataclass(frozen=True)
class StaticBF16Policy:
    workload: DecodeWorkload
    layout: str
    mtp: int
    cluster_size: int
    chunk_tokens: int
    kernel: str
    narrow: bool
    finalizer_heads: int

    @property
    def label(self) -> str:
        return f'c{self.cluster_size}t{self.chunk_tokens}/{self.kernel}'


def _validate(inputs: StaticBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ('NHD', 'HND'):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16:
        raise ValueError('q must be bfloat16')
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError('q must have shape [batch * MTP, Hq, 128]')
    if inputs.mtp not in (1, 2, 3):
        raise ValueError('S60 BF16 static decode supports MTP 1, 2, or 3')
    for name, cache in (('k_cache', inputs.k_cache), ('v_cache', inputs.v_cache)):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f'{name} must be bfloat16')
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f'{name} must have logical shape [block,64,Hkv,128]')
        if cache.stride(3) != 1:
            raise ValueError(f'{name} head dimension must be contiguous')
    if inputs.block_ids.ndim != 2 or inputs.block_ids.dtype != torch.int32:
        raise ValueError('block_ids must be rank-2 int32')
    if inputs.kv_lens.dtype != torch.int32:
        raise ValueError('kv_lens must be int32')
    hq, hkv = (int(inputs.q.shape[1]), int(inputs.k_cache.shape[2]))
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError('S60 BF16 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)')
    if hkv > 1:
        inferred_layout = 'HND' if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else 'NHD'
        if inferred_layout != inputs.layout:
            raise ValueError('explicit layout does not match K-cache strides')
        inferred_v_layout = 'HND' if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else 'NHD'
        if inferred_v_layout != inputs.layout:
            raise ValueError('explicit layout does not match V-cache strides')
    return (inputs.mtp, hq, hkv)


@dataclass
class StaticBF16Workspace:
    policy: StaticBF16Policy
    runtime_workspace: PureTritonMTP1Workspace | PureTritonMTPWorkspace


def select_static_bf16_policy(inputs: StaticBF16Inputs) -> StaticBF16Policy:
    mtp, _hq, _hkv = _validate(inputs)
    lengths = inputs.kv_lens.detach().cpu().to(torch.int32).tolist()
    if not lengths or min(lengths) < mtp:
        raise ValueError("each KV length must be at least MTP")
    return StaticBF16Policy(
        DecodeWorkload.from_lengths(lengths), inputs.layout, mtp,
        1, 512, "s60-triton-split-k", False, 1,
    )


def prepare_static_bf16_workspace(inputs: StaticBF16Inputs) -> StaticBF16Workspace:
    policy = select_static_bf16_policy(inputs)
    prepare = prepare_pure_triton_mtp1_workspace if inputs.mtp == 1 else prepare_pure_triton_mtp_workspace
    return StaticBF16Workspace(policy, prepare(inputs, "bf16"))


def attention_decode_bf16(inputs: StaticBF16Inputs, workspace) -> torch.Tensor:
    _validate(inputs)
    if isinstance(workspace, StaticBF16Workspace):
        if workspace.policy.mtp != inputs.mtp or workspace.policy.layout != inputs.layout:
            raise ValueError("workspace policy does not match inputs")
        workspace = workspace.runtime_workspace
    workspace = coerce_pure_triton_workspace(workspace, "bf16")
    return run_pure_triton_decode(inputs, workspace)


# Retain the historical public entry name without a TLE/Hopper implementation.
attention_decode_bf16_tle = attention_decode_bf16

__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "OFFICIAL_CASES",
    "StaticBF16Inputs", "StaticBF16Policy", "StaticBF16Workspace",
    "attention_decode_bf16", "attention_decode_bf16_tle",
    "prepare_static_bf16_workspace", "select_static_bf16_policy",
]
