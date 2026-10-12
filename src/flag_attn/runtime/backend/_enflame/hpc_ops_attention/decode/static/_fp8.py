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
from typing import Literal

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
class FP8DecodeInputs:
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
        if self.batch == 0 or self.q.shape[0] % self.batch:
            raise ValueError('q leading dimension must equal batch * MTP')
        return int(self.q.shape[0] // self.batch)


@dataclass(frozen=True)
class FP8DecodePolicy:
    schedule: Literal["static"]
    mtp: int
    workload: DecodeWorkload
    layout: str
    cluster_size: int
    chunk_tokens: int
    route: str

    @property
    def label(self) -> str:
        return f'c{self.cluster_size}t{self.chunk_tokens}/{self.route}'


@dataclass
class FP8DecodeWorkspace:
    policy: FP8DecodePolicy
    runtime_inputs: FP8DecodeInputs
    runtime_workspace: PureTritonMTP1Workspace | PureTritonMTPWorkspace


def _validate(inputs: FP8DecodeInputs) -> None:
    if inputs.mtp not in SUPPORTED_MTP:
        raise ValueError('S60 FP8 decode supports MTP 1, 2, or 4')
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError('q must have shape [batch * MTP,Hq,128]')
    if inputs.q.dtype != torch.float8_e4m3fn:
        raise ValueError('q must be float8_e4m3fn')
    for name, cache in (('k_cache', inputs.k_cache), ('v_cache', inputs.v_cache)):
        if cache.element_size() != 1:
            raise ValueError(f'{name} must use a 1-byte FP8 storage dtype')
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE:
            raise ValueError(f'{name} must have logical shape [block,64,Hkv,128]')
        if cache.shape[-1] != HEAD_DIM or cache.stride(-1) != 1:
            raise ValueError(f'{name} head dimension must be contiguous 128')
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if (num_head_kv, num_head_q) not in ((1, 8), (4, 32)):
        raise ValueError('S60 FP8 decode requires official GQA8 heads (Hkv,Hq)=(1,8) or (4,32)')
    if inputs.block_ids.dtype != torch.int32 or inputs.kv_lens.dtype != torch.int32:
        raise ValueError('block_ids and kv_lens must be int32')


def select_decode_policy(inputs: FP8DecodeInputs) -> FP8DecodePolicy:
    _validate(inputs)
    lengths = inputs.kv_lens.detach().cpu().to(torch.int32).tolist()
    if not lengths or min(lengths) < inputs.mtp:
        raise ValueError("each KV length must be at least MTP")
    layout = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
    return FP8DecodePolicy(
        "static", inputs.mtp, DecodeWorkload.from_lengths(lengths), layout,
        1, 512, "s60-triton-split-k",
    )


def prepare_decode_workspace(inputs: FP8DecodeInputs, quant_type: str) -> FP8DecodeWorkspace:
    policy = select_decode_policy(inputs)
    prepare = prepare_pure_triton_mtp1_workspace if inputs.mtp == 1 else prepare_pure_triton_mtp_workspace
    return FP8DecodeWorkspace(policy, inputs, prepare(inputs, quant_type))


def attention_decode_fp8(inputs: FP8DecodeInputs, workspace, quant_type: str) -> torch.Tensor:
    _validate(inputs)
    if isinstance(workspace, FP8DecodeWorkspace):
        layout = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        if workspace.policy.mtp != inputs.mtp or workspace.policy.layout != layout:
            raise ValueError("workspace policy does not match inputs")
        workspace = workspace.runtime_workspace
    workspace = coerce_pure_triton_workspace(workspace, quant_type)
    return run_pure_triton_decode(inputs, workspace)


def workspace_is_reset(workspace, quant_type: str) -> bool:
    if isinstance(workspace, FP8DecodeWorkspace):
        workspace = workspace.runtime_workspace
    coerce_pure_triton_workspace(workspace, quant_type)
    # This split-K implementation has no cooperative-finalization counters.
    return True
