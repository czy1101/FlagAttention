"""Public runtime for the final MTP=1 FP8 TLE decode implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from .compute_kernel import fp8_kvpertensor_decode_kernel
from .task_scheduler_kernel import (
    DecodeTaskSchedule,
    TASK_SLOTS,
    allocate_cluster_task_map,
    launch_cluster_task_map_assign,
    launch_cluster_task_tail_refresh,
)


BLOCK_SIZE = 64
TILE_N = 64
NUM_SEQ_Q = 1
HEAD_DIM = 128


@dataclass(frozen=True)
class DecodeConfig:
    cluster_size: int
    chunk_tokens: int

    def __post_init__(self) -> None:
        if self.cluster_size not in (2, 4, 8):
            raise ValueError("runtime policy supports cluster_size 2, 4, or 8")
        if self.chunk_tokens not in (512, 1024):
            raise ValueError("final policy supports chunk_tokens 512 or 1024")


C2_T512 = DecodeConfig(2, 512)
C8_T512 = DecodeConfig(8, 512)
C4_T1024 = DecodeConfig(4, 1024)
C8_T1024 = DecodeConfig(8, 1024)


@dataclass
class DecodeInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor

    @property
    def num_batch(self) -> int:
        return int(self.kv_lens.numel())

    @property
    def max_seq_kv(self) -> int:
        return int(self.kv_lens.max().item())


@dataclass
class DecodeWorkspace:
    config: DecodeConfig
    schedule: DecodeTaskSchedule
    q_4d: torch.Tensor
    q_scale_3d: torch.Tensor
    split_out: torch.Tensor
    lse: torch.Tensor
    completion: torch.Tensor
    last_flags: torch.Tensor
    out: torch.Tensor
    heads_per_group: int

    @property
    def stats(self) -> dict:
        return dict(self.schedule.stats or {})


CLUSTER_MESHES = {
    2: tle.device_mesh({"block_cluster": [("cluster_x", 2)]}),
    4: tle.device_mesh({"block_cluster": [("cluster_x", 4)]}),
    8: tle.device_mesh({"block_cluster": [("cluster_x", 8)]}),
}


def select_decode_config(kv_lens: torch.Tensor | list[int]) -> DecodeConfig:
    """Select the measured winner using only launch-visible sequence lengths."""
    if isinstance(kv_lens, torch.Tensor):
        lengths = kv_lens.detach().cpu().to(torch.int64).tolist()
    else:
        lengths = [int(value) for value in kv_lens]
    if not lengths or min(lengths) <= 0:
        raise ValueError("kv_lens must contain positive lengths")

    batch = len(lengths)
    max_kv = max(lengths)
    uniform = min(lengths) == max_kv
    long_32k = sum(length >= 32 * 1024 for length in lengths)

    if max_kv <= 512:
        return C2_T512
    if uniform and max_kv <= 4096:
        return C4_T1024
    if max_kv >= 128 * 1024:
        return C8_T1024
    if long_32k >= 2:
        return C4_T1024
    if max_kv >= 64 * 1024:
        return C8_T512 if batch <= 16 else C4_T1024
    return C8_T512


def _validate_inputs(inputs: DecodeInputs) -> tuple[int, int, int]:
    if not inputs.kv_lens.is_cuda:
        raise ValueError("kv_lens must be a CUDA tensor for GPU task scheduling")
    if inputs.q.ndim != 3 or inputs.q.shape[-1] != HEAD_DIM:
        raise ValueError("q must have shape [batch, num_head_q, 128] for MTP=1")
    if inputs.q.shape[0] != inputs.num_batch:
        raise ValueError("MTP=1 requires q.shape[0] == num_batch")
    for name, cache in (("k_cache", inputs.k_cache), ("v_cache", inputs.v_cache)):
        if cache.ndim != 4 or cache.shape[1] != BLOCK_SIZE or cache.shape[3] != HEAD_DIM:
            raise ValueError(f"{name} must be logical [block, 64, head, 128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    num_head_q = int(inputs.q.shape[1])
    num_head_kv = int(inputs.k_cache.shape[2])
    if num_head_q % num_head_kv:
        raise ValueError("num_head_q must be divisible by num_head_kv")
    heads_per_group = num_head_q // num_head_kv
    if heads_per_group > 8:
        raise ValueError("heads_per_group must be <= 8")
    return num_head_q, num_head_kv, heads_per_group


def make_paged_kv_descriptors(
    inputs: DecodeInputs,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    """Preserve the true NHD/HND strides in rank-4 paged descriptors."""
    _validate_inputs(inputs)
    block_shape = [1, TILE_N, 1, HEAD_DIM]
    return (
        TensorDescriptor.from_tensor(inputs.k_cache, block_shape=block_shape),
        TensorDescriptor.from_tensor(inputs.v_cache, block_shape=block_shape),
    )


def prepare_decode_workspace(
    inputs: DecodeInputs,
    config: DecodeConfig | None = None,
) -> DecodeWorkspace:
    """Allocate buffers and build the complete cluster task map on the GPU."""
    num_head_q, num_head_kv, heads_per_group = _validate_inputs(inputs)
    config = config or select_decode_config(inputs.kv_lens)
    schedule = allocate_cluster_task_map(
        inputs.kv_lens,
        num_head_kv=num_head_kv,
        max_seq_kv=inputs.max_seq_kv,
        cluster_size=config.cluster_size,
        chunk_tokens=config.chunk_tokens,
    )
    if schedule.stats is None:
        raise RuntimeError("GPU task scheduler did not publish workspace metadata")

    q_4d = inputs.q.reshape(inputs.num_batch, NUM_SEQ_Q, num_head_q, HEAD_DIM)
    q_scale_3d = inputs.q_scale.reshape(inputs.num_batch, NUM_SEQ_Q, num_head_q)
    pad_heads_per_group = ((heads_per_group + 7) // 8) * 8
    device = inputs.q.device
    split_out = torch.empty(
        (inputs.num_batch, schedule.partial_slots, NUM_SEQ_Q, num_head_q, HEAD_DIM),
        dtype=torch.float32,
        device=device,
    )
    lse = torch.empty(
        (inputs.num_batch, schedule.partial_slots, num_head_kv, NUM_SEQ_Q, pad_heads_per_group),
        dtype=torch.float32,
        device=device,
    )
    return DecodeWorkspace(
        config=config,
        schedule=schedule,
        q_4d=q_4d,
        q_scale_3d=q_scale_3d,
        split_out=split_out,
        lse=lse,
        completion=torch.zeros(
            (num_head_kv * inputs.num_batch,), dtype=torch.int32, device=device
        ),
        last_flags=torch.zeros(
            (schedule.physical_ctas,), dtype=torch.int32, device=device
        ),
        out=torch.empty(
            (inputs.num_batch, NUM_SEQ_Q, num_head_q, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        ),
        heads_per_group=heads_per_group,
    )


def refresh_decode_schedule(
    inputs: DecodeInputs,
    workspace: DecodeWorkspace,
    mode: Literal["full", "tail"] = "full",
) -> None:
    """Refresh an existing GPU schedule when its allocated topology is stable."""
    num_head_kv = int(inputs.k_cache.shape[2])
    if mode == "full":
        launch_cluster_task_map_assign(
            inputs.kv_lens,
            workspace.schedule,
            num_head_kv=num_head_kv,
            refresh_host_metadata=False,
        )
    elif mode == "tail":
        launch_cluster_task_tail_refresh(
            inputs.kv_lens, workspace.schedule, num_head_kv=num_head_kv
        )
    else:
        raise ValueError("mode must be 'full' or 'tail'")


def fp8_kvpertensor_decode(
    inputs: DecodeInputs,
    workspace: DecodeWorkspace | None = None,
    *,
    refresh_schedule: Literal["full", "tail"] | None = None,
) -> torch.Tensor:
    """Run the final log2-domain paged FP8 decode kernel."""
    workspace = workspace or prepare_decode_workspace(inputs)
    if refresh_schedule is not None:
        refresh_decode_schedule(inputs, workspace, refresh_schedule)
    k_desc, v_desc = make_paged_kv_descriptors(inputs)
    ws = workspace
    num_head_q = int(inputs.q.shape[1])

    fp8_kvpertensor_decode_kernel[(ws.schedule.num_clusters,)](
        ws.q_4d,
        k_desc,
        v_desc,
        inputs.block_ids,
        ws.schedule.task_map,
        ws.q_scale_3d,
        inputs.k_scale,
        inputs.v_scale,
        ws.split_out,
        ws.lse,
        ws.completion,
        ws.last_flags,
        ws.out,
        mesh=CLUSTER_MESHES[ws.config.cluster_size],
        B=inputs.num_batch,
        H_Q=num_head_q,
        HEADS_PER_GROUP=ws.heads_per_group,
        D=HEAD_DIM,
        DV=HEAD_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        BLOCK_N=TILE_N,
        Q_STRIDE_B=ws.q_4d.stride(0),
        Q_STRIDE_H=ws.q_4d.stride(2),
        QS_STRIDE_B=ws.q_scale_3d.stride(0),
        QS_STRIDE_H=ws.q_scale_3d.stride(2),
        SO_STRIDE_B=ws.split_out.stride(0),
        SO_STRIDE_C=ws.split_out.stride(1),
        SO_STRIDE_M=ws.split_out.stride(2),
        SO_STRIDE_H=ws.split_out.stride(3),
        LSE_STRIDE_B=ws.lse.stride(0),
        LSE_STRIDE_C=ws.lse.stride(1),
        LSE_STRIDE_HKV=ws.lse.stride(2),
        LSE_STRIDE_M=ws.lse.stride(3),
        LSE_STRIDE_HG=ws.lse.stride(4),
        O_STRIDE_B=ws.out.stride(0),
        O_STRIDE_M=ws.out.stride(1),
        O_STRIDE_H=ws.out.stride(2),
        CLUSTER_SIZE=ws.config.cluster_size,
        num_ctas=1,
        num_warps=4,
        num_stages=3,
        launch_pdl=False,
    )
    return ws.out.reshape(inputs.num_batch, num_head_q, HEAD_DIM)


__all__ = [
    "C2_T512",
    "C4_T1024",
    "C8_T512",
    "C8_T1024",
    "DecodeConfig",
    "DecodeInputs",
    "DecodeWorkspace",
    "fp8_kvpertensor_decode",
    "make_paged_kv_descriptors",
    "prepare_decode_workspace",
    "refresh_decode_schedule",
    "select_decode_config",
]
