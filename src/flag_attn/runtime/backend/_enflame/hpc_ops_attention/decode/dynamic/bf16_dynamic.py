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
import triton
import triton.language as tl


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


def _dynamic__validate(inputs: DynamicBF16Inputs) -> tuple[int, int, int]:
    if inputs.layout not in ("NHD", "HND"):
        raise ValueError("layout must be 'NHD' or 'HND'")
    if inputs.q.dtype != torch.bfloat16 or inputs.q.ndim != 3 or inputs.q.shape[-1] != 128:
        raise ValueError("q must be BF16 [batch * MTP, Hq, 128]")
    if inputs.mtp not in (1, 2, 3):
        raise ValueError("BF16 dynamic decode supports MTP 1, 2, or 3")
    for name, cache in (
        ("k_cache", inputs.k_cache),
        ("v_cache", inputs.v_cache),
    ):
        if cache.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bfloat16")
        if cache.ndim != 4 or cache.shape[1] != 64 or cache.shape[3] != 128:
            raise ValueError(f"{name} must have logical shape [block,64,Hkv,128]")
        if cache.stride(3) != 1:
            raise ValueError(f"{name} head dimension must be contiguous")
    if inputs.block_ids.dtype != torch.int32 or inputs.block_ids.ndim != 2:
        raise ValueError("block_ids must be rank-2 int32")
    if inputs.kv_lens.dtype != torch.int32:
        raise ValueError("kv_lens must be an accelerator int32 tensor")
    if inputs.kv_lens.device != inputs.q.device:
        raise ValueError("kv_lens and q must be on the same accelerator")
    hq = int(inputs.q.shape[1])
    hkv = int(inputs.k_cache.shape[2])
    if (hkv, hq) not in ((1, 8), (4, 32)):
        raise ValueError("BF16 dynamic decode requires official GQA8 heads")
    if hkv > 1:
        inferred_k = "HND" if inputs.k_cache.stride(2) > inputs.k_cache.stride(1) else "NHD"
        inferred_v = "HND" if inputs.v_cache.stride(2) > inputs.v_cache.stride(1) else "NHD"
        if inferred_k != inputs.layout or inferred_v != inputs.layout:
            raise ValueError("explicit layout does not match cache strides")
    return inputs.mtp, hq, hkv


_S60_TASK_STRIDE = 12
_S60_TILE_N = 128
_S60_BLOCK_N = 64
_S60_GROUP_M = 8
_S60_CTAS_PER_CU = 128
_S60_MAX_CTAS = 2048
_S60_MIN_PROCESS_LEN = 2048
_S60_LARGE_MIN_PROCESS_LEN = 4096
_S60_WORKSPACE_BUDGET_BYTES = 8 * 1024**3
_S60_VECTOR_TASK_MAP_MIN_BATCH = 32
_S60_VECTOR_TASK_MAP_MAX_BATCH = 128
_S60_TASK_STRIDE_JIT = tl.constexpr(_S60_TASK_STRIDE)
_S60_TILE_N_JIT = tl.constexpr(_S60_TILE_N)


@triton.jit
def _s60_assign_uniform_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    TOTAL_TILES,
    SEQ_TILES: tl.constexpr,
    MAX_TASKS,
    SCHED_INTS,
):
    """Build equal-length split-K tasks without scanning every sequence."""
    tiles_per_cta = (TOTAL_TILES + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _S60_TILE_N_JIT
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    num_chunks_base = (
        (MAX_TASKS * NUM_CTAS + 1) * _S60_TASK_STRIDE_JIT
    )
    cta = tl.program_id(0).to(tl.int32)
    cursor = cta * tiles_per_cta
    cta_end = tl.minimum(cursor + tiles_per_cta, TOTAL_TILES)
    slot = 0
    while (cursor < cta_end) & (slot < MAX_TASKS - 1):
        sequence = cursor // SEQ_TILES
        hkv = sequence // B
        batch = sequence - hkv * B
        seq_begin = sequence * SEQ_TILES
        seq_end = seq_begin + SEQ_TILES
        overlap_end = tl.minimum(cta_end, seq_end)
        start_tiles = cursor - seq_begin
        add_tiles = overlap_end - cursor
        total_len = tl.load(SEQLENS_KV + batch)
        seq_start = start_tiles * _S60_TILE_N_JIT
        seq_len = tl.minimum(
            add_tiles * _S60_TILE_N_JIT,
            total_len - seq_start,
        )
        first_cta = seq_begin // tiles_per_cta
        chunk = cta - first_cta
        task_base = (
            (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
        )
        tl.store(TASK_MAP + task_base + 0, hkv)
        tl.store(TASK_MAP + task_base + 1, batch)
        tl.store(TASK_MAP + task_base + 2, chunk)
        tl.store(TASK_MAP + task_base + 3, seq_start)
        tl.store(TASK_MAP + task_base + 4, seq_len)
        tl.store(TASK_MAP + task_base + 5, seq_len)
        tl.store(
            TASK_MAP + task_base + 6,
            (seq_len + _S60_TILE_N_JIT - 1) // _S60_TILE_N_JIT,
        )
        tl.store(
            TASK_MAP + task_base + 7,
            seq_len // _S60_TILE_N_JIT,
        )
        tl.store(TASK_MAP + task_base + 8, overlap_end >= seq_end)
        tl.store(TASK_MAP + task_base + 9, 0)
        tl.store(TASK_MAP + task_base + 10, 0)
        tl.store(TASK_MAP + task_base + 11, 0)
        if overlap_end >= seq_end:
            tl.store(
                TASK_MAP + num_chunks_base + hkv * B + batch,
                chunk + 1,
            )
        cursor = overlap_end
        slot += 1

    terminator = (
        (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
    )
    tl.store(TASK_MAP + terminator + 0, -1)
    tl.store(TASK_MAP + terminator + 1, -1)


@triton.jit
def _s60_assign_scalar_dynamic_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    TOTAL_TILES,
    MAX_TASKS,
    SCHED_INTS,
):
    """Build independent S60 split-K task lists in parallel by CTA."""
    tiles_per_cta = (TOTAL_TILES + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _S60_TILE_N_JIT
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    num_chunks_base = (
        (MAX_TASKS * NUM_CTAS + 1) * _S60_TASK_STRIDE_JIT
    )
    cta = tl.program_id(0).to(tl.int32)
    cta_start = cta * tiles_per_cta
    cta_end = tl.minimum(cta_start + tiles_per_cta, TOTAL_TILES)
    prefix = 0
    slot = 0
    hkv = 0
    while hkv < H_KV:
        batch = 0
        while batch < B:
            total_len = tl.load(SEQLENS_KV + batch)
            seq_tiles = (
                total_len + _S60_TILE_N_JIT - 1
            ) // _S60_TILE_N_JIT
            seq_begin = prefix
            seq_end = seq_begin + seq_tiles
            overlap_begin = tl.maximum(cta_start, seq_begin)
            overlap_end = tl.minimum(cta_end, seq_end)
            if (overlap_begin < overlap_end) & (slot < MAX_TASKS - 1):
                start_tiles = overlap_begin - seq_begin
                add_tiles = overlap_end - overlap_begin
                seq_start = start_tiles * _S60_TILE_N_JIT
                seq_len = tl.minimum(
                    add_tiles * _S60_TILE_N_JIT,
                    total_len - seq_start,
                )
                first_cta = seq_begin // tiles_per_cta
                chunk = cta - first_cta
                task_base = (
                    (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
                )
                tl.store(TASK_MAP + task_base + 0, hkv)
                tl.store(TASK_MAP + task_base + 1, batch)
                tl.store(TASK_MAP + task_base + 2, chunk)
                tl.store(TASK_MAP + task_base + 3, seq_start)
                tl.store(TASK_MAP + task_base + 4, seq_len)
                tl.store(TASK_MAP + task_base + 5, seq_len)
                tl.store(
                    TASK_MAP + task_base + 6,
                    (seq_len + _S60_TILE_N_JIT - 1) // _S60_TILE_N_JIT,
                )
                tl.store(
                    TASK_MAP + task_base + 7,
                    seq_len // _S60_TILE_N_JIT,
                )
                tl.store(TASK_MAP + task_base + 8, overlap_end >= seq_end)
                tl.store(TASK_MAP + task_base + 9, 0)
                tl.store(TASK_MAP + task_base + 10, 0)
                tl.store(TASK_MAP + task_base + 11, 0)
                if overlap_end >= seq_end:
                    tl.store(
                        TASK_MAP + num_chunks_base + hkv * B + batch,
                        chunk + 1,
                    )
                slot += 1
            prefix = seq_end
            batch += 1
        hkv += 1

    terminator = (
        (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
    )
    tl.store(TASK_MAP + terminator + 0, -1)
    tl.store(TASK_MAP + terminator + 1, -1)


@triton.jit
def _s60_assign_dynamic_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    TOTAL_TILES,
    MAX_TASKS,
    SCHED_INTS,
    BLOCK_B: tl.constexpr,
):
    """Build heterogeneous tasks from a vectorized sequence-tile prefix."""
    tiles_per_cta = (TOTAL_TILES + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _S60_TILE_N_JIT
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    offs_b = tl.arange(0, BLOCK_B)
    valid_b = offs_b < B
    lengths = tl.load(SEQLENS_KV + offs_b, mask=valid_b, other=0)
    seq_tiles = tl.where(
        valid_b,
        (lengths + _S60_TILE_N_JIT - 1) // _S60_TILE_N_JIT,
        0,
    )
    prefix_end = tl.cumsum(seq_tiles, axis=0)
    tiles_per_head = tl.sum(seq_tiles, axis=0)

    num_chunks_base = (
        (MAX_TASKS * NUM_CTAS + 1) * _S60_TASK_STRIDE_JIT
    )
    cta = tl.program_id(0).to(tl.int32)
    cursor = cta * tiles_per_cta
    cta_end = tl.minimum(cursor + tiles_per_cta, TOTAL_TILES)
    slot = 0
    while (cursor < cta_end) & (slot < MAX_TASKS - 1):
        hkv = cursor // tiles_per_head
        head_cursor = cursor - hkv * tiles_per_head
        batch = tl.sum(
            tl.where(valid_b & (head_cursor >= prefix_end), 1, 0),
            axis=0,
        ).to(tl.int32)
        previous_end = tl.sum(
            tl.where(valid_b & (offs_b == batch - 1), prefix_end, 0),
            axis=0,
        )
        current_end = tl.sum(
            tl.where(valid_b & (offs_b == batch), prefix_end, 0),
            axis=0,
        )
        total_len = tl.sum(
            tl.where(valid_b & (offs_b == batch), lengths, 0),
            axis=0,
        )
        seq_begin = hkv * tiles_per_head + previous_end
        seq_end = hkv * tiles_per_head + current_end
        overlap_end = tl.minimum(cta_end, seq_end)
        start_tiles = cursor - seq_begin
        add_tiles = overlap_end - cursor
        seq_start = start_tiles * _S60_TILE_N_JIT
        seq_len = tl.minimum(
            add_tiles * _S60_TILE_N_JIT,
            total_len - seq_start,
        )
        first_cta = seq_begin // tiles_per_cta
        chunk = cta - first_cta
        task_base = (
            (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
        )
        tl.store(TASK_MAP + task_base + 0, hkv)
        tl.store(TASK_MAP + task_base + 1, batch)
        tl.store(TASK_MAP + task_base + 2, chunk)
        tl.store(TASK_MAP + task_base + 3, seq_start)
        tl.store(TASK_MAP + task_base + 4, seq_len)
        tl.store(TASK_MAP + task_base + 5, seq_len)
        tl.store(
            TASK_MAP + task_base + 6,
            (seq_len + _S60_TILE_N_JIT - 1) // _S60_TILE_N_JIT,
        )
        tl.store(
            TASK_MAP + task_base + 7,
            seq_len // _S60_TILE_N_JIT,
        )
        tl.store(TASK_MAP + task_base + 8, overlap_end >= seq_end)
        tl.store(TASK_MAP + task_base + 9, 0)
        tl.store(TASK_MAP + task_base + 10, 0)
        tl.store(TASK_MAP + task_base + 11, 0)
        if overlap_end >= seq_end:
            tl.store(
                TASK_MAP + num_chunks_base + hkv * B + batch,
                chunk + 1,
            )
        cursor = overlap_end
        slot += 1

    terminator = (
        (MAX_TASKS * cta + slot + 1) * _S60_TASK_STRIDE_JIT
    )
    tl.store(TASK_MAP + terminator + 0, -1)
    tl.store(TASK_MAP + terminator + 1, -1)


def _s60_dynamic_task_map_metadata(
    kv_lens: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
) -> tuple[int, int, int]:
    lengths = kv_lens.detach().cpu().to(torch.int32).tolist()
    tiles = [
        (length + _S60_TILE_N - 1) // _S60_TILE_N
        for length in lengths
    ]
    total_tiles = sum(tiles) * num_head_kv
    tiles_per_cta = max(
        (total_tiles + num_ctas - 1) // num_ctas,
        max(1, min_process_len // _S60_TILE_N),
    )
    max_tasks = tiles_per_cta + 1
    chunk_ints = len(lengths) * num_head_kv
    chunk_pad = (
        (chunk_ints + _S60_TASK_STRIDE - 1)
        // _S60_TASK_STRIDE
        * _S60_TASK_STRIDE
    )
    sched_ints = (
        (max_tasks * num_ctas + 1) * _S60_TASK_STRIDE + chunk_pad
    )
    return max_tasks, sched_ints, sched_ints


def _launch_s60_dynamic_task_map(
    kv_lens: torch.Tensor,
    task_map: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
    total_tiles: int,
    uniform_seq_tiles: int,
    max_tasks: int,
    sched_ints: int,
) -> None:
    batch = kv_lens.numel()
    use_vector_map = (
        uniform_seq_tiles == 0
        and batch >= _S60_VECTOR_TASK_MAP_MIN_BATCH
        and batch <= _S60_VECTOR_TASK_MAP_MAX_BATCH
    )
    if uniform_seq_tiles > 0:
        kernel = _s60_assign_uniform_task_map_kernel
    elif use_vector_map:
        kernel = _s60_assign_dynamic_task_map_kernel
    else:
        kernel = _s60_assign_scalar_dynamic_task_map_kernel
    kernel[(num_ctas,)](
        kv_lens,
        task_map,
        B=batch,
        H_KV=num_head_kv,
        NUM_CTAS=num_ctas,
        MIN_PROCESS_LEN=min_process_len,
        TOTAL_TILES=total_tiles,
        **(
            {"SEQ_TILES": uniform_seq_tiles}
            if uniform_seq_tiles > 0
            else (
                {"BLOCK_B": triton.next_power_of_2(batch)}
                if use_vector_map
                else {}
            )
        ),
        MAX_TASKS=max_tasks,
        SCHED_INTS=sched_ints,
        num_warps=1,
        num_stages=1,
    )


@dataclass
class S60DynamicBF16Workspace:
    """S60 workspace for the native dynamic split-K decode route."""

    task_map: torch.Tensor
    split_out: torch.Tensor
    split_lse: torch.Tensor
    out: torch.Tensor
    mtp: int
    num_ctas: int
    total_tiles: int
    uniform_seq_tiles: int
    max_tasks: int
    sched_ints: int
    heads_per_group: int
    padded_heads_per_group: int
    min_process_len: int
    route: str = "s60-dynamic-page64-blockptr-shared-kv-m8-split128-w1"


def _s60_compute_units(device: torch.device) -> int:
    """Read the S60 compute-unit count from the available device properties."""
    backend = getattr(torch, device.type, None)
    get_properties = getattr(backend, "get_device_properties", None)
    if callable(get_properties):
        try:
            properties = get_properties(device)
        except (TypeError, OSError, RuntimeError, ValueError):
            properties = None
        if properties is not None:
            for name in (
                "multi_processor_count",
                "compute_unit_count",
                "num_compute_units",
                "cluster_count",
            ):
                value = getattr(properties, name, None)
                if value is not None and int(value) > 0:
                    return int(value)
    return 16


@triton.jit
def _s60_dynamic_bf16_splitk_kernel(
    Q,
    K,
    V,
    BLOCK_IDS,
    KV_LENS,
    TASK_MAP,
    SPLIT_OUT,
    SPLIT_LSE,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_GROUP_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    MAX_TASKS,
    BLOCK_N: tl.constexpr,
    FULL_PAGE_LOADS: tl.constexpr,
    Q_SB: tl.constexpr,
    Q_SM: tl.constexpr,
    Q_SH: tl.constexpr,
    K_SBLOCK: tl.constexpr,
    K_STOKEN: tl.constexpr,
    K_SHEAD: tl.constexpr,
    K_SD: tl.constexpr,
    V_SBLOCK: tl.constexpr,
    V_STOKEN: tl.constexpr,
    V_SHEAD: tl.constexpr,
    V_SD: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
):
    """S60 BF16 producer using safe M8 dots and shared K/V loads."""
    pid = tl.program_id(0).to(tl.int32)
    group_block = pid % NUM_GROUP_BLOCKS
    cta = pid // NUM_GROUP_BLOCKS

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    # Keep the online softmax in base 2.  This is algebraically equivalent to
    # exp(score) while mapping to the efficient S60 exp2 lowering.
    scale_log2 = tl.rsqrt(tl.full((), D, tl.float32)) * 1.4426950408889634

    slot = 0
    task = ((MAX_TASKS * cta + slot) + 1) * 12
    hkv = tl.load(TASK_MAP + task + 0)
    while (slot < MAX_TASKS) & (hkv >= 0):
        batch = tl.load(TASK_MAP + task + 1)
        chunk = tl.load(TASK_MAP + task + 2)
        seq_start = tl.load(TASK_MAP + task + 3)
        seq_len = tl.load(TASK_MAP + task + 4)
        total_len = tl.load(KV_LENS + batch)

        head_rows = group_block * GROUP_M + tl.arange(0, GROUP_M)
        hq_rows = hkv * HEADS_PER_GROUP + head_rows
        valid_rows = (head_rows < HEADS_PER_GROUP) & (hq_rows < H_Q)
        q0 = tl.load(
            Q
            + batch * Q_SB
            + hq_rows[:, None] * Q_SH
            + offs_d[None, :],
            mask=valid_rows[:, None],
            other=0.0,
        )
        if NUM_SEQ_Q >= 2:
            q1 = tl.load(
                Q
                + batch * Q_SB
                + Q_SM
                + hq_rows[:, None] * Q_SH
                + offs_d[None, :],
                mask=valid_rows[:, None],
                other=0.0,
            )
        if NUM_SEQ_Q == 3:
            q2 = tl.load(
                Q
                + batch * Q_SB
                + 2 * Q_SM
                + hq_rows[:, None] * Q_SH
                + offs_d[None, :],
                mask=valid_rows[:, None],
                other=0.0,
            )

        m0 = tl.full((GROUP_M,), -float("inf"), tl.float32)
        l0 = tl.zeros((GROUP_M,), tl.float32)
        acc0 = tl.zeros((GROUP_M, D), tl.float32)
        if NUM_SEQ_Q >= 2:
            m1 = tl.full((GROUP_M,), -float("inf"), tl.float32)
            l1 = tl.zeros((GROUP_M,), tl.float32)
            acc1 = tl.zeros((GROUP_M, D), tl.float32)
        if NUM_SEQ_Q == 3:
            m2 = tl.full((GROUP_M,), -float("inf"), tl.float32)
            l2 = tl.zeros((GROUP_M,), tl.float32)
            acc2 = tl.zeros((GROUP_M, D), tl.float32)

        start = 0
        while start < seq_len:
            logical_start = seq_start + start
            logical = logical_start + offs_n
            page = logical_start // BLOCK_SIZE
            if FULL_PAGE_LOADS:
                if NUM_SEQ_Q == 3:
                    local_n = start + offs_n
                    valid_n = (local_n < seq_len) & (logical < total_len)
                physical = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + page)
            else:
                local_n = start + offs_n
                valid_n = (local_n < seq_len) & (logical < total_len)
                physical = tl.load(
                    BLOCK_IDS + batch * MAX_BLOCKS + page,
                    mask=logical_start < total_len,
                    other=0,
                )
            k_page = tl.make_block_ptr(
                base=K + physical * K_SBLOCK + hkv * K_SHEAD,
                shape=(BLOCK_SIZE, D),
                strides=(K_STOKEN, K_SD),
                offsets=(0, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0),
            )
            if FULL_PAGE_LOADS:
                k = tl.load(k_page)
            else:
                k = tl.load(
                    k_page,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            v_page = tl.make_block_ptr(
                base=V + physical * V_SBLOCK + hkv * V_SHEAD,
                shape=(BLOCK_SIZE, D),
                strides=(V_STOKEN, V_SD),
                offsets=(0, 0),
                block_shape=(BLOCK_N, D),
                order=(1, 0),
            )
            if FULL_PAGE_LOADS:
                v = tl.load(v_page)
            else:
                v = tl.load(
                    v_page,
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            scores0 = (
                tl.dot(q0, tl.trans(k), out_dtype=tl.float32) * scale_log2
            )
            if FULL_PAGE_LOADS and NUM_SEQ_Q <= 2:
                scores0 = tl.where(
                    valid_rows[:, None]
                    & (logical[None, :] < total_len - (NUM_SEQ_Q - 1)),
                    scores0,
                    -float("inf"),
                )
            else:
                scores0 = tl.where(
                    valid_rows[:, None]
                    & valid_n[None, :]
                    & (logical[None, :] < total_len - (NUM_SEQ_Q - 1)),
                    scores0,
                    -float("inf"),
                )
            tile_max0 = tl.max(scores0, axis=1)
            m_new0 = tl.maximum(m0, tile_max0)
            active0 = m_new0 != -float("inf")
            safe_new0 = tl.where(active0, m_new0, 0.0)
            safe_old0 = tl.where(m0 == -float("inf"), safe_new0, m0)
            p0 = tl.where(
                active0[:, None],
                tl.exp2(scores0 - safe_new0[:, None]),
                0.0,
            )
            alpha0 = tl.where(
                active0,
                tl.exp2(safe_old0 - safe_new0),
                0.0,
            )
            l0 = l0 * alpha0 + tl.sum(p0, axis=1)
            acc0 = acc0 * alpha0[:, None]
            acc0 = tl.dot(
                p0.to(tl.bfloat16),
                v,
                acc=acc0,
                out_dtype=tl.float32,
            )
            m0 = m_new0

            if NUM_SEQ_Q >= 2:
                scores1 = (
                    tl.dot(q1, tl.trans(k), out_dtype=tl.float32)
                    * scale_log2
                )
                if FULL_PAGE_LOADS and NUM_SEQ_Q <= 2:
                    scores1 = tl.where(
                        valid_rows[:, None]
                        & (logical[None, :] < total_len - (NUM_SEQ_Q - 2)),
                        scores1,
                        -float("inf"),
                    )
                else:
                    scores1 = tl.where(
                        valid_rows[:, None]
                        & valid_n[None, :]
                        & (logical[None, :] < total_len - (NUM_SEQ_Q - 2)),
                        scores1,
                        -float("inf"),
                    )
                tile_max1 = tl.max(scores1, axis=1)
                m_new1 = tl.maximum(m1, tile_max1)
                active1 = m_new1 != -float("inf")
                safe_new1 = tl.where(active1, m_new1, 0.0)
                safe_old1 = tl.where(m1 == -float("inf"), safe_new1, m1)
                p1 = tl.where(
                    active1[:, None],
                    tl.exp2(scores1 - safe_new1[:, None]),
                    0.0,
                )
                alpha1 = tl.where(
                    active1,
                    tl.exp2(safe_old1 - safe_new1),
                    0.0,
                )
                l1 = l1 * alpha1 + tl.sum(p1, axis=1)
                acc1 = acc1 * alpha1[:, None]
                acc1 = tl.dot(
                    p1.to(tl.bfloat16),
                    v,
                    acc=acc1,
                    out_dtype=tl.float32,
                )
                m1 = m_new1

            if NUM_SEQ_Q == 3:
                scores2 = (
                    tl.dot(q2, tl.trans(k), out_dtype=tl.float32)
                    * scale_log2
                )
                scores2 = tl.where(
                    valid_rows[:, None]
                    & valid_n[None, :]
                    & (logical[None, :] < total_len),
                    scores2,
                    -float("inf"),
                )
                tile_max2 = tl.max(scores2, axis=1)
                m_new2 = tl.maximum(m2, tile_max2)
                active2 = m_new2 != -float("inf")
                safe_new2 = tl.where(active2, m_new2, 0.0)
                safe_old2 = tl.where(m2 == -float("inf"), safe_new2, m2)
                p2 = tl.where(
                    active2[:, None],
                    tl.exp2(scores2 - safe_new2[:, None]),
                    0.0,
                )
                alpha2 = tl.where(
                    active2,
                    tl.exp2(safe_old2 - safe_new2),
                    0.0,
                )
                l2 = l2 * alpha2 + tl.sum(p2, axis=1)
                acc2 = acc2 * alpha2[:, None]
                acc2 = tl.dot(
                    p2.to(tl.bfloat16),
                    v,
                    acc=acc2,
                    out_dtype=tl.float32,
                )
                m2 = m_new2

            start += BLOCK_N

        tl.store(
            SPLIT_OUT
            + batch * SO_SB
            + chunk * SO_SC
            + hq_rows[:, None] * SO_SH
            + offs_d[None, :],
            acc0,
            mask=valid_rows[:, None],
        )
        tl.store(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + hkv * SL_SK
            + head_rows * SL_SH,
            m0,
            mask=valid_rows,
        )
        tl.store(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + hkv * SL_SK
            + head_rows * SL_SH
            + 1,
            l0,
            mask=valid_rows,
        )

        if NUM_SEQ_Q >= 2:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + chunk * SO_SC
                + SO_SM
                + hq_rows[:, None] * SO_SH
                + offs_d[None, :],
                acc1,
                mask=valid_rows[:, None],
            )
            tl.store(
                SPLIT_LSE
                + batch * SL_SB
                + chunk * SL_SC
                + SL_SM
                + hkv * SL_SK
                + head_rows * SL_SH,
                m1,
                mask=valid_rows,
            )
            tl.store(
                SPLIT_LSE
                + batch * SL_SB
                + chunk * SL_SC
                + SL_SM
                + hkv * SL_SK
                + head_rows * SL_SH
                + 1,
                l1,
                mask=valid_rows,
            )

        if NUM_SEQ_Q == 3:
            tl.store(
                SPLIT_OUT
                + batch * SO_SB
                + chunk * SO_SC
                + 2 * SO_SM
                + hq_rows[:, None] * SO_SH
                + offs_d[None, :],
                acc2,
                mask=valid_rows[:, None],
            )
            tl.store(
                SPLIT_LSE
                + batch * SL_SB
                + chunk * SL_SC
                + 2 * SL_SM
                + hkv * SL_SK
                + head_rows * SL_SH,
                m2,
                mask=valid_rows,
            )
            tl.store(
                SPLIT_LSE
                + batch * SL_SB
                + chunk * SL_SC
                + 2 * SL_SM
                + hkv * SL_SK
                + head_rows * SL_SH
                + 1,
                l2,
                mask=valid_rows,
            )

        slot += 1
        task = ((MAX_TASKS * cta + slot) + 1) * 12
        hkv = tl.load(
            TASK_MAP + task + 0,
            mask=slot < MAX_TASKS,
            other=-1,
        )


@triton.jit
def _s60_dynamic_bf16_combine_kernel(
    SPLIT_OUT,
    SPLIT_LSE,
    TASK_MAP,
    OUT,
    NUM_SEQ_Q: tl.constexpr,
    H_Q: tl.constexpr,
    HEADS_PER_GROUP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    D: tl.constexpr,
    MAX_TASKS,
    NUM_CTAS: tl.constexpr,
    SO_SB: tl.constexpr,
    SO_SC: tl.constexpr,
    SO_SM: tl.constexpr,
    SO_SH: tl.constexpr,
    SL_SB: tl.constexpr,
    SL_SC: tl.constexpr,
    SL_SM: tl.constexpr,
    SL_SK: tl.constexpr,
    SL_SH: tl.constexpr,
    O_SB: tl.constexpr,
    O_SM: tl.constexpr,
    O_SH: tl.constexpr,
):
    """Combine unnormalized S60 online-softmax states exactly."""
    pid = tl.program_id(0)
    head_programs_per_group = tl.cdiv(HEADS_PER_GROUP, HEADS_PER_PROGRAM)
    num_head_programs = (H_Q // HEADS_PER_GROUP) * head_programs_per_group
    head_program = pid % num_head_programs
    work = pid // num_head_programs
    seq_m = work % NUM_SEQ_Q
    batch = work // NUM_SEQ_Q
    hkv = head_program // head_programs_per_group
    head_pass = head_program - hkv * head_programs_per_group
    h_in_group = head_pass * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM)
    hq = hkv * HEADS_PER_GROUP + h_in_group
    valid_h = (h_in_group < HEADS_PER_GROUP) & (hq < H_Q)
    offs_d = tl.arange(0, D)

    counts = (MAX_TASKS * NUM_CTAS + 1) * 12
    batch_count = tl.load(TASK_MAP + 3)
    n_chunks = tl.load(TASK_MAP + counts + hkv * batch_count + batch)

    max_score = tl.full((HEADS_PER_PROGRAM,), -float("inf"), tl.float32)
    denom = tl.zeros((HEADS_PER_PROGRAM,), tl.float32)
    out = tl.zeros((HEADS_PER_PROGRAM, D), tl.float32)
    chunk = 0
    while chunk < n_chunks:
        chunk_max = tl.load(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + seq_m * SL_SM
            + hkv * SL_SK
            + h_in_group * SL_SH,
            mask=valid_h,
            other=-float("inf"),
        )
        chunk_denom = tl.load(
            SPLIT_LSE
            + batch * SL_SB
            + chunk * SL_SC
            + seq_m * SL_SM
            + hkv * SL_SK
            + h_in_group * SL_SH
            + 1,
            mask=valid_h,
            other=0.0,
        )
        next_max = tl.maximum(max_score, chunk_max)
        active = next_max != -float("inf")
        safe_next = tl.where(active, next_max, 0.0)
        safe_old = tl.where(max_score == -float("inf"), safe_next, max_score)
        alpha = tl.where(active, tl.exp2(safe_old - safe_next), 0.0)
        beta = tl.where(
            chunk_max != -float("inf"),
            tl.exp2(chunk_max - safe_next),
            0.0,
        )
        partial = tl.load(
            SPLIT_OUT
            + batch * SO_SB
            + chunk * SO_SC
            + seq_m * SO_SM
            + hq[:, None] * SO_SH
            + offs_d[None, :],
            mask=valid_h[:, None],
            other=0.0,
        )
        out = out * alpha[:, None] + beta[:, None] * partial
        denom = denom * alpha + beta * chunk_denom
        max_score = next_max
        chunk += 1

    safe_denom = tl.where(denom > 0.0, denom, 1.0)
    tl.store(
        OUT
        + batch * O_SB
        + seq_m * O_SM
        + hq[:, None] * O_SH
        + offs_d[None, :],
        tl.where(denom[:, None] > 0.0, out / safe_denom[:, None], 0.0),
        mask=valid_h[:, None],
    )


def _prepare_s60_dynamic_bf16_workspace(
    inputs: DynamicBF16Inputs,
) -> S60DynamicBF16Workspace:
    mtp, hq, hkv = _dynamic__validate(inputs)
    min_kv_len = int(inputs.kv_lens.min().item())
    if min_kv_len < mtp:
        raise ValueError("each KV length must be at least MTP")
    heads_per_group = hq // hkv
    padded_heads = triton.cdiv(heads_per_group, 8) * 8
    lengths = inputs.kv_lens.detach().cpu().to(torch.int32).tolist()
    seq_tiles = [
        (int(length) + _S60_TILE_N - 1) // _S60_TILE_N
        for length in lengths
    ]
    total_tiles = sum(seq_tiles) * hkv
    uniform_seq_tiles = (
        seq_tiles[0]
        if seq_tiles and all(value == seq_tiles[0] for value in seq_tiles)
        else 0
    )
    workspace_bytes_per_cta = (
        inputs.batch
        * mtp
        * (hq * HEAD_DIM + hkv * padded_heads * 2)
        * 4
    )
    min_process_len = (
        _S60_LARGE_MIN_PROCESS_LEN
        if min_kv_len >= _S60_LARGE_MIN_PROCESS_LEN
        else _S60_MIN_PROCESS_LEN
    )
    min_tiles_per_cta = max(1, min_process_len // _S60_TILE_N)
    max_ctas_by_granularity = max(
        1,
        (total_tiles + min_tiles_per_cta - 1) // min_tiles_per_cta,
    )
    max_ctas_by_memory = max(
        1,
        _S60_WORKSPACE_BUDGET_BYTES // workspace_bytes_per_cta,
    )
    num_ctas = max(
        1,
        min(
            total_tiles,
            max_ctas_by_granularity,
            _s60_compute_units(inputs.q.device) * _S60_CTAS_PER_CU,
            _S60_MAX_CTAS,
            max_ctas_by_memory,
        ),
    )
    max_tasks, sched_ints, workspace_ints = _s60_dynamic_task_map_metadata(
        inputs.kv_lens,
        num_head_kv=hkv,
        num_ctas=num_ctas,
        min_process_len=min_process_len,
    )
    workspace = S60DynamicBF16Workspace(
        task_map=torch.full(
            (workspace_ints,),
            -1,
            dtype=torch.int32,
            device=inputs.q.device,
        ),
        split_out=torch.empty(
            (inputs.batch, num_ctas, mtp, hq, HEAD_DIM),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        split_lse=torch.empty(
            (inputs.batch, num_ctas, mtp, hkv, padded_heads * 2),
            dtype=torch.float32,
            device=inputs.q.device,
        ),
        out=torch.empty(
            (inputs.batch, mtp, hq, HEAD_DIM),
            dtype=torch.bfloat16,
            device=inputs.q.device,
        ),
        mtp=mtp,
        num_ctas=num_ctas,
        total_tiles=total_tiles,
        uniform_seq_tiles=uniform_seq_tiles,
        max_tasks=max_tasks,
        sched_ints=sched_ints,
        heads_per_group=heads_per_group,
        padded_heads_per_group=padded_heads,
        min_process_len=min_process_len,
    )
    return workspace


def _run_s60_dynamic_bf16(
    inputs: DynamicBF16Inputs,
    workspace: S60DynamicBF16Workspace,
) -> torch.Tensor:
    mtp, hq, hkv = _dynamic__validate(inputs)
    if mtp != workspace.mtp or hq // hkv != workspace.heads_per_group:
        raise ValueError("S60 workspace does not match inputs")

    _launch_s60_dynamic_task_map(
        inputs.kv_lens,
        workspace.task_map,
        num_head_kv=hkv,
        num_ctas=workspace.num_ctas,
        min_process_len=workspace.min_process_len,
        total_tiles=workspace.total_tiles,
        uniform_seq_tiles=workspace.uniform_seq_tiles,
        max_tasks=workspace.max_tasks,
        sched_ints=workspace.sched_ints,
    )
    q_4d = inputs.q.reshape(inputs.batch, mtp, hq, HEAD_DIM)
    group_m = _S60_GROUP_M
    num_group_blocks = triton.cdiv(workspace.heads_per_group, group_m)
    # Each MTP query uses an independent, proven-safe M8 dot while all of them
    # share the K/V tile loaded by the program.  Wider fused M dimensions can
    # trigger a GCU300 SIP in the current Enflame Triton compiler.
    producer_grid = workspace.num_ctas * num_group_blocks
    _s60_dynamic_bf16_splitk_kernel[(producer_grid,)](
        q_4d,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        workspace.task_map,
        workspace.split_out,
        workspace.split_lse,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        GROUP_M=group_m,
        NUM_GROUP_BLOCKS=num_group_blocks,
        D=HEAD_DIM,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_BLOCKS=inputs.block_ids.shape[1],
        MAX_TASKS=workspace.max_tasks,
        BLOCK_N=_S60_BLOCK_N,
        FULL_PAGE_LOADS=(workspace.min_process_len >= 4096),
        Q_SB=q_4d.stride(0),
        Q_SM=q_4d.stride(1),
        Q_SH=q_4d.stride(2),
        K_SBLOCK=inputs.k_cache.stride(0),
        K_STOKEN=inputs.k_cache.stride(1),
        K_SHEAD=inputs.k_cache.stride(2),
        K_SD=inputs.k_cache.stride(3),
        V_SBLOCK=inputs.v_cache.stride(0),
        V_STOKEN=inputs.v_cache.stride(1),
        V_SHEAD=inputs.v_cache.stride(2),
        V_SD=inputs.v_cache.stride(3),
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SK=workspace.split_lse.stride(3),
        SL_SH=workspace.split_lse.stride(4) * 2,
        num_warps=1,
        num_stages=2 if mtp == 1 else 3,
    )
    combine_heads_per_program = (
        8
        if (
            mtp >= 2
            or workspace.num_ctas >= 64
            or workspace.total_tiles >= workspace.num_ctas * 32
            # Small skewed batches have few combine programs relative to the
            # producer work.  HPP8 halves their MTP1 combine grid.
            or inputs.batch <= 16
        )
        and workspace.heads_per_group >= 8
        else 4 if workspace.heads_per_group >= 4
        else 2 if workspace.heads_per_group >= 2
        else 1
    )
    combine_grid = (
        inputs.batch
        * mtp
        * hkv
        * triton.cdiv(workspace.heads_per_group, combine_heads_per_program)
    )
    _s60_dynamic_bf16_combine_kernel[(combine_grid,)](
        workspace.split_out,
        workspace.split_lse,
        workspace.task_map,
        workspace.out,
        NUM_SEQ_Q=mtp,
        H_Q=hq,
        HEADS_PER_GROUP=workspace.heads_per_group,
        HEADS_PER_PROGRAM=combine_heads_per_program,
        D=HEAD_DIM,
        MAX_TASKS=workspace.max_tasks,
        NUM_CTAS=workspace.num_ctas,
        SO_SB=workspace.split_out.stride(0),
        SO_SC=workspace.split_out.stride(1),
        SO_SM=workspace.split_out.stride(2),
        SO_SH=workspace.split_out.stride(3),
        SL_SB=workspace.split_lse.stride(0),
        SL_SC=workspace.split_lse.stride(1),
        SL_SM=workspace.split_lse.stride(2),
        SL_SK=workspace.split_lse.stride(3),
        SL_SH=workspace.split_lse.stride(4) * 2,
        O_SB=workspace.out.stride(0),
        O_SM=workspace.out.stride(1),
        O_SH=workspace.out.stride(2),
        num_warps=1,
        num_stages=1,
    )
    return workspace.out.reshape_as(inputs.q)


def _refresh_s60_dynamic_bf16_task_map(
    inputs: DynamicBF16Inputs,
    workspace: S60DynamicBF16Workspace,
) -> None:
    mtp, _hq, hkv = _dynamic__validate(inputs)
    if mtp != workspace.mtp:
        raise ValueError("S60 workspace MTP does not match inputs")
    lengths = inputs.kv_lens.detach().cpu().to(torch.int32).tolist()
    seq_tiles = [
        (int(length) + _S60_TILE_N - 1) // _S60_TILE_N
        for length in lengths
    ]
    workspace.total_tiles = sum(seq_tiles) * hkv
    workspace.uniform_seq_tiles = (
        seq_tiles[0]
        if seq_tiles and all(value == seq_tiles[0] for value in seq_tiles)
        else 0
    )
    max_tasks, sched_ints, _ = _s60_dynamic_task_map_metadata(
        inputs.kv_lens,
        num_head_kv=hkv,
        num_ctas=workspace.num_ctas,
        min_process_len=workspace.min_process_len,
    )
    if max_tasks > workspace.max_tasks or sched_ints > workspace.task_map.numel():
        raise ValueError("S60 dynamic task-map capacity changed; rebuild the workspace")
    _launch_s60_dynamic_task_map(
        inputs.kv_lens,
        workspace.task_map,
        num_head_kv=hkv,
        num_ctas=workspace.num_ctas,
        min_process_len=workspace.min_process_len,
        total_tiles=workspace.total_tiles,
        uniform_seq_tiles=workspace.uniform_seq_tiles,
        max_tasks=workspace.max_tasks,
        sched_ints=workspace.sched_ints,
    )


def prepare_dynamic_bf16_workspace(
    inputs: DynamicBF16Inputs,
) -> S60DynamicBF16Workspace:
    """Allocate the S60 dynamic Split-K workspace."""
    return _prepare_s60_dynamic_bf16_workspace(inputs)


def attention_decode_bf16_dynamic(
    inputs: DynamicBF16Inputs,
    workspace: S60DynamicBF16Workspace,
) -> torch.Tensor:
    """Run BF16 dynamic decode on Enflame S60."""
    if not isinstance(workspace, S60DynamicBF16Workspace):
        raise TypeError(f"unsupported BF16 dynamic workspace: {type(workspace)!r}")
    return _run_s60_dynamic_bf16(inputs, workspace)


def refresh_dynamic_bf16_task_map(
    inputs: DynamicBF16Inputs,
    workspace: S60DynamicBF16Workspace,
) -> None:
    """Refresh scheduling metadata after dynamic KV lengths change."""
    if not isinstance(workspace, S60DynamicBF16Workspace):
        raise TypeError(f"unsupported BF16 dynamic workspace: {type(workspace)!r}")
    _refresh_s60_dynamic_bf16_task_map(inputs, workspace)


def bf16_dynamic_workspace_is_reset(
    workspace: S60DynamicBF16Workspace,
) -> bool:
    """The S60 path has no persistent cooperative-finalization counters."""
    if not isinstance(workspace, S60DynamicBF16Workspace):
        raise TypeError(f"unsupported BF16 dynamic workspace: {type(workspace)!r}")
    return True


__all__ = [
    "BLOCK_SIZE",
    "HEAD_DIM",
    "OFFICIAL_CASES",
    "DynamicBF16Inputs",
    "S60DynamicBF16Workspace",
    "attention_decode_bf16_dynamic",
    "bf16_dynamic_workspace_is_reset",
    "prepare_dynamic_bf16_workspace",
    "refresh_dynamic_bf16_task_map",
]
