"""GPU-resident cluster task-scheduler kernels for final FP8 decode.

A prefix kernel computes deterministic cluster offsets and metadata. A compact
records kernel emits only real chunks plus required cluster padding. Runtime
map generation never falls back to a CPU builder.
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
TASK_STRIDE = 12
TASK_SLOTS = 2
TILE_N = 64
DIRECT_MODE = 0
GROUP_MODE = 1
DUMMY_MODE = 2
META_NUM_CLUSTERS = 0
META_PHYSICAL_CTAS = 1
META_REDUCTION_CLUSTERS = 2
META_DIRECT_TASKS = 3
META_FINE_CHUNKS_MAX = 4
META_EFFECTIVE_CHUNKS_MAX = 5
META_COMPUTE_TASKS = 6
META_DUMMY_TASKS = 7
META_SCHED_INTS = 8
META_INVALID_LENGTHS = 9
META_SIZE = 10
_TASK_STRIDE_JIT = tl.constexpr(TASK_STRIDE)
_TASK_SLOTS_JIT = tl.constexpr(TASK_SLOTS)
_TILE_N_JIT = tl.constexpr(TILE_N)
_DIRECT_MODE_JIT = tl.constexpr(DIRECT_MODE)
_GROUP_MODE_JIT = tl.constexpr(GROUP_MODE)
_DUMMY_MODE_JIT = tl.constexpr(DUMMY_MODE)
_META_NUM_CLUSTERS_JIT = tl.constexpr(META_NUM_CLUSTERS)
_META_PHYSICAL_CTAS_JIT = tl.constexpr(META_PHYSICAL_CTAS)
_META_REDUCTION_CLUSTERS_JIT = tl.constexpr(META_REDUCTION_CLUSTERS)
_META_DIRECT_TASKS_JIT = tl.constexpr(META_DIRECT_TASKS)
_META_FINE_CHUNKS_MAX_JIT = tl.constexpr(META_FINE_CHUNKS_MAX)
_META_EFFECTIVE_CHUNKS_MAX_JIT = tl.constexpr(META_EFFECTIVE_CHUNKS_MAX)
_META_COMPUTE_TASKS_JIT = tl.constexpr(META_COMPUTE_TASKS)
_META_DUMMY_TASKS_JIT = tl.constexpr(META_DUMMY_TASKS)
_META_SCHED_INTS_JIT = tl.constexpr(META_SCHED_INTS)
_META_INVALID_LENGTHS_JIT = tl.constexpr(META_INVALID_LENGTHS)

@triton.jit
def assign_cluster_task_prefix_kernel(SEQLENS_KV, OFFSETS, META, TASK_MAP, B: tl.constexpr, H_KV: tl.constexpr, NUM_SEQ_Q: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, BLOCK_SEQ: tl.constexpr):
    """Compute per-sequence offsets and exact task-map metadata on device."""
    batch = tl.arange(0, BLOCK_SEQ)
    num_sequences = B * H_KV
    valid = batch < B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=0).to(tl.int32)
    positive = valid & (total_len > 0)
    num_chunks = tl.where(positive, (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS, 0)
    direct = (positive & (num_chunks == 1)).to(tl.int32)
    reduction_clusters = tl.where(positive & (num_chunks > 1), (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE, 0).to(tl.int32)
    effective_chunks = tl.where(num_chunks == 1, 1, reduction_clusters)
    reduction_offsets, reduction_per_head = tle.cumsum(reduction_clusters, axis=0, reverse=False)
    direct_offsets, direct_per_head = tle.cumsum(direct, axis=0, reverse=False)
    for hkv in range(H_KV):
        seq = hkv * B + batch
        tl.store(OFFSETS + seq * 2 + 0, hkv * reduction_per_head + reduction_offsets, mask=valid)
        tl.store(OFFSETS + seq * 2 + 1, hkv * direct_per_head + direct_offsets, mask=valid)
    reduction_total = reduction_per_head * H_KV
    direct_total = direct_per_head * H_KV
    direct_clusters = (direct_total + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    num_clusters = reduction_total + direct_clusters
    physical_ctas = num_clusters * CLUSTER_SIZE
    num_chunks_base = (_TASK_SLOTS_JIT * physical_ctas + 1) * _TASK_STRIDE_JIT
    chunk_pad_ints = (num_sequences + _TASK_STRIDE_JIT - 1) // _TASK_STRIDE_JIT * _TASK_STRIDE_JIT
    sched_ints = num_chunks_base + chunk_pad_ints
    fine_chunks_max = tl.max(num_chunks, axis=0)
    effective_chunks_max = tl.max(effective_chunks, axis=0)
    compute_tasks = tl.sum(num_chunks, axis=0) * H_KV
    dummy_tasks = tl.sum(tl.where(num_chunks > 1, reduction_clusters * CLUSTER_SIZE - num_chunks, 0), axis=0) * H_KV
    invalid_lengths = tl.sum((valid & (total_len <= 0)).to(tl.int32), axis=0) * H_KV
    tl.store(META + _META_NUM_CLUSTERS_JIT, num_clusters)
    tl.store(META + _META_PHYSICAL_CTAS_JIT, physical_ctas)
    tl.store(META + _META_REDUCTION_CLUSTERS_JIT, reduction_total)
    tl.store(META + _META_DIRECT_TASKS_JIT, direct_total)
    tl.store(META + _META_FINE_CHUNKS_MAX_JIT, fine_chunks_max)
    tl.store(META + _META_EFFECTIVE_CHUNKS_MAX_JIT, effective_chunks_max)
    tl.store(META + _META_COMPUTE_TASKS_JIT, compute_tasks)
    tl.store(META + _META_DUMMY_TASKS_JIT, dummy_tasks)
    tl.store(META + _META_SCHED_INTS_JIT, sched_ints)
    tl.store(META + _META_INVALID_LENGTHS_JIT, invalid_lengths)
    tl.store(TASK_MAP + 0, CHUNK_TOKENS // _TILE_N_JIT + 1)
    tl.store(TASK_MAP + 1, physical_ctas)
    tl.store(TASK_MAP + 2, H_KV)
    tl.store(TASK_MAP + 3, B)
    tl.store(TASK_MAP + 4, sched_ints * 4)

@triton.jit
def _store_task_record(TASK_MAP, cta, hkv, batch, chunk, seq_start, seq_len, seq_kvcache, num_tile_kv, num_tile_full, is_causal, mode, group_chunk, group_count, mask):
    task_base = (cta * _TASK_SLOTS_JIT + 1) * _TASK_STRIDE_JIT
    tl.store(TASK_MAP + task_base + 0, hkv, mask=mask)
    tl.store(TASK_MAP + task_base + 1, batch, mask=mask)
    tl.store(TASK_MAP + task_base + 2, chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=mask)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=mask)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=mask)
    tl.store(TASK_MAP + task_base + 6, num_tile_kv, mask=mask)
    tl.store(TASK_MAP + task_base + 7, num_tile_full, mask=mask)
    tl.store(TASK_MAP + task_base + 8, is_causal, mask=mask)
    tl.store(TASK_MAP + task_base + 9, mode, mask=mask)
    tl.store(TASK_MAP + task_base + 10, group_chunk, mask=mask)
    tl.store(TASK_MAP + task_base + 11, group_count, mask=mask)
    sentinel_base = (cta * _TASK_SLOTS_JIT + 2) * _TASK_STRIDE_JIT
    tl.store(TASK_MAP + sentinel_base + 0, -1, mask=mask)
    tl.store(TASK_MAP + sentinel_base + 1, -1, mask=mask)

@triton.jit
def assign_cluster_task_records_compact_kernel(SEQLENS_KV, OFFSETS, META, TASK_MAP, B: tl.constexpr, H_KV: tl.constexpr, NUM_SEQ_Q: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr):
    """Write only this sequence's real chunks and required cluster padding.

    A vectorized records kernel would use ``BLOCK_CHUNKS`` equal to the
    longest sequence's chunk count for every sequence. This compact kernel
    is deliberately scalar-loop based: short sequences in a long-tail batch
    write O(their own chunks) task records rather than O(max chunks in batch).
    """
    seq_id = tl.program_id(0)
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1)
    reduction_total = tl.load(META + _META_REDUCTION_CLUSTERS_JIT)
    direct_total = tl.load(META + _META_DIRECT_TASKS_JIT)
    physical_ctas = tl.load(META + _META_PHYSICAL_CTAS_JIT)
    num_chunks_base = (_TASK_SLOTS_JIT * physical_ctas + 1) * _TASK_STRIDE_JIT
    group_count = (num_chunks + CLUSTER_SIZE - 1) // CLUSTER_SIZE
    effective_chunks = tl.where(num_chunks == 1, 1, group_count)
    tl.store(TASK_MAP + num_chunks_base + seq_id, effective_chunks)
    if num_chunks <= 0:
        return
    if num_chunks == 1:
        cluster = reduction_total + direct_index // CLUSTER_SIZE
        rank = direct_index % CLUSTER_SIZE
        cta = cluster * CLUSTER_SIZE + rank
        seq_kvcache = tl.maximum(total_len - NUM_SEQ_Q, 0)
        _store_task_record(TASK_MAP, cta, hkv, batch, 0, 0, total_len, seq_kvcache, (total_len + _TILE_N_JIT - 1) // _TILE_N_JIT, seq_kvcache // _TILE_N_JIT, 1, _DIRECT_MODE_JIT, 0, 1, True)
        if direct_index == direct_total - 1:
            clear_rank = tl.arange(0, CLUSTER_SIZE)[:, None]
            clear_field = tl.arange(0, 16)[None, :]
            clear_cta = cluster * CLUSTER_SIZE + clear_rank
            clear_task_base = (clear_cta * _TASK_SLOTS_JIT + 1) * _TASK_STRIDE_JIT
            clear_mask = (clear_rank > rank) & (clear_field < _TASK_STRIDE_JIT)
            tl.store(TASK_MAP + clear_task_base + clear_field, -1, mask=clear_mask)
            clear_sentinel_base = (clear_cta * _TASK_SLOTS_JIT + 2) * _TASK_STRIDE_JIT
            sentinel_mask = (clear_rank > rank) & (clear_field < 2)
            tl.store(TASK_MAP + clear_sentinel_base + clear_field, -1, mask=sentinel_mask)
        return
    covered_chunks = group_count * CLUSTER_SIZE
    chunk = 0
    while chunk < covered_chunks:
        real = chunk < num_chunks
        group_chunk = chunk // CLUSTER_SIZE
        rank = chunk % CLUSTER_SIZE
        cluster = reduction_offset + group_chunk
        cta = cluster * CLUSTER_SIZE + rank
        logical_start = chunk * CHUNK_TOKENS
        seq_start = tl.where(real, logical_start, 0)
        seq_len = tl.where(real, tl.minimum(CHUNK_TOKENS, total_len - logical_start), 0)
        is_last = real & (chunk == num_chunks - 1)
        seq_kvcache = tl.where(is_last, tl.maximum(seq_len - NUM_SEQ_Q, 0), seq_len)
        num_tile_kv = (seq_len + _TILE_N_JIT - 1) // _TILE_N_JIT
        num_tile_full = tl.where(is_last, seq_kvcache // _TILE_N_JIT, seq_len // _TILE_N_JIT)
        mode = tl.where(real, _GROUP_MODE_JIT, _DUMMY_MODE_JIT)
        _store_task_record(TASK_MAP, cta, hkv, batch, chunk, seq_start, seq_len, seq_kvcache, num_tile_kv, num_tile_full, is_last.to(tl.int32), mode, group_chunk, group_count, True)
        chunk += 1

@triton.jit
def refresh_cluster_task_tail_kernel(SEQLENS_KV, OFFSETS, META, TASK_MAP, B: tl.constexpr, H_KV: tl.constexpr, NUM_SEQ_Q: tl.constexpr, CLUSTER_SIZE: tl.constexpr, CHUNK_TOKENS: tl.constexpr, BLOCK_SEQ: tl.constexpr):
    """Refresh length-dependent tail fields while cluster topology is stable."""
    seq_id = tl.arange(0, BLOCK_SEQ)
    valid = seq_id < B * H_KV
    hkv = seq_id // B
    batch = seq_id - hkv * B
    total_len = tl.load(SEQLENS_KV + batch, mask=valid, other=1).to(tl.int32)
    num_chunks = (total_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    reduction_offset = tl.load(OFFSETS + seq_id * 2 + 0, mask=valid, other=0)
    direct_index = tl.load(OFFSETS + seq_id * 2 + 1, mask=valid, other=0)
    reduction_total = tl.load(META + _META_REDUCTION_CLUSTERS_JIT)
    direct = num_chunks == 1
    tail_chunk = num_chunks - 1
    direct_cluster = reduction_total + direct_index // CLUSTER_SIZE
    direct_rank = direct_index % CLUSTER_SIZE
    reduction_cluster = reduction_offset + tail_chunk // CLUSTER_SIZE
    reduction_rank = tail_chunk % CLUSTER_SIZE
    cluster = tl.where(direct, direct_cluster, reduction_cluster)
    rank = tl.where(direct, direct_rank, reduction_rank)
    cta = cluster * CLUSTER_SIZE + rank
    task_base = (cta * _TASK_SLOTS_JIT + 1) * _TASK_STRIDE_JIT
    seq_start = (num_chunks - 1) * CHUNK_TOKENS
    seq_len = total_len - seq_start
    seq_kvcache = tl.maximum(seq_len - NUM_SEQ_Q, 0)
    tl.store(TASK_MAP + task_base + 3, seq_start, mask=valid)
    tl.store(TASK_MAP + task_base + 4, seq_len, mask=valid)
    tl.store(TASK_MAP + task_base + 5, seq_kvcache, mask=valid)
    tl.store(TASK_MAP + task_base + 6, (seq_len + _TILE_N_JIT - 1) // _TILE_N_JIT, mask=valid)
    tl.store(TASK_MAP + task_base + 7, seq_kvcache // _TILE_N_JIT, mask=valid)
    tl.store(TASK_MAP + task_base + 8, 1, mask=valid)

@dataclass
class DecodeTaskSchedule:
    task_workspace: torch.Tensor
    task_map: torch.Tensor
    offsets: torch.Tensor
    meta: torch.Tensor
    cluster_size: int
    chunk_tokens: int
    block_seq: int
    block_chunks: int
    capacity_clusters: int
    capacity_ints: int
    num_clusters: int = 0
    physical_ctas: int = 0
    sched_ints: int = 0
    partial_slots: int = 0
    stats: dict | None = None

def _capacity(*, num_sequences: int, max_seq_kv: int, cluster_size: int, chunk_tokens: int) -> tuple[int, int, int]:
    max_chunks = max(1, (max_seq_kv + chunk_tokens - 1) // chunk_tokens)
    max_groups = (max_chunks + cluster_size - 1) // cluster_size
    max_reduction_clusters = num_sequences * max_groups if max_chunks > 1 else 0
    max_direct_clusters = (num_sequences + cluster_size - 1) // cluster_size
    capacity_clusters = max_reduction_clusters + max_direct_clusters
    capacity_ctas = max(capacity_clusters * cluster_size, cluster_size)
    chunk_pad_ints = (num_sequences + TASK_STRIDE - 1) // TASK_STRIDE * TASK_STRIDE
    capacity_ints = (TASK_SLOTS * capacity_ctas + 1) * TASK_STRIDE + chunk_pad_ints
    block_chunks = triton.next_power_of_2(max(max_chunks, cluster_size))
    return (capacity_clusters, capacity_ints, block_chunks)

def allocate_cluster_task_map(kv_lens: torch.Tensor, *, num_head_kv: int, max_seq_kv: int, cluster_size: int, chunk_tokens: int) -> DecodeTaskSchedule:
    """Allocate capacity and populate a cluster task map entirely on GPU."""
    if not kv_lens.is_cuda:
        raise ValueError('cluster GPU assignment requires CUDA kv_lens')
    if cluster_size not in (2, 4, 8):
        raise ValueError(f'cluster_size must be 2, 4, or 8, got {cluster_size}')
    if chunk_tokens < TILE_N or chunk_tokens % TILE_N:
        raise ValueError(f'chunk_tokens must be a positive multiple of {TILE_N}, got {chunk_tokens}')
    num_sequences = kv_lens.numel() * num_head_kv
    block_seq = triton.next_power_of_2(kv_lens.numel())
    if block_seq > 1024:
        raise ValueError(f'cluster GPU assign currently supports B <= 1024, got {kv_lens.numel()}')
    capacity_clusters, capacity_ints, block_chunks = _capacity(num_sequences=num_sequences, max_seq_kv=max_seq_kv, cluster_size=cluster_size, chunk_tokens=chunk_tokens)
    task_map = torch.full((capacity_ints,), -1, dtype=torch.int32, device=kv_lens.device)
    assignment = DecodeTaskSchedule(task_workspace=task_map.view(torch.int8), task_map=task_map, offsets=torch.empty((num_sequences, 2), dtype=torch.int32, device=kv_lens.device), meta=torch.empty((META_SIZE,), dtype=torch.int32, device=kv_lens.device), cluster_size=cluster_size, chunk_tokens=chunk_tokens, block_seq=block_seq, block_chunks=block_chunks, capacity_clusters=capacity_clusters, capacity_ints=capacity_ints)
    launch_cluster_task_map_assign(kv_lens, assignment, num_head_kv=num_head_kv, refresh_host_metadata=True)
    return assignment

def launch_cluster_task_map_assign(kv_lens: torch.Tensor, assignment: DecodeTaskSchedule, *, num_head_kv: int, refresh_host_metadata: bool=False) -> None:
    """Regenerate an allocated map; fixed-topology calls need no host sync."""
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError('kv_lens/H_KV shape differs from the allocated cluster task map')
    prefix_warps = max(1, min(32, assignment.block_seq // 32))
    assign_cluster_task_prefix_kernel[1,](kv_lens, assignment.offsets, assignment.meta, assignment.task_map, B=batch, H_KV=num_head_kv, NUM_SEQ_Q=1, CLUSTER_SIZE=assignment.cluster_size, CHUNK_TOKENS=assignment.chunk_tokens, BLOCK_SEQ=assignment.block_seq, num_warps=prefix_warps, num_stages=1)
    record_warps = 1
    records_kernel = assign_cluster_task_records_compact_kernel
    record_args = {'B': batch, 'H_KV': num_head_kv, 'NUM_SEQ_Q': 1, 'CLUSTER_SIZE': assignment.cluster_size, 'CHUNK_TOKENS': assignment.chunk_tokens, 'num_warps': record_warps, 'num_stages': 1}
    records_kernel[num_sequences,](kv_lens, assignment.offsets, assignment.meta, assignment.task_map, **record_args)
    if not refresh_host_metadata:
        return
    values = assignment.meta.detach().cpu().to(torch.int64).tolist()
    invalid_lengths = values[META_INVALID_LENGTHS]
    if invalid_lengths:
        raise ValueError(f'cluster GPU assign does not support {invalid_lengths} empty KV sequences')
    num_clusters = values[META_NUM_CLUSTERS]
    sched_ints = values[META_SCHED_INTS]
    if num_clusters > assignment.capacity_clusters or sched_ints > assignment.capacity_ints:
        raise RuntimeError(f'cluster task-map capacity was underestimated: clusters={num_clusters}/{assignment.capacity_clusters}, ints={sched_ints}/{assignment.capacity_ints}')
    assignment.num_clusters = num_clusters
    assignment.physical_ctas = values[META_PHYSICAL_CTAS]
    assignment.sched_ints = sched_ints
    assignment.partial_slots = max(values[META_EFFECTIVE_CHUNKS_MAX], 1)
    assignment.stats = {'cluster_size': assignment.cluster_size, 'num_clusters': num_clusters, 'physical_ctas': values[META_PHYSICAL_CTAS], 'compute_tasks': values[META_COMPUTE_TASKS], 'fine_chunks_max': values[META_FINE_CHUNKS_MAX], 'effective_chunks_max': values[META_EFFECTIVE_CHUNKS_MAX], 'direct_tasks': values[META_DIRECT_TASKS], 'dummy_tasks': values[META_DUMMY_TASKS]}

def launch_cluster_task_tail_refresh(kv_lens: torch.Tensor, assignment: DecodeTaskSchedule, *, num_head_kv: int) -> None:
    """Update tail records without rebuilding unchanged 512-token topology.

    The caller must run ``launch_cluster_task_map_assign`` with refreshed host
    metadata whenever any ``ceil(kv_len / chunk_tokens)`` value changes.
    """
    batch = kv_lens.numel()
    num_sequences = batch * num_head_kv
    if assignment.offsets.numel() != num_sequences * 2:
        raise ValueError('kv_lens/H_KV shape differs from the allocated cluster task map')
    block_seq = triton.next_power_of_2(num_sequences)
    num_warps = max(1, min(8, block_seq // 32))
    refresh_cluster_task_tail_kernel[1,](kv_lens, assignment.offsets, assignment.meta, assignment.task_map, B=batch, H_KV=num_head_kv, NUM_SEQ_Q=1, CLUSTER_SIZE=assignment.cluster_size, CHUNK_TOKENS=assignment.chunk_tokens, BLOCK_SEQ=block_seq, num_warps=num_warps, num_stages=1)

__all__ = [
    "DecodeTaskSchedule",
    "TASK_SLOTS",
    "TASK_STRIDE",
    "TILE_N",
    "allocate_cluster_task_map",
    "launch_cluster_task_map_assign",
    "launch_cluster_task_tail_refresh",
]
