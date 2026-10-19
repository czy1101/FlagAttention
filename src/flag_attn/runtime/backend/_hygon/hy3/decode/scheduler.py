"""Pure Triton scheduling and launch-policy helpers for Hygon hy3 decode.

The task-map producer and geometry/resource policy live together so the
backend has one auxiliary module beside the attention kernels in ``core.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

PURE_TRITON_TASK_STRIDE = 12
PURE_TRITON_TILE_N = 64
_PURE_TRITON_TASK_STRIDE = tl.constexpr(12)
_PURE_TRITON_TILE_N = tl.constexpr(64)


@triton.jit
def assign_pure_triton_task_map_kernel(
    SEQLENS_KV,
    TASK_MAP,
    B: tl.constexpr,
    H_KV: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    MIN_PROCESS_LEN: tl.constexpr,
    MAX_TASKS: tl.constexpr,
    SCHED_INTS: tl.constexpr,
):
    """Build the portable MTP1 split-K schedule without TLE primitives."""
    total_tiles_per_head = 0
    b_scan = 0
    while b_scan < B:
        total_len = tl.load(SEQLENS_KV + b_scan)
        total_tiles_per_head += (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N
        b_scan += 1

    total_tiles = total_tiles_per_head * H_KV
    tiles_per_cta = (total_tiles + NUM_CTAS - 1) // NUM_CTAS
    minimum_tiles = MIN_PROCESS_LEN // _PURE_TRITON_TILE_N
    if minimum_tiles < 1:
        minimum_tiles = 1
    tiles_per_cta = tl.maximum(tiles_per_cta, minimum_tiles)
    if tl.program_id(0) == 0:
        tl.store(TASK_MAP + 0, tiles_per_cta + 1)
        tl.store(TASK_MAP + 1, NUM_CTAS)
        tl.store(TASK_MAP + 2, H_KV)
        tl.store(TASK_MAP + 3, B)
        tl.store(TASK_MAP + 4, SCHED_INTS * 4)

    num_chunks_base = (MAX_TASKS * NUM_CTAS + 1) * _PURE_TRITON_TASK_STRIDE
    hkv = 0
    batch = 0
    chunks = 0
    start_tiles = 0
    total_len = tl.load(SEQLENS_KV)
    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

    cta = 0
    while cta < NUM_CTAS:
        bucket = tiles_per_cta
        slot = 0
        while (bucket > 0) & (hkv < H_KV) & (slot < MAX_TASKS - 1):
            if tiles_left <= 0:
                batch += 1
                if batch >= B:
                    batch = 0
                    hkv += 1
                chunks = 0
                start_tiles = 0
                if hkv < H_KV:
                    total_len = tl.load(SEQLENS_KV + batch)
                    tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

            if (tiles_left > 0) & (hkv < H_KV):
                cur_hkv = hkv
                cur_batch = batch
                total_len = tl.load(SEQLENS_KV + cur_batch)
                chunk = chunks
                add_tiles = tl.minimum(tiles_left, bucket)
                if chunks == NUM_CTAS - 1:
                    add_tiles = tiles_left
                seq_start = start_tiles * _PURE_TRITON_TILE_N
                seq_len = tl.minimum(add_tiles * _PURE_TRITON_TILE_N, total_len - seq_start)
                start_tiles += add_tiles
                tiles_left -= add_tiles
                bucket -= add_tiles
                chunks += 1

                task_base = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
                tl.store(TASK_MAP + task_base + 0, cur_hkv)
                tl.store(TASK_MAP + task_base + 1, cur_batch)
                tl.store(TASK_MAP + task_base + 2, chunk)
                tl.store(TASK_MAP + task_base + 3, seq_start)
                tl.store(TASK_MAP + task_base + 4, seq_len)
                tl.store(TASK_MAP + task_base + 5, seq_len)
                tl.store(TASK_MAP + task_base + 6, (seq_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N)
                tl.store(TASK_MAP + task_base + 7, seq_len // _PURE_TRITON_TILE_N)
                tl.store(TASK_MAP + task_base + 8, tiles_left <= 0)
                tl.store(TASK_MAP + task_base + 9, 0)
                tl.store(TASK_MAP + task_base + 10, 0)
                tl.store(TASK_MAP + task_base + 11, 0)
                slot += 1

                if tiles_left <= 0:
                    tl.store(TASK_MAP + num_chunks_base + cur_hkv * B + cur_batch, chunks)
                    batch = cur_batch + 1
                    if batch >= B:
                        batch = 0
                        hkv = cur_hkv + 1
                    chunks = 0
                    start_tiles = 0
                    if hkv < H_KV:
                        total_len = tl.load(SEQLENS_KV + batch)
                        tiles_left = (total_len + _PURE_TRITON_TILE_N - 1) // _PURE_TRITON_TILE_N

        terminator = (MAX_TASKS * cta + slot + 1) * _PURE_TRITON_TASK_STRIDE
        tl.store(TASK_MAP + terminator + 0, -1)
        tl.store(TASK_MAP + terminator + 1, -1)
        cta += 1


def pure_triton_task_map_metadata(
    kv_lens: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
) -> tuple[int, int, int]:
    """Return fixed-capacity metadata for the portable scheduler."""
    lengths = kv_lens.detach().cpu().to(torch.int32).tolist()
    tiles = [(length + PURE_TRITON_TILE_N - 1) // PURE_TRITON_TILE_N for length in lengths]
    total_tiles = sum(tiles) * num_head_kv
    tiles_per_cta = max(
        (total_tiles + num_ctas - 1) // num_ctas,
        max(1, min_process_len // PURE_TRITON_TILE_N),
    )
    # Each scheduled sequence is split into chunks of at most
    # ``tiles_per_cta`` tiles.  The producer kernel statically iterates over
    # MAX_TASKS slots, so using tiles_per_cta itself can leave dozens of
    # empty slots for short/ragged batches.  Bound the real task count from
    # the exact sequence split count and keep one extra terminator slot.
    total_tasks = sum(
        (tile_count + tiles_per_cta - 1) // tiles_per_cta
        for tile_count in tiles
    ) * num_head_kv
    max_tasks = min(tiles_per_cta, total_tasks) + 1
    chunk_ints = len(lengths) * num_head_kv
    chunk_pad = (chunk_ints + PURE_TRITON_TASK_STRIDE - 1) // PURE_TRITON_TASK_STRIDE * PURE_TRITON_TASK_STRIDE
    sched_ints = (max_tasks * num_ctas + 1) * PURE_TRITON_TASK_STRIDE + chunk_pad
    return max_tasks, sched_ints, sched_ints


def launch_pure_triton_task_map(
    kv_lens: torch.Tensor,
    task_map: torch.Tensor,
    *,
    num_head_kv: int,
    num_ctas: int,
    min_process_len: int,
    max_tasks: int,
    sched_ints: int,
) -> None:
    assign_pure_triton_task_map_kernel[(1,)](
        kv_lens,
        task_map,
        B=kv_lens.numel(),
        H_KV=num_head_kv,
        NUM_CTAS=num_ctas,
        MIN_PROCESS_LEN=min_process_len,
        MAX_TASKS=max_tasks,
        SCHED_INTS=sched_ints,
        num_warps=1,
        num_stages=1,
    )


__all__ = [
    "PURE_TRITON_TASK_STRIDE",
    "PURE_TRITON_TILE_N",
    "assign_pure_triton_task_map_kernel",
    "pure_triton_task_map_metadata",
    "launch_pure_triton_task_map",
]

DEFAULT_BLOCKS_PER_CHUNK = 8
DEFAULT_HEAD_TILE = 8
DEFAULT_MAIN_WARPS = 4
DEFAULT_MAIN_STAGES = 3
DEFAULT_COMBINE_WARPS = 4
DEFAULT_COMBINE_STAGES = 1
DEFAULT_CTA_WAVES = 2


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("ceil_div expects value >= 0 and divisor > 0")
    return (value + divisor - 1) // divisor


def floor_power_of_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (value.bit_length() - 1)


def derive_num_warps(head_dim: int, *, lane_width: int = 32) -> int:
    """Derive the main-kernel warp count from the vector width.

    This is a bounded resource rule, rather than a table keyed by sequence
    length.  The bounds match the Triton/Hygon backend's supported launch
    range; the actual value scales with the amount of work per head.
    """
    if head_dim <= 0 or lane_width <= 0:
        raise ValueError("head_dim and lane_width must be positive")
    return min(8, max(2, floor_power_of_two(ceil_div(head_dim, lane_width))))


def derive_num_stages(
    blocks_per_chunk: int,
    *,
    blocks_per_stage: int = 4,
) -> int:
    """Derive pipeline depth from reduction chunk size.

    A stage covers a bounded number of cache blocks.  This keeps the launch
    policy stable when page size or chunk size changes without introducing
    sequence-length-specific constants.
    """
    if blocks_per_chunk <= 0 or blocks_per_stage <= 0:
        raise ValueError("chunk and stage block counts must be positive")
    return min(4, max(2, 1 + ceil_div(blocks_per_chunk, blocks_per_stage)))


def derive_cta_waves(
    quant_type: str,
    *,
    ragged: bool,
    uniform_chunks: int,
    blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK,
) -> int:
    """Select occupancy target from reduction pressure, not exact shapes.

    BF16 and FP8 have different register/LDS pressure on gfx936.  The
    distinction is an algorithm/resource property; raggedness determines
    whether the FP8 path should use the lower occupancy target.
    """
    if quant_type == "bf16":
        return DEFAULT_CTA_WAVES
    if quant_type not in {"qkpertoken_perhead_vperhead", "qpertoken_perhead_kvpertensor"}:
        raise ValueError(f"unsupported quant_type: {quant_type}")
    # Uniform reductions have enough regular work to hide latency with the
    # larger occupancy target; ragged task maps are intentionally bounded to
    # the lower target to avoid register/LDS pressure.  This is independent of
    # any exact sequence length.
    return DEFAULT_CTA_WAVES if ragged else 2 * DEFAULT_CTA_WAVES


@dataclass(frozen=True)
class HardwareLaunchPolicy:
    """Hardware-resource defaults kept separate from geometry policy."""

    cta_waves: int
    num_warps: int
    num_stages: int
    combine_warps: int
    combine_stages: int


def derive_hardware_policy(
    *,
    quant_type: str,
    head_dim: int,
    blocks_per_chunk: int,
    ragged: bool,
    uniform_chunks: int,
) -> HardwareLaunchPolicy:
    """Derive launch-resource defaults from workload/resource geometry."""
    warps = derive_num_warps(head_dim)
    return HardwareLaunchPolicy(
        cta_waves=derive_cta_waves(
            quant_type,
            ragged=ragged,
            uniform_chunks=uniform_chunks,
            blocks_per_chunk=blocks_per_chunk,
        ),
        num_warps=warps,
        num_stages=derive_num_stages(blocks_per_chunk),
        combine_warps=warps,
        combine_stages=DEFAULT_COMBINE_STAGES,
    )


def derive_group_m(heads_per_group: int, *, head_tile: int = DEFAULT_HEAD_TILE) -> int:
    """Use the largest power-of-two head tile that fits the GQA group."""
    if heads_per_group <= 0 or head_tile <= 0:
        raise ValueError("head counts and head_tile must be positive")
    return min(
        head_tile,
        heads_per_group,
        floor_power_of_two(heads_per_group),
    )


def derive_uniform_group_m(
    heads_per_group: int,
    uniform_chunks: int,
    *,
    blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK,
) -> int:
    """Balance head parallelism against short or deep reductions."""
    full = derive_group_m(heads_per_group)
    half = max(1, full // 2)
    short_window = max(1, blocks_per_chunk // 4)
    deep_window = blocks_per_chunk * 2
    if uniform_chunks <= short_window or uniform_chunks >= deep_window:
        return half
    return full


def derive_chunks_per_program(
    uniform_chunks: int,
    *,
    blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK,
) -> int:
    """Choose a bounded chunk grouping from chunk geometry."""
    if uniform_chunks <= 0 or blocks_per_chunk <= 0:
        raise ValueError("chunk counts must be positive")
    target = max(1, blocks_per_chunk // 2)
    return target if uniform_chunks == blocks_per_chunk else 1


def derive_chunk_tokens(block_size: int, *, blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK) -> int:
    """Derive the reduction chunk from the paged-cache block geometry."""
    if block_size <= 0 or blocks_per_chunk <= 0:
        raise ValueError("block_size and blocks_per_chunk must be positive")
    return block_size * blocks_per_chunk


def derive_block_n(
    block_size: int,
    *,
    ragged: bool,
    uniform_chunks: int,
    blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK,
) -> int:
    """Use a full page tile for short uniform work and half-page for long/ragged work."""
    if block_size <= 0 or blocks_per_chunk <= 0:
        raise ValueError("block_size and blocks_per_chunk must be positive")
    long_reduction = ragged or uniform_chunks >= blocks_per_chunk
    return block_size // (2 if long_reduction else 1)


def derive_uniform_chunks(
    uniform_length: int | None,
    *,
    chunk_tokens: int,
    num_ctas: int,
) -> int:
    """Calculate usable uniform chunks, bounded by available CTAs."""
    if uniform_length is None or uniform_length <= 0:
        return 0
    chunks = ceil_div(uniform_length, chunk_tokens)
    return chunks if chunks <= num_ctas else 0


@dataclass(frozen=True)
class DecodeGeometryPolicy:
    block_size: int
    chunk_tokens: int
    min_process_len: int
    group_m: int
    block_n: int
    uniform_chunks: int


def derive_policy(
    *,
    block_size: int,
    heads_per_group: int,
    uniform_length: int | None,
    ragged: bool,
    num_ctas: int,
    blocks_per_chunk: int = DEFAULT_BLOCKS_PER_CHUNK,
) -> DecodeGeometryPolicy:
    chunk_tokens = derive_chunk_tokens(
        block_size,
        blocks_per_chunk=blocks_per_chunk,
    )
    uniform_chunks = derive_uniform_chunks(
        uniform_length,
        chunk_tokens=chunk_tokens,
        num_ctas=num_ctas,
    )
    return DecodeGeometryPolicy(
        block_size=block_size,
        chunk_tokens=chunk_tokens,
        # The task-map granularity is one reduction chunk, not a shape-specific
        # magic sequence length.
        min_process_len=chunk_tokens,
        group_m=derive_group_m(heads_per_group),
        block_n=derive_block_n(
            block_size,
            ragged=ragged,
            uniform_chunks=uniform_chunks,
            blocks_per_chunk=blocks_per_chunk,
        ),
        uniform_chunks=uniform_chunks,
    )


__all__ = [
    "DEFAULT_BLOCKS_PER_CHUNK",
    "DEFAULT_CTA_WAVES",
    "DEFAULT_COMBINE_STAGES",
    "DEFAULT_COMBINE_WARPS",
    "DEFAULT_HEAD_TILE",
    "DEFAULT_MAIN_STAGES",
    "DEFAULT_MAIN_WARPS",
    "DecodeGeometryPolicy",
    "HardwareLaunchPolicy",
    "ceil_div",
    "derive_block_n",
    "derive_chunk_tokens",
    "derive_cta_waves",
    "derive_group_m",
    "derive_chunks_per_program",
    "derive_hardware_policy",
    "derive_num_stages",
    "derive_num_warps",
    "derive_uniform_group_m",
    "derive_policy",
    "derive_uniform_chunks",
    "floor_power_of_two",
]

