# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained Triton/TLE DiffKV attention implementation.

This module contains both the TLE-optimized and synchronous Triton kernels.
The backend is selected at import time and can be overridden per call.

The optimized path uses asynchronous TLE loads and shape-aware launch
heuristics.  The synchronous Triton implementation remains available as a
fallback when TLE is unavailable or explicitly disabled.

The caller selects either direct 2D execution or split-KV 3D execution.  The
launcher then derives tile, reducer, and resource settings for that path.  The
split-KV reducer combines the partial softmax statistics and outputs.

The implementation supports models where the V head dimension differs from
the Q/K head dimension.  The packed KV cache layout is:

    kv_cache: [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

The host splits this cache into separate K and V pointers.  The 2D path writes
the final output directly; the 3D path writes per-segment partials and uses a
reducer kernel to combine them.
"""

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# TLE is optional. Keep the standard Triton path importable when the extension
# is absent. By default the backend is selected only from the current Triton
# environment: use TLE when it imports successfully and otherwise use the
# standard Triton implementation. ``FLAG_ATTN_DIFFKV_BACKEND`` is retained as
# an explicit override for debugging and controlled benchmarks:
#
#   FLAG_ATTN_DIFFKV_BACKEND=auto    # TLE when available (default)
#   FLAG_ATTN_DIFFKV_BACKEND=tle     # require TLE; fail if unavailable
#   FLAG_ATTN_DIFFKV_BACKEND=triton  # force synchronous standard Triton
try:
    import triton.experimental.tle.language as tle

    HAS_TLE = callable(getattr(tle, "load", None))
    _TLE_IMPORT_ERROR = (
        None
        if HAS_TLE
        else RuntimeError("triton.experimental.tle.language has no load API")
    )
except Exception as exc:  # optional extension may fail on an incompatible Triton
    tle = None
    HAS_TLE = False
    _TLE_IMPORT_ERROR = exc


def _requested_backend() -> str:
    requested = os.environ.get("FLAG_ATTN_DIFFKV_BACKEND", "auto").strip().lower()
    if requested not in {"auto", "tle", "triton"}:
        raise ValueError(
            "FLAG_ATTN_DIFFKV_BACKEND must be one of auto, tle, triton; "
            f"got {requested!r}"
        )
    return requested


REQUESTED_BACKEND = _requested_backend()
if REQUESTED_BACKEND == "tle" and not HAS_TLE:
    raise RuntimeError(
        "FLAG_ATTN_DIFFKV_BACKEND=tle was requested, but the current Triton "
        "environment does not provide triton.experimental.tle.language"
    ) from _TLE_IMPORT_ERROR

USE_TLE = HAS_TLE and REQUESTED_BACKEND != "triton"
SELECTED_BACKEND = "tle" if USE_TLE else "triton"


def tle_import_error() -> Exception | None:
    """Return the import failure that caused the automatic fallback, if any."""
    return _TLE_IMPORT_ERROR


def is_tle_available() -> bool:
    """Whether the optional TLE language extension imported successfully."""
    return HAS_TLE


def get_diffkv_backend_info() -> dict[str, str | bool | None]:
    """Return the backend decision made before the first Triton launch."""
    return {
        "requested": REQUESTED_BACKEND,
        "selected": SELECTED_BACKEND,
        "has_tle": HAS_TLE,
        "tle_error": None if _TLE_IMPORT_ERROR is None else str(_TLE_IMPORT_ERROR),
    }


@lru_cache(maxsize=8)
def _get_num_sms(device_index: int) -> int:
    """Cache the device property used by the TLE launch policy."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _device_num_sms(device: torch.device) -> int:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _get_num_sms(device_index)


# ---------------------------------------------------------------------------
# Launch policy
# ---------------------------------------------------------------------------
# Keep constants grouped by responsibility: kernel defaults, workload
# classification, and occupancy/resource limits.

_SUPPORTED_PATHS = frozenset(("2d", "3d"))
_OPTIMIZED_HEAD_SIZE_QK = 192
_OPTIMIZED_HEAD_SIZE_V = 128


@dataclass(frozen=True)
class _LaunchDefaults:
    warps: int = 4
    stages: int = 3
    short_2d_stages: int = 5
    fused_segments: int = 32
    compact_fused_segments: int = 16
    fused_warps: int = 2
    fused_stages: int = 2
    persistent_segments: int = 16
    persistent_segments_per_program: int = 2


@dataclass(frozen=True)
class _WorkloadPolicy:
    short_k: int = 1024
    medium_k: int = 8192
    medium_split_k: int = 4096
    short_2d_max_k: int = 1536
    # Keep the wide short-2D tile to at most four 128-token iterations.
    short_2d_wide_tile_k: int = 512
    short_2d_light_batch: int = 4


@dataclass(frozen=True)
class _ResourcePolicy:
    long_min_programs_per_sm: int = 4
    long_min_tiles_per_segment: int = 32
    medium_min_programs_per_sm: int = 4
    medium_min_tiles_per_segment: int = 4
    dedup_min_programs_per_sm: int = 1
    two_page_dedup_min_programs_per_sm: int = 1
    # Two-page tiles still benefit from broadcasting page ids while the
    # decode grid has fewer than roughly eight CTAs per SM.  The wider bound
    # also keeps page-table reuse enabled when medium decode doubles its
    # split count to recover TLE occupancy.
    two_page_dedup_max_programs_per_sm: int = 8
    async_min_segments: int = 64
    wide_tile_max_batch: int = 32
    # Short split-KV loops cannot amortize asynchronous pipeline setup.
    # Keep the lightweight one-stage path for up to four tiles; longer loops
    # retain the overlap-oriented stage counts below.
    light_pipeline_max_tiles: int = 4


_LAUNCH = _LaunchDefaults()
_WORKLOAD = _WorkloadPolicy()
_RESOURCE = _ResourcePolicy()

# Autotune is opt-in because these decode kernels are short enough that the
# Autotuner wrapper's dispatch overhead is visible in end-to-end timings.
_DIFFKV_AUTOTUNE = os.environ.get("FLAG_ATTN_DIFFKV_AUTOTUNE", "0") == "1"
_TLE_MAIN_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
    triton.Config({}, num_warps=8, num_stages=4),
    triton.Config({}, num_warps=4, num_stages=5),
]
_TLE_REDUCER_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=1, num_stages=3),
    triton.Config({}, num_warps=2, num_stages=3),
]


def _reset_tle_autotune_state(kwargs, reset_only=False):
    """Reset fused completion state before each optional autotune trial."""
    del reset_only
    if kwargs.get("FUSED_REDUCER", False):
        kwargs["completion_counter_ptr"].zero_()


# ---------------------------------------------------------------------------
# Triton helper functions
# ---------------------------------------------------------------------------
def _tle_workload_class(max_seqlen_k: int | None) -> str | None:
    """Classify decode work by KV length for launch policy selection."""
    if max_seqlen_k is None:
        return None
    if max_seqlen_k <= _WORKLOAD.short_k:
        return "short"
    if max_seqlen_k <= _WORKLOAD.medium_k:
        return "medium"
    return "long"


def _normalize_path(path: str) -> str:
    """Normalize and validate the public 2D/3D launch-path selector."""
    requested = path.strip().lower()
    if requested not in _SUPPORTED_PATHS:
        raise ValueError("path must be 2d or 3d")
    return requested


def _resolve_3d_path(
    path: str,
    *,
    max_seqlen_q: int,
    num_par_softmax_segments: int | None,
    has_softmax_buffers: bool,
) -> bool:
    """Validate an explicit launch path and return whether it is 3D."""
    requested = _normalize_path(path)
    if requested == "2d":
        return False
    if max_seqlen_q > 1:
        raise ValueError("the 3d path currently supports decode workloads only")
    if num_par_softmax_segments is None or not has_softmax_buffers:
        raise ValueError("the 3d path requires preallocated softmax buffers")
    return True


def _has_softmax_buffers(
    segm_output: torch.Tensor | None,
    segm_max: torch.Tensor | None,
    segm_expsum: torch.Tensor | None,
) -> bool:
    """Return whether all workspaces required by the 3D path are present."""
    return all(buffer is not None for buffer in (segm_output, segm_max, segm_expsum))


def _validate_attention_inputs(q: torch.Tensor, causal: bool, sinks) -> None:
    """Validate arguments shared by the TLE and standard Triton launchers."""
    if not causal:
        raise ValueError("only causal attention is supported")
    if sinks is not None and sinks.shape[0] != q.shape[1]:
        raise ValueError("sinks must have one value per query head")


def _has_optimized_head_layout(head_size_qk: int, head_size_v: int) -> bool:
    """Whether the TLE-only launch policies support this head layout."""
    return (
        head_size_qk == _OPTIMIZED_HEAD_SIZE_QK
        and head_size_v == _OPTIMIZED_HEAD_SIZE_V
    )


def _segment_pointers(
    out: torch.Tensor,
    use_3d: bool,
    segm_output: torch.Tensor | None,
    segm_max: torch.Tensor | None,
    segm_expsum: torch.Tensor | None,
):
    """Return valid segment pointers for either kernel layout.

    Triton still requires non-null pointers for tensors that are unused by the
    2D specialization, so the output tensor is used as a harmless placeholder.
    """
    if use_3d:
        return segm_output, segm_max, segm_expsum
    return out, out, out


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    z = S / x
    p = tl.exp(z)
    n = tl.exp(-z)
    return x * (p - n) / (p + n)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left += 1
        else:
            right = mid
    return left - 1


@triton.jit
def resolve_seq_and_query_len(
    query_start_len_ptr,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )
    q_block_start = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local = q_block_global_idx - q_block_start
    cur_start = tl.load(query_start_len_ptr + seq_idx)
    cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
    return (
        seq_idx,
        q_block_local,
        cur_start,
        cur_stop - cur_start,
        tl.load(seq_lens_ptr + seq_idx),
    )


@triton.jit
def init_softmax_M(
    sink_ptr,
    query_offset_1,
    query_mask_1,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    IS_3D: tl.constexpr,
    SEGMENTS_PER_PROGRAM: tl.constexpr = 1,
):
    max_prefix = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_prefix = tl.minimum(max_prefix, seq_len)
    num_tiles = cdiv_fn(max_prefix, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qlo = q_block_local_idx * BLOCK_Q
        qhi = tl.minimum(qlo + (BLOCK_M - 1) // num_queries_per_kv,
                         cur_batch_query_len - 1)
        first_key = context_len + qlo - SLIDING_WINDOW + 1
        last_key = context_len + qhi
        tile_start = tl.maximum(0, first_key // TILE_SIZE)
        tile_end = tl.minimum((last_key // TILE_SIZE) + 1, num_tiles)
    if IS_3D:
        # A persistent fused program may own several adjacent KV segments.
        # Keeping the segments contiguous lets one CTA carry online-softmax
        # state across the whole chunk, reducing both partials and reducer
        # work without requiring a grid-wide barrier.
        segment_base = segm_idx_or_0 * SEGMENTS_PER_PROGRAM
        loop_lo = max(segment_base * tiles_per_segment_or_0, tile_start)
        loop_hi = min(
            (segment_base + SEGMENTS_PER_PROGRAM) * tiles_per_segment_or_0,
            tile_end,
        )
    else:
        loop_lo, loop_hi = tile_start, tile_end
    return loop_lo, loop_hi, max_prefix


@triton.jit
def compute_causal_kv_seq_mask(
    query_abs_pos,
    seq_offset,
    SLIDING_WINDOW: tl.constexpr,
):
    """Build the causal and optional sliding-window mask used by DiffKV."""
    seq_mask = seq_offset[None, :] <= query_abs_pos
    if SLIDING_WINDOW > 0:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
    return seq_mask


@triton.jit
def apply_alibi_to_score(
    S,
    alibi_slope,
    seq_offset,
    context_len,
    query_pos,
    USE_ALIBI_SQRT: tl.constexpr,
):
    if USE_ALIBI_SQRT:
        rel = seq_offset - (context_len + query_pos[:, None])
        bias = tl.where(rel <= 0, -tl.sqrt((-rel).to(tl.float32)), 0.0)
    else:
        bias = seq_offset - context_len
    return S + alibi_slope[:, None] * bias


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx
    )
    mask = query_mask_0 & query_mask_1
    tl.store(segm_max_ptr + offset, M, mask=mask)
    tl.store(segm_expsum_ptr + offset, L, mask=mask)


@triton.jit
def softmax_step(S, M, L):
    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.exp(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.exp(M - m_j)
    return m_j, L * alpha + l_j, P, alpha


def should_use_tle_fused_reducer(
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    block_size: int,
    use_3d: bool,
) -> bool:
    """Return whether the fused reducer supports the requested workload."""
    workload = _tle_workload_class(max_seqlen_k)
    if not (
        use_3d
        and _has_optimized_head_layout(head_size_qk, head_size_v)
        and max_seqlen_q == 1
        and workload is not None
        and block_size == 16
    ):
        return False
    # The fused path uses one completion counter per sequence/head group.  A
    # single sequence can use it across short and medium KV lengths.  For a
    # small batch, keep fusion in the lower medium range where the regular
    # launch would use the same compact split; longer medium contexts retain
    # the parallel reducer to avoid doubling partial-buffer traffic.
    if num_seqs == 1:
        return workload in {"short", "medium"}
    return (
        num_seqs <= 4
        and workload == "medium"
        and max_seqlen_k <= _WORKLOAD.medium_split_k
    )


def should_use_tle_persistent_fused_reducer(
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    use_3d: bool,
) -> bool:
    """Return whether the optional persistent reducer fits the workload."""
    if os.environ.get("FLAG_ATTN_DIFFKV_PERSISTENT_FUSED", "0") != "1":
        return False
    workload = _tle_workload_class(max_seqlen_k)
    return (
        use_3d
        and _has_optimized_head_layout(head_size_qk, head_size_v)
        and max_seqlen_q == 1
        and workload == "medium"
        and num_seqs == 1
        and num_query_heads >= num_kv_heads
        and block_size == 16
    )


def should_split_tle_decode_heads(
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    use_3d: bool,
    num_sms: int | None = None,
) -> bool:
    """Return whether decode should split the GQA group across CTAs.

    For the short 2D path, split heads only when the unsplit query/KV grid
    would occupy less than roughly one quarter of the device.  This keeps
    enough CTAs for small batches without paying the duplicate K/V load cost
    once the batch already provides useful parallelism.
    """
    num_queries_per_kv = num_query_heads // num_kv_heads
    workload = _tle_workload_class(max_seqlen_k)
    if not (
        max_seqlen_q == 1
        and workload is not None
        and _has_optimized_head_layout(head_size_qk, head_size_v)
        and block_size == 16
        and num_queries_per_kv <= 16
    ):
        return False
    if not use_3d:
        if workload != "short" or num_sms is None or num_sms <= 0:
            return False
        min_grid = max(num_sms // 4, num_kv_heads)
        return num_seqs * num_kv_heads <= min_grid
    return (
        workload == "short" and num_seqs <= 8
    ) or (
        workload == "medium" and num_seqs == 1
    )


def get_tle_reduce_num_warps(
    num_query_tokens: int,
    num_query_heads: int,
    num_sms: int,
) -> int:
    """Select the validated direct-launch reducer resource count."""
    return 2 if num_query_tokens * num_query_heads < num_sms else 1


def get_tle_main_num_warps(
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    split_heads: bool,
    fused_reducer: bool,
) -> int:
    """Select the main-kernel warp count from launch geometry.

    Medium 3D grids with at least 32 sequences already expose enough CTAs to
    trade warp-level parallelism for lower register pressure.  Keeping the
    default four-warps launch for shorter batches avoids slowing the less
    populated grids, while long workloads retain their overlap-oriented
    configuration.
    """
    if fused_reducer:
        return _LAUNCH.fused_warps
    if split_heads and num_seqs == 1:
        return 8
    if use_3d and _tle_workload_class(max_seqlen_k) == "medium" and num_seqs >= 32:
        return 2
    return _LAUNCH.warps


def should_use_async_tle_reducer(
    max_seqlen_k: int | None,
    num_query_tokens: int,
    num_query_heads: int,
    num_segments: int,
    num_sms: int,
) -> bool:
    """Prefetch reducer partials only for a sub-wave long-context grid."""
    return (
        _tle_workload_class(max_seqlen_k) == "long"
        and num_segments >= _RESOURCE.async_min_segments
        and num_query_tokens * num_query_heads < num_sms
        and num_query_heads > 64
    )


# Segment selection is a pure function of scalar launch descriptors.  Cache
# repeated decode shapes so the host does not redo the policy on every token.
@lru_cache(maxsize=128)
def get_num_par_softmax_segments(
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    total_num_q_blocks: int | None = None,
    num_kv_heads: int | None = None,
    num_sms: int | None = None,
    block_size: int | None = None,
) -> int:
    """Return a launch-parallelism-aware split-KV segment count.

    Medium decode grids use fewer segments to limit partial-output and reducer
    traffic.  Under-filled small-batch grids may receive one additional split
    when the resulting tiles remain large enough to amortize the extra
    partial-output traffic.  The final split is adjusted only from derived
    grid and tile geometry.
    """
    if not use_3d:
        return 1
    workload = _tle_workload_class(max_seqlen_k)
    if workload == "short":
        segments = 64 if num_seqs <= 1 else 16
    elif workload == "medium":
        segments = 32 if num_seqs <= 1 else 8 if num_seqs <= 8 else 4
    else:
        segments = 128 if num_seqs <= 1 else 32 if num_seqs <= 8 else 16
    tile_size = get_tle_tile_size(use_3d, num_seqs, max_seqlen_k)
    if should_use_tle_fused_reducer(
        _OPTIMIZED_HEAD_SIZE_QK,
        _OPTIMIZED_HEAD_SIZE_V,
        1,
        max_seqlen_k,
        num_seqs,
        block_size or 16,
        use_3d,
    ):
        compact_fused = (
            max_seqlen_k is not None
            and tile_size > (block_size or 16)
            and math.ceil(max_seqlen_k / (_LAUNCH.fused_segments * tile_size)) <= 2
        )
        segments = (
            _LAUNCH.compact_fused_segments
            if compact_fused
            else _LAUNCH.fused_segments
        )

    # Medium decode with a small batch can leave the TLE main kernel with
    # fewer than one full CTA wave per SM.  Add one split only when the
    # expanded split still gives each CTA useful tile work.  This keeps the
    # asynchronous path while addressing the register/occupancy bottleneck
    # seen in under-filled small-batch, medium-KV decode grids.
    if (
        use_3d
        and workload == "medium"
        and 1 < num_seqs <= 8
        and max_seqlen_k is not None
        and total_num_q_blocks is not None
        and num_kv_heads is not None
        and num_sms is not None
        and num_sms > 0
        and segments >= 2
    ):
        current_programs = total_num_q_blocks * num_kv_heads * segments
        expanded_segments = min(segments * 2, _LAUNCH.fused_segments)
        expanded_tiles = math.ceil(
            max_seqlen_k / (expanded_segments * tile_size)
        )
        if (
            current_programs
            < _RESOURCE.medium_min_programs_per_sm * num_sms
            and expanded_tiles >= _RESOURCE.medium_min_tiles_per_segment
        ):
            segments = expanded_segments

    if (
        use_3d
        and max_seqlen_k is not None
        and total_num_q_blocks is not None
        and num_kv_heads is not None
        and num_sms is not None
        and num_sms > 0
        and block_size is not None
        and segments >= 2
    ):
        candidate_segments = segments // 2
        candidate_tiles_per_segment = math.ceil(
            max_seqlen_k / (candidate_segments * tile_size)
        )
        candidate_programs = (
            total_num_q_blocks * num_kv_heads * candidate_segments
        )
        dedup_with_current = should_dedup_block_table(
            use_3d,
            tile_size,
            block_size,
            total_num_q_blocks,
            num_kv_heads,
            segments,
            num_sms,
        )
        dedup_with_candidate = should_dedup_block_table(
            use_3d,
            tile_size,
            block_size,
            total_num_q_blocks,
            num_kv_heads,
            candidate_segments,
            num_sms,
        )
        if (
            candidate_tiles_per_segment >= _RESOURCE.long_min_tiles_per_segment
            and candidate_programs >= _RESOURCE.long_min_programs_per_sm * num_sms
            and dedup_with_candidate == dedup_with_current
        ):
            return candidate_segments
    return segments


def get_tle_num_stages(
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    num_segments: int | None = None,
    tile_size: int | None = None,
) -> int:
    """Select the TLE software-pipeline depth for the shape."""
    workload = _tle_workload_class(max_seqlen_k)
    if (
        use_3d
        and max_seqlen_k is not None
        and num_segments is not None
        and tile_size is not None
        and num_segments > 0
        and tile_size > 0
        and math.ceil(max_seqlen_k / (num_segments * tile_size))
        <= _RESOURCE.light_pipeline_max_tiles
    ):
        # A one-stage pipeline avoids setup/retirement overhead when a CTA
        # only visits a few KV tiles.  Longer loops keep the normal
        # overlap-oriented stage count below.
        return 1
    if (
        not use_3d
        and max_seqlen_k is not None
        and max_seqlen_k <= _WORKLOAD.short_2d_max_k
    ):
        if (
            max_seqlen_k <= _WORKLOAD.short_2d_wide_tile_k
            and num_seqs <= _WORKLOAD.short_2d_light_batch
        ):
            return 2
        return _LAUNCH.short_2d_stages
    if (
        use_3d
        and workload == "medium"
        and num_seqs >= 8
        and max_seqlen_k is not None
        and max_seqlen_k <= _WORKLOAD.medium_split_k
    ):
        return 2
    if use_3d and workload == "medium" and num_seqs >= 8:
        return 4
    if use_3d and workload == "long":
        return 5
    return _LAUNCH.stages


def get_tle_tile_size(
    use_3d: bool,
    num_seqs: int,
    max_seqlen_k: int | None = None,
) -> int:
    """Select a KV tile width from launch parallelism and sequence length.

    Wider decode grids use smaller tiles to preserve independent CTAs.  Short
    2D batches below the wide-tile limit can use a 128-token tile to reduce
    pipeline iterations; longer workloads use 64-token tiles when that saves
    serial KV-loop work.  The thresholds are workload properties rather than
    GPU-specific shape tables.
    """
    workload = _tle_workload_class(max_seqlen_k)
    if not use_3d:
        if (
            workload == "short"
            and max_seqlen_k is not None
            and max_seqlen_k <= _WORKLOAD.short_2d_wide_tile_k
            and num_seqs < _RESOURCE.wide_tile_max_batch
        ):
            return 128
        return 64 if workload == "short" and num_seqs >= 8 else 32
    if (
        num_seqs < _RESOURCE.wide_tile_max_batch
        and max_seqlen_k is not None
        and max_seqlen_k >= _WORKLOAD.medium_k
    ):
        return 64
    if use_3d and workload == "medium" and num_seqs >= 16:
        # Larger decode batches already provide sequence parallelism.  A
        # wider tile reduces the serial KV loop while the segment policy
        # supplies enough CTAs to keep the device occupied.
        return 64
    if num_seqs >= 8:
        return 32
    if workload == "medium":
        return 64
    return 32 if workload == "long" else 16


def should_dedup_block_table(
    use_3d: bool,
    tile_size: int,
    block_size: int,
    total_num_q_blocks: int,
    num_kv_heads: int,
    num_segments: int,
    num_sms: int,
) -> bool:
    """Use page-centric block-table loads for an underfilled 3D grid."""
    total_programs = total_num_q_blocks * num_kv_heads * num_segments
    if tile_size == block_size:
        required_programs_per_sm = _RESOURCE.dedup_min_programs_per_sm
    elif tile_size == 2 * block_size:
        if use_3d:
            programs_per_sm = total_programs / max(num_sms, 1)
            return (
                programs_per_sm >= _RESOURCE.two_page_dedup_min_programs_per_sm
                and programs_per_sm < _RESOURCE.two_page_dedup_max_programs_per_sm
            )
        required_programs_per_sm = _RESOURCE.dedup_min_programs_per_sm
    elif tile_size == 4 * block_size:
        # Load the four physical page ids once and broadcast them to K/V lanes.
        required_programs_per_sm = _RESOURCE.dedup_min_programs_per_sm
    else:
        return False
    return total_programs >= required_programs_per_sm * num_sms


# ---------------------------------------------------------------------------
# TLE and fallback kernels
# ---------------------------------------------------------------------------
@triton.jit
def kernel_unified_attention_diffkv(
    # Output and synchronization pointers.  In 2D mode we write the final
    # result into ``output_ptr``; in 3D mode we write per-segment partials
    # into ``segm_*``.  ``completion_counter_ptr`` is only used by the fused
    # reducer path; callers pass a valid placeholder for other paths.
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    completion_counter_ptr,
    # Query/cache inputs and optional metadata.
    query_ptr,
    key_cache_ptr,  # view of packed cache: [..., :head_size_qk]
    value_cache_ptr,  # view of packed cache: [..., head_size_qk:hqk+hv]
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    softcap,
    # Compile-time dimensions and feature switches.
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    DEDUP_BLOCK_TABLE: tl.constexpr,
    LOOP_NUM_STAGES: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_QK_PADDED: tl.constexpr,
    USE_SPLIT_QK: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Tensor strides and query metadata.
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,  # == HEAD_SIZE_QK
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    # Strides for both cache views (they share the same packed buffer, so
    # dims 0/1/2 strides match; only the per-head extent differs).
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    # Runtime launch dimensions and path selection.
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    SPLIT_HEADS: tl.constexpr,
    FUSED_REDUCER: tl.constexpr,
    PERSISTENT_SEGMENTS_PER_PROGRAM: tl.constexpr,
    # ``IS_3D`` toggles between 2D layout (one program walks the full KV
    # sequence) and 3D layout (split-KV / FlashDecoding-style: per-segm
    # programs write partials, finalized by ``kernel_reduce_segments_diffkv``).
    IS_3D: tl.constexpr,
    # Decode-only batches have exactly one query token per sequence.  Their
    # q-block index is therefore the sequence index, avoiding the generic
    # packed-query block lookup and its conservative extra grid entries.
    IS_DECODE: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    head_program_idx = tl.program_id(1)
    # Each program owns a contiguous BLOCK_M slice of its query-head group.
    split_factor = num_queries_per_kv // BLOCK_M if SPLIT_HEADS else 1
    kv_head_idx = head_program_idx // split_factor
    segm_idx = tl.program_id(2) if IS_3D else 0

    if IS_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = seq_idx
        cur_batch_query_len = 1
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
        (
            seq_idx,
            q_block_local_idx,
            cur_batch_in_all_start_index,
            cur_batch_query_len,
            seq_len,
        ) = resolve_seq_and_query_len(
            query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
        )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(
            seq_len,
            NUM_SEGMENTS_PER_SEQ * PERSISTENT_SEGMENTS_PER_PROGRAM * TILE_SIZE,
        )
        if (
            segm_idx
            * PERSISTENT_SEGMENTS_PER_PROGRAM
            * tiles_per_segment
            * TILE_SIZE
            >= seq_len
        ):
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv
        + (head_program_idx % split_factor) * BLOCK_M
        + offs_m
        if SPLIT_HEADS
        else kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)
    query_mask = query_mask_0[:, None] & query_mask_1[:, None]

    if USE_SPLIT_QK:
        offs_d_qk_lo = tl.arange(0, 128)
        offs_d_qk_hi = 128 + tl.arange(0, 64)
        Q_lo = tl.load(
            query_ptr + query_offset_base + offs_d_qk_lo[None, :],
            mask=query_mask,
            other=0.0,
        )
        Q_hi = tl.load(
            query_ptr + query_offset_base + offs_d_qk_hi[None, :],
            mask=query_mask,
            other=0.0,
        )
    else:
        offs_d_qk = tl.arange(0, HEAD_SIZE_QK_PADDED)
        dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_QK, 1, 0).to(tl.int1)
        Q = tl.load(
            query_ptr + query_offset_base + offs_d_qk[None, :],
            mask=dim_mask_qk[None, :] & query_mask,
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr,
        query_offset_1,
        query_mask_1,
        segm_idx,
        BLOCK_M,
        USE_SINKS,
        IS_3D,
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        IS_3D,
        SEGMENTS_PER_PROGRAM=PERSISTENT_SEGMENTS_PER_PROGRAM,
    )

    for j in tl.range(loop_lo, loop_hi, num_stages=LOOP_NUM_STAGES):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if DEDUP_BLOCK_TABLE and TILE_SIZE == BLOCK_SIZE:
            # Every token in this tile belongs to one logical page.  Load the
            # physical page id once and broadcast it to K/V address lanes.
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            physical_block_idx = physical_block_idx + tl.zeros(
                [TILE_SIZE], dtype=tl.int64
            )
        elif DEDUP_BLOCK_TABLE and TILE_SIZE == 2 * BLOCK_SIZE:
            table_idx = (j * TILE_SIZE) // BLOCK_SIZE
            block_idx_0 = tl.load(
                block_tables_ptr + block_table_offset + table_idx
            ).to(tl.int64)
            block_idx_1 = tl.load(
                block_tables_ptr + block_table_offset + table_idx + 1,
                mask=j * TILE_SIZE + BLOCK_SIZE < max_seq_prefix_len,
                other=0,
            ).to(tl.int64)
            physical_block_idx = tl.where(
                offs_t < BLOCK_SIZE, block_idx_0, block_idx_1
            )
        elif DEDUP_BLOCK_TABLE and TILE_SIZE == 4 * BLOCK_SIZE:
            table_idx = (j * TILE_SIZE) // BLOCK_SIZE
            block_idx_0 = tl.load(
                block_tables_ptr + block_table_offset + table_idx
            ).to(tl.int64)
            block_idx_1 = tl.load(
                block_tables_ptr + block_table_offset + table_idx + 1
            ).to(tl.int64)
            block_idx_2 = tl.load(
                block_tables_ptr + block_table_offset + table_idx + 2
            ).to(tl.int64)
            block_idx_3 = tl.load(
                block_tables_ptr + block_table_offset + table_idx + 3,
                mask=j * TILE_SIZE + 3 * BLOCK_SIZE < max_seq_prefix_len,
                other=0,
            ).to(tl.int64)
            page_idx = offs_t // BLOCK_SIZE
            physical_block_idx = tl.where(
                page_idx == 0,
                block_idx_0,
                tl.where(
                    page_idx == 1,
                    block_idx_1,
                    tl.where(page_idx == 2, block_idx_2, block_idx_3),
                ),
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset_base = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        if USE_SPLIT_QK:
            K_lo = tle.load(
                key_cache_ptr
                + k_offset_base
                + offs_d_qk_lo[:, None] * stride_k_cache_3,
                    mask=tile_mask[None, :],
                    other=0.0,
                    is_async=True,
                ).to(Q_lo.dtype)
            K_hi = tle.load(
                key_cache_ptr
                + k_offset_base
                + offs_d_qk_hi[:, None] * stride_k_cache_3,
                    mask=tile_mask[None, :],
                    other=0.0,
                    is_async=True,
                ).to(Q_hi.dtype)
        else:
            K = tle.load(
                key_cache_ptr
                + k_offset_base
                + offs_d_qk[:, None] * stride_k_cache_3,
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
                is_async=True,
            ).to(Q.dtype)
        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED).  In the asynchronous mode the
        # producer/consumer pipe overlaps K with compute, while V remains
        # synchronous because its pointer tile is not TMA-compatible.
        V_load = tle.load(
            value_cache_ptr + v_offset,
            mask=dim_mask_v[None, :] & tile_mask[:, None],
            other=0.0,
            is_async=False,
        )
        if USE_SPLIT_QK:  # noqa: SIM108
            V = V_load.to(Q_lo.dtype)
        else:
            V = V_load.to(Q.dtype)

        if IS_DECODE:
            # Decode queries are always the final token of their sequence.
            # ``tile_mask`` already clips the loop to ``seq_len`` and the
            # causal sequence mask is therefore redundant.  Sliding-window
            # bounds are applied separately to V below.
            seq_mask = tile_mask[None, :]
        else:
            query_abs_pos = context_len + query_pos[:, None]
            seq_mask = compute_causal_kv_seq_mask(
                query_abs_pos,
                seq_offset,
                SLIDING_WINDOW,
            )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_SPLIT_QK:
            if IS_3D:
                S += tl.dot(Q_lo, K_lo)
                S += tl.dot(Q_hi, K_hi)
                S *= scale
            else:
                S += scale * tl.dot(Q_lo, K_lo)
                S += scale * tl.dot(Q_hi, K_hi)
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask,
            S,
            float("-inf"),
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]
        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue --------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; the reducer accumulates in FP32.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + segm_idx * HEAD_SIZE_V_PADDED
            + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
        )
        store_query_mask = query_mask_0 & query_mask_1
        if FUSED_REDUCER:
            # BLOCK_M repeats each GQA row to form a tensor-core-compatible
            # tile. One representative store per real query head is enough.
            store_query_mask &= offs_m < num_queries_per_kv
        if FUSED_REDUCER:
            # Publish partials before the release atomic.  The write-through
            # stores and CTA barrier ensure the last CTA observes all peers'
            # partial output and scalar softmax values.
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask_v[None, :] & store_query_mask[:, None],
                cache_modifier=".wt",
            )
            fused_scalar_store_offset = (
                query_offset_0.to(tl.int64)
                * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                + query_offset_1 * NUM_SEGMENTS_PER_SEQ
                + segm_idx
            )
            tl.store(
                segm_max_ptr + fused_scalar_store_offset,
                M,
                mask=store_query_mask,
                cache_modifier=".wt",
            )
            tl.store(
                segm_expsum_ptr + fused_scalar_store_offset,
                L,
                mask=store_query_mask,
                cache_modifier=".wt",
            )
            tl.debug_barrier()
        else:
            tl.store(
                segm_output_ptr + segm_output_offset,
                acc,
                mask=dim_mask_v[None, :] & store_query_mask[:, None],
            )
            store_segm_reduce_scalars(
                segm_max_ptr,
                segm_expsum_ptr,
                query_offset_0,
                query_offset_1,
                segm_idx,
                M,
                L,
                store_query_mask,
                store_query_mask,
                num_query_heads,
                NUM_SEGMENTS_PER_SEQ,
            )
        if FUSED_REDUCER:
            if SPLIT_HEADS:
                counter_offset = (
                    seq_idx * (num_query_heads // BLOCK_M) + head_program_idx
                )
                local_head_start = (head_program_idx % split_factor) * BLOCK_M
                local_head_count: tl.constexpr = BLOCK_M
            else:
                counter_offset = (
                    seq_idx * (num_query_heads // num_queries_per_kv)
                    + kv_head_idx
                )
                local_head_start = 0
                local_head_count: tl.constexpr = num_queries_per_kv
            old_count = tl.atomic_add(
                completion_counter_ptr + counter_offset,
                1,
                sem="acq_rel",
                scope="gpu",
            )
            if old_count == NUM_SEGMENTS_PER_SEQ - 1:
                fused_dim_mask = offs_d_v < HEAD_SIZE_V
                for local_head in range(0, local_head_count):
                    query_head_idx = (
                        kv_head_idx * num_queries_per_kv
                        + local_head_start
                        + local_head
                    )
                    segm_ids = tl.arange(0, NUM_SEGMENTS_PER_SEQ)
                    fused_scalar_offset = (
                        seq_idx.to(tl.int64)
                        * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                        + query_head_idx * NUM_SEGMENTS_PER_SEQ
                        + segm_ids
                    )
                    fused_output_offset = (
                        seq_idx.to(tl.int64)
                        * (
                            num_query_heads
                            * NUM_SEGMENTS_PER_SEQ
                            * HEAD_SIZE_V_PADDED
                        )
                        + query_head_idx
                        * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
                        + segm_ids[:, None] * HEAD_SIZE_V_PADDED
                        + offs_d_v[None, :]
                    )
                    fused_output = tl.load(
                        segm_output_ptr + fused_output_offset,
                        mask=fused_dim_mask[None, :],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    fused_max = tl.load(
                        segm_max_ptr + fused_scalar_offset,
                        cache_modifier=".cg",
                    )
                    overall_max = tl.max(fused_max)
                    fused_expsum = tl.load(
                        segm_expsum_ptr + fused_scalar_offset,
                        cache_modifier=".cg",
                    )
                    fused_scale = tl.exp(fused_max - overall_max)
                    fused_expsum *= fused_scale
                    overall_expsum = tl.sum(fused_expsum)
                    fused_output *= fused_scale[:, None]
                    fused_acc = tl.sum(fused_output, axis=0) / overall_expsum
                    final_output_offset = (
                        seq_idx * output_stride_0
                        + query_head_idx * output_stride_1
                        + offs_d_v
                    )
                    tl.store(
                        output_ptr + final_output_offset,
                        fused_acc,
                        mask=fused_dim_mask,
                    )
                tl.atomic_xchg(
                    completion_counter_ptr + counter_offset,
                    0,
                    sem="release",
                    scope="gpu",
                )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d_v[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def kernel_reduce_segments_diffkv(
    output_ptr,  # [num_tokens, num_query_heads, head_size_v]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size_v]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    """Combine per-segment partials into the final softmax output.

    Mirrors ``reduce_segments`` from triton_unified_attention.py but
    indexes V's head size (``HEAD_SIZE_V``) instead of the shared one.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if IS_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
        )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


@triton.jit
def kernel_reduce_segments_diffkv_async(
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    """Reduce split-KV partials with an overlapped TLE streaming read."""
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if IS_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
        )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    # Read reducer partials through the asynchronous TLE path.
    segm_output = tle.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
        is_async=True,
    )

    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)



# ---- Inlined synchronous Triton fallback -------------------------------
@triton.jit
def _fallback_kernel_unified_attention_diffkv(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into
    # ``segm_*`` and ``output_ptr`` is unused (callers may pass any
    # non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    query_ptr,
    key_cache_ptr,  # view of packed cache: [..., :head_size_qk]
    value_cache_ptr,  # view of packed cache: [..., head_size_qk:hqk+hv]
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    scale,
    softcap,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,  # == HEAD_SIZE_QK
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_QK: tl.constexpr,
    HEAD_SIZE_QK_PADDED: tl.constexpr,
    USE_SPLIT_QK: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    # Strides for both cache views (they share the same packed buffer, so
    # dims 0/1/2 strides match; only the per-head extent differs).
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    # ``IS_3D`` toggles between 2D layout (one program walks the full KV
    # sequence) and 3D layout (split-KV / FlashDecoding-style: per-segm
# programs write partials, finalized by the fallback reducer).
    IS_3D: tl.constexpr,
    # Decode has exactly one query token per sequence, so program axis 0 maps
    # directly to the sequence without scanning packed-query boundaries.
    IS_DECODE: tl.constexpr,
    SPLIT_HEADS: tl.constexpr,
    DEDUP_BLOCK_TABLE: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    head_program_idx = tl.program_id(1)
    split_factor = num_queries_per_kv // BLOCK_M if SPLIT_HEADS else 1
    kv_head_idx = head_program_idx // split_factor
    segm_idx = tl.program_id(2) if IS_3D else 0

    if IS_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_in_all_start_index = seq_idx
        cur_batch_query_len = 1
        seq_len = tl.load(seq_lens_ptr + seq_idx)
    else:
        (
            seq_idx,
            q_block_local_idx,
            cur_batch_in_all_start_index,
            cur_batch_query_len,
            seq_len,
        ) = resolve_seq_and_query_len(
            query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
        )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d_qk = tl.arange(0, HEAD_SIZE_QK_PADDED)
    offs_d_v = tl.arange(0, HEAD_SIZE_V_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = (
        kv_head_idx * num_queries_per_kv
        + (head_program_idx % split_factor) * BLOCK_M
        + offs_m
        if SPLIT_HEADS
        else kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    )
    query_offset_base = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
    )

    dim_mask_qk = tl.where(offs_d_qk < HEAD_SIZE_QK, 1, 0).to(tl.int1)
    dim_mask_v = tl.where(offs_d_v < HEAD_SIZE_V, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_QK_PADDED).  The split layout avoids padding
    # overhead for the supported 192-wide query head.
    if USE_SPLIT_QK:
        offs_d_qk_lo = tl.arange(0, 128)
        offs_d_qk_hi = 128 + tl.arange(0, 64)
        Q_lo = tl.load(
            query_ptr + query_offset_base + offs_d_qk_lo[None, :],
            mask=(offs_d_qk_lo < HEAD_SIZE_QK)[None, :]
            & query_mask_0[:, None]
            & query_mask_1[:, None],
            other=0.0,
        )
        Q_hi = tl.load(
            query_ptr + query_offset_base + offs_d_qk_hi[None, :],
            mask=(offs_d_qk_hi < HEAD_SIZE_QK)[None, :]
            & query_mask_0[:, None]
            & query_mask_1[:, None],
            other=0.0,
        )
    else:
        query_offset = query_offset_base + offs_d_qk[None, :]
        Q = tl.load(
            query_ptr + query_offset,
            mask=dim_mask_qk[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
            other=0.0,
        )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_V_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_V_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if DEDUP_BLOCK_TABLE and TILE_SIZE == 2 * BLOCK_SIZE:
            # Broadcast page ids for two-page tiles instead of issuing one
            # scalar page-table load per token lane.
            table_idx = (j * TILE_SIZE) // BLOCK_SIZE
            block_idx_0 = tl.load(
                block_tables_ptr + block_table_offset + table_idx,
                cache_modifier=".ca",
            ).to(tl.int64)
            block_idx_1 = tl.load(
                block_tables_ptr + block_table_offset + table_idx + 1,
                mask=j * TILE_SIZE + BLOCK_SIZE < max_seq_prefix_len,
                other=0,
                cache_modifier=".ca",
            ).to(tl.int64)
            physical_block_idx = tl.where(
                offs_t < BLOCK_SIZE, block_idx_0, block_idx_1
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d_v[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset_base = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        if USE_SPLIT_QK:
            K_lo = tl.load(
                key_cache_ptr
                + k_offset_base
                + offs_d_qk_lo[:, None] * stride_k_cache_3,
                mask=tile_mask[None, :],
                other=0.0,
            ).to(Q_lo.dtype)
            K_hi = tl.load(
                key_cache_ptr
                + k_offset_base
                + offs_d_qk_hi[:, None] * stride_k_cache_3,
                mask=tile_mask[None, :],
                other=0.0,
            ).to(Q_hi.dtype)
        else:
            k_offset = k_offset_base + offs_d_qk[:, None] * stride_k_cache_3
            # K : (HEAD_SIZE_QK_PADDED, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
            )
            K = K_load.to(Q.dtype)
        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask_v[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = V_load.to(Q_lo.dtype if USE_SPLIT_QK else Q.dtype)

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_causal_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            SLIDING_WINDOW,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_SPLIT_QK:
            S += scale * tl.dot(Q_lo, K_lo)
            S += scale * tl.dot(Q_hi, K_hi)
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue --------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by reduce_segments_diffkv.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
            + segm_idx * HEAD_SIZE_V_PADDED
            + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d_v[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def _fallback_kernel_reduce_segments_diffkv(
    output_ptr,  # [num_tokens, num_query_heads, head_size_v]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size_v]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,
    num_query_heads: tl.constexpr,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,  # == HEAD_SIZE_V
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_V: tl.constexpr,
    HEAD_SIZE_V_PADDED: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_DECODE: tl.constexpr,
):
    """Combine per-segment partials into the final softmax output.

    Mirrors ``reduce_segments`` from triton_unified_attention.py but
    indexes V's head size (``HEAD_SIZE_V``) instead of the shared one.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if IS_DECODE:
        seq_idx = query_token_idx
    else:
        seq_idx = find_seq_idx(
            query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
        )
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_V_PADDED) < HEAD_SIZE_V, 1, 0).to(
        tl.int1
    )

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_V_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


if _DIFFKV_AUTOTUNE:
    _kernel_diffkv_autotuned = triton.autotune(
        configs=_TLE_MAIN_AUTOTUNE_CONFIGS,
        key=[
            "HEAD_SIZE_QK",
            "HEAD_SIZE_V",
            "BLOCK_M",
            "NUM_SEGMENTS_PER_SEQ",
            "IS_3D",
            "SPLIT_HEADS",
            "FUSED_REDUCER",
        ],
        pre_hook=_reset_tle_autotune_state,
    )(kernel_unified_attention_diffkv)
    _kernel_reduce_autotuned = triton.autotune(
        configs=_TLE_REDUCER_AUTOTUNE_CONFIGS,
        key=["HEAD_SIZE_V", "NUM_SEGMENTS_PER_SEQ", "IS_DECODE"],
    )(kernel_reduce_segments_diffkv)
    _kernel_reduce_async_autotuned = triton.autotune(
        configs=_TLE_REDUCER_AUTOTUNE_CONFIGS,
        key=["HEAD_SIZE_V", "NUM_SEGMENTS_PER_SEQ", "IS_DECODE"],
    )(kernel_reduce_segments_diffkv_async)
else:
    _kernel_diffkv_autotuned = kernel_unified_attention_diffkv
    _kernel_reduce_autotuned = kernel_reduce_segments_diffkv
    _kernel_reduce_async_autotuned = kernel_reduce_segments_diffkv_async


def should_use_split_qk_diffkv(
    head_size_qk: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
) -> bool:
    workload = _tle_workload_class(max_seqlen_k)
    return (
        head_size_qk == _OPTIMIZED_HEAD_SIZE_QK
        and max_seqlen_q == 1
        and workload is not None
        and (not use_3d or num_seqs <= 8 or num_seqs >= 16)
    )


@dataclass(frozen=True)
class _DiffKVLaunchConfig:
    """Fully derived launch values for one DiffKV invocation.

    The dispatch code is intentionally split into three layers: module
    constants describe the supported policy, this selector derives values
    from the current workload, and the backend launcher passes the result to
    Triton.  Keeping the derived values together prevents individual kernel
    arguments from growing their own shape-specific policy.
    """

    workload: str | None
    use_3d: bool
    block_m: int
    block_q: int
    tile_size: int
    num_segments: int
    launch_num_q_blocks: int
    use_decode_fastpath: bool
    split_heads: bool
    use_split_qk: bool
    dedup_block_table: bool
    fuse_reducer: bool
    persistent_reducer: bool
    loop_num_stages: int


# All inputs are scalar launch descriptors, so reuse the immutable result for
# repeated decode calls instead of recomputing the policy on every invocation.
@lru_cache(maxsize=128)
def _select_tle_launch_config(
    *,
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    num_query_tokens: int,
    is_decode: bool,
    path: str,
    num_par_softmax_segments: int | None,
    has_softmax_buffers: bool,
    num_sms: int,
    fused_reducer_available: bool,
) -> _DiffKVLaunchConfig:
    """Derive geometry and resource choices from one workload description."""
    workload = _tle_workload_class(max_seqlen_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_m = (
        16
        if num_queries_per_kv <= 16
        else triton.next_power_of_2(num_queries_per_kv)
    )
    block_q = block_m // num_queries_per_kv

    use_3d = _resolve_3d_path(
        path,
        max_seqlen_q=max_seqlen_q,
        num_par_softmax_segments=num_par_softmax_segments,
        has_softmax_buffers=has_softmax_buffers,
    )

    fused_reducer = fused_reducer_available and should_use_tle_fused_reducer(
        head_size_qk,
        head_size_v,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        block_size,
        use_3d,
    )
    persistent_reducer = fused_reducer and should_use_tle_persistent_fused_reducer(
        head_size_qk,
        head_size_v,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        num_query_heads,
        num_kv_heads,
        block_size,
        use_3d,
    )

    split_heads = is_decode and should_split_tle_decode_heads(
        head_size_qk,
        head_size_v,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        num_query_heads,
        num_kv_heads,
        block_size,
        use_3d,
        num_sms=num_sms,
    )
    if split_heads:
        if not use_3d and num_seqs >= 16:
            block_m = 8
        elif not use_3d or (workload == "short" and num_seqs == 1):
            block_m = 4
        elif (
            workload == "medium"
            and num_seqs == 1
            and max_seqlen_k is not None
            and max_seqlen_k > _WORKLOAD.medium_split_k
        ):
            block_m = 4
        else:
            block_m = 2
        block_q = 1
    elif fused_reducer and workload == "medium":
        block_m = num_queries_per_kv
        block_q = 1

    total_num_q_blocks = num_query_tokens // block_q + num_seqs
    use_decode_fastpath = is_decode and total_num_q_blocks > num_seqs
    launch_num_q_blocks = num_seqs if use_decode_fastpath else total_num_q_blocks
    tile_size = get_tle_tile_size(
        use_3d, num_seqs, max_seqlen_k
    )

    if not use_3d:
        num_segments = 1
    elif persistent_reducer:
        num_segments = _LAUNCH.persistent_segments
    else:
        if num_par_softmax_segments is None:
            raise ValueError("3D DiffKV launch requires segment storage")
        num_segments = num_par_softmax_segments

    use_split_qk = should_use_split_qk_diffkv(
        head_size_qk,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        use_3d,
    )
    dedup_block_table = should_dedup_block_table(
        use_3d,
        tile_size,
        block_size,
        total_num_q_blocks,
        num_kv_heads,
        num_segments,
        num_sms,
    )
    loop_num_stages = (
        _LAUNCH.fused_stages
        if fused_reducer
        else get_tle_num_stages(
            max_seqlen_k,
            num_seqs,
            use_3d,
            num_segments=num_segments,
            tile_size=tile_size,
        )
    )
    return _DiffKVLaunchConfig(
        workload=workload,
        use_3d=use_3d,
        block_m=block_m,
        block_q=block_q,
        tile_size=tile_size,
        num_segments=num_segments,
        launch_num_q_blocks=launch_num_q_blocks,
        use_decode_fastpath=use_decode_fastpath,
        split_heads=split_heads,
        use_split_qk=use_split_qk,
        dedup_block_table=dedup_block_table,
        fuse_reducer=fused_reducer,
        persistent_reducer=persistent_reducer,
        loop_num_stages=loop_num_stages,
    )


# ---------------------------------------------------------------------------
# Backend launchers
# ---------------------------------------------------------------------------
def _unified_attention_diffkv_fallback(
    q,  # [num_tokens, num_query_heads, head_size_qk]
    k,  # view: [num_blocks, block_size, num_kv_heads, head_size_qk]
    v,  # view: [num_blocks, block_size, num_kv_heads, head_size_v]
    out,  # [num_tokens, num_query_heads, head_size_v]
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    max_seqlen_q: int = 1,
    alibi_slopes=None,
    sinks=None,
    use_alibi_sqrt=False,
    # 3D / split-KV softmax buffers.  They are required when ``path="3d"``.
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    path: str = "2d",
):
    _validate_attention_inputs(q, causal, sinks)

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]
    head_size_qk_padded = triton.next_power_of_2(head_size_qk)
    head_size_v_padded = triton.next_power_of_2(head_size_v)
    workload = _tle_workload_class(max_seqlen_k)

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    is_decode = max_seqlen_q == 1 and q.shape[0] == num_seqs
    launch_num_q_blocks = num_seqs if is_decode else total_num_q_blocks

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    use_3d = _resolve_3d_path(
        path,
        max_seqlen_q=max_seqlen_q,
        num_par_softmax_segments=num_par_softmax_segments,
        has_softmax_buffers=_has_softmax_buffers(
            softmax_segm_output, softmax_segm_max, softmax_segm_expsum
        ),
    )

    # Decode uses smaller tiles to expose KV parallelism; medium workloads
    # can afford a wider tile when the grid already has enough CTAs.
    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
    if use_3d and workload == "medium" and num_seqs >= 8:
        tile_size = 32
    fallback_split_qk = (
        use_3d
        and num_seqs <= 8
        and workload == "medium"
        and _has_optimized_head_layout(head_size_qk, head_size_v)
    )
    fallback_num_segments = num_par_softmax_segments
    fallback_split_heads = False
    fallback_dedup_block_table = (
        fallback_split_qk and tile_size == 2 * block_size
    )

    segm_output_ptr, segm_max_ptr, segm_expsum_ptr = _segment_pointers(
        out,
        use_3d,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
    )
    grid: tuple[Any, ...]
    if use_3d:
        grid = (
            launch_num_q_blocks,
            num_kv_heads,
            fallback_num_segments,
        )
        num_segments = fallback_num_segments
    else:
        grid = (launch_num_q_blocks, num_kv_heads)
        num_segments = 1

    _fallback_kernel_unified_attention_diffkv[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=head_size_qk_padded,
        USE_SPLIT_QK=fallback_split_qk,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=head_size_v_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        IS_3D=use_3d,
        IS_DECODE=is_decode,
        SPLIT_HEADS=False,
        DEDUP_BLOCK_TABLE=fallback_dedup_block_table,
    )

    if use_3d:
        # Use fewer warps for the compact reduction tile.
        reduce_num_warps = (
            2 if workload == "medium" and num_seqs >= 8
            else 1 if workload == "short" and num_seqs >= 8
            else 4
        )
        _fallback_kernel_reduce_segments_diffkv[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE_V=head_size_v,
            HEAD_SIZE_V_PADDED=head_size_v_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            IS_DECODE=is_decode,
            num_warps=reduce_num_warps,
        )


def _unified_attention_diffkv_tle(
    q,  # [num_tokens, num_query_heads, head_size_qk]
    k,  # view: [num_blocks, block_size, num_kv_heads, head_size_qk]
    v,  # view: [num_blocks, block_size, num_kv_heads, head_size_v]
    out,  # [num_tokens, num_query_heads, head_size_v]
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    max_seqlen_q: int = 1,
    alibi_slopes=None,
    sinks=None,
    use_alibi_sqrt=False,
    # 3D / split-KV softmax buffers.  They are required when ``path="3d"``.
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    fused_reducer_counter: torch.Tensor | None = None,
    path: str = "2d",
):
    _validate_attention_inputs(q, causal, sinks)

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]
    head_size_qk_padded = triton.next_power_of_2(head_size_qk)
    head_size_v_padded = triton.next_power_of_2(head_size_v)
    is_decode = max_seqlen_q == 1 and q.shape[0] == num_seqs
    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0
    num_sms = _device_num_sms(q.device)
    launch = _select_tle_launch_config(
        head_size_qk=head_size_qk,
        head_size_v=head_size_v,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        block_size=block_size,
        num_query_tokens=q.shape[0],
        is_decode=is_decode,
        path=path,
        num_par_softmax_segments=num_par_softmax_segments,
        has_softmax_buffers=_has_softmax_buffers(
            softmax_segm_output, softmax_segm_max, softmax_segm_expsum
        ),
        num_sms=num_sms,
        fused_reducer_available=fused_reducer_counter is not None,
    )
    workload = launch.workload
    use_3d = launch.use_3d
    BLOCK_M = launch.block_m
    BLOCK_Q = launch.block_q
    tile_size = launch.tile_size
    num_segments = launch.num_segments
    launch_num_q_blocks = launch.launch_num_q_blocks
    use_decode_fastpath = launch.use_decode_fastpath
    split_heads = launch.split_heads
    use_split_qk = launch.use_split_qk
    dedup_block_table = launch.dedup_block_table
    fuse_reducer = launch.fuse_reducer
    persistent_fused_requested = launch.persistent_reducer
    main_kernel_kwargs = {}
    if not _DIFFKV_AUTOTUNE:
        main_kernel_kwargs = {
            "num_warps": get_tle_main_num_warps(
                max_seqlen_k,
                num_seqs,
                use_3d,
                split_heads,
                fuse_reducer,
            ),
            "num_stages": launch.loop_num_stages,
        }
    segm_output_ptr, segm_max_ptr, segm_expsum_ptr = _segment_pointers(
        out,
        use_3d,
        softmax_segm_output,
        softmax_segm_max,
        softmax_segm_expsum,
    )
    if use_3d:
        grid = (
            launch_num_q_blocks,
            (
                num_kv_heads * (num_queries_per_kv // BLOCK_M)
                if split_heads
                else num_kv_heads
            ),
            num_par_softmax_segments,
        )
    else:
        grid = (
            (
                launch_num_q_blocks,
                num_kv_heads * (num_queries_per_kv // BLOCK_M),
            )
            if split_heads
            else (launch_num_q_blocks, num_kv_heads)
        )
    _kernel_diffkv_autotuned[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        completion_counter_ptr=(fused_reducer_counter if fuse_reducer else out),
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        scale=softmax_scale,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        DEDUP_BLOCK_TABLE=dedup_block_table,
        LOOP_NUM_STAGES=launch.loop_num_stages,
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=head_size_qk_padded,
        USE_SPLIT_QK=use_split_qk,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=head_size_v_padded,
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        SLIDING_WINDOW=sliding_window_val,
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        IS_3D=use_3d,
        IS_DECODE=use_decode_fastpath,
        SPLIT_HEADS=split_heads,
        FUSED_REDUCER=fuse_reducer,
        PERSISTENT_SEGMENTS_PER_PROGRAM=(
            _LAUNCH.persistent_segments_per_program
            if persistent_fused_requested
            else 1
        ),
        **main_kernel_kwargs,
    )

    if use_3d and not fuse_reducer:
        use_async_reducer = should_use_async_tle_reducer(
            max_seqlen_k,
            q.shape[0],
            num_query_heads,
            num_segments,
            num_sms,
        )
        reduce_kernel = (
            _kernel_reduce_async_autotuned
            if use_async_reducer
            else _kernel_reduce_autotuned
        )
        reducer_kwargs = {}
        if not _DIFFKV_AUTOTUNE:
            reducer_kwargs["num_warps"] = get_tle_reduce_num_warps(
                q.shape[0], num_query_heads, num_sms
            )
        reduce_kernel[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            TILE_SIZE=tile_size,
            HEAD_SIZE_V=head_size_v,
            HEAD_SIZE_V_PADDED=head_size_v_padded,
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_segments,
            IS_DECODE=use_decode_fastpath,
            **reducer_kwargs,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _resolve_backend(backend: str | None) -> str:
    """Resolve a per-call backend override."""
    selected = SELECTED_BACKEND if backend is None else backend.strip().lower()
    if selected == "auto":
        selected = SELECTED_BACKEND
    if selected not in {"tle", "triton"}:
        raise ValueError(
            "backend must be one of auto, tle, or triton; "
            f"got {backend!r}"
        )
    if selected == "tle" and not HAS_TLE:
        raise RuntimeError(
            "TLE backend requested but triton.experimental.tle.language is "
            f"unavailable: {tle_import_error()}"
        )
    return selected


def unified_attention_diffkv(
    *args: Any,
    backend: str | None = None,
    path: str = "2d",
    **kwargs: Any,
):
    """Run DiffKV through the selected inlined TLE or standard Triton path."""
    selected = _resolve_backend(backend)
    # Accept the legacy vLLM-style threshold argument while keeping path
    # selection explicit and centralized in the launcher.
    kwargs.pop("seq_threshold_3D", None)
    if selected == "tle":
        return _unified_attention_diffkv_tle(*args, path=path, **kwargs)
    # The fallback launcher has no fused-reducer argument.
    kwargs.pop("fused_reducer_counter", None)
    return _unified_attention_diffkv_fallback(*args, path=path, **kwargs)


def unified_attention_diffkv_tle(*args: Any, **kwargs: Any):
    """Explicit TLE implementation entry point."""
    return unified_attention_diffkv(*args, backend="tle", **kwargs)


def unified_attention_diffkv_fallback(*args: Any, **kwargs: Any):
    """Explicit standard non-TLE Triton entry point."""
    return unified_attention_diffkv(*args, backend="triton", **kwargs)


@torch.no_grad()
def diffkv_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    attn_scale: float | None = None,
    window_size: int = -1,
    path: str = "2d",
    num_segments: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Run paged DiffKV attention for decode workloads.

    Args:
        query: Query tensor with shape ``[B, Hq, Dqk]``.
        key_cache: Paged key cache with shape ``[NB, BS, Hkv, Dqk]``.
        value_cache: Paged value cache with shape ``[NB, BS, Hkv, Dv]``.
        context_lens: Number of cached tokens for each sequence, shape ``[B]``.
        block_tables: Physical KV block indices, shape ``[B, max_blocks]``.
        attn_scale: Optional softmax scale; defaults to ``Dqk ** -0.5``.
        window_size: Number of recent KV tokens to attend to, or ``-1`` for all.
        path: Launch path: ``"2d"`` or ``"3d"``. Defaults to ``"2d"``.
        num_segments: Optional split-KV segment count for the 3D path.
        backend: Optional backend override: ``"auto"``, ``"tle"``, or
            ``"triton"``.

    Returns:
        Attention output with shape ``[B, Hq, Dv]``.
    """
    selected = _resolve_backend(backend)
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("expected query [B,H,D] and paged key/value [NB,BS,Hkv,D]")
    batch, num_query_heads, head_size_qk = query.shape
    if value_cache.shape[:3] != key_cache.shape[:3]:
        raise ValueError("key/value cache leading dimensions must match")
    num_kv_heads = key_cache.shape[2]
    head_size_v = value_cache.shape[3]
    if num_query_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if context_lens.numel() != batch or block_tables.shape[0] != batch:
        raise ValueError("context_lens and block_tables must have batch dimension B")
    if attn_scale is None:
        attn_scale = head_size_qk**-0.5
    context_lens = context_lens.to(device=query.device, dtype=torch.int32)
    block_tables = block_tables.to(device=query.device, dtype=torch.int32)
    out = torch.empty(
        (batch, num_query_heads, head_size_v), device=query.device, dtype=query.dtype
    )
    cu_seqlens_q = torch.arange(
        batch + 1, device=query.device, dtype=torch.int32
    )
    max_seqlen_k = int(context_lens.max().item())
    workload = _tle_workload_class(max_seqlen_k)
    path = _normalize_path(path)
    use_3d = path == "3d"
    if use_3d:
        if num_segments is None:
            if selected == "tle":
                num_segments = get_num_par_softmax_segments(
                    max_seqlen_k,
                    batch,
                    True,
                    total_num_q_blocks=2 * batch,
                    num_kv_heads=num_kv_heads,
                    num_sms=_device_num_sms(query.device),
                    block_size=key_cache.shape[1],
                )
            else:
                num_segments = 64 if workload == "short" and batch <= 1 else 16
        padded_v = triton.next_power_of_2(head_size_v)
        segm_output = torch.empty(
            (batch, num_query_heads, num_segments, padded_v),
            device=query.device,
            dtype=query.dtype if selected == "tle" else torch.float32,
        )
        segm_max = torch.empty(
            (batch, num_query_heads, num_segments),
            device=query.device,
            dtype=torch.float32,
        )
        segm_expsum = torch.empty_like(segm_max)
        seq_threshold = batch
    else:
        num_segments = None
        segm_output = segm_max = segm_expsum = None
        seq_threshold = None
    fused_reducer_counter = None
    if selected == "tle" and use_3d and should_use_tle_fused_reducer(
        head_size_qk,
        head_size_v,
        1,
        max_seqlen_k,
        batch,
        key_cache.shape[1],
        use_3d,
    ):
        fused_reducer_counter = torch.zeros(
            batch * num_query_heads, device=query.device, dtype=torch.int32
        )
    triton_window = (window_size - 1, 0) if window_size > 0 else (-1, -1)
    unified_attention_diffkv(
        q=query,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=context_lens,
        softmax_scale=float(attn_scale),
        causal=True,
        window_size=triton_window,
        block_table=block_tables,
        softcap=0.0,
        max_seqlen_q=1,
        seq_threshold_3D=seq_threshold,
        num_par_softmax_segments=num_segments,
        softmax_segm_output=segm_output,
        softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_expsum,
        max_seqlen_k=max_seqlen_k,
        fused_reducer_counter=fused_reducer_counter,
        path=path,
    )
    return out


__all__ = [
    "diffkv_attention",
    "unified_attention_diffkv",
    "unified_attention_diffkv_tle",
    "unified_attention_diffkv_fallback",
    "HAS_TLE",
    "USE_TLE",
    "REQUESTED_BACKEND",
    "SELECTED_BACKEND",
    "is_tle_available",
    "tle_import_error",
    "get_diffkv_backend_info",
]
