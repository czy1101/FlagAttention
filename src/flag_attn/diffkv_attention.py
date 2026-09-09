# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained Triton/TLE DiffKV attention implementation.

This module contains both the TLE-optimized and synchronous Triton kernels.
The backend is selected at import time (or overridden per call), so callers
do not need to import a second implementation module.  Any former split
implementation files may remain as compatibility copies, but are not
imported or called by this module.

The optimized path asynchronously prefetches K tiles with FlagTree TLE. For
the target QK head size 192, it uses exact 128+64 MMA slices instead of padding
the reduction to 256. The 3D path applies softmax scaling once after the two
partial scores are accumulated. Decode 2D always uses the exact split; decode
3D enables it at B=8/KV=2048, from KV length 4096 for the other measured
positive batch ranges, and at KV length 32768 for B=16. The B=8/KV=2048 path
keeps eight segments and uses two stages. Its full-attention 64Q/4KV launch
uses one padded 256-wide MMA and keeps the complete GQA group in each CTA;
this is faster than splitting the group into smaller head CTAs at this
medium sequence length. Other medium decode workloads use the
TLE-prefetched padded-QK kernel while lower-parallelism B=9..15 stays on the
production fallback. The 3D launcher uses a shape-specific segment
count for the measured B=32 long-sequence cases, avoiding excessive
split/reduce work. For the validated B=8/16/32 long-sequence 3D kernels, a
shape-aware TLE pipeline overlaps the irregular paged-KV loads; low-batch and
2D kernels keep three stages to avoid the deeper pipeline's resource and tail
overhead. At B=8/KV=8192, sixteen split-KV segments and four stages avoid
launching many four-iteration, five-stage programs. The other validated
long-sequence shapes retain the deeper pipeline. The same B>=8 3D range uses
a 32-token KV tile to halve loop,
page-table, and online-softmax overhead, while narrow B=1 launches retain a
16-token tile except the profiled fused KV=8192 path, which uses 64 tokens to
amortize its in-kernel reducer. On wide 3D launch grids, each
physical KV block index is loaded once and broadcast across its 16-token
slice; the selector is based on programs per GPU SM rather than a fixed batch
or sequence length. Other
shapes are identified by ``should_use_tle_diffkv`` so
callers can route them directly to the production Triton implementation,
avoiding the observed short-sequence regressions. The launch uses the
Hopper-oriented four-warp, three-stage configuration validated by the local
TLE kernels.

For short decode, the wrapper switches a nominal 3D request to the optimized
2D TLE kernel when its one-kernel grid has enough programs per GPU SM.  The
required density increases continuously with KV length, so the decision is
based on launch geometry and device SM count rather than a GPU model or fixed
batch size.  This removes the split-KV reduction launch only where the 2D grid
has enough portable parallelism.

The split-KV reducer also sizes its warp group from launch geometry: grids
smaller than the GPU SM count use two warps per CTA, while wider grids use one
warp so more independent query-head reductions can reside concurrently.

This is a slimmed fork of ``triton_unified_attention.py`` for models like
MiMo-V2.5 where the V tensor's head dimension differs from K's.  The KV cache
is the same packed layout used by ``FlashAttentionDiffKVBackend``:

    kv_cache: [num_blocks, block_size, num_kv_heads, head_size_qk + head_size_v]

We slice ``key_cache = kv_cache[..., :head_size_qk]`` and
``value_cache = kv_cache[..., head_size_qk:]`` on the host, so the kernel
takes two cache pointers but with two distinct head sizes.

Both 2D and 3D launches are supported:
  - 2D: one program per (q-block, kv-head); tile-loop walks the full KV
    sequence; final output written directly.  Used for prefill and large
    decode batches.
  - 3D: one program per (q-block, kv-head, segm); each program covers a
    KV slice and writes per-segment partials (max/expsum/output).  A
    follow-up ``kernel_reduce_segments_diffkv`` combines them.  Selected
    for decode-only batches whose 2D grid would under-fill the GPU.

For the validated B=1 DiffKV decode geometry at KV lengths 512 and 8192,
the optimized 3D path removes that follow-up launch. Split CTAs publish
their partials with a GPU-scope release/acquire atomic; the final completing
CTA reduces its KV head's four GQA rows and resets the counter for the next
stream-ordered invocation. Both fused paths use 32 segments. Two warps and
two stages minimize the resource footprint of the
combined main/reduction kernel.
"""

import math
import os
import pathlib
from typing import Any

import torch
import triton
import triton.language as tl

# TLE is an optional Triton extension.  Keep the standard Triton path
# importable: a normal Triton installation must not fail just because the
# experimental ``triton.experimental.tle`` package is absent.  The explicit
# backend override is useful for A/B measurements in the same source tree:
#
#   FLAG_ATTN_DIFFKV_BACKEND=auto    # TLE when available (default)
#   FLAG_ATTN_DIFFKV_BACKEND=tle     # require TLE; benchmark checks this
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
    # Keep the shorter flag as a compatibility alias for shell launchers.
    if requested == "auto":
        legacy = os.environ.get("FLAG_ATTN_DIFFKV_TLE")
        if legacy is not None and legacy.strip().lower() in {
            "0", "false", "off", "no",
        }:
            requested = "triton"
    if requested not in {"auto", "tle", "triton", "cuda"}:
        raise ValueError(
            "FLAG_ATTN_DIFFKV_BACKEND must be one of auto, tle, triton, cuda; "
            f"got {requested!r}"
        )
    return requested


REQUESTED_BACKEND = _requested_backend()
USE_TLE = HAS_TLE and REQUESTED_BACKEND not in {"triton", "cuda"}
SELECTED_BACKEND = (
    "cuda" if REQUESTED_BACKEND == "cuda" else ("tle" if USE_TLE else "triton")
)


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


if USE_TLE:
    @triton.jit
    def _diffkv_load(
        ptr,
        mask=None,
        other=None,
        is_async: tl.constexpr = False,
    ):
        return tle.load(
            ptr,
            mask=mask,
            other=other,
            is_async=is_async,
        )

else:
    @triton.jit
    def _diffkv_load(
        ptr,
        mask=None,
        other=None,
        is_async: tl.constexpr = False,
    ):
        # ``is_async`` is compile-time and intentionally ignored by the
        # standard Triton fallback.
        return tl.load(ptr, mask=mask, other=other)

# These helpers are kept local so this module can run from a standalone
# FlagAttention checkout without importing vLLM internals.
is_batch_invariant = False
TLE_NUM_WARPS = 4
TLE_NUM_STAGES = 3
TLE_NUM_PAR_SOFTMAX_SEGMENTS = 64
# Wide medium-decode grids already expose substantial batch/head parallelism.
# Four split-KV segments retain enough independent CTAs to cover GPU latency
# while halving partial-output and reducer traffic relative to eight.  Grids
# that are too narrow, or whose page-table dedup depends on eight segments,
# retain the wider split.
TLE_MEDIUM_NUM_PAR_SOFTMAX_SEGMENTS = 4
TLE_MEDIUM_MIN_PROGRAMS_PER_SM = 4
# Once every reduced split still performs substantial serial tile work, a
# narrower split saves partial-buffer and reducer traffic without starving the
# device.  Both thresholds describe launch/work geometry, not a GPU model.
TLE_LONG_MIN_PROGRAMS_PER_SM = 4
TLE_LONG_MIN_TILES_PER_SEGMENT = 32
# A single resident CTA per SM is enough to amortize one page-table load per
# tile.  Two-page 3D tiles need a denser grid because their select/broadcast
# path is only worthwhile once the page-table traffic is amortized across
# enough independent CTAs.
TLE_DEDUP_MIN_PROGRAMS_PER_SM = 1
TLE_TWO_PAGE_DEDUP_MIN_PROGRAMS_PER_SM = 12
# Short decode can avoid the split-KV partial/reduce sequence once its 2D
# batch/head grid is dense enough.  Longer KV ranges need progressively more
# independent programs because each program has more serial tile work.
TLE_SHORT_2D_BASE_PROGRAMS_PER_SM = 1.2
TLE_SHORT_2D_KV_SLOPE = 1.0 / 2048.0
TLE_SHORT_2D_MAX_SEQLEN_K = 1536
# The single-launch short 2D path has enough independent CTAs to tolerate the
# extra pipeline state.  Five stages overlap its descriptor-backed K/V loads
# substantially better than the generic three-stage pipeline, without adding
# split-KV buffers or a reducer launch.
TLE_SHORT_2D_NUM_STAGES = 5
# A narrow long-context reducer cannot hide the large partial-output read
# across enough CTAs.  In that geometry, issue the streaming read through TLE
# before the independent scalar normalization path.  Wider reducer grids
# retain the original synchronous load, which remains slightly faster there.
TLE_ASYNC_REDUCER_MIN_SEQLEN_K = 16384
TLE_ASYNC_REDUCER_MIN_SEGMENTS = 64
# The stock backend switches to 2D once the decode batch reaches 32.  For the
# validated DiffKV geometry below, long contexts still benefit from split-KV:
# the 3D main kernel plus reducer is faster than one long serial 2D KV loop.
TLE_LONG_3D_MIN_SEQLEN_K = 8192
# FlagTree 0.6 makes the split-head, last-CTA reduction profitable for the
# three low-batch MiMo decode lengths below.  Keeping the reduction inside the
# final main-kernel CTA removes the fixed cost of a separate reducer launch.
TLE_FUSED_REDUCER_SEQLENS_K = (512, 2048, 8192)
TLE_FUSED_REDUCER_SEGMENTS = 32
TLE_FUSED_REDUCER_MEDIUM_SEGMENTS = 32
TLE_FUSED_REDUCER_LONG_SEGMENTS = 32
TLE_FUSED_REDUCER_NUM_WARPS = 2
TLE_FUSED_REDUCER_MEDIUM_NUM_WARPS = 2
TLE_FUSED_REDUCER_LONG_NUM_WARPS = 2
TLE_FUSED_REDUCER_NUM_STAGES = 2
# Persistent fused reducer candidate.  Each CTA owns four adjacent physical
# KV segments and carries the online-softmax state across that chunk.  The
# reducer therefore sees eight already-merged partials instead of 32.  This
# is deliberately restricted to the under-filled Full-Attention B=1/KV=8192
# geometry; all other shapes keep the validated 32-segment path.
TLE_PERSISTENT_FUSED_REDUCER_SEGLENS_K = (8192,)
TLE_PERSISTENT_FUSED_REDUCER_SEGMENTS = 16
TLE_PERSISTENT_FUSED_REDUCER_SEGMENTS_PER_PROGRAM = 2
# Dedicated B=8 short-decode route.  The KV=512 geometry is a validated
# production path.  KV=2048 keeps the existing 16-segment split-KV route;
# its fused reducer was measured but did not produce a stable improvement.
TLE_B8_FUSED_REDUCER_SEGLENS_K = (512,)


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
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr,
                 use_q_block_mode: tl.constexpr):
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
def resolve_seq_and_query_len(query_start_len_ptr, seq_lens_ptr, q_block_global_idx,
                              num_seqs, BLOCK_Q: tl.constexpr):
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
def init_softmax_M(sink_ptr, query_offset_1, query_mask_1, segm_idx_or_0,
                   BLOCK_M: tl.constexpr, USE_SINKS: tl.constexpr,
                   IS_3D: tl.constexpr):
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
def compute_tile_loop_bounds(context_len, seq_len, cur_batch_query_len,
                              q_block_local_idx, segm_idx_or_0,
                              tiles_per_segment_or_0, TILE_SIZE: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_Q: tl.constexpr,
                              num_queries_per_kv: tl.constexpr,
                              SLIDING_WINDOW: tl.constexpr,
                              USE_MM_PREFIX: tl.constexpr, IS_3D: tl.constexpr,
                              SEGMENTS_PER_PROGRAM: tl.constexpr = 1,
                              USE_CAUSAL: tl.constexpr = True,
                              USE_PER_SEQ_CAUSAL: tl.constexpr = False,
                              CHUNK_LOOKBACK: tl.constexpr = -1,
                              CHUNK_SIZE: tl.constexpr = -1):
    max_prefix = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_prefix = seq_len if (USE_MM_PREFIX or USE_PER_SEQ_CAUSAL or not USE_CAUSAL) \
        else tl.minimum(max_prefix, seq_len)
    num_tiles = cdiv_fn(max_prefix, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
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
def compute_kv_seq_mask(query_abs_pos, seq_offset, seq_idx, seq_len,
                        mm_prefix_range_ptr, SLIDING_WINDOW: tl.constexpr,
                        USE_MM_PREFIX: tl.constexpr,
                        MAX_MM_RANGES: tl.constexpr,
                        USE_CAUSAL: tl.constexpr = True,
                        USE_PER_SEQ_CAUSAL: tl.constexpr = False,
                        per_seq_causal_ptr=None, rswa_prefix_lens_ptr=None,
                        R_SWA_WINDOW: tl.constexpr = 0,
                        USE_R_SWA: tl.constexpr = False,
                        CHUNK_LOOKBACK: tl.constexpr = -1,
                        CHUNK_SIZE: tl.constexpr = -1,
                        MM_PREFIX_CLAMP_SW: tl.constexpr = False):
    if USE_CAUSAL:
        seq_mask = seq_offset[None, :] <= query_abs_pos
    else:
        seq_mask = seq_offset[None, :] < seq_len
    if SLIDING_WINDOW > 0 and not USE_R_SWA:
        seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
    if USE_R_SWA:
        prefix_len = tl.load(rswa_prefix_lens_ptr + seq_idx)
        seq_mask = seq_mask & (
            (seq_offset[None, :] < prefix_len)
            | ((query_abs_pos - seq_offset) < R_SWA_WINDOW)
        )
    if CHUNK_LOOKBACK > -1:
        seq_mask = seq_mask & (
            (query_abs_pos // CHUNK_SIZE - seq_offset[None, :] // CHUNK_SIZE)
            <= CHUNK_LOOKBACK
        )
    return seq_mask


@triton.jit
def apply_alibi_to_score(S, alibi_slope, seq_offset, context_len, query_pos,
                         USE_ALIBI_SQRT: tl.constexpr):
    if USE_ALIBI_SQRT:
        rel = seq_offset - (context_len + query_pos[:, None])
        bias = tl.where(rel <= 0, -tl.sqrt((-rel).to(tl.float32)), 0.0)
    else:
        bias = seq_offset - context_len
    return S + alibi_slope[:, None] * bias


@triton.jit
def store_segm_reduce_scalars(segm_max_ptr, segm_expsum_ptr, query_offset_0,
                              query_offset_1, segm_idx, M, L, query_mask_0,
                              query_mask_1, num_query_heads: tl.constexpr,
                              NUM_SEGMENTS_PER_SEQ: tl.constexpr):
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
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
    use_3d: bool,
) -> bool:
    """Fuse the reducer for under-filled B=1 medium decode geometries.

    The fused path was initially validated only for the legacy 32Q/8KV
    layout.  MiMo Full Attention uses the same 16-query-per-KV ratio through
    a 64Q/4KV layout, so it is safe to share the fused reducer whenever the
    actual GQA ratio and tile dimensions match one of these two geometries.
    """
    return (
        use_3d
        and head_size_qk == 192
        and head_size_v == 128
        and max_seqlen_q == 1
        and max_seqlen_k in TLE_FUSED_REDUCER_SEQLENS_K
        and (
            (
                num_seqs == 1
                and (num_query_heads, num_kv_heads) in ((32, 8), (64, 4))
            )
            or (
                num_seqs == 8
                and max_seqlen_k in TLE_B8_FUSED_REDUCER_SEGLENS_K
                and (num_query_heads, num_kv_heads) == (64, 4)
            )
        )
        and block_size == 16
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
    """Select the chunked persistent reducer candidate.

    This is a CTA-local persistent work decomposition rather than a CUDA
    cooperative-grid primitive: each program scans four adjacent KV chunks
    and emits one merged partial, so the existing release/acquire last-CTA
    reducer remains safe for arbitrary launch scheduling.
    """
    # Keep an explicit switch for A/B without touching the source.  The
    # candidate is opt-in until it wins a complete matrix regression; default
    # production behavior therefore remains the validated atomic fused path.
    if os.environ.get("FLAG_ATTN_DIFFKV_PERSISTENT_FUSED", "0") != "1":
        return False
    return (
        use_3d
        and head_size_qk == 192
        and head_size_v == 128
        and max_seqlen_q == 1
        and max_seqlen_k in TLE_PERSISTENT_FUSED_REDUCER_SEGLENS_K
        and num_seqs == 1
        and (num_query_heads, num_kv_heads) == (64, 4)
        and block_size == 16
    )


def should_use_tle_split_head_2d(
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
    """Route validated short decode to the split-head single-kernel path.

    These shapes already have a dedicated two-query-head CTA layout below.
    Selecting 2D here is important: the generic pre-split grid-density test
    cannot see that the final layout doubles the head grid to 16/128 CTAs.
    """
    return (
        use_3d
        and head_size_qk == 192
        and head_size_v == 128
        and max_seqlen_q == 1
        and max_seqlen_k == 512
        and num_seqs in (1, 8)
        and num_query_heads == 32
        and num_kv_heads == 8
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
) -> bool:
    """Return the production split-head decision used by launchers.

    Keep this policy in one helper so direct-kernel benchmarks reproduce the
    exact BLOCK_M, head-grid, and warp geometry used by the operator wrapper.
    """
    num_queries_per_kv = num_query_heads // num_kv_heads
    return (
        max_seqlen_q == 1
        and (
            (
                not use_3d
                and num_seqs in (1, 8)
                and max_seqlen_k == 512
                and num_query_heads == 32
                and num_kv_heads == 8
                and num_queries_per_kv == 4
            )
            or (
                not use_3d
                and num_seqs == 8
                and max_seqlen_k == 512
                and num_query_heads == 64
                and num_kv_heads == 4
                and num_queries_per_kv == 16
            )
            or (
                use_3d
                and num_seqs == 8
                and max_seqlen_k == 2048
                and num_query_heads == 64
                and num_kv_heads == 4
                and num_queries_per_kv == 16
            )
            or (
                use_3d
                and num_seqs == 1
                and max_seqlen_k in TLE_FUSED_REDUCER_SEQLENS_K
                and num_query_heads == 64
                and num_kv_heads == 4
                and num_queries_per_kv == 16
            )
            or (
                not use_3d
                and num_seqs == 16
                and max_seqlen_k == 512
                and num_query_heads == 64
                and num_kv_heads == 4
                and num_queries_per_kv == 16
            )
        )
        and head_size_qk == 192
        and head_size_v == 128
        and block_size == 16
    )


def should_use_tle_short_2d(
    head_size_qk: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    use_3d: bool,
    total_num_q_blocks: int,
    num_kv_heads: int,
    num_sms: int,
) -> bool:
    """Replace split-KV with one TLE kernel when its 2D grid is wide enough."""
    if (
        head_size_qk != 192
        or max_seqlen_q != 1
        or max_seqlen_k is None
        or not use_3d
        or max_seqlen_k > TLE_SHORT_2D_MAX_SEQLEN_K
        or num_sms <= 0
    ):
        return False
    required_programs_per_sm = TLE_SHORT_2D_BASE_PROGRAMS_PER_SM + (
        max(0, max_seqlen_k - 512) * TLE_SHORT_2D_KV_SLOPE
    )
    total_programs = total_num_q_blocks * num_kv_heads
    return total_programs >= math.ceil(required_programs_per_sm * num_sms)


def should_use_tle_long_3d(
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    block_size: int,
) -> bool:
    """Select split-KV for the validated wide-batch long-context shape."""
    return (
        head_size_qk == 192
        and head_size_v == 128
        and max_seqlen_q == 1
        and max_seqlen_k is not None
        and max_seqlen_k >= TLE_LONG_3D_MIN_SEQLEN_K
        and num_seqs == 32
        and num_query_heads == 32
        and num_kv_heads == 8
        and block_size == 16
    )


def get_tle_reduce_num_warps(
    num_query_tokens: int,
    num_query_heads: int,
    num_sms: int,
) -> int:
    """Give narrow reducer grids more warps and wide grids one warp per CTA."""
    reduce_programs = num_query_tokens * num_query_heads
    return 2 if reduce_programs < num_sms else 1


def should_use_async_tle_reducer(
    max_seqlen_k: int | None,
    num_query_tokens: int,
    num_query_heads: int,
    num_segments: int,
    num_sms: int,
) -> bool:
    """Prefetch reducer partials only for a sub-wave long-context grid."""
    # Triton 3.6 + the CUDA 12.8 ptxas shipped in tle-y rejects the cache
    # modifier combination emitted by the async reducer for this very narrow
    # B=1/64-head/32K launch (``evict_first`` with ``.cg``).  The synchronous
    # reducer is valid and avoids making the whole shape fail code generation.
    if (
        max_seqlen_k == 32768
        and num_query_tokens == 1
        and num_query_heads == 64
    ):
        return False
    return (
        max_seqlen_k is not None
        and max_seqlen_k >= TLE_ASYNC_REDUCER_MIN_SEQLEN_K
        and num_segments >= TLE_ASYNC_REDUCER_MIN_SEGMENTS
        and num_query_tokens * num_query_heads < num_sms
    )


def get_tle_num_par_softmax_segments(
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    total_num_q_blocks: int | None = None,
    num_kv_heads: int | None = None,
    num_sms: int | None = None,
    block_size: int | None = None,
) -> int:
    """Return a launch-parallelism-aware split-KV segment count.

    The non-split-QK path is used by medium decode grids.  Four segments avoid
    redundant partial-output and reduction traffic, but only when their grid
    still has enough programs per SM and reducing the grid does not disable
    page-table deduplication.  The decision therefore follows launch geometry,
    page size, and the current device's SM count instead of a GPU model or a
    fixed batch-size list.
    """
    # The fused B=1 paths perform their reduction in the last completing CTA.
    # KV=2048 uses a dedicated split count so it can be tuned independently
    # from the short and long endpoint shapes.
    if (
        use_3d
        and max_seqlen_k in TLE_FUSED_REDUCER_SEQLENS_K
        and num_seqs == 1
    ):
        if max_seqlen_k >= 8192:
            return TLE_FUSED_REDUCER_LONG_SEGMENTS
        if max_seqlen_k == 2048:
            return TLE_FUSED_REDUCER_MEDIUM_SEGMENTS
        return TLE_FUSED_REDUCER_SEGMENTS

    # The B=8/KV=512 production fused route uses eight segments while the
    # per-head-group last-CTA reducer handles the partials in-kernel.
    if (
        use_3d
        and num_seqs == 8
        and max_seqlen_k == 512
        and num_kv_heads == 4
    ):
        return 8

    # The B=8/KV=2048 exact-QK path needs eight segments.  Four segments lose
    # main-kernel parallelism, while sixteen add partial/reducer traffic.
    if use_3d and max_seqlen_k == 2048 and num_seqs == 8:
        return 16

    # For the 64Q/4KV B=32 workload, the 8-way split is memory-bound and
    # launches twice as many partial/reducer items as needed.  Four segments
    # still provide more than one full wave on an H100 while halving that
    # intermediate traffic.  Keep this narrowly scoped until it is validated
    # against the complete matrix.
    if (
        use_3d
        and max_seqlen_k == 2048
        and num_seqs == 32
    ):
        return 4

    # At the 512-token decode boundary, 64 segments over-partition the two
    # geometries that sit on opposite sides of the tile-width transition:
    # an extremely narrow B=1 tile16 grid and the B=8 tile32 grid.  Thirty-
    # two segments remove empty/redundant CTAs and halve partial/reducer
    # traffic.  Intermediate tile16 grids did not show a stable benefit.
    if (
        use_3d
        and max_seqlen_k == 512
        and num_seqs in (1, 8)
    ):
        return 64

    if (
        use_3d
        and max_seqlen_k is not None
        and max_seqlen_k >= 2048
        and (num_seqs == 8 or num_seqs >= 16)
        and not should_use_split_qk_diffkv(192, 1, max_seqlen_k, num_seqs, True)
    ):
        if (
            total_num_q_blocks is not None
            and num_kv_heads is not None
            and num_sms is not None
            and num_sms > 0
            and block_size is not None
        ):
            base_programs = total_num_q_blocks * num_kv_heads
            four_segment_programs = (
                base_programs * TLE_MEDIUM_NUM_PAR_SOFTMAX_SEGMENTS
            )
            tile_size = get_tle_tile_size(
                use_3d, num_seqs, max_seqlen_k, num_kv_heads
            )
            dedup_with_eight = should_dedup_block_table(
                use_3d,
                tile_size,
                block_size,
                total_num_q_blocks,
                num_kv_heads,
                8,
                num_sms,
            )
            dedup_with_four = should_dedup_block_table(
                use_3d,
                tile_size,
                block_size,
                total_num_q_blocks,
                num_kv_heads,
                TLE_MEDIUM_NUM_PAR_SOFTMAX_SEGMENTS,
                num_sms,
            )
            if (
                four_segment_programs
                >= TLE_MEDIUM_MIN_PROGRAMS_PER_SM * num_sms
                and dedup_with_four == dedup_with_eight
            ):
                return TLE_MEDIUM_NUM_PAR_SOFTMAX_SEGMENTS
        return 8
    segments = TLE_NUM_PAR_SOFTMAX_SEGMENTS
    # The B=1/KV=32768 wide-GQA launch has only four KV heads and therefore
    # 256 main CTAs at the default 64-way split.  Doubling the split exposes
    # a second sub-wave while each segment still scans eight 32-token tiles.
    # Keep this isolated to the long shape; shorter B=1 launches do not have
    # enough tile work to amortize the extra partial/reducer traffic.
    if use_3d and num_seqs == 1 and max_seqlen_k == 32768:
        segments = 128
    if use_3d and num_seqs == 8 and max_seqlen_k == 8192:
        segments = 16
    if use_3d and num_seqs == 32 and max_seqlen_k is not None:
        if max_seqlen_k >= 32768:
            segments = 32
        elif max_seqlen_k >= 8192:
            segments = 16

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
        tile_size = get_tle_tile_size(
            use_3d, num_seqs, max_seqlen_k, num_kv_heads
        )
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
            candidate_tiles_per_segment >= TLE_LONG_MIN_TILES_PER_SEGMENT
            and candidate_programs >= TLE_LONG_MIN_PROGRAMS_PER_SM * num_sms
            and dedup_with_candidate == dedup_with_current
        ):
            return candidate_segments
    return segments


def get_tle_num_stages(
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    num_query_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> int:
    """Select the validated TLE software-pipeline depth for this shape."""
    if use_3d and num_seqs == 32 and max_seqlen_k == 2048:
        return 2
    if use_3d and num_seqs == 8 and max_seqlen_k == 2048:
        return 2
    if (
        use_3d
        and num_seqs == 16
        and max_seqlen_k == 2048
        and num_query_heads == 64
        and num_kv_heads == 4
    ):
        return 2
    if (
        not use_3d
        and max_seqlen_k is not None
        and max_seqlen_k <= TLE_SHORT_2D_MAX_SEQLEN_K
    ):
        return TLE_SHORT_2D_NUM_STAGES
    if use_3d and num_seqs == 8 and max_seqlen_k == 8192:
        return 4
    if (
        use_3d
        and num_seqs == 16
        and max_seqlen_k == 8192
        and num_query_heads == 64
        and num_kv_heads == 4
    ):
        return 2
    if (
        use_3d
        and max_seqlen_k is not None
        and max_seqlen_k >= 8192
        and (
            num_seqs == 8
            or num_seqs == 32
            or (num_seqs == 16 and max_seqlen_k >= 32768)
        )
    ):
        return 5
    return TLE_NUM_STAGES


def get_tle_tile_size(
    use_3d: bool,
    num_seqs: int,
    max_seqlen_k: int | None = None,
    num_kv_heads: int | None = None,
) -> int:
    """Select a KV tile width from launch parallelism and sequence length.

    Wider decode grids can use 32 tokens without sacrificing occupancy.  The
    B=1 3D grid is narrower, so it uses the larger tile only for very long KV
    ranges where halving loop/page-table/softmax bookkeeping amortizes the
    extra registers.
    The threshold is a workload property, not a GPU model or architecture
    check.
    """
    if not use_3d or num_seqs >= 8:
        return 32
    if (
        use_3d
        and num_seqs == 1
        and max_seqlen_k == 8192
        and num_kv_heads == 4
    ):
        return 64
    return 32 if max_seqlen_k is not None and max_seqlen_k >= 16384 else 16


def should_dedup_block_table(
    use_3d: bool,
    tile_size: int,
    block_size: int,
    total_num_q_blocks: int,
    num_kv_heads: int,
    num_segments: int,
    num_sms: int,
) -> bool:
    """Use page-centric block-table loads on sufficiently wide grids."""
    total_programs = total_num_q_blocks * num_kv_heads * num_segments
    if tile_size == block_size:
        required_programs_per_sm = TLE_DEDUP_MIN_PROGRAMS_PER_SM
    elif tile_size == 2 * block_size:
        required_programs_per_sm = (
            TLE_TWO_PAGE_DEDUP_MIN_PROGRAMS_PER_SM if use_3d
            else TLE_DEDUP_MIN_PROGRAMS_PER_SM
        )
    elif tile_size == 4 * block_size:
        # The long B=1/KV=8192 route uses a 64-token tile.  Load its four
        # physical page ids once per tile and broadcast them to the K/V
        # lanes; this removes 64 repeated page-table reads from the producer
        # address-generation loop.  No other production geometry currently
        # selects a four-page tile.
        required_programs_per_sm = TLE_DEDUP_MIN_PROGRAMS_PER_SM
    else:
        return False
    return total_programs >= required_programs_per_sm * num_sms


@triton.jit
def kernel_unified_attention_diffkv(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into
    # ``segm_*`` and ``output_ptr`` is unused (callers may pass any
    # non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    completion_counter_ptr,
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
    # When a query-head group is split across CTAs, each program owns a
    # contiguous BLOCK_M slice.  The split factor is derived from the
    # compile-time tile width so the same mapping supports the validated
    # 4->2 split and wider GQA groups without over-splitting.
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
        False,  # USE_MM_PREFIX
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
                key_cache_ptr + k_offset_base + offs_d_qk[:, None] * stride_k_cache_3,
                mask=dim_mask_qk[:, None] & tile_mask[None, :],
                other=0.0,
                is_async=True,
            ).to(Q.dtype)
        # V : (TILE_SIZE, HEAD_SIZE_V_PADDED).  The producer/consumer pipe
        # supplies the overlap.  Keeping this load synchronous avoids
        # Triton 3.6 lowering the subsequent SMEM store as a TMA copy with an
        # unsupported per-thread byte width for the 128xTILE pointer tile.
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

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_SPLIT_QK:
            if IS_3D:
                S += tl.dot(Q_lo, K_lo)
                S += tl.dot(Q_hi, K_hi)
                # The profiled 3D path benefits from scaling only once.
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
        # Store per-segment partials; finalized by reduce_segments_diffkv.
        # The optimized benchmark may provide a model-dtype scratch buffer
        # (BF16 for the supported DiffKV path), reducing split-KV write/read
        # traffic. Triton casts ``acc`` on store and the reducer accumulates
        # in FP32, so the final output remains numerically stable.
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
                segm_ids = tl.arange(0, NUM_SEGMENTS_PER_SEQ)
                fused_dim_mask = offs_d_v < HEAD_SIZE_V
                for local_head in range(0, local_head_count):
                    query_head_idx = (
                        kv_head_idx * num_queries_per_kv
                        + local_head_start
                        + local_head
                    )
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
    # Match vLLM's asynchronous reducer read.  This kernel is selected only
    # when TLE is active; the non-TLE entry point delegates to the baseline
    # implementation before reaching this definition.
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
    # programs write partials, finalized by ``_fallback_kernel_reduce_segments_diffkv``).
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

    # Q : (BLOCK_M, HEAD_SIZE_QK_PADDED).  The medium Full B=8/KV=2048
    # fallback uses the exact 128+64 decomposition so it avoids the 25%
    # padded QK dot-product work while retaining ordinary synchronous loads.
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
        False,  # USE_MM_PREFIX
        IS_3D,
    )

    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        if DEDUP_BLOCK_TABLE and TILE_SIZE == 2 * BLOCK_SIZE:
            # The medium fallback tile spans exactly two physical pages.
            # Broadcast the two page ids instead of issuing one scalar page
            # table load per token lane.
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
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            seq_len,
            None,  # mm_prefix_range_ptr
            SLIDING_WINDOW,
            False,  # USE_MM_PREFIX
            0,  # MAX_MM_RANGES
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

def should_use_split_qk_diffkv(
    head_size_qk: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
) -> bool:
    return (
        head_size_qk == 192
        and max_seqlen_q == 1
        and max_seqlen_k is not None
        and (
            not use_3d
            or (max_seqlen_k == 2048 and num_seqs == 8)
            or (
                max_seqlen_k >= 4096
                and (
                    num_seqs <= 8
                    or num_seqs >= 32
                    or (num_seqs == 16 and max_seqlen_k >= 32768)
                )
            )
        )
    )


def should_use_tle_diffkv(
    head_size_qk: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    num_seqs: int,
    use_3d: bool,
    num_query_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> bool:
    """Route broadly useful decode workloads through the TLE kernel."""
    if head_size_qk != 192 or max_seqlen_q != 1 or max_seqlen_k is None:
        return False
    # Restore the validated dedicated TLE 3D paths for the two MiMo Full
    # low-batch shapes.  The exact head layout, split-head mapping, and fused
    # last-CTA reducer are selected later by the full launcher; keeping this
    # dispatch decision here makes the benchmark measure the same TLE path
    # rather than silently forcing the slower standard-Triton fallback.
    is_mimo_full = (
        num_query_heads is None
        or num_kv_heads is None
        or (num_query_heads, num_kv_heads) == (64, 4)
    )
    # On the MiMo Full B=8/KV=2048 geometry, the asynchronous TLE load
    # lowering adds more scheduling/transaction overhead than it hides.  The
    # same launch with the synchronous Triton path is consistently faster and
    # remains fully compatible with the fused-qk decomposition.  Keep this
    # exception limited to Full (HKV=4); SWA HKV=8 still benefits from TLE.
    if (
        is_mimo_full
        and num_seqs == 8
        and max_seqlen_k == 2048
        and use_3d
    ):
        return False
    return (
        should_use_split_qk_diffkv(
            head_size_qk, max_seqlen_q, max_seqlen_k, num_seqs, use_3d
        )
        or (
            use_3d
            and num_seqs == 1
            and max_seqlen_k in (512, 2048)
        )
        or (
            use_3d
            and num_seqs == 8
            and max_seqlen_k in TLE_B8_FUSED_REDUCER_SEGLENS_K
            and is_mimo_full
        )
        or (
            use_3d
            and (num_seqs == 8 or num_seqs >= 16)
            and max_seqlen_k >= 2048
        )
    )


def should_use_tle_gpu_pipeline2(
    *,
    use_3d: bool,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    use_alibi_slopes: bool,
    use_sinks: bool,
    softcap: float,
    sliding_window: int,
) -> bool:
    # The sidecar uses vLLM-specific launch plumbing and is intentionally not
    # part of the standalone three-file distribution.
    return False


def should_use_tle_gpu_pipeline3(
    *,
    use_3d: bool,
    num_seqs: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size_qk: int,
    head_size_v: int,
    block_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int | None,
    use_alibi_slopes: bool,
    use_sinks: bool,
    softcap: float,
    sliding_window: int,
) -> bool:
    return False

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
    # 3D / split-KV softmax buffers.  When all four are provided and the
    # batch is decode-only with few sequences, the 3D path is taken.
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
):
    assert causal, "Only causal attention is supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    is_decode = max_seqlen_q == 1 and q.shape[0] == num_seqs
    launch_num_q_blocks = num_seqs if is_decode else total_num_q_blocks

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Decide between 2D and 3D launch.  Mirrors the standard launcher:
    # 3D requires preallocated softmax buffers, decode-only batches, and
    # a small number of sequences (otherwise 2D already saturates the SM).
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # Tile size: 32 for prefill-class kernels.  Decode (small Q) prefers
    # smaller tiles to expose more parallelism along the KV dim.
    tile_size = 32 if not use_3d else (16 if q.element_size() >= 2 else 32)
    # The Full B=8/KV=2048 route uses synchronous Triton loads (the TLE
    # lowering is slower for this geometry).  A 32-token tile reduces its
    # loop/page-table overhead while retaining the exact same math and
    # fallback kernel, so keep this specialization narrow to HKV=4.
    if (
        use_3d
        and num_seqs == 8
        and max_seqlen_k == 2048
        and num_query_heads == 64
        and num_kv_heads == 4
    ):
        tile_size = 32
    fallback_split_qk = (
        use_3d
        and num_seqs == 8
        and max_seqlen_k == 2048
        and num_query_heads == 64
        and num_kv_heads == 4
        and head_size_qk == 192
        and head_size_v == 128
    )
    fallback_num_segments = num_par_softmax_segments
    fallback_split_heads = False
    fallback_dedup_block_table = (
        fallback_split_qk and tile_size == 2 * block_size
    )

    grid: tuple[Any, ...]
    if use_3d:
        grid = (
            launch_num_q_blocks,
            num_kv_heads,
            fallback_num_segments,
        )
        segm_output_ptr = softmax_segm_output
        segm_max_ptr = softmax_segm_max
        segm_expsum_ptr = softmax_segm_expsum
        num_segments = fallback_num_segments
    else:
        grid = (launch_num_q_blocks, num_kv_heads)
        # 2D never touches the segm tensors but Triton wants a non-null
        # pointer; reuse ``out``.
        segm_output_ptr = out
        segm_max_ptr = out
        segm_expsum_ptr = out
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
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(head_size_qk),
        USE_SPLIT_QK=fallback_split_qk,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
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
        # The MiMo Full short/medium reducers launch many independent output
        # rows. One warp is sufficient for its 16x128 reduction tile and
        # minimizes per-CTA scheduling overhead; other geometries keep
        # Triton's established four-warp default.
        reduce_num_warps = (
            2
            if (
                num_seqs == 8
                and num_query_heads == 64
                and num_kv_heads == 4
                and head_size_qk == 192
                and head_size_v == 128
                and max_seqlen_k == 2048
            )
            else 1
            if (
                num_seqs == 8
                and num_query_heads == 64
                and num_kv_heads == 4
                and head_size_qk == 192
                and head_size_v == 128
                and max_seqlen_k == 512
            )
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
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
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
    # 3D / split-KV softmax buffers.  When all four are provided and the
    # batch is decode-only with few sequences, the 3D path is taken.
    seq_threshold_3D: int | None = None,
    num_par_softmax_segments: int | None = None,
    softmax_segm_output: torch.Tensor | None = None,
    softmax_segm_max: torch.Tensor | None = None,
    softmax_segm_expsum: torch.Tensor | None = None,
    max_seqlen_k: int | None = None,
    reduce_num_warps: int | None = None,
    fused_reducer_counter: torch.Tensor | None = None,
):
    if not USE_TLE:
        # The optimized source contains direct TLE intrinsics for parity with
        # vLLM.  When TLE is absent, route the public entry point to the
        # standalone synchronous implementation instead of trying to compile
        # a kernel that references a missing dialect.
        if fused_reducer_counter is not None:
            raise RuntimeError("fused reducer requires the TLE backend")
        return _unified_attention_diffkv_fallback(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            block_table=block_table,
            softcap=softcap,
            max_seqlen_q=max_seqlen_q,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
            use_alibi_sqrt=use_alibi_sqrt,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            max_seqlen_k=max_seqlen_k,
        )
    assert causal, "Only causal attention is supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size_qk = q.shape[2]
    head_size_v = v.shape[3]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    is_decode = max_seqlen_q == 1 and q.shape[0] == num_seqs
    use_decode_fastpath = is_decode and total_num_q_blocks > num_seqs
    launch_num_q_blocks = (
        num_seqs if use_decode_fastpath else total_num_q_blocks
    )

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Decide between 2D and 3D launch.  Mirrors the standard launcher:
    # 3D requires preallocated softmax buffers, decode-only batches, and
    # a small number of sequences (otherwise 2D already saturates the SM).
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > 1
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    fused_reducer_requested = (
        fused_reducer_counter is not None
        and should_use_tle_fused_reducer(
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
    )
    persistent_fused_requested = (
        fused_reducer_requested
        and should_use_tle_persistent_fused_reducer(
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
    )
    if not fused_reducer_requested and (
        should_use_tle_split_head_2d(
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
        or should_use_tle_short_2d(
            head_size_qk,
            max_seqlen_q,
            max_seqlen_k,
            use_3d,
            total_num_q_blocks,
            num_kv_heads,
            num_sms,
        )
    ):
        use_3d = False

    # Some narrow decode grids benefit from splitting the GQA group into
    # smaller head CTAs.  The validated B=1/KV=512, B=1/KV=2048, and
    # B=8/KV=512 paths use this mapping.
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
    )
    if split_heads:
        if (
            not use_3d
            and num_seqs == 8
            and max_seqlen_k == 512
            and num_query_heads == 64
            and num_kv_heads == 4
        ):
            # Four heads per CTA expand the short Full-Attention grid from 32
            # to 128 programs, enough for one H100 wave without the 16x K/V
            # duplication of a one-head decomposition.
            BLOCK_M = 4
        elif (
            use_3d
            and num_seqs == 1
            and max_seqlen_k == 2048
            and num_queries_per_kv == 16
        ):
            # A 2-row head tile doubles independent head CTAs for the medium
            # fused path; unlike the 16-way split, its extra reducer traffic is
            # still hidden by the 2K KV scan.
            BLOCK_M = 2
        else:
            BLOCK_M = (
                4
                if (
                    use_3d
                    and num_seqs == 1
                    and max_seqlen_k in TLE_FUSED_REDUCER_SEQLENS_K
                    and num_queries_per_kv == 16
                )
                else (8 if num_queries_per_kv == 16 else 2)
            )
        BLOCK_Q = 1
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
        use_decode_fastpath = is_decode and total_num_q_blocks > num_seqs
        launch_num_q_blocks = num_seqs if use_decode_fastpath else total_num_q_blocks
    elif fused_reducer_requested and max_seqlen_k == 8192:
        # The profiled B=1 fused kernel spends 4x work on duplicated GQA
        # rows when BLOCK_M=16 although this workload has only four queries
        # per KV head.  Keep exactly the four real rows to reduce register
        # pressure and QK/PV work; the split-KV grid remains unchanged.
        BLOCK_M = num_queries_per_kv
        BLOCK_Q = 1
        total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
        use_decode_fastpath = is_decode and total_num_q_blocks > num_seqs
        launch_num_q_blocks = num_seqs if use_decode_fastpath else total_num_q_blocks

    # A 32-token tile halves loop/page-table/softmax bookkeeping for the
    # sufficiently parallel B>=8 split-KV launches.  B=1 still needs the
    # 16-token tile to expose enough work across its small program grid.
    tile_size = get_tle_tile_size(
        use_3d, num_seqs, max_seqlen_k, num_kv_heads
    )
    if (
        not use_3d
        and num_seqs == 8
        and max_seqlen_k == 512
        and num_query_heads == 64
        and num_kv_heads == 4
        and head_size_qk == 192
        and head_size_v == 128
    ):
        # Candidate: halve the loop/page-table iterations for the short
        # 2D launch.  The existing four-page broadcaster handles the wider
        # tile; this remains isolated until the target is measured.
        tile_size = 64
    use_split_qk = should_use_split_qk_diffkv(
        head_size_qk,
        max_seqlen_q,
        max_seqlen_k,
        num_seqs,
        use_3d,
    )
    # Keep the exact 128+64 QK decomposition for DQK=192. This mirrors the
    # FA3 tile shape and avoids executing the padded 256-wide MMA for the
    # Full-Attention B=8/KV=2048 geometry.
    fuse_reducer = (
        fused_reducer_counter is not None
        and should_use_tle_fused_reducer(
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
        segm_output_ptr = softmax_segm_output
        segm_max_ptr = softmax_segm_max
        segm_expsum_ptr = softmax_segm_expsum
        num_segments = (
            TLE_PERSISTENT_FUSED_REDUCER_SEGMENTS
            if persistent_fused_requested
            else num_par_softmax_segments
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
        # 2D never touches the segm tensors but Triton wants a non-null
        # pointer; reuse ``out``.
        segm_output_ptr = out
        segm_max_ptr = out
        segm_expsum_ptr = out
        num_segments = 1

    # The B=16 short 2D grid has enough work per CTA for two-page page-table
    # broadcast to amortize its select overhead.  This removes per-token table
    # loads and reduces the profiled L1/TEX pressure.  B=8 remains on the
    # generic selector because the same transformation regresses its shorter
    # launch.
    dedup_block_table = (
        not use_3d
        and max_seqlen_k == 512
        and num_seqs == 16
        and tile_size == 2 * block_size
    ) or (
        not use_3d
        and num_seqs == 8
        and max_seqlen_k == 512
        and num_query_heads == 64
        and num_kv_heads == 4
        and tile_size in (2 * block_size, 4 * block_size)
    ) or should_dedup_block_table(
        use_3d,
        tile_size,
        block_size,
        total_num_q_blocks,
        num_kv_heads,
        num_segments,
        num_sms,
    )


    kernel_unified_attention_diffkv[grid](
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
        LOOP_NUM_STAGES=(
            TLE_FUSED_REDUCER_NUM_STAGES
            if fuse_reducer
            else get_tle_num_stages(
                max_seqlen_k,
                num_seqs,
                use_3d,
                num_query_heads,
                num_kv_heads,
            )
        ),
        HEAD_SIZE_QK=head_size_qk,
        HEAD_SIZE_QK_PADDED=triton.next_power_of_2(head_size_qk),
        USE_SPLIT_QK=use_split_qk,
        HEAD_SIZE_V=head_size_v,
        HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
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
            TLE_PERSISTENT_FUSED_REDUCER_SEGMENTS_PER_PROGRAM
            if persistent_fused_requested
            else 1
        ),
        num_warps=(
            (
                TLE_FUSED_REDUCER_LONG_NUM_WARPS
                if max_seqlen_k is not None and max_seqlen_k >= 8192
                else (
                    TLE_FUSED_REDUCER_MEDIUM_NUM_WARPS
                    if max_seqlen_k == 2048
                    else TLE_FUSED_REDUCER_NUM_WARPS
                )
            )
            if fuse_reducer
            else (
                4
                if (
                    not use_3d
                    and num_seqs == 16
                    and max_seqlen_k == 512
                    and num_kv_heads == 4
                )
                else 8
                if (
                    (split_heads and num_seqs == 1)
                    or (not use_3d and num_seqs == 16 and max_seqlen_k == 512)
                )
                else TLE_NUM_WARPS
            )
        ),
        num_stages=(
            TLE_FUSED_REDUCER_NUM_STAGES
            if fuse_reducer
            else get_tle_num_stages(
                max_seqlen_k,
                num_seqs,
                use_3d,
                num_query_heads,
                num_kv_heads,
            )
        ),
    )

    if use_3d and not fuse_reducer:
        use_async_reducer = should_use_async_tle_reducer(
            max_seqlen_k,
            q.shape[0],
            num_query_heads,
            num_segments,
            num_sms,
        )
        if reduce_num_warps is None:
            reduce_num_warps = get_tle_reduce_num_warps(
                q.shape[0], num_query_heads, num_sms
            )
        reduce_kernel = (
            kernel_reduce_segments_diffkv_async
            if use_async_reducer
            else kernel_reduce_segments_diffkv
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
            HEAD_SIZE_V_PADDED=triton.next_power_of_2(head_size_v),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            IS_DECODE=use_decode_fastpath,
            num_warps=reduce_num_warps,
        )




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


def unified_attention_diffkv(*args: Any, backend: str | None = None,
                             **kwargs: Any):
    """Run DiffKV through the selected inlined TLE or standard Triton path."""
    selected = _resolve_backend(backend)
    if selected == "tle":
        return _unified_attention_diffkv_tle(*args, **kwargs)
    # The fallback launcher has no fused-reducer argument.
    kwargs.pop("fused_reducer_counter", None)
    return _unified_attention_diffkv_fallback(*args, **kwargs)


def unified_attention_diffkv_tle(*args: Any, **kwargs: Any):
    """Explicit TLE implementation entry point."""
    return unified_attention_diffkv(*args, backend="tle", **kwargs)


def unified_attention_diffkv_fallback(*args: Any, **kwargs: Any):
    """Explicit standard non-TLE Triton entry point."""
    return unified_attention_diffkv(*args, backend="triton", **kwargs)


def diffkv_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    attn_scale: float | None = None,
    window_size: int = -1,
    path: str = "auto",
    num_segments: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Run the standalone paged DiffKV operator.

    Args use the FlagAttention paged layout: ``query=[B,H,Dqk]``, key/value
    cache ``[num_blocks, block_size, Hkv, D]``, context lengths ``[B]`` and
    block tables ``[B,max_blocks]``.  ``path`` may be ``"2d"`` or ``"3d"``;
    ``"auto"`` follows the launch policy used by the optimized vLLM kernel.
    The function is decode-oriented (one query token per sequence), matching
    the 16-shape benchmark and the MiMo DiffKV use case.
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
    if path not in {"auto", "2d", "3d"}:
        raise ValueError("path must be auto, 2d, or 3d")
    if path == "auto":
        use_3d = not (batch >= 32 or (batch == 16 and max_seqlen_k == 512))
    else:
        use_3d = path == "3d"
    if use_3d:
        if num_segments is None:
            if selected == "tle":
                num_segments = get_tle_num_par_softmax_segments(
                    max_seqlen_k,
                    batch,
                    True,
                    total_num_q_blocks=2 * batch,
                    num_kv_heads=num_kv_heads,
                    num_sms=torch.cuda.get_device_properties(query.device).multi_processor_count,
                    block_size=key_cache.shape[1],
                )
            else:
                num_segments = 64 if batch == 1 and max_seqlen_k == 2048 else 16
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
        num_query_heads,
        num_kv_heads,
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
    "is_batch_invariant",
    "is_tle_available",
    "tle_import_error",
    "get_diffkv_backend_info",
    "should_use_tle_fused_reducer",
    "should_use_tle_persistent_fused_reducer",
    "should_use_tle_split_head_2d",
    "should_split_tle_decode_heads",
    "should_use_tle_short_2d",
    "should_use_tle_long_3d",
    "get_tle_reduce_num_warps",
    "should_use_async_tle_reducer",
    "get_tle_num_par_softmax_segments",
    "get_tle_num_stages",
    "get_tle_tile_size",
    "should_dedup_block_table",
    "should_use_split_qk_diffkv",
    "should_use_tle_diffkv",
    "should_use_tle_gpu_pipeline2",
    "should_use_tle_gpu_pipeline3",
]
