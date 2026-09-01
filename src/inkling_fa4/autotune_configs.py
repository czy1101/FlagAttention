"""Workload-aware launch presets for Inkling paged relative attention.

Launch configurations are specialized by GPU architecture, query workload,
KV work, head dimensions, and GQA packing. H100 values are benchmark seeds and
must be validated against the workload benchmark and Nsight reports.
"""

from __future__ import annotations

import torch


def _detect_arch() -> str:
    """Return the active CUDA compute-capability key."""
    if not torch.cuda.is_available():
        return "sm90"

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def get_preset(
    max_seqlen_q: int,
    *,
    max_seqlen_k: int | None = None,
    head_dim_q: int = 128,
    head_dim_v: int = 128,
    q_heads_per_kv_head: int = 1,
    arch: str | None = None,
) -> dict[str, int]:
    """Return a launch configuration specialized by workload class.

    Args:
        max_seqlen_q:
            Maximum query length in the batch.
        max_seqlen_k:
            Synchronization-free upper bound for KV length. The wrapper may use
            page-table capacity instead of reading cache_seqlens back to CPU.
        head_dim_q:
            Q/K head dimension.
        head_dim_v:
            V/output head dimension.
        q_heads_per_kv_head:
            GQA group size.
        arch:
            Explicit architecture such as "sm90", or None for auto-detection.
    """
    arch = _detect_arch() if arch is None else arch
    k = max_seqlen_q if max_seqlen_k is None else max_seqlen_k
    wide = max(head_dim_q, head_dim_v) > 128
    scheduler_rows = max_seqlen_q * q_heads_per_kv_head

    if arch == "sm90":
        # Single-token decode:
        # Pack the GQA heads into the M dimension so one paged K/V traversal
        # serves multiple Q heads. Long KV uses a wider K tile.
        if max_seqlen_q == 1:
            block_q = 16 if q_heads_per_kv_head <= 16 else 32
            block_k = 64 if k < 8192 else 128

            return {
                "BLOCK_Q": block_q,
                "BLOCK_K": block_k,
                "num_warps": 4 if not wide else 8,
                "num_stages": 3,
            }

        # Small-Q/chunked workloads:
        # Keep the FP32 output accumulator small and use Split-KV to expose
        # additional parallelism where the caller selects num_splits > 1.
        if scheduler_rows <= 64:
            return {
                "BLOCK_Q": 32,
                "BLOCK_K": 64,
                "num_warps": 4,
                "num_stages": 3,
            }

        # H100 prefill A/B configuration:
        #
        # The former 128x64, 8-warp, 4-stage configuration produced:
        #   254 registers/thread
        #   131.58 KiB dynamic shared memory/CTA
        #   12.5% theoretical occupancy
        #
        # Reducing BLOCK_Q to 64 shortens the live FP32 accumulator and score
        # fragments. Four warps and three stages reduce register/shared-memory
        # pressure and increase the query-tile grid.
        return {
            "BLOCK_Q": 64,
            "BLOCK_K": 64,
            "num_warps": 4,
            "num_stages": 3,
        }

    # Portable fallback for non-SM90 devices.
    if max_seqlen_q <= 16:
        return {
            "BLOCK_Q": 16,
            "BLOCK_K": 32,
            "num_warps": 4,
            "num_stages": 2,
        }

    return {
        "BLOCK_Q": 32,
        "BLOCK_K": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


def recommend_num_splits(
    *,
    batch_size: int,
    num_kv_heads: int,
    max_seqlen_q: int,
    max_seqlen_k_bound: int,
    block_k: int,
    sm_count: int,
) -> int:
    """Return a conservative synchronization-free Split-KV seed.

    This is only a fallback heuristic. Explicit benchmarked split counts should
    take priority for known serving workloads.
    """
    if max_seqlen_k_bound < 64:
        return 1

    # Calibrated H100 seeds for the currently benchmarked batch-1/GQA shapes.
    # These conditions deliberately remain narrow; unknown production shapes
    # should use a workload table or an offline autotuning result.
    if batch_size == 1 and num_kv_heads == 2:
        if max_seqlen_q == 1:
            if max_seqlen_k_bound <= 1024:
                return 16
            if max_seqlen_k_bound <= 8192:
                return 64

        if max_seqlen_q <= 32 and max_seqlen_k_bound >= 512:
            return 8

        if max_seqlen_q <= 64 and max_seqlen_k_bound >= 512:
            return 8

        if max_seqlen_q <= 512 and max_seqlen_k_bound >= 512:
            return 4

    # General fallback: estimate how many additional KV partitions are useful
    # before exceeding roughly two CTA waves.
    query_tiles = max(1, (max_seqlen_q + 63) // 64)
    base_ctas = max(1, batch_size * num_kv_heads * query_tiles)
    target_ctas = min(sm_count * 2, 256)

    by_occupancy = max(1, target_ctas // base_ctas)
    by_work = max(
        1,
        (max_seqlen_k_bound + block_k - 1) // block_k,
    )

    return min(128, by_occupancy, by_work)