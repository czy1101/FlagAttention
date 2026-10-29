# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Batched KKT / stitch launches for the Raw GDN inverse prep.

Both kernels are chunk-parallel over ``B*NT*H`` independent jobs.  The affine
prep kernel already amortises its per-program scalar/scheduling cost by
processing four consecutive chunks per program; the KKT and stitch kernels are
limited the same way.  Measured on 910B (B4/T4096/H64 BF16, 2026-10-27):

* KKT: flat grid with ``BATCH=8`` 1.31-1.34 -> 1.05-1.07 ms (-20%), bit-exact;
* stitch: 2-D grid ``(cdiv(NT, 32), B*H)`` with ``BATCH=32`` 1.42 -> 1.12 ms
  (-20.3%), bit-exact.

The Raw diagonal inverse itself is *not* per-program-overhead bound (2/4 jobs
per CTA were 2-6% slower and 4 jobs exceed the 192 KB UB budget), so it stays
as is.  Small grids keep the previous batch sizes because the extra batching is
lost in noise there.  ``FLAG_ATTN_ASCEND_GDN_INVBATCH=0`` restores the previous
launches.
"""

from __future__ import annotations

import os

import triton
import triton.language as tl

from .raw_inverse_prep import (
    _stitch_inverse_one_tile,
    launch_kkt_blocks_batched,
    launch_stitch_inverse_batch4,
)

_OFF = {"0", "false", "off", "no"}

# Crossovers measured above: below these job counts the previous batching wins
# or is within noise.
_KKT_BATCH8_MIN_JOBS = 2048
# Keep the 2-D stitch for grids where the measured win is clear (>=2048 jobs:
# B4/T4096 -20%, B2/T16384 -27%, B4/T2048 -21%); at 1024 jobs it is noise.
_STITCH_2D_MIN_JOBS = 2048
_STITCH_2D_BATCH = 32


def _enabled() -> bool:
    return os.environ.get(
        "FLAG_ATTN_ASCEND_GDN_INVBATCH", "1"
    ).lower() not in _OFF


@triton.jit
def _stitch_2d_batched_kernel(
    L_FULL, DIAG, INVERSE,
    H: tl.constexpr, NT: tl.constexpr,
    BATCH: tl.constexpr,
):
    i_g, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    for k in tl.static_range(0, BATCH):
        i_t = i_g * BATCH + k
        if i_t < NT:
            tile = (i_b * NT + i_t) * H + i_h
            _stitch_inverse_one_tile(L_FULL, DIAG, INVERSE, tile)


def launch_kkt_blocks_dispatch(ke, kr, L_full, T, H, K, NT, BT, jobs):
    """Flat batched KKT; BATCH=8 for large grids, 4 otherwise."""
    batch = 8 if (_enabled() and jobs >= _KKT_BATCH8_MIN_JOBS) else 4
    launch_kkt_blocks_batched(
        ke, kr, L_full, T, H, K, NT, BT, batch_size=batch,
    )


def launch_stitch_inverse_dispatch(L_full, diag, inverse, H, NT, jobs):
    """2-D batched stitch for large grids, previous flat batch4 otherwise."""
    if _enabled() and jobs >= _STITCH_2D_MIN_JOBS:
        B_H = L_full.shape[0] // NT
        _stitch_2d_batched_kernel[
            (triton.cdiv(NT, _STITCH_2D_BATCH), B_H)
        ](
            L_full, diag, inverse, H, NT, _STITCH_2D_BATCH,
            num_warps=4, num_stages=1,
        )
        return
    launch_stitch_inverse_batch4(L_full, diag, inverse)


__all__ = ["launch_kkt_blocks_dispatch", "launch_stitch_inverse_dispatch"]
