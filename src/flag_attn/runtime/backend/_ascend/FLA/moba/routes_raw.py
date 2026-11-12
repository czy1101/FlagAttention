# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""TLE-Raw route compaction kernels (AscendC on the vector core).

The Triton route builder needs, per route slot, a cursor update inside its
(``token tile, head, chunk``) cell.  Triton has no scalar top-level loop over UB
here and no GM scatter at all -- measured, a scalarized GM scatter costs
2.6 ns/element and `tle.atomic_add` with a consumed result hangs the vector
core -- so the compact layout is built by AscendC kernels that only exchange
*contiguous* buffers with Triton:

  * :func:`route_cell_hist` fills ``counts[b, h, c]`` (slots of tile ``b`` of
    head ``h`` routed to chunk ``c``);
  * :func:`route_cell_bucket` turns one (tile, head) into runs in chunk order
    (``out_rows``, the packed query row ``t * num_heads + h``) plus the compact
    position of every slot (``positions``).

Both are exported as ``@dialect(name="cann")`` custom ops, see
``_moba_routes_raw.cpp``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

#: The C++ kernels stage at most this many rank segments (routed top-k).
MAX_TOPK = 8
_MAX_TOPK = tl.constexpr(MAX_TOPK)
#: UB budget: 8 rank buffers + out + positions + start/base cursors.
_MAX_TILE_TOKENS = 2048
#: Tokens staged per raw bucket call; a larger tile is streamed in several calls
#: (the run and position cursors persist between them).
_MAX_CALL_TOKENS = 1024
_CALL_TOKENS = tl.constexpr(_MAX_CALL_TOKENS)
_BUCKET_BLOCK = 256

try:  # FlagTree builds expose the TLE-Raw channel; plain Triton does not.
    import triton.experimental.tle as tle
    import triton.experimental.tle.language.raw as tle_raw
    from triton.experimental.tle.raw import dialect

    _HAS_TLE_RAW = True
except Exception:  # pragma: no cover - non-FlagTree Ascend builds
    tle = None
    tle_raw = None
    dialect = None
    _HAS_TLE_RAW = False

_RAW_CPP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_moba_routes_raw.cpp")

if _HAS_TLE_RAW:

    @dialect(name="cann", file=_RAW_CPP, extern_func_name="moba_routes_hist",
             pipeline={"core": "vector", "pipe": "PIPE_V"})
    def _hist_op(*args, **kwargs):
        ...

    @dialect(name="cann", file=_RAW_CPP, extern_func_name="moba_routes_bucket",
             pipeline={"core": "vector", "pipe": "PIPE_V"})
    def _bucket_op(*args, **kwargs):
        ...


def _raw_available(selected_chunks) -> bool:
    if not _HAS_TLE_RAW:
        return False
    if selected_chunks.device.type != "npu":
        return False
    if not 1 <= selected_chunks.shape[0] <= MAX_TOPK:
        return False
    if selected_chunks.dtype != torch.int32:
        return False
    return True


if _HAS_TLE_RAW:

    @triton.jit
    def _route_hist_kernel(
        sel_ptr,
        counts_ptr,
        num_tokens,
        BT: tl.constexpr,
        NC: tl.constexpr,
        H: tl.constexpr,
        TOPK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, BT)
        u0 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u1 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u2 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u3 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u4 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u5 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u6 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u7 = tle.dsa.alloc([BT], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        src = sel_ptr + head * num_tokens + tile * BT + offs
        rank_stride = H * num_tokens
        # Every staged buffer needs a producer (an undefined UB operand makes
        # the planner emit a device-side malloc and the link fails), so unused
        # ranks are filled with rank 0 and ignored by the kernel's `topk`.
        for rank in tl.static_range(_MAX_TOPK):
            use_rank = rank if rank < TOPK else TOPK - 1
            src_rank = src + use_rank * rank_stride
            if rank == 0:
                tle.dsa.copy(src_rank, u0, [BT])
            elif rank == 1:
                tle.dsa.copy(src_rank, u1, [BT])
            elif rank == 2:
                tle.dsa.copy(src_rank, u2, [BT])
            elif rank == 3:
                tle.dsa.copy(src_rank, u3, [BT])
            elif rank == 4:
                tle.dsa.copy(src_rank, u4, [BT])
            elif rank == 5:
                tle.dsa.copy(src_rank, u5, [BT])
            elif rank == 6:
                tle.dsa.copy(src_rank, u6, [BT])
            else:
                tle.dsa.copy(src_rank, u7, [BT])
        cnt = tl.zeros([NC], dtype=tl.int32)
        cnt = tle_raw.call(
            _hist_op,
            [
                tle.dsa.to_tensor(u0),
                tle.dsa.to_tensor(u1),
                tle.dsa.to_tensor(u2),
                tle.dsa.to_tensor(u3),
                tle.dsa.to_tensor(u4),
                tle.dsa.to_tensor(u5),
                tle.dsa.to_tensor(u6),
                tle.dsa.to_tensor(u7),
                TOPK,
                BT,
                NC,
                cnt,
            ],
            output_indices=[11],
        )
        tl.store(
            counts_ptr + (tile * H + head) * NC + tl.arange(0, NC), cnt
        )

    @triton.jit
    def _stage_ids(sel_ptr, head, num_tokens, first, rank_stride,
                   TOPK: tl.constexpr, CALL: tl.constexpr,
                   u0, u1, u2, u3, u4, u5, u6, u7):
        """Stage the CALL tokens of every routed rank into the 8 UB buffers."""
        src = sel_ptr + head * num_tokens + first + tl.arange(0, CALL)
        for rank in tl.static_range(_MAX_TOPK):
            use_rank = rank if rank < TOPK else TOPK - 1
            src_rank = src + use_rank * rank_stride
            if rank == 0:
                tle.dsa.copy(src_rank, u0, [CALL])
            elif rank == 1:
                tle.dsa.copy(src_rank, u1, [CALL])
            elif rank == 2:
                tle.dsa.copy(src_rank, u2, [CALL])
            elif rank == 3:
                tle.dsa.copy(src_rank, u3, [CALL])
            elif rank == 4:
                tle.dsa.copy(src_rank, u4, [CALL])
            elif rank == 5:
                tle.dsa.copy(src_rank, u5, [CALL])
            elif rank == 6:
                tle.dsa.copy(src_rank, u6, [CALL])
            else:
                tle.dsa.copy(src_rank, u7, [CALL])

    @triton.jit
    def _route_bucket_kernel(
        sel_ptr,
        start_ptr,
        base_ptr,
        out_rows_ptr,
        pos_ptr,
        num_tokens,
        BT: tl.constexpr,
        NC: tl.constexpr,
        H: tl.constexpr,
        TOPK: tl.constexpr,
        CALL: tl.constexpr,
        NCALL: tl.constexpr,
    ):
        """Bucket one (tile, head); a 2*CALL tile is streamed as two calls.

        `start`/`base` double as the run-destination and compact-position
        cursors, so the second call continues the first one's runs; the run
        output is chained through the raw call and the per-call position buffers
        are stitched by the caller.
        """
        tile = tl.program_id(0)
        head = tl.program_id(1)
        cell = tile * H + head
        u0 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u1 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u2 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u3 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u4 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u5 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u6 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        u7 = tle.dsa.alloc([CALL], dtype=tl.int32, mem_addr_space=tle.dsa.ascend.UB)
        offs_c = tl.arange(0, NC)
        start_ub = tle.dsa.alloc([NC], dtype=tl.int32,
                                 mem_addr_space=tle.dsa.ascend.UB)
        base_ub = tle.dsa.alloc([NC], dtype=tl.int32,
                                mem_addr_space=tle.dsa.ascend.UB)
        tle.dsa.copy(start_ptr + cell * NC + offs_c, start_ub, [NC])
        tle.dsa.copy(base_ptr + cell * NC + offs_c, base_ub, [NC])
        rank_stride = H * num_tokens
        # -1 padding is set once: the raw kernel only writes occupied slots.
        out_t = tl.full([TOPK * BT], -1, dtype=tl.int32)
        pos_t = tl.full([TOPK * CALL], -1, dtype=tl.int32)
        offs_pos = tl.arange(0, TOPK * CALL)
        first = tile * BT
        _stage_ids(sel_ptr, head, num_tokens, first, rank_stride,
                   TOPK=TOPK, CALL=CALL, u0=u0, u1=u1, u2=u2, u3=u3,
                   u4=u4, u5=u5, u6=u6, u7=u7)
        # 18 operands exactly: the raw channel silently drops the custom op at 19
        # (measured with an unconditional marker write that never landed).
        out_t, pos_t = tle_raw.call(
            _bucket_op,
            [
                tle.dsa.to_tensor(u0),
                tle.dsa.to_tensor(u1),
                tle.dsa.to_tensor(u2),
                tle.dsa.to_tensor(u3),
                tle.dsa.to_tensor(u4),
                tle.dsa.to_tensor(u5),
                tle.dsa.to_tensor(u6),
                tle.dsa.to_tensor(u7),
                tle.dsa.to_tensor(start_ub),
                tle.dsa.to_tensor(base_ub),
                TOPK,
                CALL,
                head,
                H,
                first * H,
                out_t,
                pos_t,
            ],
            output_indices=[15, 16],
        )
        tl.store(pos_ptr + (cell * NCALL + 0) * (TOPK * CALL) + offs_pos, pos_t)
        if BT > CALL:
            _stage_ids(sel_ptr, head, num_tokens, first + CALL, rank_stride,
                       TOPK=TOPK, CALL=CALL, u0=u0, u1=u1, u2=u2, u3=u3,
                       u4=u4, u5=u5, u6=u6, u7=u7)
            pos_t = tl.full([TOPK * CALL], -1, dtype=tl.int32)
            out_t, pos_t = tle_raw.call(
                _bucket_op,
                [
                    tle.dsa.to_tensor(u0),
                    tle.dsa.to_tensor(u1),
                    tle.dsa.to_tensor(u2),
                    tle.dsa.to_tensor(u3),
                    tle.dsa.to_tensor(u4),
                    tle.dsa.to_tensor(u5),
                    tle.dsa.to_tensor(u6),
                    tle.dsa.to_tensor(u7),
                    tle.dsa.to_tensor(start_ub),
                    tle.dsa.to_tensor(base_ub),
                    TOPK,
                    CALL,
                    head,
                    H,
                    (first + CALL) * H,
                    out_t,
                    pos_t,
                ],
                output_indices=[15, 16],
            )
            tl.store(pos_ptr + (cell * NCALL + 1) * (TOPK * CALL) + offs_pos, pos_t)
        tl.store(
            out_rows_ptr + cell * (TOPK * BT) + tl.arange(0, TOPK * BT), out_t
        )


#: Below this many packed tokens a single 1024-token call is faster than the
#: streamed 2048-token tile (two raw calls plus a second position buffer), so the
#: larger tile is only worth its extra complexity on the big benchmark shapes.
_SMALL_SEQLEN = 32768


def tile_tokens_for_raw(seqlen: int) -> int:
    """Largest UB-safe token tile that still divides ``seqlen``."""
    tile = _MAX_TILE_TOKENS if seqlen >= _SMALL_SEQLEN else _MAX_CALL_TOKENS
    while tile > 1 and seqlen % tile != 0:
        tile //= 2
    return tile


def route_cell_hist(selected_chunks, num_chunks: int) -> torch.Tensor | None:
    """``counts[tile, head, chunk]`` via the raw histogram kernel."""
    topk, num_heads, num_tokens = selected_chunks.shape
    tile = tile_tokens_for_raw(num_tokens)
    if tile < 1 or not _raw_available(selected_chunks):
        return None
    n_blocks = num_tokens // tile
    counts = torch.empty(
        (n_blocks * num_heads * num_chunks,),
        device=selected_chunks.device,
        dtype=torch.int32,
    )
    _route_hist_kernel[(n_blocks, num_heads)](
        selected_chunks,
        counts,
        num_tokens,
        BT=tile,
        NC=num_chunks,
        H=num_heads,
        TOPK=topk,
        num_warps=1,
    )
    return counts.view(n_blocks, num_heads, num_chunks)


def route_cell_bucket(selected_chunks, start3, base3) -> tuple | None:
    """Per-cell runs (``out_rows``) and compact per-slot positions."""
    topk, num_heads, num_tokens = selected_chunks.shape
    n_blocks = start3.shape[0]
    tile = num_tokens // n_blocks
    if not _raw_available(selected_chunks) or tile > _MAX_TILE_TOKENS:
        return None
    device = selected_chunks.device
    # tiles are powers of two, so the staged call is either the whole tile or
    # exactly half of a 2*CALL tile: no partial sub-block ever reaches the kernel
    call = min(tile, _MAX_CALL_TOKENS)
    n_calls = tile // call
    out_rows = torch.empty(
        (n_blocks * num_heads, topk * tile), device=device, dtype=torch.int32
    )
    pos_tile = torch.empty(
        (n_blocks * num_heads, n_calls, topk * call),
        device=device, dtype=torch.int32,
    )
    _route_bucket_kernel[(n_blocks, num_heads)](
        selected_chunks,
        start3,
        base3,
        out_rows,
        pos_tile,
        num_tokens,
        BT=tile,
        NC=start3.shape[2],
        H=num_heads,
        TOPK=topk,
        CALL=call,
        NCALL=n_calls,
        num_warps=1,
    )
    # [tile, head, call, rank, token] -> [rank, token * num_heads + head]
    positions = (
        pos_tile.view(n_blocks, num_heads, n_calls, topk, call)
        .permute(3, 0, 2, 4, 1)
        .contiguous()
        .view(topk, num_heads * num_tokens)
    )
    return out_rows, positions
