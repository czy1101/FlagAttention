"""Frozen A34 TLE/WGMMA attention core used by the integrated provider."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


BT = 128
BM = 64
BN = 64
RCP_LN2 = 1.4426950216


def allocator(size: int, alignment: int, stream: int | None):
    del alignment, stream
    return torch.empty(size, dtype=torch.uint8, device="cuda")


@triton.jit
def producer(q_desc, k_desc, v_desc, q0, q1, k_stages, v_stages,
             q0_full, q1_full, k_full, v_full, empty, query_tile, tile_count,
             BM_VALUE: tl.constexpr, BN_VALUE: tl.constexpr,
             D_VALUE: tl.constexpr, CAPACITY_VALUE: tl.constexpr):
    q_start = query_tile * (2 * BM_VALUE)
    tle.gpu.copy(
        q_desc, q0, [BM_VALUE, D_VALUE], [q_start, 0], barrier=q0_full,
    )
    tle.gpu.copy(
        q_desc, q1, [BM_VALUE, D_VALUE], [q_start + BM_VALUE, 0],
        barrier=q1_full,
    )
    for tile in range(0, tile_count):
        slot = tile % CAPACITY_VALUE
        generation = tile // CAPACITY_VALUE
        tle.gpu.barrier_wait(empty[slot], phaseIdx=generation)
        tle.gpu.copy(
            k_desc, k_stages.slot(slot), [BN_VALUE, D_VALUE],
            [tile * BN_VALUE, 0], barrier=k_full[slot],
        )
        tle.gpu.copy(
            v_desc, v_stages.slot(slot), [BN_VALUE, D_VALUE],
            [tile * BN_VALUE, 0], barrier=v_full[slot],
        )


@triton.jit
def consumer(output, lse, q_smem, p_smem, q_full, k_stages, v_stages,
             k_full, v_full, empty, query_tile, head,
             row_in_cta: tl.constexpr, scale_log2: tl.constexpr,
             T_VALUE: tl.constexpr, HQ_VALUE: tl.constexpr,
             D_VALUE: tl.constexpr, BM_VALUE: tl.constexpr,
             BN_VALUE: tl.constexpr, CAPACITY_VALUE: tl.constexpr,
             IS_BF16: tl.constexpr):
    rows = tl.arange(0, BM_VALUE)
    dims = tl.arange(0, D_VALUE)
    query_start = query_tile * (2 * BM_VALUE) + row_in_cta
    query_rows = query_start + rows
    tile_count = (query_tile + 1) * 2

    tle.gpu.barrier_wait(q_full, phaseIdx=0)
    running_max = tl.full((BM_VALUE,), float("-inf"), tl.float32)
    running_sum = tl.zeros((BM_VALUE,), tl.float32)
    output_acc = tl.zeros((BM_VALUE, D_VALUE), tl.float32)

    for tile in range(0, tile_count):
        slot = tile % CAPACITY_VALUE
        generation = tile // CAPACITY_VALUE
        tle.gpu.barrier_wait(k_full[slot], phaseIdx=generation)
        score = tle.gpu.wgmma(
            q_smem, k_stages.slot(slot), out_dtype=tl.float32, trans_b=True,
        )
        score = tle.gpu.wgmma_wait(0, score) * scale_log2
        key_rows = tile * BN_VALUE + tl.arange(0, BN_VALUE)
        score = tl.where(
            query_rows[:, None] >= key_rows[None, :],
            score,
            float("-inf"),
        )
        tile_max = tl.max(score, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp2(running_max - next_max)
        probability = tl.exp2(score - next_max[:, None])
        running_sum = running_sum * alpha + tl.sum(probability, axis=1)
        output_acc *= alpha[:, None]
        if IS_BF16:
            tl.store(tle.gpu.local_ptr(p_smem), probability.to(tl.bfloat16))
        else:
            tl.store(tle.gpu.local_ptr(p_smem), probability.to(tl.float16))

        tle.gpu.barrier_wait(v_full[slot], phaseIdx=generation)
        tile_output = tle.gpu.wgmma(
            p_smem, v_stages.slot(slot), out_dtype=tl.float32,
        )
        output_acc += tle.gpu.wgmma_wait(0, tile_output)
        running_max = next_max
        tle.gpu.barrier_arrive(empty[slot], phaseIdx=generation)

    output_acc /= running_sum[:, None]
    output_ptrs = (
        output + query_rows[:, None] * (HQ_VALUE * D_VALUE)
        + head * D_VALUE + dims[None, :]
    )
    lse_ptrs = lse + query_rows * HQ_VALUE + head
    if IS_BF16:
        tl.store(output_ptrs, output_acc.to(tl.bfloat16))
    else:
        tl.store(output_ptrs, output_acc.to(tl.float16))
    tl.store(lse_ptrs, running_max + tl.log2(running_sum))


@triton.jit
def consumer0(output, lse, q0, p0, q0_full, k_stages, v_stages,
              k_full, v_full, empty, query_tile, head,
              scale_log2: tl.constexpr, T_VALUE: tl.constexpr,
              HQ_VALUE: tl.constexpr, D_VALUE: tl.constexpr,
              BM_VALUE: tl.constexpr, BN_VALUE: tl.constexpr,
              CAPACITY_VALUE: tl.constexpr, IS_BF16: tl.constexpr):
    consumer(
        output, lse, q0, p0, q0_full, k_stages, v_stages,
        k_full, v_full, empty, query_tile, head, 0, scale_log2,
        T_VALUE, HQ_VALUE, D_VALUE, BM_VALUE, BN_VALUE, CAPACITY_VALUE,
        IS_BF16,
    )


@triton.jit
def consumer1(output, lse, q1, p1, q1_full, k_stages, v_stages,
              k_full, v_full, empty, query_tile, head,
              scale_log2: tl.constexpr, T_VALUE: tl.constexpr,
              HQ_VALUE: tl.constexpr, D_VALUE: tl.constexpr,
              BM_VALUE: tl.constexpr, BN_VALUE: tl.constexpr,
              CAPACITY_VALUE: tl.constexpr, IS_BF16: tl.constexpr):
    consumer(
        output, lse, q1, p1, q1_full, k_stages, v_stages,
        k_full, v_full, empty, query_tile, head, BM_VALUE, scale_log2,
        T_VALUE, HQ_VALUE, D_VALUE, BM_VALUE, BN_VALUE, CAPACITY_VALUE,
        IS_BF16,
    )


@triton.jit
def a34_attention_kernel(q_cache, k_cache, v, output, lse,
                         scale_log2: tl.constexpr, T_VALUE: tl.constexpr,
                         HQ_VALUE: tl.constexpr, H_VALUE: tl.constexpr,
                         G_VALUE: tl.constexpr,
                         D_VALUE: tl.constexpr, BM_VALUE: tl.constexpr,
                         BN_VALUE: tl.constexpr, CAPACITY_VALUE: tl.constexpr,
                         IS_BF16: tl.constexpr):
    query_tile = tl.program_id(0)
    head = tl.program_id(1)
    tile_count = (query_tile + 1) * 2
    q_desc = tl.make_tensor_descriptor(
        q_cache + head * T_VALUE * D_VALUE,
        shape=[T_VALUE, D_VALUE], strides=[D_VALUE, 1],
        block_shape=[BM_VALUE, D_VALUE],
    )
    k_desc = tl.make_tensor_descriptor(
        k_cache + head * T_VALUE * D_VALUE,
        shape=[T_VALUE, D_VALUE], strides=[D_VALUE, 1],
        block_shape=[BN_VALUE, D_VALUE],
    )
    kv_head = head // G_VALUE
    v_desc = tl.make_tensor_descriptor(
        v + kv_head * D_VALUE,
        shape=[T_VALUE, D_VALUE], strides=[H_VALUE * D_VALUE, 1],
        block_shape=[BN_VALUE, D_VALUE],
    )

    if IS_BF16:
        k_stages = tle.gpu.alloc(
            (CAPACITY_VALUE, BN_VALUE, D_VALUE), dtype=tl.bfloat16,
            layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
        )
        v_stages = tle.gpu.alloc(
            (CAPACITY_VALUE, BN_VALUE, D_VALUE), dtype=tl.bfloat16,
            layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
        )
        q0 = tle.gpu.alloc((BM_VALUE, D_VALUE), dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        q1 = tle.gpu.alloc((BM_VALUE, D_VALUE), dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        p0 = tle.gpu.alloc((BM_VALUE, BN_VALUE), dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        p1 = tle.gpu.alloc((BM_VALUE, BN_VALUE), dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    else:
        k_stages = tle.gpu.alloc(
            (CAPACITY_VALUE, BN_VALUE, D_VALUE), dtype=tl.float16,
            layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
        )
        v_stages = tle.gpu.alloc(
            (CAPACITY_VALUE, BN_VALUE, D_VALUE), dtype=tl.float16,
            layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
        )
        q0 = tle.gpu.alloc((BM_VALUE, D_VALUE), dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        q1 = tle.gpu.alloc((BM_VALUE, D_VALUE), dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        p0 = tle.gpu.alloc((BM_VALUE, BN_VALUE), dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
        p1 = tle.gpu.alloc((BM_VALUE, BN_VALUE), dtype=tl.float16, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True)
    q0_full = tle.gpu.alloc_barrier(expect_bytes=BM_VALUE * D_VALUE * 2)
    q1_full = tle.gpu.alloc_barrier(expect_bytes=BM_VALUE * D_VALUE * 2)
    k_full = tle.gpu.alloc_barriers(
        num_barriers=CAPACITY_VALUE, expect_bytes=BN_VALUE * D_VALUE * 2,
    )
    v_full = tle.gpu.alloc_barriers(
        num_barriers=CAPACITY_VALUE, expect_bytes=BN_VALUE * D_VALUE * 2,
    )
    empty = tle.gpu.alloc_barriers(
        num_barriers=CAPACITY_VALUE, arrive_count=2, init=tle.gpu.READY,
    )
    tle.gpu.warp_specialize(
        [
            (producer, (q_desc, k_desc, v_desc, q0, q1, k_stages, v_stages,
                        q0_full, q1_full, k_full, v_full, empty, query_tile,
                        tile_count, BM_VALUE, BN_VALUE, D_VALUE, CAPACITY_VALUE)),
            (consumer0, (output, lse, q0, p0, q0_full, k_stages, v_stages,
                         k_full, v_full, empty, query_tile, head, scale_log2,
                         T_VALUE, HQ_VALUE, D_VALUE, BM_VALUE, BN_VALUE,
                         CAPACITY_VALUE, IS_BF16)),
            (consumer1, (output, lse, q1, p1, q1_full, k_stages, v_stages,
                         k_full, v_full, empty, query_tile, head, scale_log2,
                         T_VALUE, HQ_VALUE, D_VALUE, BM_VALUE, BN_VALUE,
                         CAPACITY_VALUE, IS_BF16)),
        ],
        [4, 4],
        [224, 224],
    )


def launch(q_cache, k_cache, v, output, lse, capacity: int, scale: float):
    b, hq, t, d = q_cache.shape
    h = v.shape[2]
    if b != 1 or t % BT != 0:
        raise ValueError("optimized attention requires B=1 and T divisible by 128")
    return a34_attention_kernel[(t // BT, hq)](
        q_cache, k_cache, v, output, lse,
        scale_log2=float(scale) * RCP_LN2,
        T_VALUE=t, HQ_VALUE=hq, H_VALUE=h, G_VALUE=hq // h, D_VALUE=d,
        BM_VALUE=BM, BN_VALUE=BN, CAPACITY_VALUE=capacity,
        IS_BF16=q_cache.dtype == torch.bfloat16,
        num_warps=4, num_stages=1,
    )


__all__ = ["allocator", "launch", "a34_attention_kernel"]
