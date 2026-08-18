"""Final SM90 FP8 decode compute kernel.

This production snapshot fixes the validated direct NHD/HND, log2-domain,
fused LDSM/PRMT/register-source WGMMA path. The measured cluster2, cluster4,
and cluster8 merge specializations remain selectable.
"""

from pathlib import Path
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
import triton.experimental.tle.language.raw as tle_raw
from triton.experimental.tle.raw import dialect
TASK_STRIDE = 12
TASK_SLOTS = 2
TILE_N = 64
ROWS_Q = 8
DIRECT_MODE = 0
DUMMY_MODE = 2
_TASK_STRIDE_JIT = tl.constexpr(TASK_STRIDE)
_TASK_SLOTS_JIT = tl.constexpr(TASK_SLOTS)
_ROWS_Q_JIT = tl.constexpr(ROWS_Q)
_TMA_STAGES_JIT = tl.constexpr(2)
_DIRECT_MODE_JIT = tl.constexpr(DIRECT_MODE)
_DUMMY_MODE_JIT = tl.constexpr(DUMMY_MODE)
_K_FRAGMENT_JIT = tl.constexpr(32)

@triton.jit
def _fused_ldmatrix_pv_wgmma(acc, p_smem_ptr, v_smem_ptr):
    """Fuse V LDSM/PRMT and register-source PV WGMMA.

    ``acc`` already contains the online-softmax alpha rescale.  PV is accumulated
    from zero in a separate eight-register WGMMA result and merged into ``acc``
    with FP32 adds after the wait.  This matches the CUDA kernel and avoids
    assuming that Triton's tensor accumulator order is identical to the raw
    RS-WGMMA output constraint order.

    No joined LDSM tensor is exposed to Triton's layout propagation, so it
    cannot materialize a shared-memory round trip between LDSM and WGMMA.

    The scalar shared pointers are broadcast over ``acc``.  With ``pack=8``
    each asm invocation receives the eight FP32 accumulator registers owned by
    one Hopper WGMMA thread, followed by eight identical P pointers and eight
    identical V pointers.
    """
    return tl.inline_asm_elementwise(asm="\n        {\n            .reg .u32 tid, lane, warp, dbase, row, col, off;\n            .reg .u64 voff, vaddr, pdesc0, pdesc1;\n            .reg .b32 raw<4>, hold<4>, a<4>;\n            .reg .pred accumulate;\n\n            // All warps must finish publishing the P tile before WGMMA reads\n            // it.  Place the sole CTA barrier before LDSM so the fused\n            // LDSM -> PRMT -> WGMMA sequence itself remains uninterrupted.\n            bar.sync 0;\n\n            // CUTE's V TiledCopy source layout gives each warp sixteen\n            // adjacent D bytes while the lane selects N.  The CUDA kernel's\n            // exact LDSM order is (D0,N0), (D64,N0), (D0,N32), (D64,N32).\n            // RS WGMMA consumes those fragments in first, third, second,\n            // fourth order: K advances before the output-D half advances.\n            // Hopper's 128-byte TMA swizzle is Swizzle<3,4,3>.  Match the\n            // CUDA SASS literally: form the complete shared address first,\n            // then apply addr ^= (addr & 0x380) >> 3.  Applying the XOR only\n            // to a logical row offset incorrectly assumes that every V slot\n            // has zero address bits 7:9.\n            // The target CUDA SASS first reduces threadIdx.x modulo 128 and\n            // then forms R60 = (tid % 32) * 128 + (tid / 32) * 16.  Keep the\n            // physical CUDA lane/warp split here; the CUTE TiledCopy layout\n            // is already reflected in that source-address expression.\n            mov.u32 tid, %tid.x;\n            and.b32 lane, tid, 31;\n            shr.u32 warp, tid, 5;\n            shl.b32 dbase, warp, 4;\n            shl.b32 row, lane, 7;\n\n            // Build P descriptors before the first V fragment so each\n            // completed LDSM/PRMT group can immediately launch its async\n            // WGMMA.  This shortens A-register live ranges and overlaps the\n            // following fragment conversion with tensor-core execution.\n            shr.u64 pdesc0, $16, 4;\n            add.u64 pdesc0, pdesc0, 0x8000000000000000;\n            add.u64 pdesc1, pdesc0, 2;\n            setp.ne.u32 accumulate, tid, 0xffffffff;\n\n            add.u32 off, row, dbase;\n            cvt.u64.u32 voff, off;\n            add.u64 vaddr, $24, voff;\n            and.b64 voff, vaddr, 0x380;\n            shr.u64 voff, voff, 3;\n            xor.b64 vaddr, vaddr, voff;\n            ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16\n                {raw0, raw1, raw2, raw3}, [vaddr];\n            prmt.b32 a0, raw0, raw1, 0x6240;\n            prmt.b32 a1, raw0, raw1, 0x7351;\n            prmt.b32 a2, raw2, raw3, 0x6240;\n            prmt.b32 a3, raw2, raw3, 0x7351;\n            // Publish the freshly written RS operand registers to the\n            // warpgroup MMA async proxy.  The CUDA kernel emits one\n            // WARPGROUP.ARRIVE between every PRMT group and QGMMA.\n            wgmma.fence.sync.aligned;\n            wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n                {$0, $1, $2, $3}, {a0, a1, a2, a3}, pdesc0, 0, 1, 1;\n\n            // Preserve the D64/N0 fragment while D0/N32 is issued first.\n            add.u32 col, dbase, 64;\n            add.u32 off, row, col;\n            cvt.u64.u32 voff, off;\n            add.u64 vaddr, $24, voff;\n            and.b64 voff, vaddr, 0x380;\n            shr.u64 voff, voff, 3;\n            xor.b64 vaddr, vaddr, voff;\n            ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16\n                {hold0, hold1, hold2, hold3}, [vaddr];\n\n            add.u32 row, row, 4096;\n            mov.u32 col, dbase;\n            add.u32 off, row, col;\n            cvt.u64.u32 voff, off;\n            add.u64 vaddr, $24, voff;\n            and.b64 voff, vaddr, 0x380;\n            shr.u64 voff, voff, 3;\n            xor.b64 vaddr, vaddr, voff;\n            ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16\n                {raw0, raw1, raw2, raw3}, [vaddr];\n            prmt.b32 a0, raw0, raw1, 0x6240;\n            prmt.b32 a1, raw0, raw1, 0x7351;\n            prmt.b32 a2, raw2, raw3, 0x6240;\n            prmt.b32 a3, raw2, raw3, 0x7351;\n            wgmma.fence.sync.aligned;\n            wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n                {$0, $1, $2, $3}, {a0, a1, a2, a3}, pdesc1, accumulate, 1, 1;\n\n            prmt.b32 a0, hold0, hold1, 0x6240;\n            prmt.b32 a1, hold0, hold1, 0x7351;\n            prmt.b32 a2, hold2, hold3, 0x6240;\n            prmt.b32 a3, hold2, hold3, 0x7351;\n            wgmma.fence.sync.aligned;\n            wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n                {$4, $5, $6, $7}, {a0, a1, a2, a3}, pdesc0, 0, 1, 1;\n\n            add.u32 col, dbase, 64;\n            add.u32 off, row, col;\n            cvt.u64.u32 voff, off;\n            add.u64 vaddr, $24, voff;\n            and.b64 voff, vaddr, 0x380;\n            shr.u64 voff, voff, 3;\n            xor.b64 vaddr, vaddr, voff;\n            ldmatrix.sync.aligned.m8n8.x4.trans.shared::cta.b16\n                {raw0, raw1, raw2, raw3}, [vaddr];\n            prmt.b32 a0, raw0, raw1, 0x6240;\n            prmt.b32 a1, raw0, raw1, 0x7351;\n            prmt.b32 a2, raw2, raw3, 0x6240;\n            prmt.b32 a3, raw2, raw3, 0x7351;\n            wgmma.fence.sync.aligned;\n            wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n                {$4, $5, $6, $7}, {a0, a1, a2, a3}, pdesc1, accumulate, 1, 1;\n\n            wgmma.commit_group.sync.aligned;\n            wgmma.wait_group.sync.aligned 0;\n\n            // CUDA keeps the online-softmax accumulator separate from the\n            // transient PV WGMMA result, then merges it after DEPBAR.LE.\n            add.f32 $0, $0, $8;\n            add.f32 $1, $1, $9;\n            add.f32 $2, $2, $10;\n            add.f32 $3, $3, $11;\n            add.f32 $4, $4, $12;\n            add.f32 $5, $5, $13;\n            add.f32 $6, $6, $14;\n            add.f32 $7, $7, $15;\n        }\n        ", constraints='=&f,=&f,=&f,=&f,=&f,=&f,=&f,=&f,f,f,f,f,f,f,f,f,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l,l', args=[acc, p_smem_ptr.to(tl.uint64), v_smem_ptr.to(tl.uint64)], dtype=tl.float32, is_pure=False, pack=8)

@dialect(name='cuda', file=Path(__file__).parent / 'dynamic_splitk_finalize.cu')
def _raw_cuda_dynamic_splitk_finalize(*args, **kwargs):
    ...

@triton.jit
def fp8_kvpertensor_decode_kernel(Q, K_DESC, VT_DESC, BLOCK_IDS, TASK_MAP, QSCALE, KSCALE, VSCALE, SPLIT_OUT, LSE, COMPLETION, LAST_FLAGS, OUT, mesh: tl.constexpr, B: tl.constexpr, H_Q: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, D: tl.constexpr, DV: tl.constexpr, BLOCK_SIZE: tl.constexpr, MAX_BLOCKS: tl.constexpr, BLOCK_N: tl.constexpr, Q_STRIDE_B: tl.constexpr, Q_STRIDE_H: tl.constexpr, QS_STRIDE_B: tl.constexpr, QS_STRIDE_H: tl.constexpr, SO_STRIDE_B: tl.constexpr, SO_STRIDE_C: tl.constexpr, SO_STRIDE_M: tl.constexpr, SO_STRIDE_H: tl.constexpr, LSE_STRIDE_B: tl.constexpr, LSE_STRIDE_C: tl.constexpr, LSE_STRIDE_HKV: tl.constexpr, LSE_STRIDE_M: tl.constexpr, LSE_STRIDE_HG: tl.constexpr, O_STRIDE_B: tl.constexpr, O_STRIDE_M: tl.constexpr, O_STRIDE_H: tl.constexpr, CLUSTER_SIZE: tl.constexpr):
    cta = tl.program_id(0)
    cluster_rank = tle.shard_id(mesh, 'cluster_x')
    task_base = (cta * _TASK_SLOTS_JIT + 1) * _TASK_STRIDE_JIT
    hkv = tl.load(TASK_MAP + task_base + 0)
    batch = tl.load(TASK_MAP + task_base + 1)
    if hkv < 0:
        return
    seq_start = tl.load(TASK_MAP + task_base + 3)
    seq_len = tl.load(TASK_MAP + task_base + 4)
    seq_kvcache = tl.load(TASK_MAP + task_base + 5)
    is_causal = tl.load(TASK_MAP + task_base + 8)
    task_mode = tl.load(TASK_MAP + task_base + 9)
    group_chunk = tl.load(TASK_MAP + task_base + 10)
    group_count = tl.load(TASK_MAP + task_base + 11)
    has_work = task_mode != _DUMMY_MODE_JIT
    q_smem = tle.gpu.alloc([_ROWS_Q_JIT, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    p_smem = tle.gpu.alloc([_ROWS_Q_JIT, BLOCK_N], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_raw_smem = tle.gpu.alloc([_TMA_STAGES_JIT, 1, BLOCK_N, 1, D], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    v_raw_smem = tle.gpu.alloc([_TMA_STAGES_JIT, 1, BLOCK_N, 1, DV], dtype=tl.float8e4nv, layout=None, scope=tle.gpu.smem)
    k_full = tle.gpu.alloc_barriers(num_barriers=_TMA_STAGES_JIT, arrive_count=1, expect_bytes=BLOCK_N * D)
    vt_full = tle.gpu.alloc_barriers(num_barriers=_TMA_STAGES_JIT, arrive_count=1, expect_bytes=DV * BLOCK_N)
    if CLUSTER_SIZE == 2:
        peer_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    elif CLUSTER_SIZE == 4:
        peer1_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer2_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer3_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer1_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer2_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        peer3_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    else:
        partial_acc_smem = tle.gpu.alloc([DV, _ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
        partial_lse_smem = tle.gpu.alloc([_ROWS_Q_JIT], dtype=tl.float32, layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=False)
    offs_q = tl.arange(0, _ROWS_Q_JIT)
    offs_d = tl.arange(0, D)
    offs_v = tl.arange(0, DV)
    offs_n = tl.arange(0, BLOCK_N)
    q_rows = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[:, None], (_ROWS_Q_JIT, D))
    q_cols = tl.broadcast_to(tl.arange(0, D)[None, :], (_ROWS_Q_JIT, D))
    p_rows = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[:, None], (_ROWS_Q_JIT, BLOCK_N))
    p_cols = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (_ROWS_Q_JIT, BLOCK_N))
    acc_rows = tl.broadcast_to(tl.arange(0, DV)[:, None], (DV, _ROWS_Q_JIT))
    acc_cols = tl.broadcast_to(tl.arange(0, _ROWS_Q_JIT)[None, :], (DV, _ROWS_Q_JIT))
    store_offs_v = (offs_v & -16) + ((offs_v & 7) << 1) + (offs_v >> 3 & 1)
    q_smem_ptr = tle.gpu.local_ptr(q_smem, (q_rows, q_cols))
    p_smem_ptr = tle.gpu.local_ptr(p_smem, (p_rows, p_cols))
    if CLUSTER_SIZE == 8:
        partial_acc_ptr = tle.gpu.local_ptr(partial_acc_smem, (acc_rows, acc_cols))
        partial_lse_ptr = tle.gpu.local_ptr(partial_lse_smem, (offs_q,))
    inv_sqrt_d = tl.rsqrt(tl.full((), D, tl.float32))
    kscale = tl.load(KSCALE + 0).to(tl.float32)
    vscale = tl.load(VSCALE + 0).to(tl.float32) / 256.0
    hq = hkv * HEADS_PER_GROUP + offs_q
    valid_q = has_work & (offs_q < HEADS_PER_GROUP) & (hq < H_Q)
    acc = tl.zeros((DV, _ROWS_Q_JIT), tl.float32)
    lse = tl.full((_ROWS_Q_JIT,), -float('inf'), tl.float32)
    if has_work:
        q = tl.load(Q + batch * Q_STRIDE_B + hq[:, None] * Q_STRIDE_H + offs_d[None, :], mask=valid_q[:, None], other=0.0)
        tl.store(q_smem_ptr, q)
        qscale = tl.load(QSCALE + batch * QS_STRIDE_B + hq * QS_STRIDE_H, mask=valid_q, other=1.0).to(tl.float32)
        m_i = tl.full((_ROWS_Q_JIT,), -float('inf'), tl.float32)
        l_i = tl.zeros((_ROWS_Q_JIT,), tl.float32)
        copy_iter = 0
        start = 0
        if start < seq_len:
            block_no = seq_start // BLOCK_SIZE
            phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
            tle.gpu.copy(K_DESC, k_raw_smem.slot(0), [1, BLOCK_N, 1, D], [phys, 0, hkv, 0], barrier=k_full[0])
            tle.gpu.copy(VT_DESC, v_raw_smem.slot(0), [1, BLOCK_N, 1, DV], [phys, 0, hkv, 0], barrier=vt_full[0])
        while start < seq_len:
            local_n = start + offs_n
            valid_cols = local_n < seq_len
            buf = copy_iter % _TMA_STAGES_JIT
            phase = copy_iter // _TMA_STAGES_JIT & 1
            next_start = start + BLOCK_N
            if next_start < seq_len:
                next_iter = copy_iter + 1
                next_buf = next_iter % _TMA_STAGES_JIT
                aligned_logical = seq_start + next_start
                block_no = aligned_logical // BLOCK_SIZE
                phys = tl.load(BLOCK_IDS + batch * MAX_BLOCKS + block_no)
                tle.gpu.copy(K_DESC, k_raw_smem.slot(next_buf), [1, BLOCK_N, 1, D], [phys, 0, hkv, 0], barrier=k_full[next_buf])
                tle.gpu.copy(VT_DESC, v_raw_smem.slot(next_buf), [1, BLOCK_N, 1, DV], [phys, 0, hkv, 0], barrier=vt_full[next_buf])
            tle.gpu.barrier_wait(k_full[buf], phaseIdx=phase)
            k_page = tle.gpu.reshape_wgmma_smem_operand(k_raw_smem.slot(buf), [BLOCK_N, D])
            scores = tle.gpu.wgmma(k_page, q_smem, trans_b=True, out_dtype=tl.float32)
            scores = tle.gpu.wgmma_wait(0, scores)
            scores = scores * (inv_sqrt_d * kscale * 1.4426950408889634)
            scores = scores * qscale[None, :]
            causal = (is_causal == 0) | (local_n[:, None] < seq_kvcache + 1)
            scores = tl.where(valid_cols[:, None] & causal & valid_q[None, :], scores, -float('inf'))
            m_tile = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, m_tile)
            valid_update = m_new != -float('inf')
            safe_m_new = tl.where(valid_update, m_new, 0.0)
            safe_m_i = tl.where(m_i == -float('inf'), safe_m_new, m_i)
            p = tl.exp2(scores - safe_m_new[None, :])
            p = tl.where(valid_update[None, :], p, 0.0)
            alpha = tl.exp2(safe_m_i - safe_m_new)
            alpha = tl.where(valid_update, alpha, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=0)
            p_scaled_t = tl.trans((p * 256.0).to(tl.float8e4nv))
            p_scaled_t = tl.reshape(p_scaled_t, (_ROWS_Q_JIT, BLOCK_N // 16, 2, 8))
            p_scaled_t = tl.permute(p_scaled_t, (0, 1, 3, 2))
            p_scaled_t = tl.reshape(p_scaled_t, (_ROWS_Q_JIT, BLOCK_N))
            tl.store(p_smem_ptr, p_scaled_t)
            tle.gpu.barrier_wait(vt_full[buf], phaseIdx=phase)
            v_page = tle.gpu.reshape_wgmma_smem_operand(v_raw_smem.slot(buf), [BLOCK_N, DV])
            p_base = tle.gpu.local_ptr(p_smem, (0, 0))
            v_base = tle.gpu.local_ptr(v_page, (0, 0))
            acc = _fused_ldmatrix_pv_wgmma(acc * alpha[None, :], p_base, v_base)
            m_i = m_new
            l_i = l_new
            start = next_start
            copy_iter += 1
        has_value = l_i > 0.0
        acc = tl.where(has_value[None, :], acc / l_i[None, :] * vscale, 0.0)
        lse = tl.where(has_value, tl.log2(l_i) + m_i, -float('inf'))
    if task_mode == _DIRECT_MODE_JIT:
        tl.store(OUT + batch * O_STRIDE_B + hq[None, :] * O_STRIDE_H + store_offs_v[:, None], acc, mask=valid_q[None, :])
        return
    group_acc = acc
    group_lse = lse
    if CLUSTER_SIZE == 2:
        peer_acc_remote = tle.remote(peer_acc_smem, 0, scope=mesh)
        peer_lse_remote = tle.remote(peer_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            peer_acc = tl.load(tle.gpu.local_ptr(peer_acc_smem, (acc_rows, acc_cols)))
            peer_lse = tl.load(tle.gpu.local_ptr(peer_lse_smem, (offs_q,)))
            max_lse = tl.maximum(lse, peer_lse)
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            own_weight = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            peer_weight = tl.where(peer_lse != -float('inf'), tl.exp2(peer_lse - safe_max), 0.0)
            denom = own_weight + peer_weight
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            group_acc = (acc * own_weight[None, :] + peer_acc * peer_weight[None, :]) / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
    elif CLUSTER_SIZE == 4:
        peer1_acc_remote = tle.remote(peer1_acc_smem, 0, scope=mesh)
        peer2_acc_remote = tle.remote(peer2_acc_smem, 0, scope=mesh)
        peer3_acc_remote = tle.remote(peer3_acc_smem, 0, scope=mesh)
        peer1_lse_remote = tle.remote(peer1_lse_smem, 0, scope=mesh)
        peer2_lse_remote = tle.remote(peer2_lse_smem, 0, scope=mesh)
        peer3_lse_remote = tle.remote(peer3_lse_smem, 0, scope=mesh)
        if cluster_rank == 1:
            tl.store(tle.gpu.local_ptr(peer1_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer1_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 2:
            tl.store(tle.gpu.local_ptr(peer2_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer2_lse_remote, (offs_q,)), lse)
        elif cluster_rank == 3:
            tl.store(tle.gpu.local_ptr(peer3_acc_remote, (acc_rows, acc_cols)), acc)
            tl.store(tle.gpu.local_ptr(peer3_lse_remote, (offs_q,)), lse)
        tle.distributed_barrier(mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse_smem, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse_smem, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse_smem, (offs_q,)))
            max_lse = tl.maximum(tl.maximum(lse, lse1), tl.maximum(lse2, lse3))
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float('inf'), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float('inf'), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float('inf'), tl.exp2(lse3 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer1_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight1[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer2_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight2[None, :]
            peer_acc = tl.load(tle.gpu.local_ptr(peer3_acc_smem, (acc_rows, acc_cols)))
            weighted_acc += peer_acc * weight3[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
    else:
        tl.store(partial_acc_ptr, acc)
        tl.store(partial_lse_ptr, lse)
        tle.distributed_barrier(mesh)
        peer1_acc = tle.remote(partial_acc_smem, 1, scope=mesh)
        peer2_acc = tle.remote(partial_acc_smem, 2, scope=mesh)
        peer3_acc = tle.remote(partial_acc_smem, 3, scope=mesh)
        peer4_acc = tle.remote(partial_acc_smem, 4, scope=mesh)
        peer5_acc = tle.remote(partial_acc_smem, 5, scope=mesh)
        peer6_acc = tle.remote(partial_acc_smem, 6, scope=mesh)
        peer7_acc = tle.remote(partial_acc_smem, 7, scope=mesh)
        peer1_lse = tle.remote(partial_lse_smem, 1, scope=mesh)
        peer2_lse = tle.remote(partial_lse_smem, 2, scope=mesh)
        peer3_lse = tle.remote(partial_lse_smem, 3, scope=mesh)
        peer4_lse = tle.remote(partial_lse_smem, 4, scope=mesh)
        peer5_lse = tle.remote(partial_lse_smem, 5, scope=mesh)
        peer6_lse = tle.remote(partial_lse_smem, 6, scope=mesh)
        peer7_lse = tle.remote(partial_lse_smem, 7, scope=mesh)
        if cluster_rank == 0:
            lse1 = tl.load(tle.gpu.local_ptr(peer1_lse, (offs_q,)))
            lse2 = tl.load(tle.gpu.local_ptr(peer2_lse, (offs_q,)))
            lse3 = tl.load(tle.gpu.local_ptr(peer3_lse, (offs_q,)))
            lse4 = tl.load(tle.gpu.local_ptr(peer4_lse, (offs_q,)))
            lse5 = tl.load(tle.gpu.local_ptr(peer5_lse, (offs_q,)))
            lse6 = tl.load(tle.gpu.local_ptr(peer6_lse, (offs_q,)))
            lse7 = tl.load(tle.gpu.local_ptr(peer7_lse, (offs_q,)))
            max_lse = tl.maximum(lse, lse1)
            max_lse = tl.maximum(max_lse, lse2)
            max_lse = tl.maximum(max_lse, lse3)
            max_lse = tl.maximum(max_lse, lse4)
            max_lse = tl.maximum(max_lse, lse5)
            max_lse = tl.maximum(max_lse, lse6)
            max_lse = tl.maximum(max_lse, lse7)
            valid_group = max_lse != -float('inf')
            safe_max = tl.where(valid_group, max_lse, 0.0)
            weight0 = tl.where(lse != -float('inf'), tl.exp2(lse - safe_max), 0.0)
            weight1 = tl.where(lse1 != -float('inf'), tl.exp2(lse1 - safe_max), 0.0)
            weight2 = tl.where(lse2 != -float('inf'), tl.exp2(lse2 - safe_max), 0.0)
            weight3 = tl.where(lse3 != -float('inf'), tl.exp2(lse3 - safe_max), 0.0)
            weight4 = tl.where(lse4 != -float('inf'), tl.exp2(lse4 - safe_max), 0.0)
            weight5 = tl.where(lse5 != -float('inf'), tl.exp2(lse5 - safe_max), 0.0)
            weight6 = tl.where(lse6 != -float('inf'), tl.exp2(lse6 - safe_max), 0.0)
            weight7 = tl.where(lse7 != -float('inf'), tl.exp2(lse7 - safe_max), 0.0)
            denom = weight0 + weight1 + weight2 + weight3 + weight4 + weight5 + weight6 + weight7
            safe_denom = tl.where(denom > 0.0, denom, 1.0)
            weighted_acc = acc * weight0[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer1_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight1[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer2_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight2[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer3_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight3[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer4_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight4[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer5_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight5[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer6_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight6[None, :]
            peer_acc_value = tl.load(tle.gpu.local_ptr(peer7_acc, (acc_rows, acc_cols)))
            weighted_acc += peer_acc_value * weight7[None, :]
            group_acc = weighted_acc / safe_denom[None, :]
            group_acc = tl.where(valid_group[None, :], group_acc, 0.0)
            group_lse = tl.where(valid_group, tl.log2(safe_denom) + safe_max, -float('inf'))
        tle.distributed_barrier(mesh)
    if cluster_rank == 0:
        output_mask = (offs_q < HEADS_PER_GROUP)[None, :] & (hq < H_Q)[None, :]
        if group_count == 1:
            tl.store(OUT + batch * O_STRIDE_B + hq[None, :] * O_STRIDE_H + store_offs_v[:, None], group_acc, mask=output_mask)
        else:
            tl.store(SPLIT_OUT + batch * SO_STRIDE_B + group_chunk * SO_STRIDE_C + hq[None, :] * SO_STRIDE_H + store_offs_v[:, None], group_acc, mask=output_mask)
            tl.store(LSE + batch * LSE_STRIDE_B + group_chunk * LSE_STRIDE_C + hkv * LSE_STRIDE_HKV + offs_q * LSE_STRIDE_HG, group_lse, mask=(offs_q < HEADS_PER_GROUP) & (hq < H_Q))
            tle_raw.call(_raw_cuda_dynamic_splitk_finalize, [SPLIT_OUT, LSE, COMPLETION, LAST_FLAGS, OUT, hkv, batch, group_count, tl.full((), B, tl.int32), tl.full((), H_Q, tl.int32), tl.full((), HEADS_PER_GROUP, tl.int32), tl.full((), DV, tl.int32), tl.full((), SO_STRIDE_B, tl.int64), tl.full((), SO_STRIDE_C, tl.int64), tl.full((), SO_STRIDE_M, tl.int64), tl.full((), SO_STRIDE_H, tl.int64), tl.full((), LSE_STRIDE_B, tl.int64), tl.full((), LSE_STRIDE_C, tl.int64), tl.full((), LSE_STRIDE_HKV, tl.int64), tl.full((), LSE_STRIDE_M, tl.int64), tl.full((), LSE_STRIDE_HG, tl.int64), tl.full((), O_STRIDE_B, tl.int64), tl.full((), O_STRIDE_M, tl.int64), tl.full((), O_STRIDE_H, tl.int64)])

__all__ = ["fp8_kvpertensor_decode_kernel"]
