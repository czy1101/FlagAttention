import triton
import triton.language as tl
import triton.experimental.tle.language as tle

@triton.jit
def _tle_h100_kernel(QDESC,KDESC,VDESC,LOG_LAMBDA,START_INDEX,O,sm_scale,SLOTS:tl.constexpr,
                        M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,BN:tl.constexpr,
                        BM:tl.constexpr,STATIC:tl.constexpr):
    BLOCK_M:tl.constexpr=BM
    BLOCK_N:tl.constexpr=BN
    BLOCK_DMODEL:tl.constexpr=D
    STORE_L:tl.constexpr=False
    stride_log_lambda_z:tl.constexpr=H*N
    stride_log_lambda_h:tl.constexpr=N
    stride_log_lambda_n:tl.constexpr=1
    stride_start_index_z:tl.constexpr=H*tl.cdiv(M,128)
    stride_start_index_h:tl.constexpr=tl.cdiv(M,128)
    stride_start_index_mb:tl.constexpr=1
    stride_oz:tl.constexpr=M*H*D
    stride_oh:tl.constexpr=D
    stride_om:tl.constexpr=H*D
    stride_ok:tl.constexpr=1
    # General packed causal inference. ACP remains grouped by 128 Q.
    dtype = O.dtype.element_ty
    h = tl.program_id(0)
    hk = h // (H // HK)
    b = tl.program_id(1)
    mblock = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e
    qbuf = tle.gpu.alloc([BLOCK_M, BLOCK_DMODEL], dtype)
    kbuf = tle.gpu.alloc([SLOTS, BLOCK_N, BLOCK_DMODEL], dtype)
    vbuf = tle.gpu.alloc([SLOTS, BLOCK_N, BLOCK_DMODEL], dtype)
    qbar = tle.gpu.alloc_barrier(expect_bytes=BLOCK_M * BLOCK_DMODEL * 2)
    kbars = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)
    vbars = tle.gpu.alloc_barriers(SLOTS, expect_bytes=BLOCK_N * BLOCK_DMODEL * 2)

    tle.gpu.copy(QDESC, qbuf, [BLOCK_M, BLOCK_DMODEL],
                 [b * M + mblock * BLOCK_M, h * BLOCK_DMODEL], barrier=qbar)
    lo = tl.load(START_INDEX + b * stride_start_index_z + h * stride_start_index_h
                 + (mblock * BLOCK_M // 128) * stride_start_index_mb)
    lo = (lo // BLOCK_N) * BLOCK_N
    hi = tl.minimum(N, N - M + (mblock + 1) * BLOCK_M)
    if lo < hi:
        tle.gpu.copy(KDESC, kbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                     [b * N + lo, hk * BLOCK_DMODEL], barrier=kbars[0])
        tle.gpu.copy(VDESC, vbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                     [b * N + lo, hk * BLOCK_DMODEL], barrier=vbars[0])

    rm = mblock * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rd = tl.arange(0, BLOCK_DMODEL)
    prefix = LOG_LAMBDA + b * stride_log_lambda_z + h * stride_log_lambda_h
    if STATIC and M % BM == 0:
        gq = tl.load(prefix + N - M + rm, cache_modifier=".cg")
    else:
        gq = tl.load(prefix + N - M + rm, rm < M, other=0, cache_modifier=".cg")
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)
    diagonal_start = ((N - M + mblock * BLOCK_M + 1) // BLOCK_N) * BLOCK_N
    tle.gpu.barrier_wait(qbar, phaseIdx=0)

    for start_n in range(lo, hi, BLOCK_N):
        it = (start_n - lo) // BLOCK_N
        slot = it % SLOTS
        phase = it // SLOTS
        next_n = start_n + BLOCK_N
        tle.gpu.barrier_wait(kbars[slot], phaseIdx=phase)
        qk_async = tle.gpu.wgmma(qbuf, kbuf.slot(slot), trans_b=True, input_precision="ieee")
        # The next slot was released by the previous iteration's PV wait.
        if SLOTS == 2:
            if next_n < hi:
                nxt = (it + 1) % SLOTS
                tle.gpu.copy(KDESC, kbuf.slot(nxt), [BLOCK_N, BLOCK_DMODEL],
                             [b * N + next_n, hk * BLOCK_DMODEL], barrier=kbars[nxt])
                tle.gpu.copy(VDESC, vbuf.slot(nxt), [BLOCK_N, BLOCK_DMODEL],
                             [b * N + next_n, hk * BLOCK_DMODEL], barrier=vbars[nxt])
        if STATIC and N % BN == 0:
            gk = tl.load(prefix + start_n + rn, cache_modifier=".cg")
        else:
            gk = tl.load(prefix + start_n + rn, start_n + rn < N, other=0, cache_modifier=".cg")
        s = tle.gpu.wgmma_wait(0, qk_async) * qk_scale
        decay_bias = gq[:, None] - gk[None, :]
        # Upstream PTX fuses the decay product, not the QK scaling product.
        s = tl.fma(decay_bias, log2e, s)
        if not STATIC or N % BN != 0:
            s = tl.where((start_n + rn)[None, :] < N, s, -float("inf"))
        if start_n >= diagonal_start:
            s = tl.where((N - M + rm[:, None]) >= (start_n + rn)[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        p_sum = tl.sum(p, 1)
        tle.gpu.barrier_wait(vbars[slot], phaseIdx=phase)
        # Match upstream TTGIR: rescaled old acc is the WGMMA accumulator.
        acc *= alpha[:, None]
        pv_async = tle.gpu.wgmma(p.to(dtype), vbuf.slot(slot), acc=acc, input_precision="ieee")
        l_i = l_i * alpha + p_sum
        m_i = m_new
        acc = tle.gpu.wgmma_wait(0, pv_async)
        if SLOTS == 1:
            if next_n < hi:
                tle.gpu.copy(KDESC, kbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                             [b * N + next_n, hk * BLOCK_DMODEL], barrier=kbars[0])
                tle.gpu.copy(VDESC, vbuf.slot(0), [BLOCK_N, BLOCK_DMODEL],
                             [b * N + next_n, hk * BLOCK_DMODEL], barrier=vbars[0])

    acc = acc * (1.0 / l_i[:, None])
    op = O + b * stride_oz + h * stride_oh + rm[:, None] * stride_om + rd[None, :] * stride_ok
    if STATIC and M % BM == 0:
        tl.store(op, acc.to(dtype), cache_modifier=".cg")
    else:
        tl.store(op, acc.to(dtype), rm[:, None] < M, cache_modifier=".cg")
