"""
Implementation of Forgetting Attention.

Our code is adapted from https://github.com/FlagOpen/FlagAttention/blob/ee91638dec6da8c00c4113d179f469e0ffcd5852/src/flag_attn/flash.py. The code is modified to implement Forgetting Attention.

The original license info from FlagAttention:

Copyright 2023 BAAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, PRUNE_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, REAL_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    log2e: tl.constexpr = 1.4426950408889634
    loge2: tl.constexpr = 0.6931471805599453
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    O += off_z * stride_oz + off_h * stride_oh
    L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)


    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        # .ca permits 4/8-byte async loads too; .cg requires 16-byte copies.
        q = tl.load(q_ptrs, mask=offs_k[None, :] < REAL_D, other=0, cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".ca")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < REAL_D), other=0, cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".ca")

    #Dot I trick: to place q in registers, it saves shared memory
    # if BLOCK_DMODEL < 128:
    #     I = tl.where(offs_k[:, None] == offs_k,
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_DMODEL, BLOCK_DMODEL), 0.0, dtype=input_dtype))
    #     q = tl.dot(q, I, input_precision="ieee").to(input_dtype)
    # else:
    #     I = tl.where(offs_m_base[:, None] == offs_m_base,
    #                  tl.full((BLOCK_M, BLOCK_M), 1.0, dtype=input_dtype),
    #                  tl.full((BLOCK_M, BLOCK_M), 0.0, dtype=input_dtype))
    #     q = tl.dot(I, q, input_precision="ieee").to(input_dtype)

    # NOTE: Loop-Bound-For-N
    # The indices in m-dimension that this block may access is in `[start_m * BLOCK_M, (start_m + 1) * BLOCK_M)`.
    # According to the rule of causal masking, then max index in n-dimension that this block may access
    # is `P_SEQ + (start_m + 1) * BLOCK_M`.
    # However, the upper bound of index in n-dimension should never exceed the sequence length of k/v(`P_SEQ + N_CTX`).
    # `P_SEQ + (start_m + 1) * BLOCK_M` may be larger than `N`.
    # At this case, there would be illegal memory access when loading k & v tiles
    # if mask_n is not applied for loading(only when `DIVISIBLE_N`` is true).
    # See also https://github.com/FlagOpen/FlagAttention/pull/8
    if IS_CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    offs_n_init = offs_n_base

    if HAS_SEQ_START:
        SEQ_START += off_z
        seq_start = tl.load(SEQ_START)
        lo = tl.minimum(seq_start, hi)
    else:
        lo = 0
        seq_start = 0

    if IS_ADAPTIVE:
        # No need to multiple start_m by BLOCK_M here
        START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h + ((start_m * BLOCK_M) // PRUNE_M) * stride_start_index_mb
        start_index = tl.load(START_INDEX)
        lo = tl.maximum(start_index, lo)
    lo = (lo // BLOCK_N) * BLOCK_N
    offs_n_init += lo

    # loop over k, v and update accumulators
    k_ptrs = K + (offs_k[:, None] * stride_kk + offs_n_init[None, :] * stride_kn) # (BLOCK_DMODEL, BLOCK_N)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk) # (BLOCK_N, BLOCK_DMODEL)
    log_lambda_in_ptrs = LOG_LAMBDA + (offs_n_init * stride_log_lambda_n) # (BLOCK_N, BLOCK_DMODEL)
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + offs_n_base

        # -- load k, v --
        if DIVISIBLE_N:
            k = tl.load(k_ptrs, mask=offs_k[:, None] < REAL_D, other=0, cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=offs_k[None, :] < REAL_D, other=0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".ca")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :] & (offs_k[:, None] < REAL_D), other=0, cache_modifier=".ca")
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < REAL_D), other=0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".ca")

        # -- compute qk ---
        # s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if BLOCK_M > 1:
            s = tl.dot(q, k, input_precision="ieee") * qk_scale
        else:
            # (1, D), (D, T)
            s = tl.sum((q.T * k).to(tl.float32), axis=0, keep_dims=True) * qk_scale
        decay_bias = log_lambda_out[:, None] - log_lambda_in[None, :]
        s += decay_bias * log2e

        if not DIVISIBLE_N:
            s = tl.where(mask_n[None, :], s, float("-inf"))
        if IS_CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= offs_n[None, :]
            s = tl.where(causal_mask, s, float("-inf"))
        if HAS_SEQ_START:
            s = tl.where(offs_n[None, :] >= seq_start, s, float("-inf"))


        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2((m_i - m_i_new))
        p = tl.math.exp2(s - m_i_new[:, None])

        # -- compute partial sumexpn before applying dropout
        p_sum = tl.sum(p, 1)


        # -- scale and update acc: acc *= alpha[:, None]--
        acc *= alpha[:, None]
        if BLOCK_M > 1:
            acc += tl.dot(p.to(input_dtype), v, input_precision="ieee")
        else:
            acc += tl.sum(p.T * v, axis=0, keep_dims=True)

        # -- update m_i and l_i --
        l_i = l_i * alpha + p_sum
        m_i = m_i_new
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        log_lambda_in_ptrs += BLOCK_N * stride_log_lambda_n

    # write back l & o
    if IS_CAUSAL and (LARGER_M or HAS_SEQ_START):
        is_empty_line = (offs_m + P_SEQ) < seq_start
        acc = tl.where(is_empty_line[:, None], 0.0, acc * (1.0 / l_i[:, None]))
        l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        l = m_i * loge2 + tl.log(l_i) # log(normalizer)


    if DIVISIBLE_M:
        tl.store(l_ptrs, l, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=offs_k[None, :] < REAL_D, cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & (offs_k[None, :] < REAL_D), cache_modifier=".cg")

@triton.jit
def _find_start_index_kernel(
    LOG_LAMBDA,
    START_INDEX,
    THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    # -- grid id --
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)

    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    start_index = 0

    log_lambda_out_ptr = LOG_LAMBDA + P_SEQ * stride_log_lambda_n
    start_index_ptr = START_INDEX



    threshold = THRESHOLD
    for start_m in range(0, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)

        log_lambda_out = tl.load(log_lambda_out_ptr)

        offset_n = start_index + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in

        while decay < threshold:
            start_index += BLOCK_N

            offset_n = start_index + BLOCK_N - 1
            if not DIVISIBLE_N:
                offset_n = tl.minimum(N - 1, offset_n)
            log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
            decay = log_lambda_out - log_lambda_in


        tl.store(start_index_ptr, start_index.to(START_INDEX.dtype.element_ty))
        start_index_ptr += stride_start_index_mb
        log_lambda_out_ptr += stride_log_lambda_n * BLOCK_M
