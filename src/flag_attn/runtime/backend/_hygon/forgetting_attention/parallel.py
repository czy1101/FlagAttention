"""Hygon Triton implementation of Forgetting Attention."""

import math
from collections import defaultdict
from typing import Literal, Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from flag_attn.runtime import get_tuned_config
from flag_attn.runtime.backend._hygon.heuristics_config_utils import get_heuristics_config


def maybe_contiguous(x):
    # Keep the input layout; kernels use the provided tensor strides.
    return x

class ForgettingAttention(torch.autograd.Function):
    events = defaultdict(lambda: {
        "fwd_start_event": torch.cuda.Event(enable_timing=True),
        "fwd_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_end_event": torch.cuda.Event(enable_timing=True),

        "fwd_find_index_start_event": torch.cuda.Event(enable_timing=True),
        "fwd_find_index_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_kv_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_kv_end_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_q_start_event": torch.cuda.Event(enable_timing=True),
        "bwd_find_index_q_end_event": torch.cuda.Event(enable_timing=True),
    })
    info = defaultdict(lambda: {
        "fwd_time": 0.0,
        "fwd_count": 0,
        "bwd_time": 0.0,
        "bwd_count": 0,

        "fwd_find_index_time": 0.0,
        "fwd_find_index_count": 0,
        "bwd_find_index_kv_time": 0.0,
        "bwd_find_index_kv_count": 0,
        "bwd_find_index_q_time": 0.0,
        "bwd_find_index_q_count": 0,
    })

    @staticmethod
    def forward(ctx, q, k, v, log_fgate, seq_start, causal, sm_scale, adaptive_threshold, return_log_normalizer, return_start_index, record_time_key, record_attention_time, record_find_index_time):
        if record_attention_time:
            ForgettingAttention.events[record_time_key]["fwd_start_event"].record()


        assert causal, "Only causal attention is supported"
        Dq, Dk, Dv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Dq == Dk == Dv, "feature size of q, k, v should be equal"
        assert Dk in {16, 32, 60, 64, 100, 128}, "We only support head dims in {16, 32, 60, 64, 100, 128}"


        B, H, M, D = q.shape
        N = k.shape[2]
        assert log_fgate.shape == (B, H, N)
        assert log_fgate.dtype == torch.float32, "log_fgate must be in torch.float32. It is best to do this cast before logsigmoid, e.g., torch.nn.functional.logsigmoid(fgate_logit.float())."
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        if adaptive_threshold is not None:
            if isinstance(adaptive_threshold, str):
                assert adaptive_threshold == "auto", f'adaptive_threshold must be either the string "auto", a float, or a Tensor, but got {adaptive_threshold}.'
                max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                assert max_q_norm.size() == max_k_norm.size() == (B, H)

                logit_upper_bound = max_q_norm * max_k_norm * sm_scale
                tolerance = -10
                # Use the KV sequence length in the logarithmic bound.
                adaptive_threshold = -(2 * logit_upper_bound + math.log(N)) + tolerance

            adaptive_threshold = torch.as_tensor(adaptive_threshold, dtype=torch.float, device=q.device)
            try:
                adaptive_threshold = torch.broadcast_to(adaptive_threshold, (B, H))
            except RuntimeError:
                raise RuntimeError(f'adaptive_threshold must be either the string "auto" or broadcastable to (batch_size, num_heads) = ({B}, {H}), but got {adaptive_threshold.size()}.')
            assert adaptive_threshold.size() == (B, H)

        if seq_start is not None:
            has_seq_start = True
            assert seq_start.shape == (B,)
        else:
            has_seq_start = False
            # HAS_SEQ_START is a compile-time false branch in the kernel, so
            # use an existing device pointer instead of allocating a dummy tensor.
            seq_start = q
        log_fgate = log_fgate.float()
        if has_seq_start:
            log_fgate = log_fgate.clone()
            # Keep masked positions at zero so they do not affect the prefix sum.
            mask_index = (torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None])
            mask_index = torch.broadcast_to(mask_index, log_fgate.size())
            log_fgate[mask_index] = 0.0

        # The inclusive prefix sum of a one-element sequence is the input itself.
        # Reuse it directly instead of launching a separate cumsum kernel.
        log_lambda = (
            log_fgate
            if N == 1
            else torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()
        )

        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N
        assert (not larger_m), "The key/value tensors must be longer than the query tensor"


        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # Launch under the device context associated with the query tensor.
        device = torch.cuda.device_of(q)

        with torch.cuda.device(device):
            heuristics = get_heuristics_config("forgetting_attention")

            if M > 1:
                config = get_fwd_config(B, H, M, N, D, causal)
                BLOCK_M, BLOCK_N, num_stages, num_warps = config
            else:
                BLOCK_N, num_stages, num_warps = get_decode_config(N)
                BLOCK_M = 1

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0


            # Fuse threshold search into the attention CTAs for short and
            # moderate queries, avoiding a separate launch and scratch tensor.
            # Use the serial search kernel for longer queries, where independent
            # per-tile searches add too much work.
            can_fuse_start_index = (
                adaptive_threshold is not None
                and not record_find_index_time
                and not return_start_index
            )
            fuse_binary_start_index = (
                can_fuse_start_index
                and M >= int(heuristics.get("fused_binary_start_min_m", 129))
                and triton.cdiv(M, BLOCK_M)
                <= int(heuristics.get("fused_binary_start_max_query_tiles", 16))
            )
            fuse_start_index = can_fuse_start_index and (
                M <= int(heuristics.get("fused_start_max_m", 128))
                or fuse_binary_start_index
            )
            need_start_index = return_start_index or (
                adaptive_threshold is not None and not fuse_start_index
            )
            if need_start_index:
                start_index = torch.empty(
                    (B, H, triton.cdiv(M, BLOCK_M)), dtype=torch.long, device=q.device
                )
            else:
                # START_INDEX is unused when adaptation is disabled or the
                # start index is computed inside the attention kernel.
                start_index = q
            if adaptive_threshold is not None and not fuse_start_index:
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].record()
                query_tiles = triton.cdiv(M, BLOCK_M)
                search_num_stages, search_num_warps = get_search_config()
                if (
                    M >= int(heuristics.get("fused_binary_start_min_m", 129))
                    and query_tiles
                    <= int(heuristics.get("fused_binary_start_max_query_tiles", 16))
                ):
                    # The serial search kernel processes query tiles one by
                    # one. For a small number of tiles, launch one independent
                    # lower-bound search per tile.
                    _find_start_index_parallel_kernel[(query_tiles, H, B)](
                        log_lambda,
                        start_index,
                        adaptive_threshold,
                        log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                        start_index.stride(0), start_index.stride(1), start_index.stride(2),
                        adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                        B, H, M, N, P_SEQ,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        DIVISIBLE_N=divisible_n,
                        num_warps=search_num_warps,
                        num_stages=search_num_stages,
                    )
                else:
                    _find_start_index_kernel[(H, B)](
                        log_lambda,
                        start_index,
                        adaptive_threshold,
                        log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                        start_index.stride(0), start_index.stride(1), start_index.stride(2),
                        adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                        B, H, M, N, P_SEQ,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                        num_warps=search_num_warps,
                        num_stages=search_num_stages,
                    )
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].elapsed_time(ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"])
                    ForgettingAttention.info[record_time_key]["fwd_find_index_time"] += elapsed
                    ForgettingAttention.info[record_time_key]["fwd_find_index_count"] += 1

            def grid(META):
                return (triton.cdiv(M, META["BLOCK_M"]), H, B)
            o = torch.empty_like(q)
            L = (
                torch.empty((B, H, M), device=q.device, dtype=torch.float32)
                if return_log_normalizer else q
            )
            threshold = adaptive_threshold if adaptive_threshold is not None else q
            stride_threshold_z = threshold.stride(0) if adaptive_threshold is not None else 0
            stride_threshold_h = threshold.stride(1) if adaptive_threshold is not None else 0
            _fwd_kernel[grid](
                q, k, v, log_lambda, seq_start, start_index, threshold, sm_scale,
                L, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                stride_threshold_z, stride_threshold_h,
                start_index.stride(0), start_index.stride(1), start_index.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, M, N, P_SEQ, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=triton.next_power_of_2(D),
                D_HEAD=D, DIVISIBLE_D=(D == triton.next_power_of_2(D)),
                IS_CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                IS_ADAPTIVE=adaptive_threshold is not None,
                FUSE_START_INDEX=fuse_start_index,
                FUSE_BINARY_START_INDEX=fuse_binary_start_index,
                RETURN_LOG_NORMALIZER=return_log_normalizer,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_warps=num_warps, num_stages=num_stages,
            )

        if return_log_normalizer:
            ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start, adaptive_threshold)
        else:
            ctx.save_for_backward(q, k, v, o, log_lambda, seq_start, adaptive_threshold)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.has_seq_start = has_seq_start
        ctx.record_time_key = record_time_key
        ctx.record_attention_time = record_attention_time
        ctx.record_find_index_time = record_find_index_time

        has_extra_return = return_log_normalizer or return_start_index

        if record_attention_time:
            ForgettingAttention.events[record_time_key]["fwd_end_event"].record()
            torch.cuda.synchronize()
            elapsed = ForgettingAttention.events[record_time_key]["fwd_start_event"].elapsed_time(ForgettingAttention.events[record_time_key]["fwd_end_event"])
            ForgettingAttention.info[record_time_key]["fwd_time"] += elapsed
            ForgettingAttention.info[record_time_key]["fwd_count"] += 1
        if has_extra_return:
            outs = (
                o,
                L if return_log_normalizer else None,
                start_index if return_start_index else None
            )
            return outs
        return o

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("This bundled ACP reference is forward-only")

def _forgetting_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_fgate: torch.Tensor,
    *,
    head_first: bool = False,
    seq_start: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    adaptive_threshold: Optional[Union[Literal["auto"], float, torch.Tensor]] = None,
):
    """
    A tiled Triton implementation of Forgetting Attention.

    Note:
    - q, k, v should be in bfloat16/float16 and log_fgate should be in float32.
    - We only support seqlen_q <= seqlen_k
    - We only support causal attention
    - Head dimension must be in one of {16, 32, 60, 64, 100, 128}

    Arguments:
        - q: (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
        - k: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - v: (batch_size, seqlen_k, num_heads, head_dim) unless head_first=True.
        - log_fgate: (batch_size, seqlen_k, num_heads) unless head_first=True.
              This should be the **log** of the forget gates. This is typically the
              output of torch.nn.functional.logsigmoid.
        - head_first: if True, the order the num_heads and seqlen_* axis of the all
              FloatTensor inputs and outputs should be (num_heads, seq_len_*) instead of
              (seq_len_*, num_heads)
        - seq_start: If not None, should be LongTensor with shape (batch_size,)
              and range in [0, seq_len_k). For each batch index batch_id, no attention
              will be allocated to tokens before the token index seq_start[batch_id].
              This is useful for left-padded inputs.
        - sm_scale: The scaling of attention scores before applying softmax. If
              None, it defaults to (1.0 / math.sqrt(head_dim))
        - adaptive_threshold: The threshold for adaptive computation pruning. This
              should be either the string "auto", a float, or a Tensor that is
              broadcastable to (batch_size, num_heads). If "auto", the threshold would
              be computed automatically based on the L2 norms of queries and keys.

    Returns:
        out (torch.Tensor): (batch_size, seqlen_q, num_heads, head_dim) unless head_first=True.
    """
    for name, entry in dict(q=q, k=k, v=v).items():
        assert entry.dtype in [torch.float16, torch.bfloat16], f"Only torch.float16 or torch.bfloat16 are supported for q/k/v, but got {entry.dtype} for {name}."
    if not head_first:
        q, k, v = [rearrange(item, "b t h d -> b h t d") for item in (q, k, v)]
        log_fgate = rearrange(log_fgate, "b t h -> b h t")
    out = ForgettingAttention.apply(q, k, v, log_fgate, seq_start, True, sm_scale, adaptive_threshold, False, False, None, False, False)
    if not head_first:
        out = rearrange(out, "b h t d -> b t h d")
    return out

def _first_tuned_config(op_name):
    configs = get_tuned_config(op_name)
    return configs[0] if configs else None


def get_fwd_config(B, H, M, N, D, causal):
    """Use the curated Hygon launch from the backend tuning configuration."""
    del B, H, M, N, D
    assert causal
    config = _first_tuned_config("_forgetting_attention_fwd")
    if config is None:
        return (64, 32, 1, 4)
    meta = config.kwargs
    return (
        int(meta.get("BLOCK_M", 64)),
        int(meta.get("BLOCK_N", 32)),
        config.num_stages or 1,
        config.num_warps or 4,
    )


def get_search_config():
    """Return search-kernel stages and warps from the Hygon tuning file."""
    config = _first_tuned_config("_forgetting_attention_search")
    if config is None:
        return 1, 1
    return config.num_stages or 1, config.num_warps or 1


def get_decode_config(N):
    """Choose the single-query launch from the Hygon tuning file."""
    config = _first_tuned_config("_forgetting_attention_decode")
    if config is None:
        block_n, num_stages, num_warps = 128, 1, 4
    else:
        block_n = int(config.kwargs.get("BLOCK_N", 128))
        num_stages = config.num_stages or 1
        num_warps = config.num_warps or 4
    return min(block_n, max(16, triton.next_power_of_2(N))), num_stages, num_warps

@triton.jit
def _fwd_kernel(
    Q, K, V, LOG_LAMBDA, SEQ_START, START_INDEX, THRESHOLD, sm_scale,
    L, O,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_threshold_z, stride_threshold_h,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, D_HEAD: tl.constexpr,
    IS_CAUSAL: tl.constexpr, LARGER_M: tl.constexpr, HAS_SEQ_START: tl.constexpr,
    IS_ADAPTIVE: tl.constexpr, FUSE_START_INDEX: tl.constexpr,
    FUSE_BINARY_START_INDEX: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr, DIVISIBLE_D: tl.constexpr,
    RETURN_LOG_NORMALIZER: tl.constexpr,
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
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    O += off_z * stride_oz + off_h * stride_oh
    if RETURN_LOG_NORMALIZER:
        L += (off_z * H + off_h) * M # l's shape is (B, H, M)

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_m = start_m * BLOCK_M + offs_m_base
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_DMODEL)


    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk) # (BLOCK_M, BLOCK_DMODEL)
    log_lambda_out_ptrs = LOG_LAMBDA + (P_SEQ + offs_m) * stride_log_lambda_n
    o_ptrs = O + (offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok) # (BLOCK_M, BLOCK_DMODEL)
    if RETURN_LOG_NORMALIZER:
        l_ptrs = L + offs_m

    # initialize pointer to m and l, fp32 for accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q
    if DIVISIBLE_M:
        if DIVISIBLE_D:
            q = tl.load(q_ptrs, cache_modifier=".ca")
        else:
            q = tl.load(q_ptrs, mask=offs_k[None, :] < D_HEAD, other=0.0, cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".ca")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < D_HEAD), other=0.0, cache_modifier=".ca")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".ca")

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
        if FUSE_START_INDEX:
            threshold = tl.load(THRESHOLD)
            log_lambda_out_start = tl.load(
                LOG_LAMBDA + (P_SEQ + start_m * BLOCK_M) * stride_log_lambda_n
            )
            if FUSE_BINARY_START_INDEX:
                left = tl.full((), 0, tl.int32)
                right = tl.cdiv(N, BLOCK_N) - 1
                while left < right:
                    mid = (left + right) // 2
                    offset_n = mid * BLOCK_N + BLOCK_N - 1
                    if not DIVISIBLE_N:
                        offset_n = tl.minimum(N - 1, offset_n)
                    log_lambda_in_start = tl.load(
                        LOG_LAMBDA + offset_n * stride_log_lambda_n
                    )
                    decay = log_lambda_out_start - log_lambda_in_start
                    if decay < threshold:
                        left = mid + 1
                    else:
                        right = mid
                start_index = left * BLOCK_N
            else:
                start_index = 0
                offset_n = start_index + BLOCK_N - 1
                if not DIVISIBLE_N:
                    offset_n = tl.minimum(N - 1, offset_n)
                log_lambda_in_start = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
                decay = log_lambda_out_start - log_lambda_in_start
                while decay < threshold:
                    start_index += BLOCK_N
                    offset_n = start_index + BLOCK_N - 1
                    if not DIVISIBLE_N:
                        offset_n = tl.minimum(N - 1, offset_n)
                    log_lambda_in_start = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
                    decay = log_lambda_out_start - log_lambda_in_start
        else:
            # The start-index buffer has one entry per query tile.
            START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h + start_m * stride_start_index_mb
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
            if DIVISIBLE_D:
                k = tl.load(k_ptrs, cache_modifier=".ca")
                v = tl.load(v_ptrs, cache_modifier=".ca")
            else:
                k = tl.load(k_ptrs, mask=offs_k[:, None] < D_HEAD, other=0.0, cache_modifier=".ca")
                v = tl.load(v_ptrs, mask=offs_k[None, :] < D_HEAD, other=0.0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".ca")
        else:
            mask_n = offs_n < N
            if DIVISIBLE_D:
                k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".ca")
                v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".ca")
            else:
                k = tl.load(k_ptrs, mask=(offs_k[:, None] < D_HEAD) & mask_n[None, :], other=0.0, cache_modifier=".ca")
                v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_k[None, :] < D_HEAD), other=0.0, cache_modifier=".ca")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".ca")

        # -- compute qk ---
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

        # Sum the unnormalized attention weights for the row normalization.
        p_sum = tl.sum(p, 1)


        # Rescale the previous accumulator before adding this block.
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
        if RETURN_LOG_NORMALIZER:
            l = tl.where(is_empty_line, float("-inf"), m_i * loge2 + tl.log(l_i))
    else:
        acc = acc * (1.0 / l_i[:, None])
        if RETURN_LOG_NORMALIZER:
            l = m_i * loge2 + tl.log(l_i) # log(normalizer)


    if RETURN_LOG_NORMALIZER:
        if DIVISIBLE_M:
            tl.store(l_ptrs, l, cache_modifier=".cg")
        else:
            tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")

    if DIVISIBLE_M:
        if DIVISIBLE_D:
            tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
        else:
            tl.store(o_ptrs, acc.to(input_dtype), mask=offs_k[None, :] < D_HEAD, cache_modifier=".cg")
    else:
        if DIVISIBLE_D:
            tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")
        else:
            tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None] & (offs_k[None, :] < D_HEAD), cache_modifier=".cg")

@triton.jit
def _find_start_index_kernel(
    LOG_LAMBDA,
    START_INDEX,
    THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_threshold_z, stride_threshold_h,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    # -- grid id --
    off_h = tl.program_id(0)
    off_z = tl.program_id(1)

    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h
    start_index = 0

    log_lambda_out_ptr = LOG_LAMBDA + P_SEQ * stride_log_lambda_n
    start_index_ptr = START_INDEX



    threshold = tl.load(THRESHOLD)
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


@triton.jit
def _find_start_index_parallel_kernel(
    LOG_LAMBDA,
    START_INDEX,
    THRESHOLD,
    stride_log_lambda_z, stride_log_lambda_h, stride_log_lambda_n,
    stride_start_index_z, stride_start_index_h, stride_start_index_mb,
    stride_threshold_z, stride_threshold_h,
    Z, H, M, N, P_SEQ,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
):
    # One CTA computes one query tile's first BLOCK_N-aligned threshold
    # crossing. log_fgate is a log-retention value, so its prefix is
    # nonincreasing and the crossing can be found with a lower-bound search.
    block_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    LOG_LAMBDA += off_z * stride_log_lambda_z + off_h * stride_log_lambda_h
    START_INDEX += off_z * stride_start_index_z + off_h * stride_start_index_h
    THRESHOLD += off_z * stride_threshold_z + off_h * stride_threshold_h

    start_m = block_m * BLOCK_M
    log_lambda_out = tl.load(
        LOG_LAMBDA + (P_SEQ + start_m) * stride_log_lambda_n
    )
    threshold = tl.load(THRESHOLD)

    left = tl.full((), 0, tl.int32)
    right = tl.cdiv(N, BLOCK_N) - 1
    while left < right:
        mid = (left + right) // 2
        offset_n = mid * BLOCK_N + BLOCK_N - 1
        if not DIVISIBLE_N:
            offset_n = tl.minimum(N - 1, offset_n)
        log_lambda_in = tl.load(LOG_LAMBDA + offset_n * stride_log_lambda_n)
        decay = log_lambda_out - log_lambda_in
        if decay < threshold:
            left = mid + 1
        else:
            right = mid

    start_index = left * BLOCK_N
    start_index_ptr = (
        START_INDEX + block_m * stride_start_index_mb
    )
    tl.store(start_index_ptr, start_index.to(START_INDEX.dtype.element_ty))
_SUPPORTED_DIMS = (16, 32, 60, 64, 100, 128)


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    dimension = q.shape[-1]
    if dimension not in _SUPPORTED_DIMS:
        raise ValueError("head dimension must be one of 16, 32, 60, 64, 100, 128")
    if k.shape[-1] != dimension or v.shape[-1] != dimension:
        raise ValueError("q, k and v must have the same head dimension")

    head_axis = 1 if head_first else 2
    heads, kv_heads = q.shape[head_axis], k.shape[head_axis]
    if kv_heads <= 0 or heads <= 0 or heads % kv_heads or v.shape[head_axis] != kv_heads:
        raise ValueError("query heads must be a positive multiple of key/value heads")

    # Scale by the logical head dimension, including non-power-of-two D.
    scale = 1.0 / math.sqrt(dimension) if sm_scale is None else sm_scale

    with torch.cuda.device(q.device):
        output = _forgetting_attention_impl(
            q, k, v, log_fgate, head_first=head_first, seq_start=seq_start,
            sm_scale=scale, adaptive_threshold=adaptive_threshold,
        )
    return output
