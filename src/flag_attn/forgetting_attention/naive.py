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

# Unoptimized Triton ACP reference, not a dense PyTorch implementation.
# Forward code is copied from zhixuan-lin/forgetting-transformer:
# commit 883f260c636e87971339749b0d8310004794a5cc
# src/forgetting_transformer/ops/forgetting_attention.py
# Original complete source SHA256:
# 6a71b9ebef4dbfaf489ca32fb4ad32ed51a3dc474b6d681c21ddf026eef1b5d2
# This forward-only extraction retains the original forward operations and
# launch configuration. Backward and standalone upstream tests are excluded.
import math
from collections import defaultdict, OrderedDict
from typing import Literal, Optional, Union

import torch
import triton
import triton.language as tl

# The optimized path is optional even without TLE: older Triton releases keep
# the reference adapter rather than failing at package import.
try:
    from triton import knobs as _plain_knobs
    from triton.runtime import jit as _plain_runtime_jit
    from triton.tools.tensor_descriptor import TensorDescriptor as _PlainTensorDescriptor
except ImportError:
    _plain_knobs = _plain_runtime_jit = _PlainTensorDescriptor = None
from einops import rearrange

OFFICIAL_COMMIT = "883f260c636e87971339749b0d8310004794a5cc"
__all__ = ["forgetting_attention", "triton_forgetting_attention"]

def maybe_contiguous(x):
    # only when the inner most dimension is contiguous can LDGSTS be used
    # so inner-dimension contiguity is enforced.
    return x.contiguous() if x.stride(-1) != 1 else x

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
        assert Dk in {16, 32, 64, 128}, "We only support head dims in {16, 32, 64, 128}"


        B, H, M, D = q.shape
        N = k.shape[2]
        assert log_fgate.shape == (B, H, N)
        assert log_fgate.dtype == torch.float32, "log_fgate must be in torch.float32. It is best to do this cast before logsigmoid, e.g., torch.nn.functional.logsigmoid(fgate_logit.float())."
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)

        if adaptive_threshold is not None:
            if isinstance(adaptive_threshold, str):
                assert adaptive_threshold == "auto", f'adaptive_threshold must be either the string "auto", a float, or a Tensor, but got {adaptive_threshold}.'
                # Otherwise we could calculate the max L2 norms manually 
                max_q_norm = torch.linalg.vector_norm(q, dim=-1).max(dim=-1).values
                max_k_norm = torch.linalg.vector_norm(k, dim=-1).max(dim=-1).values
                assert max_q_norm.size() == max_k_norm.size() == (B, H)

                logit_upper_bound = max_q_norm * max_k_norm * sm_scale
                tolerance = -10
                # Note we should use N instead of M here
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
            seq_start = torch.zeros((B,), device=q.device, dtype=torch.long)
        log_fgate = log_fgate.float()
        if has_seq_start:
            log_fgate = log_fgate.clone()
            # We absolutely don't want masked value to affect result. If we
            # don't do this then it could via affecting numerical precision of
            # cumsum
            mask_index = (torch.arange(N, device=q.device)[None, None, :] < seq_start[:, None, None])
            mask_index = torch.broadcast_to(mask_index, log_fgate.size())
            log_fgate[mask_index] = 0.0

        log_lambda = torch.cumsum(log_fgate, dim=-1, dtype=log_fgate.dtype).float()

        Hk, Hv = k.shape[1], v.shape[1]
        assert Hk == Hv, "num of heads in k and v should be equal"
        assert H == Hk, "groupped query attention has not been tested. You can uncomment this if you know what you are doing."
        assert H % Hk == 0, "number of heads in q must be a multiple of that in k & v"
        num_groups = H // Hk

        P_SEQ = N - M
        larger_m = M > N
        assert (not larger_m), "The key/value tensors must be longer than the query tensor"


        # contiguity
        q, k, v = maybe_contiguous(q), maybe_contiguous(k), maybe_contiguous(v)

        # to work around https://github.com/openai/triton/issues/2441
        device = torch.cuda.device_of(q)

        with torch.cuda.device(device):

            if M > 1:
                config = get_fwd_config(B, H, M, N, D, causal)
                BLOCK_M, BLOCK_N, num_stages, num_warps = config
            else:
                BLOCK_N, num_stages, num_warps = min(128, max(16, triton.next_power_of_2(N))), 1, 4
                BLOCK_M = 1

            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0

            
            start_index = torch.empty((B, H, triton.cdiv(M, BLOCK_M)), dtype=torch.long, device=q.device)
            if adaptive_threshold is not None:
                grid = (H, B)
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].record()
                _find_start_index_kernel[grid](
                    log_lambda,
                    start_index,
                    adaptive_threshold,
                    log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                    start_index.stride(0), start_index.stride(1), start_index.stride(2),
                    adaptive_threshold.stride(0), adaptive_threshold.stride(1),
                    B, H, M, N, P_SEQ,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                    num_warps=1
                )
                if record_find_index_time:
                    ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"].record()
                    torch.cuda.synchronize()
                    elapsed = ForgettingAttention.events[record_time_key]["fwd_find_index_start_event"].elapsed_time(ForgettingAttention.events[record_time_key]["fwd_find_index_end_event"])
                    ForgettingAttention.info[record_time_key]["fwd_find_index_time"] += elapsed
                    ForgettingAttention.info[record_time_key]["fwd_find_index_count"] += 1

            # Actual forward
            # consider using 3d grid to avoid div & rem
            # grid = (triton.cdiv(M, BLOCK_M), H, B)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), H, B)
            o = torch.empty_like(q)
            L = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
            _fwd_kernel[grid](
                q, k, v, log_lambda, seq_start, start_index, sm_scale,
                L, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                log_lambda.stride(0), log_lambda.stride(1), log_lambda.stride(2),
                start_index.stride(0), start_index.stride(1), start_index.stride(2),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                B, H, M, N, P_SEQ, num_groups,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=D,
                IS_CAUSAL=causal, LARGER_M=larger_m, HAS_SEQ_START=has_seq_start,
                IS_ADAPTIVE=adaptive_threshold is not None,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_warps=num_warps, num_stages=num_stages,
            )

        # autograd context maintenance
        ctx.save_for_backward(q, k, v, o, L, log_lambda, seq_start, adaptive_threshold)
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

def forgetting_attention(
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
    A FlashAttention-based implementation of Forgetting Attention. 

    Note:
    - q, k, v should be in bfloat16/float16 and log_fgate should be in float32.
    - We only support seqlen_q <= seqlen_k
    - We only support causal attention
    - Head dimension must be in one of {16, 32, 64, 128}

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

def triton_forgetting_attention(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """No-TLE ACP inference, with an upstream-Triton fast path on Hopper.

    H100/Triton 3.6 uses fused exact prefix/boundary preparation and direct GQA.
    Multi-query D=60/100 is masked in-kernel instead of padding whole tensors.
    Other devices/compiler versions and longer sequences retain the original
    explicit padding/GQA adapter. The official reference is unchanged.
    This is not a CPU or training fallback.
    """
    if head_first or seq_start is not None or not torch.is_inference_mode_enabled():
        raise NotImplementedError("general profile supports BTHD inference without seq_start")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("require 4D Q/K/V and matching K/V shapes")
    b, m, h, d = q.shape
    n, hk = k.shape[1:3]
    if (d not in (16, 32, 60, 64, 100, 128) or k.shape != (b, n, hk, d)
            or hk <= 0 or h % hk):
        raise NotImplementedError("require supported D and Q heads divisible by KV heads")
    if not (0 < m <= n and b > 0 and h > 0):
        raise ValueError("require 0 < M <= N and positive batch/heads")
    if not (q.dtype in (torch.bfloat16, torch.float16) and q.is_cuda
            and all(t.dtype == q.dtype and t.device == q.device and t.is_contiguous()
                    and t.data_ptr() % 16 == 0 for t in (q, k, v))):
        raise NotImplementedError("require contiguous, 16-byte-aligned CUDA BF16/FP16 inputs")
    if (log_fgate.shape != (b, n, h) or log_fgate.dtype != torch.float32
            or log_fgate.device != q.device or not log_fgate.is_contiguous()):
        raise NotImplementedError("gate must be contiguous FP32 BNH")
    if (not isinstance(adaptive_threshold, (int, float))
            or not math.isfinite(adaptive_threshold) or adaptive_threshold > 0):
        raise NotImplementedError("require a finite nonpositive scalar ACP threshold")
    # Compute from the original D, before padding changes the reference's input.
    scale = 1 / math.sqrt(d) if sm_scale is None else sm_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise ValueError("scale must be a finite scalar")
    padded = triton.next_power_of_2(d)
    with torch.cuda.device(q.device):
        if _plain_fast_supported(n):
            return _triton_forward(q, k, v, log_fgate, scale, float(adaptive_threshold))
        if padded != d:
            q, k, v = (torch.nn.functional.pad(t, (0, padded - d)) for t in (q, k, v))
        groups = h // hk
        if groups != 1:
            k, v = (t.repeat_interleave(groups, dim=2) for t in (k, v))
        out = forgetting_attention(
            q, k, v, log_fgate, head_first=False, seq_start=None,
            sm_scale=scale, adaptive_threshold=adaptive_threshold,
        )
        return out if padded == d else out[..., :d].contiguous()


def get_fwd_config(B, H, M, N, D, causal):
    assert causal
    if torch.cuda.get_device_capability() == (8, 0):
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 32, 3, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 4, 4
    elif torch.cuda.get_device_capability() == (9, 0):
        # H100
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 8
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 128, 2, 8
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
        else: # causal
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 3, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    elif torch.cuda.get_device_capability() == (8, 9):
        # L40S
        if D <= 64:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 64, 2, 4
        else:
            BLOCK_M, BLOCK_N, num_stages, num_warps = 128, 32, 2, 4
    else:
        raise ValueError(f"Unsupported device capability {torch.cuda.get_device_capability()}. Please open an issue.")
        # BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)

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
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
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
        q = tl.load(q_ptrs, cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, cache_modifier=".cg")
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None], cache_modifier=".cg")
        log_lambda_out = tl.load(log_lambda_out_ptrs, mask=mask_m, cache_modifier=".cg")

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
            k = tl.load(k_ptrs, cache_modifier=".cg")
            v = tl.load(v_ptrs, cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, cache_modifier=".cg")
        else:
            mask_n = offs_n < N
            k = tl.load(k_ptrs, mask=mask_n[None, :], cache_modifier=".cg")
            v = tl.load(v_ptrs, mask=mask_n[:, None], cache_modifier=".cg")
            log_lambda_in = tl.load(log_lambda_in_ptrs, mask=mask_n, cache_modifier=".cg")

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
        tl.store(o_ptrs, acc.to(input_dtype), cache_modifier=".cg")
    else:
        tl.store(l_ptrs, l, mask=mask_m, cache_modifier=".cg")
        tl.store(o_ptrs, acc.to(input_dtype), mask=mask_m[:, None], cache_modifier=".cg")

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


# No-TLE Hopper specialization. All original reference kernels above are intact.

def _plain_fast_supported(n):
    # The tested specialization is deliberately bounded. Unsupported targets
    # keep the existing reference adapter; no compilation errors are swallowed.
    return (_plain_knobs is not None
            and triton.__version__.startswith("3.6.")
            and torch.cuda.get_device_capability() == (9, 0)
            and n <= 8256)


class _PlainLaunch:
    """Cache code/launch metadata only. Pass current pointers on every call."""
    def __init__(self, fn, limit=256):
        self.fn, self.limit, self.plans = fn, limit, OrderedDict()
        self.dist_context = getattr(_plain_runtime_jit, 'DistributedRtContext', None)

    def __call__(self, key, grid, args, kwargs):
        fn = self.fn
        key = (key, tuple(grid), tuple(sorted(kwargs.items())))
        allowed = (not fn.pre_run_hooks and not _plain_knobs.runtime.debug
                   and not _plain_knobs.compilation.instrumentation_mode)
        if self.dist_context is not None and self.dist_context().is_lite_mode:
            allowed = False
        plan = self.plans.get(key) if allowed else None
        if plan is None:
            compiled = fn[grid](*args, **kwargs)
            self.last_compiled = compiled
            if allowed:
                self.plans[key] = compiled[tuple(grid) + (1,) * (3 - len(grid))]
                if len(self.plans) > self.limit:
                    self.plans.popitem(last=False)
        else:
            plan(*args)


@triton.jit
def _plain_add(a,b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;",constraints="=f,f,f",
        args=[a,b],dtype=tl.float32,is_pure=True,pack=1)

@triton.jit
def _plain_prepare(X,L,S,THR, XB:tl.constexpr,XH:tl.constexpr,XT:tl.constexpr,
                   H:tl.constexpr,N:tl.constexpr,
                   C:tl.constexpr,LOGC:tl.constexpr,NQ:tl.constexpr,NK:tl.constexpr,M:tl.constexpr,
                   BN:tl.constexpr,QM:tl.constexpr):
    h=tl.program_id(0);b=tl.program_id(1)
    i=tl.arange(0,C);qi=tl.arange(0,NQ);ki=tl.arange(0,NK)
    anchor=tl.full((NQ,),0,tl.float32);ends=tl.full((NK,),0,tl.float32)
    carry=tl.full((),0,tl.float32)
    pos_q=N-M+qi*QM
    pos_k=tl.minimum(ki*BN+BN-1,N-1)
    for chunk in range(tl.cdiv(N,C)):
        t=chunk*C+i
        x=tl.load(X+b*XB+h*XH+t*XT,t<N,other=0)
        x=tl.where(i==0,_plain_add(x,carry),x)
        for m in tl.static_range(LOGC):
            src=(i//(2<<m))*(2<<m)+(1<<m)-1
            left=tl.gather(x,src,axis=0)
            x=tl.where((i&(1<<m))!=0,_plain_add(x,left),x)
        tl.store(L+(b*H+h)*N+t,x,t<N)
        carry=tl.sum(tl.where(i==C-1,x,0),0)
        av=tl.gather(x,pos_q%C,axis=0)
        ev=tl.gather(x,pos_k%C,axis=0)
        anchor=tl.where(pos_q//C==chunk,av,anchor)
        ends=tl.where(pos_k//C==chunk,ev,ends)
    threshold=THR
    old=(ki[None,:]<tl.cdiv(N,BN))&((anchor[:,None]-ends[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+qi,starts,qi<tl.cdiv(M,QM))

@triton.jit
def _plain_starts(P, T, S, M:tl.constexpr,N:tl.constexpr,H:tl.constexpr,
                    BN:tl.constexpr,QM:tl.constexpr,NK:tl.constexpr):
    h,b,group=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    r=group*16+tl.arange(0,16)
    j=tl.arange(0,NK)
    pp=P+(b*H+h)*N
    a=tl.load(pp+N-M+r*QM,r<tl.cdiv(M,QM),other=0)
    end=tl.minimum(j*BN+BN-1,N-1)
    e=tl.load(pp+end,j<tl.cdiv(N,BN),other=0)
    threshold=T
    old=(j[None,:]<tl.cdiv(N,BN))&((a[:,None]-e[None,:])<threshold)
    starts=tl.sum(old.to(tl.int32),1)*BN
    tl.store(S+(b*H+h)*tl.cdiv(M,QM)+r,starts,r<tl.cdiv(M,QM))


@triton.jit
def _plain_attention(Q, K, V, P, S, O, sm_scale,
                     M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
                     REAL_D: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    h, b, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = h // (H // HK)
    rm = block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    qptr = Q + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    q = tl.load(qptr, (rm[:, None] < M) & (rd[None, :] < REAL_D), other=0)
    pp = P + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, rm < M, other=0)
    lo = tl.load(S + (b * H + h) * tl.cdiv(M, 128) + block * BM // 128)
    hi = tl.minimum(N, N - M + (block + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    diagonal = ((N - M + block * BM + 1) // BN) * BN
    for start in range(lo, hi, BN):
        kn = start + rn
        kp = K + (b * N * HK + kn[None, :] * HK + hk) * REAL_D + rd[:, None]
        vp = V + (b * N * HK + kn[:, None] * HK + hk) * REAL_D + rd[None, :]
        k = tl.load(kp, (kn[None, :] < N) & (rd[:, None] < REAL_D), other=0)
        v = tl.load(vp, (kn[:, None] < N) & (rd[None, :] < REAL_D), other=0)
        gk = tl.load(pp + kn, kn < N, other=0)
        score = tl.dot(q, k, input_precision="ieee") * scale
        score = tl.fma(gq[:, None] - gk[None, :], log2e, score)
        score = tl.where(kn[None, :] < N, score, -float("inf"))
        if start >= diagonal:
            score = tl.where((N - M + rm[:, None]) >= kn[None, :], score, -float("inf"))
        mnew = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mnew)
        prob = tl.exp2(score - mnew[:, None])
        psum = tl.sum(prob, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(prob.to(Q.dtype.element_ty), v, acc, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    out = (acc * (1.0 / li[:, None])).to(Q.dtype.element_ty)
    op = O + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    tl.store(op, out, (rm[:, None] < M) & (rd[None, :] < REAL_D))


@triton.jit
def _plain_tma_attention(Q, K, V, P, S, O, sm_scale,
                     M: tl.constexpr, N: tl.constexpr, H: tl.constexpr, HK: tl.constexpr,
                     REAL_D: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    h, b, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    hk = h // (H // HK)
    rm = block * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, D)
    q = Q.load([b * M + block * BM, h * D])
    pp = P + (b * H + h) * N
    gq = tl.load(pp + N - M + rm, rm < M, other=0)
    lo = tl.load(S + (b * H + h) * tl.cdiv(M, 128) + block * BM // 128)
    hi = tl.minimum(N, N - M + (block + 1) * BM)
    mi = tl.full((BM,), -float("inf"), tl.float32)
    li = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    log2e: tl.constexpr = 1.4426950408889634
    scale = sm_scale * log2e
    diagonal = ((N - M + block * BM + 1) // BN) * BN
    for start in range(lo, hi, BN):
        kn = start + rn
        k = K.load([b * N + start, hk * D]).T
        v = V.load([b * N + start, hk * D])
        gk = tl.load(pp + kn, kn < N, other=0)
        score = tl.dot(q, k, input_precision="ieee") * scale
        score = tl.fma(gq[:, None] - gk[None, :], log2e, score)
        score = tl.where(kn[None, :] < N, score, -float("inf"))
        if start >= diagonal:
            score = tl.where((N - M + rm[:, None]) >= kn[None, :], score, -float("inf"))
        mnew = tl.maximum(mi, tl.max(score, 1))
        alpha = tl.exp2(mi - mnew)
        prob = tl.exp2(score - mnew[:, None])
        psum = tl.sum(prob, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(prob.to(O.dtype.element_ty), v, acc, input_precision="ieee")
        li = li * alpha + psum
        mi = mnew
    out = (acc * (1.0 / li[:, None])).to(O.dtype.element_ty)
    op = O + (b * M * H + rm[:, None] * H + h) * REAL_D + rd[None, :]
    tl.store(op, out, (rm[:, None] < M) & (rd[None, :] < REAL_D))


_plain_prep_launch = _PlainLaunch(_plain_prepare)
_plain_starts_launch = _PlainLaunch(_plain_starts)
_plain_attn_launch = _PlainLaunch(_plain_attention)
_plain_decode_launch = _PlainLaunch(_fwd_kernel)
_plain_tma_launch = _PlainLaunch(_plain_tma_attention)

def _plain_config(b, m, h, real_d):
    if _PlainTensorDescriptor is not None and real_d == 128 and b * m * h >= 131072:
        return 64, 4, 1, True
    if real_d <= 64 and m >= 1024 and b * h >= 64:
        return 128, 4, 3, False
    return 64, 4, 1, False


def _triton_forward(q, k, v, gate, scale, threshold):
    b, m, h, real_d = q.shape
    n, hk = k.shape[1:3]
    d = triton.next_power_of_2(real_d)
    bn = min(128, max(16, triton.next_power_of_2(n))) if m == 1 else (64 if d <= 64 else 128)
    qm = 1 if m == 1 else 128
    nq, nk = triton.cdiv(m, qm), triton.cdiv(n, bn)
    starts = torch.empty((b, h, nq), device=q.device, dtype=torch.int32)
    if b * h > 1 and torch.__version__.startswith("2.10."):
        prefix = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        ln, lr = (n-1).bit_length(), (b*h-1).bit_length()
        chunk = 2 * (1 << min(9, max(4, (9 + ln - lr) // 2)))
        args = (gate, prefix, starts, float(threshold),
                n*h, 1, h, h, n, chunk, chunk.bit_length()-1,
                triton.next_power_of_2(nq), triton.next_power_of_2(nk), m, bn, qm)
        key = (q.device.index, gate.data_ptr() % 16, args[4:])
        _plain_prep_launch(key, (h, b), args, dict(num_warps=4))
    else:
        prefix = torch.cumsum(gate.transpose(1, 2), -1, dtype=torch.float32)
        args = (prefix, float(threshold), starts, m, n, h, bn, qm, triton.next_power_of_2(nk))
        _plain_starts_launch((q.device.index, b, args[3:]), (h, b, triton.cdiv(nq, 16)),
                             args, dict(num_warps=4))
    out = torch.empty_like(q)
    key = (q.device.index, q.dtype, b, m, n, h, hk, real_d, type(scale), float(scale))
    if m > 1:
        bm, warps, stages, tma = _plain_config(b, m, h, real_d)
        if tma and real_d == d:
            qd = _PlainTensorDescriptor(q, [b*m, h*d], [h*d, 1], [bm, d])
            kd = _PlainTensorDescriptor(k, [b*n, hk*d], [hk*d, 1], [bn, d])
            vd = _PlainTensorDescriptor(v, [b*n, hk*d], [hk*d, 1], [bn, d])
            args = (qd, kd, vd, prefix, starts, out, scale, m, n, h, hk, real_d, d, bm, bn)
            launch = _plain_tma_launch
        else:
            args = (q, k, v, prefix, starts, out, scale, m, n, h, hk, real_d, d, bm, bn)
            launch = _plain_attn_launch
        launch(key, (h, b, triton.cdiv(m, bm)), args,
               dict(num_warps=warps, num_stages=stages))
    else:
        if real_d != d:
            q, k, v = (torch.nn.functional.pad(t, (0, d-real_d)) for t in (q, k, v))
            out = torch.empty((b, m, h, d), device=q.device, dtype=q.dtype)
        norm = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
        args = (q, k, v, prefix, starts, starts, scale, norm, out,
                m*h*d, d, h*d, 1, n*hk*d, d, hk*d, 1, n*hk*d, d, hk*d, 1,
                h*n, n, 1, h*nq, nq, 1, m*h*d, d, h*d, 1,
                b, h, m, n, n-m, h//hk, 1, d, bn, True, False, False, True,
                True, n % bn == 0)
        _plain_decode_launch(key, (1, h, b), args, dict(num_warps=4, num_stages=1))
        if real_d != d:
            out = out[..., :real_d].contiguous()
    return out

