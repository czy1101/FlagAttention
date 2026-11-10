"""MetaX forward-only Forgetting Attention with ordinary Triton kernels.

Inputs must be finite, with nonpositive FP32 log gates. ACP grouping follows
the ordinary Ampere path: 64 query rows for D<=64, 128 otherwise. Prefix
rounding is FP32 and is not a bitwise guarantee against another backend.
"""
import math

import torch
import triton
import triton.language as tl

from .kernels import _fwd_kernel, _find_start_index_kernel


@triton.jit
def _prefix_kernel(X, Y, N: tl.constexpr, H: tl.constexpr,
                   XB: tl.constexpr, XH: tl.constexpr, XT: tl.constexpr,
                   C: tl.constexpr):
    h = tl.program_id(0)
    b = tl.program_id(1)
    i = tl.arange(0, C)
    carry = tl.full((), 0, tl.float32)
    for start in range(0, N, C):
        t = start + i
        x = tl.load(X + b * XB + h * XH + t * XT, t < N, other=0)
        x = tl.where(i == 0, x + carry, x)
        x = tl.cumsum(x)
        tl.store(Y + (b * H + h) * N + t, x, t < N)
        carry = tl.sum(tl.where(i == C - 1, x, 0), 0)


def parallel_forgetting_attn(
    q, k, v, log_fgate, *, head_first=False, seq_start=None,
    sm_scale=None, adaptive_threshold=None,
):
    """Forward causal gated attention; no automatic or tensor ACP thresholds."""
    if seq_start is not None:
        raise NotImplementedError("MetaX Forgetting Attention does not support seq_start")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, log_fgate)):
        raise NotImplementedError("MetaX Forgetting Attention is forward-only")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("require rank-four Q/K/V and identical K/V shapes")
    if not head_first:
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        if log_fgate.ndim != 3:
            raise ValueError("log_fgate must have rank three")
        log_fgate = log_fgate.transpose(1, 2)
    b, h, m, d = q.shape
    n, hk = k.shape[2], k.shape[1]
    if d not in (16, 32, 60, 64, 100, 128):
        raise ValueError("supported head dimensions are 16, 32, 60, 64, 100, 128")
    if b <= 0 or h <= 0 or hk <= 0 or h % hk or not 0 < m <= n or k.shape != (b, hk, n, d):
        raise ValueError("invalid batch, sequence, head grouping or feature dimensions")
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise NotImplementedError("require C550 CUDA-compatible FP16/BF16 tensors")
    if any(x.device != q.device or x.dtype != q.dtype for x in (k, v)):
        raise ValueError("Q/K/V must share device and dtype")
    if log_fgate.shape != (b, h, n) or log_fgate.dtype != torch.float32 or log_fgate.device != q.device:
        raise ValueError("log_fgate must be same-device FP32 with matching batch, query heads and KV length")
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    if not isinstance(scale, (float, int)) or not math.isfinite(scale):
        raise ValueError("sm_scale must be finite")
    if adaptive_threshold is not None and (
        not isinstance(adaptive_threshold, (float, int))
        or not math.isfinite(adaptive_threshold) or adaptive_threshold > 0
    ):
        raise NotImplementedError("adaptive_threshold must be None or a finite nonpositive scalar")
    block_d = triton.next_power_of_2(d)
    bm = 1 if m == 1 else (64 if block_d <= 64 else 128)
    compute_bm = 32 if m > 1 and block_d > 64 else bm
    bn = min(128, max(16, triton.next_power_of_2(n))) if m == 1 else 32
    with torch.cuda.device(q.device):
        prefix = torch.empty((b, h, n), device=q.device, dtype=torch.float32)
        _prefix_kernel[(h, b)](log_fgate, prefix, n, h, *log_fgate.stride(), 256,
                               num_warps=4, enable_fp_fusion=False)
        starts = torch.empty((b, h, triton.cdiv(m, bm)), device=q.device, dtype=torch.int32)
        if adaptive_threshold is not None:
            _find_start_index_kernel[(h, b)](
                prefix, starts, float(adaptive_threshold), *prefix.stride(), *starts.stride(),
                b, h, m, n, n - m, BLOCK_M=bm, BLOCK_N=bn,
                DIVISIBLE_M=m % bm == 0, DIVISIBLE_N=n % bn == 0, num_warps=1,
            )
        out = torch.empty((b, h, m, d), device=q.device, dtype=q.dtype)
        normalizer = torch.empty((b, h, m), device=q.device, dtype=torch.float32)
        _fwd_kernel[(triton.cdiv(m, compute_bm), h, b)](
            q, k, v, prefix, None, starts, float(scale), normalizer, out,
            *q.stride(), *k.stride(), *v.stride(), *prefix.stride(), *starts.stride(), *out.stride(),
            b, h, m, n, n - m, h // hk,
            BLOCK_M=compute_bm, PRUNE_M=bm, BLOCK_N=bn, BLOCK_DMODEL=block_d, REAL_D=d,
            IS_CAUSAL=True, LARGER_M=False, HAS_SEQ_START=False,
            IS_ADAPTIVE=adaptive_threshold is not None,
            DIVISIBLE_M=m % compute_bm == 0, DIVISIBLE_N=n % bn == 0,
            num_warps=4, num_stages=1,
        )
    return out if head_first else out.transpose(1, 2)


forgetting_attention = parallel_forgetting_attn
