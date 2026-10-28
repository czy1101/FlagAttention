# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend recurrent (decode) kernels for GDN v1.

Two interfaces, matching the FLA/vLLM ``gated_delta_rule`` decode API:

* :func:`fused_recurrent_gated_delta_rule_fwd` -- per-token recurrence over a
  (possibly variable-length) sequence; state layout ``[N, HV, V, K]``
  (value-major, as the decode path uses), optional in-place final state and
  continuous-batching state indices.
* :func:`fused_recurrent_gated_delta_rule_packed_decode` -- single-token decode
  from a packed ``mixed_qkv`` projection with the gate/beta computed in-kernel
  from ``a``/``b``/``A_log``/``dt_bias`` (the vLLM Qwen3-Next decode path).

Math is the GDN v1 recurrence used by the chunk path (see
``test_script/reference.py`` and the FLA kernel)::

    S_t = diag(exp(g_t)) S_{t-1}
    v_new = beta_t (v_t - k_t^T S_t)
    S_t = S_t + k_t v_new^T
    o_t = scale q_t^T S_t

The kernels keep the state ``[BV, BK]`` in fp32 registers, use int32 offsets
(the Ascend AIV path is much better with int32), and do not use dots: the
recurrent step is bandwidth/latency bound and the two reductions are vector
reductions over ``BK``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _fused_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
    if T == 0:
        return

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_g = g + bos * HV + i_hv
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    p_o = o + ((bos * HV) + i_hv) * V + o_v

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int32) - 1
            else:
                i_t = 0
            state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int32)
            if state_idx <= 0:
                # Padding slot: nothing to decode, leave ``o`` untouched.
                return
        else:
            state_idx = i_n
        p_h0 = h0 + state_idx * stride_init_state_token
        p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_g = tl.load(p_g).to(tl.float32)
        b_h = b_h * tl.exp(b_g)
        b_v = b_v - tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0.0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v = b_v * b_beta
        b_h = b_h + b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            # Speculative decoding / token-by-token decode: every token has its
            # own state slot, so write the running state out each step.
            final_state_idx = tl.load(
                ssm_state_indices + i_n * stride_indices_seq + i_t
            ).to(tl.int32)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token
                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_g += HV
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
        p_o += HV * V

    if STORE_FINAL_STATE:
        if INPLACE_FINAL_STATE:
            final_idx = i_n if not IS_CONTINUOUS_BATCHING else -1
        else:
            final_idx = i_n
        if final_idx >= 0:
            p_ht = ht + final_idx * stride_final_state_token
            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recurrent (decode) forward with a value-major state ``[N, HV, V, K]``.

    ``q``/``k`` are ``[B, T, H, K]``, ``v`` is ``[B, T, HV, V]``, ``g`` is
    ``[B, T, HV]`` and ``beta`` is ``[B, T, HV]`` or ``[B, T, HV, V]``.
    Returns ``(o, final_state)``; with ``inplace_final_state`` the final state
    is written back to ``initial_state``.
    """
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[-1]
    scale = float(scale)
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError("cu_seqlens requires batch size 1 (packed sequences)")
        N = len(cu_seqlens) - 1
    else:
        N = B

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError("recurrent kernel requires K in one key tile")
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    if inplace_final_state and initial_state is not None:
        final_state = initial_state
    else:
        state_dtype = initial_state.dtype if initial_state is not None else v.dtype
        final_state = q.new_empty(N, HV, V, K, dtype=state_dtype)

    stride_init_state_token = (
        initial_state.stride(0) if initial_state is not None else 0
    )
    stride_final_state_token = final_state.stride(0)
    if ssm_state_indices is None:
        stride_indices_seq = 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq = ssm_state_indices.stride(0)
    else:
        stride_indices_seq = ssm_state_indices.stride(0)

    grid = (NV, N * HV)
    _fused_recurrent_fwd_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o,
        initial_state,
        final_state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens,
        scale,
        T,
        H,
        HV,
        K,
        V,
        BK,
        BV,
        stride_init_state_token,
        stride_final_state_token,
        stride_indices_seq,
        initial_state is not None,
        inplace_final_state,
        beta.ndim == v.ndim,
        use_qk_l2norm_in_kernel,
        cu_seqlens is not None,
        ssm_state_indices is not None,
        num_accepted_tokens is not None,
        True,
        num_warps=4,
        num_stages=2,
    )
    return o, final_state


@triton.jit
def _fused_recurrent_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int32)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx <= 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0.0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(
        p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_v = tl.load(
        p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0.0
    ).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    b_h = b_h * tl.exp(g_val)
    b_v = b_v - tl.sum(b_h * b_k[None, :], 1)
    b_v = b_v * beta_val
    b_h = b_h + b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-token decode from a packed q/k/v projection.

    ``mixed_qkv`` is ``[B, 2*H*K + HV*V]`` (q, k, v packed in the last dim,
    row stride may be larger than the width), ``a``/``b`` are ``[B, HV]`` gate
    logits, ``A_log``/``dt_bias`` are ``[HV]``, ``initial_state`` is
    ``[B, HV, V, K]`` (updated in place), ``out`` is ``[B, 1, HV, V]`` and
    ``ssm_state_indices`` is ``[B]``.  A non-positive state index writes zeros
    to that row and leaves the state untouched (padding slot).
    """
    if mixed_qkv.ndim != 2:
        raise ValueError("mixed_qkv must be 2D [B, packed_dim]")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be contiguous in the last dim")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D [B, HV]")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("a and b must be contiguous in the last dim")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("A_log and dt_bias must be 1D")
    if ssm_state_indices.ndim != 1:
        raise ValueError("ssm_state_indices must be 1D for packed decode")
    if initial_state.ndim != 4:
        raise ValueError("initial_state must be 4D [B, HV, V, K]")
    if initial_state.stride(-1) != 1:
        raise ValueError("initial_state must be contiguous in the last dim")

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError("a/b batch size mismatch")
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError("a/b head count mismatch")
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError("A_log/dt_bias must have HV elements")
    if out.shape != (B, 1, HV, V):
        raise ValueError(f"out must have shape {(B, 1, HV, V)}")
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2:
        raise ValueError("invalid packed mixed_qkv width")
    q_dim = qk_dim // 2
    if q_dim % K:
        raise ValueError("packed Q size must be divisible by K")
    H = q_dim // K
    if H <= 0 or HV % H:
        raise ValueError("invalid packed head configuration")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError("packed decode requires K in one key tile")
    BV = min(triton.next_power_of_2(V), 64)
    NV = triton.cdiv(V, BV)

    grid = (NV, B * HV)
    _fused_recurrent_packed_decode_kernel[grid](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        initial_state,
        initial_state,
        ssm_state_indices,
        float(scale),
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        initial_state.stride(0),
        initial_state.stride(0),
        ssm_state_indices.stride(0),
        H,
        HV,
        K,
        V,
        BK,
        BV,
        20.0,
        use_qk_l2norm_in_kernel,
        num_warps=4,
        num_stages=2,
    )
    return out, initial_state


__all__ = [
    "fused_recurrent_gated_delta_rule_fwd",
    "fused_recurrent_gated_delta_rule_packed_decode",
]
