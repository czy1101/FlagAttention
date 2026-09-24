# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Forward orchestration for GDN-2 chunkwise training.

import os

import torch

from ..chunk_delta_h import chunk_gated_delta_rule_fwd_h
from ...gla.chunk_gla import chunk_gla_fwd_o_gk
from ...cumsum import chunk_local_cumsum

from .chunk_intra import chunk_gdn2_fwd_intra
from .gate import kda_gate_chunk_cumsum

RCP_LN2 = 1.4426950408889634


def chunk_gdn2_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w_gate: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool | None = None,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
):
    """Top-level GDN-2 forward pipeline.

    The pipeline is:
      1. Compute the base-2 log-decay cumsum within each chunk
         (``kda_gate_chunk_cumsum`` if ``use_gate_in_kernel`` else
         ``chunk_local_cumsum``).
      2. Build the intra-chunk score matrices (Aqk, Akk_inv) and the WY
         auxiliaries (w_wy, u_wy, qg, kg) via ``chunk_gdn2_fwd_intra``.
      3. Run the inter-chunk state recurrence (shared with KDA / GDN v1).
      4. Compose the output via the GLA-style ``chunk_gla_fwd_o_gk``.

    Returns ``(o, final_state, g_cumsum, Aqk, Akk, w_wy, u_wy, qg, kg, v_new,
    h, initial_state)``.
    """
    T = q.shape[1]
    K = k.shape[-1]
    # The Hygon sub-chunk path reuses the K/gate tile across four BC=16
    # sub-chunks.  It is beneficial for large head dimensions or long
    # sequences.  Keep the token-parallel fallback for small K/short T, where
    # the extra sub-chunk scheduling cost can dominate.  Callers can still
    # force either schedule with an explicit bool.
    if safe_gate is None:
        safe_gate = True
    if use_gate_in_kernel:
        g = kda_gate_chunk_cumsum(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(
            # Fuse the base-2 conversion into the cumsum kernel.  The old
            # path materialized ``g.float() * RCP_LN2`` before launching the
            # prefix-sum kernel, creating an avoidable full-size intermediate.
            g=g,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            scale=RCP_LN2,
        )

    # The fused H->O path uses a fixed-length head-major kg layout so the H
    # kernel can read each head's K tile with a compact token stride.
    fuse_h_o = (
        os.environ.get("FLAG_ATTN_GDN2_FUSE_HO", "1") == "1"
        and cu_seqlens is None
        and not return_intermediate_states
        and not disable_recompute
        and K <= 128
        and v.shape[-1] <= 128
        # The fused path is beneficial for the short-sequence launch-bound
        # class and for high-head-count state traffic.  For long sequences
        # with only a few heads, the extra q/g/A work in the H kernel can
        # outweigh the saved state-history traffic, so retain the fallback.
        and (T <= 2048 or q.shape[-2] >= 32)
    )
    # Head-major kg helps when the K tile or head parallelism is large enough
    # to amortize its alternate layout.  Small K=64/H=8 workloads are launch
    # bound and keep the contiguous generic layout instead.
    kg_head_major = (
        fuse_h_o
        and os.environ.get("FLAG_ATTN_GDN2_KG_HEAD_MAJOR", "1") == "1"
        and (K >= 128 or q.shape[-2] >= 16)
    )
    akk_head_major = (
        fuse_h_o
        and safe_gate
        and os.environ.get("FLAG_ATTN_GDN2_AKK_HEAD_MAJOR", "1") == "1"
        and (K >= 128 or q.shape[-2] >= 16)
    )
    w_wy, u_wy, qg, kg, Aqk, Akk = chunk_gdn2_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        b=b,
        w_gate=w_gate,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        store_kg_head_major=kg_head_major,
        store_akk_head_major=akk_head_major,
    )

    # In forward-only fixed-length inference, H and O consume the same
    # chunk-start state.  Stream O directly from the H recurrence to avoid
    # materializing the full [B, num_chunks, H, K, V] history and v_new.
    # Keep the fallback for backward, varlen, and intermediate-state callers.
    o = torch.empty_like(v) if fuse_h_o else None
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w_wy,
        u=u_wy,
        # ``g`` is already a base-2 cumulative gate.  Let the Hygon state
        # kernel consume it directly with exp2, avoiding a full g*ln(2)
        # intermediate tensor and its global write/read.
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        gk_base2=True,
        scale=scale,
        q=q if fuse_h_o else None,
        A=Aqk if fuse_h_o else None,
        o=o,
        fuse_o=fuse_h_o,
        kg_head_major=kg_head_major,
    )
    if state_v_first:
        if h is not None:
            h = h.transpose(-1, -2).contiguous()
        if final_state is not None:
            final_state = final_state.transpose(-1, -2).contiguous()

    if not fuse_h_o:
        o = chunk_gla_fwd_o_gk(
            q=q,
            v=v_new,
            g=g,
            A=Aqk,
            h=h,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )

    if disable_recompute is False:
        # Free intermediates that the backward will recompute.
        w_wy, u_wy, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w_wy, u_wy, qg, kg, v_new, h, initial_state

