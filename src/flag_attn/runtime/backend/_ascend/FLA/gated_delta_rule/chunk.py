# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend orchestration for the GDN v1 chunk forward.

Three paths:

* **fused path** (bf16/fp16, ``K <= 128``, ``V <= 128``, default chunk 64): three
  Triton launches -- ``cumsum(+fp32 beta)`` -> ``prep_w_u`` -> ``fwd_h_o``.
  ``fused_ascend.py`` merges the KKT/Newton inverse with ``recompute_w_u``
  (``A_inv`` never hits GM) and the state scan with the output stage (``h`` and
  ``v_new`` never hit GM).  This is the default because the staged pipeline was
  launch/host bound (0.867 of 0.905 ms was host enqueue time on T=1024/H=8),
  not compute bound.  ``FLAG_ATTN_ASCEND_GDN_FUSE=0`` selects the staged fast
  path (5 launches) instead.
* **staged fast path**: the same math in five Ascend kernels
  (``wy_fast_ascend.py``, ``fwd_h_ascend.py``, ``chunk_o_ascend.py``), kept for
  A/B and as a fallback.
* **fallback path** (``K``/``V`` > 128, or
  ``FLAG_ATTN_ASCEND_GDN_FAST=0``): the previous implementation -- ``kkt.py``
  prep plus the generic ``wy_fast``/``chunk_delta_h``/``chunk_o`` kernels at
  ``BT=32``, so fp16 and K=256 keep working.

All Ascend kernels here share the same measured rules (see the module
docstrings): no autotune, K in a single tile, and per-token decay applied
through *2D factor tiles gathered from fp32 buffers* -- a 1D->2D broadcast
multiply costs ~6 us/program and a bf16 2D gather ~0.5 ms/128 tiles on 910B,
vs ~0.07 ms for an fp32 gather.

``FLAG_ATTN_ASCEND_GDN_BT`` selects the internal chunk size (default 64, the
same chunk CANN's AscendC fused op uses); BT=32 remains available as an
explicit request and as the fallback chunk size.
"""

from __future__ import annotations

import os

import torch
import triton

from flag_attn.FLA.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from flag_attn.FLA.index import prepare_chunk_indices
from flag_attn.FLA.chunk_o import chunk_fwd_o
from flag_attn.FLA.gated_delta_rule.wy_fast import recompute_w_u_fwd

from .chunk_o_ascend import chunk_fwd_o_ascend
from .fused_ascend import (
    fold_ascend,
    fwd_h_o_2pass_ascend,
    fwd_h_o_ascend,
    prep_affine_ascend,
    prep_w_u_ascend,
)
from .fwd_h_ascend import chunk_gated_delta_rule_fwd_h_ascend
from .kkt import chunk_gated_delta_rule_prep
from .wy_fast_ascend import (
    chunk_gated_delta_rule_prep_ascend,
    chunk_local_cumsum_ascend,
    recompute_w_u_fwd_ascend,
)

_BT_ENV = "FLAG_ATTN_ASCEND_GDN_BT"
_FAST_ENV = "FLAG_ATTN_ASCEND_GDN_FAST"
_FUSE_ENV = "FLAG_ATTN_ASCEND_GDN_FUSE"
_RAW_INVERSE_ENV = "FLAG_ATTN_ASCEND_GDN_RAW_INVERSE"
_RAW_INVERSE_MIN_JOBS = 2048

_OFF = {"0", "false", "off", "no"}

# Chunk count from which the exact two-pass state scan beats the fused
# sequential one (measured crossover; see fused_ascend._affine_kernel).
_TWOPASS_MIN_CHUNKS = 32

def _use_fused() -> bool:
    """Fused (fold+prep_w_u+fwd_h_o) is the default; the staged 5-launch path
    remains available via ``FLAG_ATTN_ASCEND_GDN_FUSE=0`` (A/B and fallback)."""
    forced = os.environ.get(_FUSE_ENV, "").lower()
    return forced not in _OFF if forced else True


def ascend_chunk_size() -> int:
    return int(os.environ.get(_BT_ENV, "64"))


def _use_fast_path(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if os.environ.get(_FAST_ENV, "1").lower() in _OFF:
        return False
    # The fused fold kernel tiles v_beta with the K slice, so it requires
    # K == V; other widths use the staged fast path.
    return (
        q.dtype in (torch.bfloat16, torch.float16)
        and k.shape[-1] <= 128
        and v.shape[-1] <= 128
        and k.shape[-1] == v.shape[-1]
    )


def _ascend_wu_tile(dim: int) -> int:
    """Return a dot-compatible power-of-two tile, capped at 128."""
    return min(128, max(16, 1 << (dim - 1).bit_length()))


def _fwd_legacy(q, k, v, g, beta, scale, initial_state, output_final_state,
                cu_seqlens, bt):
    g_cum, A = chunk_gated_delta_rule_prep(
        g=g,
        k=k,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
        output_dtype=k.dtype,
    )
    block_v = 128 if v.shape[-1] == 128 else 64
    pipe_stages = 5 if v.shape[-1] >= 256 else (3 if v.shape[-1] == 128 else 2)
    wu_ascend = (
        q.dtype in (torch.bfloat16, torch.float16)
        and max(k.shape[-1], v.shape[-1]) > 128
        and k.shape[-1] in (64, 128, 256)
        and 0 < v.shape[-1] <= 256
    )
    if wu_ascend:
        # This only replaces legacy W/U with an API key width accepted by the
        # full-K prep dot and dimensions <=256. The kernel supports fixed and
        # packed-varlen W/U (including chunk-index
        # lookup) and independent K/V tails; round each tile up to a valid
        # power of two (capped at 128) so e.g. V=192 uses masked 128-wide
        # tiles.  The <=128 bf16 fused path never reaches here.
        w, u = recompute_w_u_fwd_ascend(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g_cum,
            cu_seqlens=cu_seqlens,
            bk=_ascend_wu_tile(k.shape[-1]),
            bv=_ascend_wu_tile(v.shape[-1]),
            num_warps=4,
            num_stages=2,
        )
    else:
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g_cum,
            cu_seqlens=cu_seqlens,
            block_v=block_v,
            pipe_stages=pipe_stages,
        )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_cum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=bt,
        cu_seqlens=cu_seqlens,
    )
    output_bv = min(128, max(16, triton.next_power_of_2(v.shape[-1])))
    fixed_output_supported = (
        q.dtype in (torch.bfloat16, torch.float16)
        and k.shape[-1] in (64, 128, 256)
        and 0 < v.shape[-1] <= 256
        and cu_seqlens is None
    )
    # Use the measured one-warp output kernel for fixed-length FP16 shapes in
    # this supported width range.  Besides staying within ACL's coreDim limit
    # on large grids, this avoids the generic autotuner's much slower output
    # configs (the B8/T1024/H8/K=V=64 FP16 output stage is ~19% faster here).
    # Keep the established K=V=256 bf16 route as well.
    needs_fixed_output = q.dtype == torch.float16 or (
        k.shape[-1] == 256 and v.shape[-1] == 256
    )
    if fixed_output_supported and needs_fixed_output:
        # The fixed-config output kernel bounds each launch to the ACL budget;
        # oversized supported grids are covered by multiple offset flat-PID
        # launches.  A single full-K dot avoids the generic loop-carried fp32
        # dot accumulator, while V is split into 128-wide tiles.  All
        # preceding legacy stages remain unchanged.
        o = chunk_fwd_o_ascend(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cum,
            scale=scale,
            cu_seqlens=None,
            chunk_size=bt,
            bv=output_bv,
            num_warps=1,
        )
    else:
        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g_cum,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=bt,
        )
    return g_cum, o, A, final_state


def _fwd_fast(q, k, v, g, beta, scale, initial_state, output_final_state,
              cu_seqlens, bt, head_first_strided=False):
    # One vector-only pass produces the chunk-local cumsum and an fp32 copy of
    # ``beta``; the fp32 copy matters because 2D gathers of bf16 data are ~10x
    # slower than fp32 gathers on this part (see wy_fast_ascend.py).
    if _use_fused():
        # The normal path uses vector folds -> prep (KKT/Newton+w/u+A) ->
        # state scan + output. Long fixed sequences fuse prep with the affine
        # precompute below, removing the W/U GM round trip, the A reload, and
        # one launch. A is still written because this function returns it.
        g_cum, qe, ke, kr, vb = fold_ascend(
            g, q, k, v, beta, cu_seqlens, bt,
            head_first_strided=head_first_strided,
        )
        # Optional exact two-pass state scan (pass1 chunk-parallel affine
        # precompute, pass2 with two dots per chunk; see _affine_kernel in
        # fused_ascend.py).  Measured: fwd_h_o device 0.256 -> 0.170 ms at
        # T2048/H8, while the extra pass costs 0.109 ms of device + ~0.10 ms of
        # host; it only pays off once the sequential chain is long, so it is
        # enabled for NT >= 32 (T2048/H8) and kept off below that.  Override
        # with FLAG_ATTN_ASCEND_GDN_TWOPASS=1/0.
        two_pass_env = os.environ.get("FLAG_ATTN_ASCEND_GDN_TWOPASS", "")
        if two_pass_env:
            two_pass = two_pass_env.lower() not in _OFF
        else:
            nt = len(prepare_chunk_indices(cu_seqlens, bt)) if cu_seqlens is not None else -(-g.shape[1] // bt)
            two_pass = cu_seqlens is None and nt >= _TWOPASS_MIN_CHUNKS
        if two_pass and cu_seqlens is None:
            raw_inverse = (
                os.environ.get(_RAW_INVERSE_ENV, "1").lower() not in _OFF
                and bt == 64
                and q.shape[0] * triton.cdiv(q.shape[1], bt) * beta.shape[2]
                >= _RAW_INVERSE_MIN_JOBS
                and k.shape[-1] == v.shape[-1]
                and 0 < k.shape[-1] <= 128
                and q.dtype in (torch.bfloat16, torch.float16)
            )
            if raw_inverse:
                # This experimental route defaults on only for workloads with
                # at least 2048 independent chunk/head jobs. Smaller grids lose
                # to fused prep because Raw adds inverse materialization work.
                # One-pass and varlen remain on the established fused prep.
                from .raw_inverse_prep import raw_inverse_prep_affine_ascend

                qe2, au, cu, mw, a_masked = raw_inverse_prep_affine_ascend(
                    qe, ke, kr, vb, bt, output_dtype=q.dtype
                )
            else:
                qe2, au, cu, mw, a_masked = prep_affine_ascend(
                    qe, ke, kr, vb, bt, output_dtype=q.dtype
                )
            o, final_state = fwd_h_o_2pass_ascend(
                qe2, au, cu, mw, g_cum, scale, initial_state,
                output_final_state, bt,
            )
            return g_cum, o, a_masked, final_state
        w, u, a_masked = prep_w_u_ascend(
            qe, ke, kr, vb, cu_seqlens, bt, output_dtype=q.dtype
        )
        o, final_state = fwd_h_o_ascend(
            qe, a_masked, kr, w, u, g_cum, scale, initial_state,
            output_final_state, cu_seqlens, bt,
        )
        return g_cum, o, a_masked, final_state
    g_cum, beta32 = chunk_local_cumsum_ascend(
        g, beta=beta, cu_seqlens=cu_seqlens, chunk_size=bt
    )
    g_cum, A = chunk_gated_delta_rule_prep_ascend(
        g=g,
        k=k,
        beta=beta32,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
        output_dtype=k.dtype,
        g_cum=g_cum,
    )
    w, u = recompute_w_u_fwd_ascend(
        k=k,
        v=v,
        beta=beta32,
        A=A,
        g_cumsum=g_cum,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_ascend(
        k=k,
        w=w,
        u=u,
        g=g_cum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=bt,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o_ascend(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=bt,
    )
    return g_cum, o, A, final_state


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int | None = None,
    head_first_strided: bool = False,
) -> tuple:
    """Forward pass; returns the generic ``(g, o, A, final_state, w, h, v_new)``."""
    chunk_size = int(chunk_size or ascend_chunk_size())
    if _use_fast_path(q, k, v):
        g_cum, o, A, final_state = _fwd_fast(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, chunk_size, head_first_strided,
        )
    else:
        # fp16 / wide head dims: the generic path is only UB-safe at BT<=32.
        g_cum, o, A, final_state = _fwd_legacy(
            q, k, v, g, beta, scale, initial_state, output_final_state,
            cu_seqlens, min(chunk_size, 32),
        )
    return g_cum, o, A, final_state, None, None, None


__all__ = ["chunk_gated_delta_rule_fwd", "ascend_chunk_size"]
