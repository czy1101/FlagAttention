# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hygon public GDN-2 entry point.

The Hygon backend has its own native Triton pipeline.  This adapter keeps the
public ``flag_attn.chunk_gdn2`` contract while hiding the backend pipeline's
training-style intermediate return tuple.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from flag_attn.FLA.index import prepare_chunk_indices

from .native.chunk_fwd import chunk_gdn2_fwd

__all__ = ["chunk_gdn2"]


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    return_intermediate_states: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    """Run Hygon's native GDN-2 implementation with the public API shape."""
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    (
        o,
        final_state,
        _g,
        _Aqk,
        _Akk,
        _w_wy,
        _u_wy,
        _qg,
        _kg,
        _v_new,
        h,
        _initial_state,
    ) = chunk_gdn2_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w_gate=w,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        state_v_first=state_v_first,
        use_gate_in_kernel=use_gate_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        return_intermediate_states=return_intermediate_states,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
    )
    if return_intermediate_states:
        return o, final_state, h
    return o, final_state
