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

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in ((8, 1), (8, 2), (8, 3))
    ],
    key=["L", "D", "N_BUCKET", "HAS_ONORM", "RETURN_WEIGHTS"],
)
@triton.jit
def _fused_attnres_fwd_kernel(
    query,
    residuals,
    rms_weight,
    output_rms_weight,
    output,
    logit,
    lse,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    N_BUCKET: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BL: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    RETURN_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask_d = offsets_d < D

    query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    query_weight = query_values * weight_values

    # At most 16 source scores: normalize once before mixing the residuals.
    offsets_l = tl.arange(0, BL)
    source_score = tl.full([BL], -float("inf"), dtype=tl.float32)
    for source in tl.static_range(L):
        residual_values = tl.load(residuals[source] + row * D + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(residual_values * residual_values, axis=0) / D + eps)
        source_logit = tl.sum(residual_values * query_weight, axis=0) * rstd
        source_score = tl.where(offsets_l == source, source_logit * scale, source_score)
        if RETURN_WEIGHTS:
            tl.store(logit + source * N + row, source_logit)

    maximum = tl.max(source_score, axis=0)
    probability = tl.exp(source_score - maximum)
    denominator = tl.sum(probability, axis=0)
    probability /= denominator
    output_values = tl.zeros([BD], dtype=tl.float32)
    for source in tl.static_range(L):
        source_probability = tl.sum(tl.where(offsets_l == source, probability, 0.0), axis=0)
        # Reread real input instead of keeping all source vectors live.
        residual_values = tl.load(
            residuals[source] + row * D + offsets_d, mask=mask_d, other=0.0, volatile=True
        ).to(tl.float32)
        output_values += source_probability * residual_values

    if HAS_ONORM:
        output_rstd = tl.rsqrt(tl.sum(tl.where(mask_d, output_values * output_values, 0.0), axis=0) / D + eps)
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values *= output_rstd * output_weight

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        tl.store(lse + row, maximum + tl.log(denominator))


def _padded_residual_tuple(residuals: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    padded_length = max(8, triton.next_power_of_2(len(residuals)))
    return tuple(residuals) + (residuals[0],) * (padded_length - len(residuals))


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
) -> None:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if len(residuals) > 16:
        raise ValueError(f"fused_attnres supports at most 16 residual sources, got {len(residuals)}")

    reference = residuals[0]
    if reference.device.type != "musa":
        raise ValueError("fused_attnres requires MUSA tensors")
    if reference.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference.shape[-1]
    if not 0 < hidden_size <= 8192:
        raise ValueError(f"fused_attnres requires 0 < hidden_size <= 8192, got {hidden_size}")

    for index, residual in enumerate(residuals):
        if residual.shape != reference.shape:
            raise ValueError(
                f"residual {index} has shape {tuple(residual.shape)}, expected {tuple(reference.shape)}"
            )
        if residual.device != reference.device or residual.dtype != reference.dtype:
            raise ValueError("all residual tensors must have the same device and dtype")

    for name, tensor in (("query", query), ("rms_weight", rms_weight)):
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same device and dtype as residuals")
        if tensor.numel() != hidden_size:
            raise ValueError(f"{name} must contain {hidden_size} elements, got {tensor.numel()}")
    if output_rms_weight is not None:
        if output_rms_weight.device != reference.device or output_rms_weight.dtype != reference.dtype:
            raise ValueError("output_rms_weight must have the same device and dtype as residuals")
        if output_rms_weight.numel() != hidden_size:
            raise ValueError(
                f"output_rms_weight must contain {hidden_size} elements, got {output_rms_weight.numel()}"
            )


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply forward-only Attention Residuals aggregation.

    Each residual source is RMS-normalized for scoring against ``query``. A
    softmax over the source/depth axis then mixes the original residuals. The
    MUSA implementation computes all source scores before softmax, then
    rereads residuals for the weighted sum. ``output_rms_weight``
    optionally fuses the RMSNorm required by the following sublayer.

    This API is intended for inference and does not provide an autograd formula.
    It is optimized for Kimi K3 shapes (``hidden_size=7168``, up to 9 sources),
    and supports up to 16 sources and hidden sizes up to 8192.
    """
    _validate_inputs(query, residuals, rms_weight, output_rms_weight)

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    flat_residuals = tuple(residual.reshape(-1, hidden_size).contiguous() for residual in residuals)
    num_rows = flat_residuals[0].shape[0]
    padded_residuals = _padded_residual_tuple(flat_residuals)

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    if num_rows == 0:
        if return_weights:
            return output.view(output_shape), logit.view(len(residuals), *output_shape[:-1])
        return output.view(output_shape)

    _fused_attnres_fwd_kernel[(num_rows,)](
        query=query.reshape(-1).contiguous(),
        residuals=padded_residuals,
        rms_weight=rms_weight.reshape(-1).contiguous(),
        output_rms_weight=None if output_rms_weight is None else output_rms_weight.reshape(-1).contiguous(),
        output=output,
        logit=logit,
        lse=lse,
        N=num_rows,
        L=len(residuals),
        D=hidden_size,
        N_BUCKET=0 if num_rows <= 128 else (1 if num_rows <= 1024 else 2),
        eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=output_rms_weight is not None,
        RETURN_WEIGHTS=return_weights,
        BL=triton.next_power_of_2(len(residuals)),
    )

    output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
