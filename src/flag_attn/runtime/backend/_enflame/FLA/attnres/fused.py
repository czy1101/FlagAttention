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

from collections.abc import Sequence
import os

import torch
import triton
import triton.language as tl

from flag_attn.utils import has_triton_tle

from ..compat import libtuner


HAS_TLE_ATTNRES = has_triton_tle()
if HAS_TLE_ATTNRES:
    import triton.experimental.tle.language as tle
else:
    tle = None


_ATTNRES_AUTOTUNE_ENABLED = os.environ.get("FLAG_ATTN_ATTNRES_AUTOTUNE", "0") == "1"


def _attnres_fwd_configs():
    return [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in ((1, 1), (2, 1), (4, 1), (4, 2))
    ]


def _attnres_fwd_autotune(kernel):
    # A single fixed launch does not need the autotuner's host-side dispatch.
    if _ATTNRES_AUTOTUNE_ENABLED:
        return libtuner(
            configs=_attnres_fwd_configs(),
            key=["L", "D", "N_BUCKET", "HAS_ONORM", "RETURN_WEIGHTS", "ASYNC_LOAD"],
        )(kernel)
    return kernel


if HAS_TLE_ATTNRES:

    @_attnres_fwd_autotune
    @triton.jit
    def _fused_attnres_fwd_kernel(
        query,
        residual_0,
        residual_1,
        residual_2,
        residual_3,
        residual_4,
        residual_5,
        residual_6,
        residual_7,
        residual_8,
        residual_9,
        residual_10,
        residual_11,
        residual_12,
        residual_13,
        residual_14,
        residual_15,
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
        BD: tl.constexpr,
        HAS_ONORM: tl.constexpr,
        RETURN_WEIGHTS: tl.constexpr,
        ASYNC_LOAD: tl.constexpr,
    ):
        # The Enflame launcher accepts individual pointers, but cannot lower
        # a tuple-valued launch argument. Keep the tuple inside the JIT body.
        residuals = (
            residual_0, residual_1, residual_2, residual_3,
            residual_4, residual_5, residual_6, residual_7,
            residual_8, residual_9, residual_10, residual_11,
            residual_12, residual_13, residual_14, residual_15,
        )
        row = tl.program_id(0).to(tl.int64)
        offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
        mask_d = offsets_d < D

        query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        query_weight = query_values * weight_values

        running_max = tl.full([], -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros([], dtype=tl.float32)
        output_values = tl.zeros([BD], dtype=tl.float32)

        # Each statically selected source has its own scalar base pointer.
        # Pointer-valued tl.where lowers to an unsupported arith.select on GCU.
        for source in tl.static_range(0, L):
            residual_values = tle.load(
                residuals[source] + row * D + offsets_d,
                mask=mask_d,
                other=0.0,
                eviction_policy="evict_first",
                is_async=ASYNC_LOAD,
            ).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(residual_values * residual_values, axis=0) / D + eps)
            source_logit = tl.sum(residual_values * query_weight, axis=0) * rstd
            source_score = source_logit * scale

            previous_max = running_max
            running_max = tl.maximum(running_max, source_score)
            rescale = tl.exp(previous_max - running_max)
            probability = tl.exp(source_score - running_max)
            running_sum = running_sum * rescale + probability
            output_values = output_values * rescale + probability * residual_values

            if RETURN_WEIGHTS:
                tl.store(logit + source * N + row, source_logit)

        output_values /= running_sum
        if HAS_ONORM:
            output_rstd = tl.rsqrt(tl.sum(tl.where(mask_d, output_values * output_values, 0.0), axis=0) / D + eps)
            output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
            output_values *= output_rstd * output_weight

        tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
        if RETURN_WEIGHTS:
            tl.store(lse + row, running_max + tl.log(running_sum))


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
) -> None:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if not HAS_TLE_ATTNRES:
        raise RuntimeError("fused_attnres requires a Triton build with triton.experimental.tle")
    if len(residuals) > 16:
        raise ValueError(f"fused_attnres supports at most 16 residual sources, got {len(residuals)}")

    reference = residuals[0]
    if reference.device.type != "gcu":
        raise ValueError("fused_attnres requires GCU tensors")
    if reference.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference.shape[-1]
    if hidden_size > 8192:
        raise ValueError(f"fused_attnres supports hidden_size <= 8192, got {hidden_size}")

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
    implementation keeps an online softmax and the mixed hidden vector in
    registers, so every residual element is read once. ``output_rms_weight``
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
    residual_args = {
        f"residual_{index}": flat_residuals[index] if index < len(flat_residuals) else flat_residuals[0]
        for index in range(16)
    }

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    launch_options = {} if _ATTNRES_AUTOTUNE_ENABLED else {"num_warps": 1, "num_stages": 1}
    _fused_attnres_fwd_kernel[(num_rows,)](
        query=query.reshape(-1).contiguous(),
        **residual_args,
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
        ASYNC_LOAD=len(residuals) <= 2 and output_rms_weight is None,
        **launch_options,
    )

    output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
