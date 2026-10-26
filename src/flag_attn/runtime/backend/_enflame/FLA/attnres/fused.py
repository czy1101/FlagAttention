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

"""Forward-only Attention Residuals for the Enflame GCU backend.

One program aggregates one flattened row using FP32 online softmax. Residual
sources are passed as individual pointers to satisfy the GCU launcher ABI.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flag_attn.utils import has_triton_tle

_MAX_RESIDUAL_SOURCES = 16
_MAX_HIDDEN_SIZE = 8192
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)

HAS_TLE_ATTNRES = has_triton_tle()
if HAS_TLE_ATTNRES:
    import triton.experimental.tle.language as tle
else:
    tle = None


if HAS_TLE_ATTNRES:

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
        logits,
        logsumexp,
        N,
        L: tl.constexpr,
        D: tl.constexpr,
        eps: tl.constexpr,
        scale: tl.constexpr,
        BD: tl.constexpr,
        HAS_ONORM: tl.constexpr,
        RETURN_WEIGHTS: tl.constexpr,
        ASYNC_LOAD: tl.constexpr,
    ):
        """Fuse RMS scoring, online source mixing and optional output RMSNorm."""
        # The Enflame launcher accepts individual pointers, but cannot lower
        # a tuple-valued launch argument. Keep all 16 ABI slots here; only the
        # first L sources participate in the statically unrolled loop.
        residuals = (
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
            source_weight = tl.exp(source_score - running_max)
            running_sum = running_sum * rescale + source_weight
            output_values = output_values * rescale + source_weight * residual_values

            if RETURN_WEIGHTS:
                # Store unscaled logits; logsumexp below includes scale.
                tl.store(logits + source * N + row, source_logit)

        output_values /= running_sum
        if HAS_ONORM:
            output_rstd = tl.rsqrt(tl.sum(tl.where(mask_d, output_values * output_values, 0.0), axis=0) / D + eps)
            output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
            output_values *= output_rstd * output_weight

        tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
        if RETURN_WEIGHTS:
            tl.store(logsumexp + row, running_max + tl.log(running_sum))


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
) -> None:
    """Check backend, shape and dtype requirements without reading device data."""
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if not HAS_TLE_ATTNRES:
        raise RuntimeError("fused_attnres requires a Triton build with triton.experimental.tle")
    if len(residuals) > _MAX_RESIDUAL_SOURCES:
        raise ValueError(
            f"fused_attnres supports at most {_MAX_RESIDUAL_SOURCES} residual sources, got {len(residuals)}"
        )

    reference = residuals[0]
    if reference.device.type != "gcu":
        raise ValueError("fused_attnres requires GCU tensors")
    if reference.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"fused_attnres supports float16 and bfloat16 residuals, got {reference.dtype}")
    if reference.ndim == 0:
        raise ValueError("residual tensors must have at least one dimension")
    hidden_size = reference.shape[-1]
    if hidden_size > _MAX_HIDDEN_SIZE:
        raise ValueError(f"fused_attnres supports hidden_size <= {_MAX_HIDDEN_SIZE}, got {hidden_size}")

    for index, residual in enumerate(residuals):
        if residual.shape != reference.shape:
            raise ValueError(f"residual {index} has shape {tuple(residual.shape)}, expected {tuple(reference.shape)}")
        if residual.device != reference.device or residual.dtype != reference.dtype:
            raise ValueError("all residual tensors must have the same device and dtype")

    vectors = (("query", query), ("rms_weight", rms_weight))
    if output_rms_weight is not None:
        vectors += (("output_rms_weight", output_rms_weight),)
    for name, tensor in vectors:
        if tensor.device != reference.device or tensor.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same device and dtype as residuals")
        if tensor.numel() != hidden_size:
            raise ValueError(f"{name} must contain {hidden_size} elements, got {tensor.numel()}")


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

    Each source is RMS-normalized for scoring against ``query`` and scaled by
    ``scale``. A softmax over sources mixes the original, unnormalized residuals.
    Scoring, online softmax and mixing accumulate in FP32; the output is cast
    back to the residual dtype.

    This API is intended for inference and does not provide an autograd formula.
    It is optimized for Kimi K3 shapes (``hidden_size=7168``, up to 9 sources),
    and supports up to 16 sources and hidden sizes up to 8192.

    Args:
        query: Shared scoring query containing D elements.
        residuals: Between 1 and 16 GCU tensors of shape ``(..., D)``, with the
            same shape, device and FP16/BF16 dtype. Noncontiguous inputs are
            copied to contiguous buffers before launching the kernel.
        rms_weight: Scoring RMSNorm weight containing D elements.
        output_rms_weight: Optional output RMSNorm weight containing D elements.
            All query and weight tensors must match the residual device/dtype.
        rms_eps: Epsilon added to the mean square in both RMSNorm operations.
        scale: Multiplier applied to source logits before softmax.
        return_weights: Whether to also return FP32 source probabilities.

    Returns:
        The aggregated tensor, with the same shape and dtype as each residual.
        If ``return_weights`` is true, returns ``(output, probabilities)``, with
        probabilities of shape ``(len(residuals), *residuals[0].shape[:-1])``.
        This option allocates FP32 logits/logsumexp buffers and reconstructs
        probabilities with PyTorch operations after the fused kernel.

    Raises:
        ValueError: The source count, tensor shapes, devices or dtypes do not
            satisfy the input requirements.
        RuntimeError: The Triton build does not provide TLE support.
    """
    _validate_inputs(query, residuals, rms_weight, output_rms_weight)

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    num_sources = len(residuals)
    has_output_norm = output_rms_weight is not None
    flat_residuals = tuple(residual.reshape(-1, hidden_size).contiguous() for residual in residuals)
    num_rows = flat_residuals[0].shape[0]
    # Unused ABI slots alias the first source; the kernel visits only L slots.
    residual_args = {
        f"residual_{index}": flat_residuals[index] if index < num_sources else flat_residuals[0]
        for index in range(_MAX_RESIDUAL_SOURCES)
    }

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logits = torch.empty((num_sources, num_rows), device=output.device, dtype=torch.float32)
        logsumexp = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logits = logsumexp = None

    # Preserve the measured fixed GCU launch without autotuner dispatch.
    _fused_attnres_fwd_kernel[(num_rows,)](
        query=query.reshape(-1).contiguous(),
        **residual_args,
        rms_weight=rms_weight.reshape(-1).contiguous(),
        output_rms_weight=None if output_rms_weight is None else output_rms_weight.reshape(-1).contiguous(),
        output=output,
        logits=logits,
        logsumexp=logsumexp,
        N=num_rows,
        L=num_sources,
        D=hidden_size,
        eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(hidden_size),
        HAS_ONORM=has_output_norm,
        RETURN_WEIGHTS=return_weights,
        ASYNC_LOAD=num_sources <= 2 and not has_output_norm,
        num_warps=1,
        num_stages=1,
    )

    output = output.view(output_shape)
    if return_weights:
        probabilities = (logits * scale - logsumexp[None, :]).exp()
        return output, probabilities.view(num_sources, *output_shape[:-1])
    return output


__all__ = ["fused_attnres"]
