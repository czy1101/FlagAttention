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

import os
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

# Hygon gfx936 uses the regular Triton/HIP backend.
HAS_TRITON_ATTNRES = True


@triton.jit
def _single_source_attnres_fwd_kernel(
    residual,
    output_rms_weight,
    output_rms_bias,
    post_scale,
    output,
    D,
    eps: tl.constexpr,
    BD: tl.constexpr,
    HAS_ONORM: tl.constexpr,
    HAS_ONORM_BIAS: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
):
    """Low-latency path for one residual source without probability output.

    The general kernel still has to materialize query/rms inputs and initialize
    online-softmax state even though L=1 makes the result a copy.  This path
    keeps only the residual load and an optional fused output RMSNorm.
    """
    row = tl.program_id(0).to(tl.int64)
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask_d = offsets_d < D
    residual_ptr = tl.multiple_of(residual + row * D + offsets_d, 16)
    values = tl.load(residual_ptr, mask=mask_d, other=0.0)
    if HAS_ONORM:
        values_f32 = values.to(tl.float32)
        output_rstd = tl.rsqrt(
            tl.sum(values_f32 * values_f32, axis=0) / D + eps
        )
        output_weight = tl.load(
            output_rms_weight + offsets_d, mask=mask_d, other=0.0
        ).to(tl.float32)
        values_f32 *= output_rstd * output_weight
    else:
        # Keep the copy in the source dtype; the FP32 conversion is only
        # needed by the optional RMSNorm reduction.
        values_f32 = (
            values.to(tl.float32)
            if HAS_ONORM_BIAS or HAS_POST_SCALE
            else values
        )
    if HAS_POST_SCALE:
        post_scale_values = tl.load(
            post_scale + offsets_d, mask=mask_d, other=1.0
        ).to(tl.float32)
        values_f32 = values_f32.to(tl.float32) * post_scale_values
    if HAS_ONORM_BIAS:
        output_bias = tl.load(
            output_rms_bias + offsets_d, mask=mask_d, other=0.0
        ).to(tl.float32)
        values_f32 += output_bias
    tl.store(
        output + row * D + offsets_d,
        values_f32.to(output.dtype.element_ty),
        mask=mask_d,
    )


@triton.jit
def _fused_attnres_multrow_kernel(
    query,
    residuals,
    rms_weight,
    output,
    N,
    L: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
    BD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Experimental two-row-per-program source mixer for moderate D."""
    pid = tl.program_id(0).to(tl.int64)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask = row_mask[:, None] & (offsets_d[None, :] < D)

    query_values = tl.load(query + offsets_d, mask=offsets_d < D, other=0.0).to(tl.float32)
    weight_values = tl.load(rms_weight + offsets_d, mask=offsets_d < D, other=0.0).to(tl.float32)
    query_weight = query_values * weight_values
    running_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    output_values = tl.zeros([BLOCK_M, BD], dtype=tl.float32)

    for source in tl.static_range(0, L):
        values = tl.load(
            residuals[source] + rows[:, None] * D + offsets_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(values * values, axis=1) / D + eps)
        source_score = tl.sum(values * query_weight[None, :], axis=1) * rstd * scale
        if source == 0:
            running_max = source_score
            running_sum = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            output_values = values
        else:
            previous_max = running_max
            running_max = tl.maximum(running_max, source_score)
            rescale = tl.exp(previous_max - running_max)
            probability = tl.exp(source_score - running_max)
            running_sum = running_sum * rescale + probability
            output_values = (
                output_values * rescale[:, None]
                + probability[:, None] * values
            )

    output_values /= running_sum[:, None]
    tl.store(
        output + rows[:, None] * D + offsets_d[None, :],
        output_values.to(output.dtype.element_ty),
        mask=mask,
    )


def _prune_attnres_fwd_configs(configs, named_args, **kwargs):
    del named_args
    if kwargs["N_BUCKET"] == 0:
        # Small row buckets are launch/parallelism sensitive.  The no-norm,
        # no-weight-output path has less per-row work and benefits from four
        # waves; output RMSNorm or probability output retains the wider
        # eight/sixteen-wave candidates.
        allowed_warps = (
            (4, 8)
            if not kwargs.get("HAS_ONORM", False)
            and not kwargs.get("RETURN_WEIGHTS", False)
            else (8, 16)
        )
        return [config for config in configs if config.num_warps in allowed_warps]
    return configs


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps, num_stages in (
            (4, 1),
            (4, 2),
            (4, 3),
            (8, 2),
            (16, 2),
        )
    ],
    key=[
        "L", "D", "N_BUCKET", "HAS_ONORM", "HAS_ONORM_BIAS",
        "HAS_POST_SCALE", "RETURN_WEIGHTS",
    ],
    prune_configs_by={"early_config_prune": _prune_attnres_fwd_configs},
)
@triton.jit
def _fused_attnres_fwd_kernel(
    query,
    residuals,
    rms_weight,
    output_rms_weight,
    output_rms_bias,
    post_scale,
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
    HAS_ONORM_BIAS: tl.constexpr,
    HAS_POST_SCALE: tl.constexpr,
    RETURN_WEIGHTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets_d = tl.max_contiguous(tl.multiple_of(tl.arange(0, BD), BD), BD)
    mask_d = offsets_d < D

    query_values = tl.load(query + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    weight_values = tl.load(rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
    query_weight = query_values * weight_values

    running_max = tl.full([], -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros([], dtype=tl.float32)
    output_values = tl.zeros([BD], dtype=tl.float32)

    if L == 1 and not RETURN_WEIGHTS:
        residual_ptr = tl.multiple_of(
            residuals[0] + row * D + offsets_d,
            16,
        )
        output_values = tl.load(
            residual_ptr,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        running_max = tl.zeros([], dtype=tl.float32)
        running_sum = tl.full([], 1.0, dtype=tl.float32)
    elif L == 2:
        residual0_ptr = tl.multiple_of(
            residuals[0] + row * D + offsets_d,
            16,
        )
        residual1_ptr = tl.multiple_of(
            residuals[1] + row * D + offsets_d,
            16,
        )
        residual0 = tl.load(
            residual0_ptr,
            mask=mask_d,
            other=0.0,
        )
        residual1 = tl.load(
            residual1_ptr,
            mask=mask_d,
            other=0.0,
        )
        residual0_norm = residual0.to(tl.float32)
        residual1_norm = residual1.to(tl.float32)

        rstd0 = tl.rsqrt(
            tl.sum(residual0_norm * residual0_norm, axis=0) / D + eps
        )
        rstd1 = tl.rsqrt(
            tl.sum(residual1_norm * residual1_norm, axis=0) / D + eps
        )
        logit0 = tl.sum(residual0_norm * query_weight, axis=0)
        logit1 = tl.sum(residual1_norm * query_weight, axis=0)
        logit0 = logit0 * rstd0
        logit1 = logit1 * rstd1

        score0 = logit0 * scale
        score1 = logit1 * scale
        running_max = tl.maximum(score0, score1)
        probability0 = tl.exp(score0 - running_max)
        probability1 = tl.exp(score1 - running_max)
        running_sum = probability0 + probability1
        output_values = (
            probability0 * residual0_norm
            + probability1 * residual1_norm
        )

        if RETURN_WEIGHTS:
            tl.store(logit + row, logit0)
            tl.store(logit + N + row, logit1)
    else:
        for source in tl.static_range(0, L):
            residual_ptr = tl.multiple_of(
                residuals[source] + row * D + offsets_d,
                16,
            )
            residual_values = tl.load(
                residual_ptr,
                mask=mask_d,
                other=0.0,
            )
            residual_norm = residual_values.to(tl.float32)
            rstd = tl.rsqrt(
                tl.sum(residual_norm * residual_norm, axis=0) / D + eps
            )
            source_logit = (
                tl.sum(residual_norm * query_weight, axis=0) * rstd
            )
            source_score = source_logit * scale

            if source == 0:
                running_max = source_score
                running_sum = 1.0
                output_values = residual_norm
            else:
                previous_max = running_max
                running_max = tl.maximum(running_max, source_score)
                rescale = tl.exp(previous_max - running_max)
                probability = tl.exp(source_score - running_max)
                running_sum = running_sum * rescale + probability
                output_values = (
                    output_values * rescale
                    + probability * residual_norm
                )

            if RETURN_WEIGHTS:
                tl.store(logit + source * N + row, source_logit)
    output_values /= running_sum
    if HAS_ONORM:
        output_rstd = tl.rsqrt(tl.sum(output_values * output_values, axis=0) / D + eps)
        output_weight = tl.load(output_rms_weight + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values *= output_rstd * output_weight
    if HAS_POST_SCALE:
        output_scale = tl.load(post_scale + offsets_d, mask=mask_d, other=1.0).to(tl.float32)
        output_values *= output_scale
    if HAS_ONORM_BIAS:
        output_bias = tl.load(output_rms_bias + offsets_d, mask=mask_d, other=0.0).to(tl.float32)
        output_values += output_bias

    tl.store(output + row * D + offsets_d, output_values.to(output.dtype.element_ty), mask=mask_d)
    if RETURN_WEIGHTS:
        tl.store(lse + row, running_max + tl.log(running_sum))


def _validate_inputs(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None,
    output_rms_bias: torch.Tensor | None,
    post_scale: torch.Tensor | None,
) -> None:
    if not residuals:
        raise ValueError("residuals must contain at least one tensor")
    if not HAS_TRITON_ATTNRES:
        raise RuntimeError("fused_attnres requires the Triton/HIP backend")
    if len(residuals) > 16:
        raise ValueError(f"fused_attnres supports at most 16 residual sources, got {len(residuals)}")

    reference = residuals[0]
    if not reference.is_cuda:
        raise ValueError("fused_attnres requires CUDA tensors")
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
    if output_rms_bias is not None:
        if output_rms_bias.device != reference.device or output_rms_bias.dtype != reference.dtype:
            raise ValueError("output_rms_bias must have the same device and dtype as residuals")
        if output_rms_bias.numel() != hidden_size:
            raise ValueError(
                f"output_rms_bias must contain {hidden_size} elements, got {output_rms_bias.numel()}"
            )
    if post_scale is not None:
        if post_scale.device != reference.device or post_scale.dtype != reference.dtype:
            raise ValueError("post_scale must have the same device and dtype as residuals")
        if post_scale.numel() != hidden_size:
            raise ValueError(
                f"post_scale must contain {hidden_size} elements, got {post_scale.numel()}"
            )


def fused_attnres(
    query: torch.Tensor,
    residuals: Sequence[torch.Tensor],
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    output_rms_bias: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply forward-only Attention Residuals aggregation.

    Each residual source is RMS-normalized for scoring against ``query``. A
    softmax over the source/depth axis then mixes the original residuals. The
    implementation keeps an online softmax and the mixed hidden vector in
    registers, so every residual element is read once. ``output_rms_weight``
    optionally fuses the RMSNorm required by the following sublayer, and
    ``post_scale`` and ``output_rms_bias`` fuse following elementwise scale
    and bias operations into the same output tile.

    This API is intended for inference and does not provide an autograd formula.
    It is optimized for Kimi K3 shapes (``hidden_size=7168``, up to 9 sources),
    and supports up to 16 sources and hidden sizes up to 8192.
    """
    _validate_inputs(
        query, residuals, rms_weight, output_rms_weight, output_rms_bias, post_scale
    )

    output_shape = residuals[0].shape
    hidden_size = output_shape[-1]
    flat_residuals = tuple(residual.reshape(-1, hidden_size).contiguous() for residual in residuals)
    num_rows = flat_residuals[0].shape[0]
    # For a single source, the attention weights are identically one.  Avoid
    # launching the full online-softmax kernel for tiny workloads where its
    # fixed query/weight loads dominate the useful copy (and optional norm).
    small_single_source = (
        len(flat_residuals) == 1
        and not return_weights
        and output_rms_bias is None
        and post_scale is None
        and num_rows <= 4
        and hidden_size <= 1024
        and os.environ.get("FLAG_ATTN_ATTNRES_SMALL_PATH", "1").lower()
        not in {"0", "false", "off"}
    )
    output_rms_weight_flat = (
        None
        if output_rms_weight is None
        else output_rms_weight.reshape(-1).contiguous()
    )
    output_rms_bias_flat = (
        None
        if output_rms_bias is None
        else output_rms_bias.reshape(-1).contiguous()
    )
    post_scale_flat = (
        None if post_scale is None else post_scale.reshape(-1).contiguous()
    )

    if small_single_source:
        if output_rms_weight is None:
            # L=1 has no attention work left: the result is an independent
            # copy of the only residual.  A native device copy avoids paying
            # a Triton launch for this degenerate case.
            copy_output = torch.empty_like(flat_residuals[0])
            copy_output.copy_(flat_residuals[0])
            return copy_output.view(output_shape)
        output = torch.empty_like(flat_residuals[0])
        _single_source_attnres_fwd_kernel[(num_rows,)](
            residual=flat_residuals[0],
            output_rms_weight=output_rms_weight_flat,
            output_rms_bias=output_rms_bias_flat,
            post_scale=post_scale_flat,
            output=output,
            D=hidden_size,
            eps=rms_eps,
            BD=triton.next_power_of_2(hidden_size),
            HAS_ONORM=output_rms_weight is not None,
            HAS_ONORM_BIAS=output_rms_bias is not None,
            HAS_POST_SCALE=post_scale is not None,
            num_warps=1,
            num_stages=1,
        )
        return output.view(output_shape)

    multirow = (
        not return_weights
        and output_rms_weight is None
        and output_rms_bias is None
        and post_scale is None
        and len(flat_residuals) >= 3
        # Two rows per program is useful while the row count is large enough
        # to amortize the launch, but the regular one-row schedule wins once
        # the grid is already fully occupied.  The range is a gfx936 tuning
        # heuristic, not a public-shape special case.
        and 768 <= num_rows <= 1536
        and hidden_size <= 2048
        and os.environ.get("FLAG_ATTN_ATTNRES_MULTIROW", "1").lower()
        not in {"0", "false", "off"}
    )
    # When D is small and naturally aligned, the four-row tile stays within a
    # modest VGPR footprint and amortizes the per-program query/weight loads
    # better than the one-row schedule.  The 128-element alignment avoids a
    # padded tail that can erase the benefit for irregular hidden sizes.  This
    # is a shape-derived schedule rule (D controls the live vector size), not
    # a case-specific lookup.  Keep small N on the regular path because the
    # fixed four-row tile cannot amortize its launch cost there.
    small_d_multirow = (
        not return_weights
        and output_rms_weight is None
        and output_rms_bias is None
        and post_scale is None
        and len(flat_residuals) >= 3
        and num_rows >= 128
        and hidden_size <= 512
        and hidden_size % 128 == 0
        and os.environ.get("FLAG_ATTN_ATTNRES_SMALL_D_MULTIROW", "1").lower()
        not in {"0", "false", "off"}
    )
    if small_d_multirow:
        output = torch.empty_like(flat_residuals[0])
        _fused_attnres_multrow_kernel[(triton.cdiv(num_rows, 4),)](
            query=query.reshape(-1).contiguous(),
            residuals=flat_residuals,
            rms_weight=rms_weight.reshape(-1).contiguous(),
            output=output,
            N=num_rows,
            L=len(residuals),
            D=hidden_size,
            eps=rms_eps,
            scale=scale,
            BD=triton.next_power_of_2(hidden_size),
            BLOCK_M=4,
            num_warps=4,
            num_stages=2,
        )
        return output.view(output_shape)

    if multirow:
        output = torch.empty_like(flat_residuals[0])
        _fused_attnres_multrow_kernel[(triton.cdiv(num_rows, 2),)](
            query=query.reshape(-1).contiguous(),
            residuals=flat_residuals,
            rms_weight=rms_weight.reshape(-1).contiguous(),
            output=output,
            N=num_rows,
            L=len(residuals),
            D=hidden_size,
            eps=rms_eps,
            scale=scale,
            BD=triton.next_power_of_2(hidden_size),
            BLOCK_M=2,
            num_warps=4,
            num_stages=2,
        )
        return output.view(output_shape)

    output = torch.empty_like(flat_residuals[0])
    if return_weights:
        logit = torch.empty((len(residuals), num_rows), device=output.device, dtype=torch.float32)
        lse = torch.empty((num_rows,), device=output.device, dtype=torch.float32)
    else:
        logit = lse = None

    _fused_attnres_fwd_kernel[(num_rows,)](
        query=query.reshape(-1).contiguous(),
        residuals=flat_residuals,
        rms_weight=rms_weight.reshape(-1).contiguous(),
        output_rms_weight=None if output_rms_weight is None else output_rms_weight.reshape(-1).contiguous(),
        output_rms_bias=None if output_rms_bias is None else output_rms_bias.reshape(-1).contiguous(),
        post_scale=None if post_scale is None else post_scale.reshape(-1).contiguous(),
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
        HAS_ONORM_BIAS=output_rms_bias is not None,
        HAS_POST_SCALE=post_scale is not None,
        RETURN_WEIGHTS=return_weights,
    )

    output = output.view(output_shape)
    if return_weights:
        probability = (logit * scale - lse[None, :]).exp()
        return output, probability.view(len(residuals), *output_shape[:-1])
    return output


__all__ = ["HAS_TRITON_ATTNRES", "fused_attnres"]
