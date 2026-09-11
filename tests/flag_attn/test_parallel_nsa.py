# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import pytest
import torch
import triton
from einops import repeat

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

from flag_attn.runtime.backend._mthreads.gated_linear_attention.index import (
    prepare_token_indices,
)
from flag_attn.runtime.backend._mthreads.nsa import parallel_nsa


MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()
pytestmark = pytest.mark.skipif(not MUSA_AVAILABLE, reason="parallel NSA tests require MUSA")

logger = logging.getLogger(__name__)


# ===========================================================================
# Naive reference implementation (from flash-linear-attention)
# ===========================================================================


def naive_nsa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""
    Naive PyTorch reference for NSA selected-sparse attention.

    Args:
        q: queries of shape ``[B, T, HQ, K]``.
        k: keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        block_indices: block indices of shape ``[B, T, H, S]``.
        block_size: selected block size. Default: 64.
        scale: scale factor. If None, defaults to ``K ** -0.5``.
        cu_seqlens: cumulative sequence lengths for variable-length sequences.

    Returns:
        o: outputs of shape ``[B, T, HQ, V]``.
    """
    if "head_first" in kwargs:
        raise DeprecationWarning("head_first has been removed.")
    if scale is None:
        scale = k.shape[-1] ** -0.5

    dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    BS = block_size
    k, v, block_indices = (
        repeat(x, "b t h d -> b t (h g) d", g=G)
        for x in (k, v, block_indices)
    )
    q, k, v = map(lambda x: x.float(), (q, k, v))

    o = torch.zeros_like(v)
    varlen = True
    if cu_seqlens is None:
        varlen = False
        B, T = q.shape[:2]
        cu_seqlens = torch.cat(
            [
                block_indices.new_tensor(range(0, B * T, T)),
                block_indices.new_tensor([B * T]),
            ]
        )

    for i in range(len(cu_seqlens) - 1):
        if not varlen:
            q_b, k_b, v_b, i_b = q[i], k[i], v[i], block_indices[i]
        else:
            T = cu_seqlens[i + 1] - cu_seqlens[i]
            q_b, k_b, v_b, i_b = map(
                lambda x: x[0][cu_seqlens[i] : cu_seqlens[i + 1]],
                (q, k, v, block_indices),
            )

        i_b = i_b.unsqueeze(-1) * BS + i_b.new_tensor(range(BS))
        i_b = i_b.view(T, block_indices.shape[2], -1).transpose(1, 2)
        for i_q in range(T):
            q_i = q_b[i_q] * scale
            i_i = i_b[i_q]
            k_i, v_i = map(
                lambda x: x.gather(0, i_i.clamp(0, T - 1).unsqueeze(-1).expand(*i_i.shape, x.shape[-1])),
                (k_b, v_b),
            )
            attn = torch.einsum("h d, n h d -> n h", q_i, k_i).masked_fill(i_i > i_q, float("-inf")).softmax(0)
            if not varlen:
                o[i, i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)
            else:
                o[0][cu_seqlens[i] + i_q] = torch.einsum("n h, n h v -> h v", attn, v_i)

    return o.to(dtype)


def cpu_naive_nsa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    block_indices: torch.LongTensor,
    block_size: int,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the naive reference and its backward pass on CPU for MUSA tests."""
    q_cpu = q.detach().cpu().requires_grad_(True)
    k_cpu = k.detach().cpu().requires_grad_(True)
    v_cpu = v.detach().cpu().requires_grad_(True)
    do_cpu = do.detach().cpu()

    ref_cpu = naive_nsa(
        q=q_cpu,
        k=k_cpu,
        v=v_cpu,
        block_indices=block_indices.cpu(),
        block_size=block_size,
        scale=scale,
        cu_seqlens=None if cu_seqlens is None else cu_seqlens.cpu(),
    )
    ref_cpu.backward(do_cpu)

    return (
        ref_cpu.to(q.device),
        q_cpu.grad.to(q.device),
        k_cpu.grad.to(q.device),
        v_cpu.grad.to(q.device),
    )


# ===========================================================================
# Testing utilities (from fla.utils._testing)
# ===========================================================================


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    error_rate = get_err_ratio(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {error_rate:.6f}"
    logger.info(msg)
    if abs_atol <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{prefix}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{prefix}: NaN detected in tri"
    if warning:
        if error_rate > ratio:
            logger.warning(msg)
    else:
        assert error_rate < ratio, msg


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("B", "T", "H", "HQ", "D", "S", "block_size", "scale", "dtype"),
    [
        pytest.param(*test, id="B{}-T{}-H{}-HQ{}-D{}-S{}-block_size{}-scale{}-{}".format(*test))
        for test in [
            (1, 63, 1, 16, 64, 16, 32, 1.0, torch.float16),
            (3, 111, 1, 32, 100, 16, 32, 1.0, torch.float16),
            (3, 1024, 2, 32, 60, 16, 32, 0.1, torch.float16),
            (3, 1024, 2, 32, 128, 16, 32, 0.1, torch.float16),
            (4, 2048, 2, 32, 64, 16, 32, 0.1, torch.float16),
        ]
    ],
)
def test_parallel(
    B: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    scale: float,
    dtype: torch.dtype,
):
    """Compare FlagAttention parallel_nsa with the naive reference (forward + backward)."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = torch.device("musa")

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device).requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    if device.type == "musa":
        ref, ref_dq, ref_dk, ref_dv = cpu_naive_nsa_reference(
            q=q,
            k=k,
            v=v,
            do=do,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
        )
    else:
        ref = naive_nsa(q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
        ref.backward(do)
        ref_dq, q.grad = q.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        scale=scale,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


@pytest.mark.parallel_nsa
@pytest.mark.parametrize(
    ("H", "HQ", "D", "S", "block_size", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-HQ{}-D{}-S{}-block_size{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (1, 16, 64, 16, 32, [0, 15], torch.float16),
            (2, 32, 64, 16, 32, [0, 256, 500, 1000], torch.float16),
            (2, 32, 100, 16, 32, [0, 15, 100, 300, 1200, 2000], torch.float16),
        ]
    ],
)
def test_parallel_varlen(
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    """Compare FlagAttention parallel_nsa with naive reference for variable-length sequences."""
    torch.manual_seed(42)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

    device = torch.device("musa")

    T = cu_seqlens[-1]
    cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    q = torch.randn((1, T, HQ, D), dtype=dtype, device=device).requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device=device).requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device=device)

    block_indices = torch.full((1, T, H, S), T, dtype=torch.long, device=device)
    seq_indices = prepare_token_indices(cu_seqlens_t).tolist()

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            block_indices[0, i, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    if device.type == "musa":
        ref, ref_dq, ref_dk, ref_dv = cpu_naive_nsa_reference(
            q=q,
            k=k,
            v=v,
            do=do,
            block_indices=block_indices,
            block_size=block_size,
            cu_seqlens=cu_seqlens_t,
        )
    else:
        ref = naive_nsa(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_size=block_size,
            cu_seqlens=cu_seqlens_t,
        )
        ref.backward(do)
        ref_dq, q.grad = q.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dv, v.grad = v.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=S,
        block_size=block_size,
        cu_seqlens=cu_seqlens_t,
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

    assert_close("o", ref, tri, 0.004)
    assert_close("dq", ref_dq, tri_dq, 0.005)
    assert_close("dk", ref_dk, tri_dk, 0.005)
    assert_close("dv", ref_dv, tri_dv, 0.005)


def _report_stage(name, cpu, musa):
    musa_cpu = musa.detach().cpu()
    if not cpu.is_floating_point():
        print(name, "equal:", torch.equal(cpu, musa_cpu))
        return

    finite = torch.isfinite(cpu) & torch.isfinite(musa_cpu)
    diff = (cpu[finite] - musa_cpu[finite]).abs()
    print(
        name,
        "same_nonfinite:",
        torch.equal(~torch.isfinite(cpu), ~torch.isfinite(musa_cpu)),
        "max_abs:",
        diff.max().item(),
        "mean_abs:",
        diff.mean().item(),
    )


def _naive_nsa_token_stages(q, k, v, block_indices, b, t, block_size, scale):
    output_dtype = q.dtype
    G = q.shape[2] // k.shape[2]
    k, v, block_indices = (
        repeat(x, "b t h d -> b t (h g) d", g=G)
        for x in (k, v, block_indices)
    )
    q, k, v = map(lambda x: x.float(), (q, k, v))

    T = q.shape[1]
    indices = block_indices[b, t].unsqueeze(-1) * block_size
    indices = indices + block_indices.new_tensor(range(block_size))
    indices = indices.view(q.shape[2], -1).transpose(0, 1)

    gather_indices = indices.clamp(0, T - 1).unsqueeze(-1)
    gather_indices = gather_indices.expand(*indices.shape, k.shape[-1])
    k_i = k[b].gather(0, gather_indices)
    v_i = v[b].gather(0, gather_indices)

    q_i = q[b, t] * scale
    logits = torch.einsum("h d, n h d -> n h", q_i, k_i)
    mask = indices > t
    masked_logits = logits.masked_fill(mask, float("-inf"))
    attn = masked_logits.softmax(0)
    out_fp32 = torch.einsum("n h, n h v -> h v", attn, v_i)

    return {
        "indices": indices,
        "mask": mask,
        "q_i": q_i,
        "k_i": k_i,
        "v_i": v_i,
        "logits": logits,
        "masked_logits": masked_logits,
        "attn": attn,
        "out_fp32": out_fp32,
        "out_fp16": out_fp32.to(output_dtype),
    }


@pytest.mark.parallel_nsa
def test_debug_musa_naive_reference_forward():
    """Temporary diagnostic: compare the same inputs on CPU and MUSA naive_nsa."""
    torch.manual_seed(42)
    B, T, H, HQ, D, S = 1, 63, 1, 16, 64, 16
    block_size, scale = 32, 1.0
    dtype = torch.float16
    device = torch.device("musa")

    q = torch.randn((B, T, HQ, D), dtype=dtype, device=device)
    k = torch.randn((B, T, H, D), dtype=dtype, device=device)
    v = torch.randn((B, T, H, D), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, : len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    with torch.no_grad():
        ref_musa = naive_nsa(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
        )
        ref_cpu = naive_nsa(
            q=q.cpu(),
            k=k.cpu(),
            v=v.cpu(),
            block_indices=block_indices.cpu(),
            block_size=block_size,
            scale=scale,
        )

    musa_cpu = ref_musa.cpu()
    diff = (ref_cpu - musa_cpu).abs()
    coord = tuple(x.item() for x in torch.unravel_index(diff.argmax(), diff.shape))

    print("max_abs:", diff.max().item())
    print("max_index [B, T, HQ, D]:", coord)
    print("cpu value:", ref_cpu[coord].item())
    print("musa value:", musa_cpu[coord].item())

    b, t, _, _ = coord
    stages_cpu = _naive_nsa_token_stages(
        q.cpu(),
        k.cpu(),
        v.cpu(),
        block_indices.cpu(),
        b,
        t,
        block_size,
        scale,
    )
    stages_musa = _naive_nsa_token_stages(
        q,
        k,
        v,
        block_indices,
        b,
        t,
        block_size,
        scale,
    )

    for name in stages_cpu:
        _report_stage(name, stages_cpu[name], stages_musa[name])
    cpu_mask = stages_cpu["mask"]
    musa_mask = stages_musa["mask"].detach().cpu()
    cpu_masked = stages_cpu["masked_logits"]
    musa_masked = stages_musa["masked_logits"].detach().cpu()

    print("mask equal:", torch.equal(cpu_mask, musa_mask))
    print(
        "cpu -inf matches mask:",
        torch.equal(torch.isneginf(cpu_masked), cpu_mask),
    )
    print(
        "musa -inf matches mask:",
        torch.equal(torch.isneginf(musa_masked), musa_mask),
    )

    masked_values = musa_masked[musa_mask]
    print(
        "musa values at masked positions:",
        "neg_inf=",
        torch.isneginf(masked_values).sum().item(),
        "finite=",
        torch.isfinite(masked_values).sum().item(),
        "pos_inf=",
        torch.isposinf(masked_values).sum().item(),
        "nan=",
        torch.isnan(masked_values).sum().item(),
    )

    missing_neg_inf = musa_mask & ~torch.isneginf(musa_masked)
    if missing_neg_inf.any():
        n, h = missing_neg_inf.nonzero()[0].tolist()
        print("first missing -inf [N, HQ]:", (n, h))
        print("token index:", stages_cpu["indices"][n, h].item())
        print("cpu masked value:", cpu_masked[n, h].item())
        print("musa masked value:", musa_masked[n, h].item())

    musa_logits = stages_musa["logits"]
    musa_mask = stages_musa["mask"]

    print(
        "musa mask layout:",
        "shape=",
        tuple(musa_mask.shape),
        "stride=",
        musa_mask.stride(),
        "contiguous=",
        musa_mask.is_contiguous(),
    )

    musa_masked_contiguous = musa_logits.masked_fill(
        musa_mask.contiguous(),
        float("-inf"),
    )
    musa_masked_where = torch.where(
        musa_mask,
        torch.full_like(musa_logits, float("-inf")),
        musa_logits,
    )

    _report_stage(
        "masked_fill_contiguous_mask",
        cpu_masked,
        musa_masked_contiguous,
    )
    _report_stage(
        "where_mask",
        cpu_masked,
        musa_masked_where,
    )

    print(
        "contiguous mask -inf matches:",
        torch.equal(
            torch.isneginf(musa_masked_contiguous.detach().cpu()),
            cpu_mask,
        ),
    )
    print(
        "where -inf matches:",
        torch.equal(
            torch.isneginf(musa_masked_where.detach().cpu()),
            cpu_mask,
        ),
    )
