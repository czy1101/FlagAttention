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

"""NVIDIA HY3 FP8 block-sparse prefill contract on Ascend."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass

import pytest
import torch
import triton
import torch_npu  # noqa: F401


BLOCK = 128
HEAD_DIM = 128
Q_HEADS = 8
KV_HEADS = 2


@dataclass
class _Panel:
    args: tuple[torch.Tensor | int, ...]
    q_fp8: torch.Tensor
    k_fp8: torch.Tensor
    v_fp8: torch.Tensor
    k_scale_fp32: torch.Tensor
    block_mask_cpu: torch.Tensor | None


def _implementation():
    try:
        module = importlib.import_module(
            "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill"
        )
    except ModuleNotFoundError:
        pytest.fail("Ascend HY3 FP8 prefill module is not implemented")
    return module.attention_with_kvcache_blocksparse_prefill_fp8


def _prepare_workspace():
    module = importlib.import_module(
        "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill"
    )
    try:
        return module.prepare_attention_blocksparse_prefill_fp8_workspace
    except AttributeError:
        pytest.fail("Ascend HY3 FP8 prefill reusable workspace is not implemented")


def _make_mask(q_len: int, kv_len: int, masked: bool) -> torch.Tensor | None:
    if not masked:
        return None
    q_tiles = math.ceil(q_len / BLOCK)
    kv_tiles = math.ceil(kv_len / BLOCK)
    rows = torch.arange(q_tiles).view(q_tiles, 1)
    cols = torch.arange(kv_tiles).view(1, kv_tiles)
    valid = cols <= rows + (kv_tiles - q_tiles)
    mask = valid.view(1, 1, q_tiles, kv_tiles).expand(
        1, Q_HEADS, -1, -1
    ).clone()
    mask[:, :, :, 0] = False
    return mask.to(dtype=torch.uint8).contiguous()


def _as_hnd(cache: torch.Tensor) -> torch.Tensor:
    return cache.transpose(1, 2).contiguous().transpose(1, 2)


def _make_inputs(
    quant_type: int,
    kv_layout: str,
    masked: bool,
    page_size: int,
    *,
    q_len: int = 129,
    kv_len: int = 257,
) -> _Panel:
    torch.manual_seed(10086)
    pages = math.ceil(kv_len / page_size)
    q_fp8 = (
        torch.randn(q_len, Q_HEADS, HEAD_DIM, dtype=torch.bfloat16)
        / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    k_fp8 = (
        torch.randn(
            pages, page_size, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
        )
        / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    v_fp8 = torch.randn(
        pages, page_size, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    k_bits = k_fp8.view(torch.uint8)
    v_bits = v_fp8.view(torch.uint8)
    if kv_layout == "hnd":
        k_bits = _as_hnd(k_bits)
        v_bits = _as_hnd(v_bits)

    q_scale = torch.rand(1, Q_HEADS, max(256, q_len), dtype=torch.float32)
    q_scale[:, :, q_len:] = 0
    if quant_type == 1:
        k_scale_fp32 = torch.rand(1, dtype=torch.float32) + 0.5
        k_scale = k_scale_fp32
        v_scale = torch.rand(1, dtype=torch.float32) + 0.5
    else:
        k_scale_fp32 = torch.rand(
            pages,
            page_size // 32,
            KV_HEADS,
            HEAD_DIM // 4,
            dtype=torch.float32,
        ).clamp_min_(1e-6)
        k_scale = k_scale_fp32.view(torch.uint8)
        v_scale = torch.rand(KV_HEADS, dtype=torch.float32) + 0.5

    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32)
    block_ids = torch.arange(pages, dtype=torch.int32).view(1, -1)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32)
    block_mask_cpu = _make_mask(q_len, kv_len, masked)
    block_mask = None if block_mask_cpu is None else block_mask_cpu.npu()
    args = (
        q_fp8.view(torch.uint8).npu(),
        k_bits.npu(),
        v_bits.npu(),
        q_scale.npu(),
        k_scale.npu(),
        v_scale.npu(),
        cu_seqlens_q.npu(),
        block_ids.npu(),
        kv_lens.npu(),
        q_len,
        quant_type,
        block_mask,
    )
    return _Panel(
        args=args,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        k_scale_fp32=k_scale_fp32,
        block_mask_cpu=block_mask_cpu,
    )


@torch.no_grad()
def _reference(panel: _Panel) -> torch.Tensor:
    (
        _q,
        _k,
        _v,
        q_scale_npu,
        _k_scale,
        v_scale_npu,
        _cu_seqlens_q,
        block_ids_npu,
        kv_lens_npu,
        q_len,
        quant_type,
        _block_mask,
    ) = panel.args
    q_scale = q_scale_npu.cpu()
    v_scale = v_scale_npu.cpu()
    block_ids = block_ids_npu.cpu()
    kv_len = int(kv_lens_npu.cpu()[0])
    page_ids = block_ids[0]
    paged_k = panel.k_fp8[page_ids].reshape(
        -1, KV_HEADS, HEAD_DIM
    )[:kv_len]
    paged_v = panel.v_fp8[page_ids].reshape(
        -1, KV_HEADS, HEAD_DIM
    )[:kv_len]
    token_k_scale = None
    if quant_type == 0:
        token_k_scale = (
            panel.k_scale_fp32[page_ids]
            .permute(0, 1, 3, 2)
            .reshape(-1, KV_HEADS)[:kv_len]
        )

    output = torch.empty_like(panel.q_fp8, dtype=torch.bfloat16)
    group_size = Q_HEADS // KV_HEADS
    key_positions = torch.arange(kv_len)
    key_tiles = torch.div(key_positions, BLOCK, rounding_mode="floor")
    q_positions = torch.arange(q_len)
    causal = key_positions.unsqueeze(0) <= (
        q_positions.unsqueeze(1) + kv_len - q_len
    )
    for kv_head in range(KV_HEADS):
        head_start = kv_head * group_size
        head_end = head_start + group_size
        q_group = panel.q_fp8[:, head_start:head_end].permute(
            1, 0, 2
        ).float()
        k_head = paged_k[:, kv_head].float()
        v_head = paged_v[:, kv_head].float()
        scores = torch.stack(
            [q_group[index] @ k_head.T for index in range(group_size)]
        )
        query_scale = q_scale[
            0, head_start:head_end, :q_len
        ].unsqueeze(-1)
        if quant_type == 0:
            key_scale = token_k_scale[:, kv_head].view(1, 1, kv_len)
            value_scale = v_scale[kv_head]
        else:
            key_scale = panel.k_scale_fp32[0]
            value_scale = v_scale[0]
        scores *= query_scale * key_scale / math.sqrt(HEAD_DIM)
        valid = causal.unsqueeze(0)
        if panel.block_mask_cpu is not None:
            query_tiles = torch.div(q_positions, BLOCK, rounding_mode="floor")
            sparse = panel.block_mask_cpu[
                0, head_start:head_end
            ].bool()
            sparse = sparse.index_select(1, query_tiles).index_select(
                2, key_tiles
            )
            valid = valid & sparse
        scores.masked_fill_(~valid, -float("inf"))
        probabilities = torch.exp(
            scores - scores.max(dim=-1, keepdim=True).values
        )
        denominator = probabilities.sum(dim=-1, keepdim=True)
        probabilities = (
            probabilities * 256.0
        ).to(torch.float8_e4m3fn).float()
        result = torch.stack(
            [probabilities[index] @ v_head for index in range(group_size)]
        )
        result = result * (value_scale / 256.0) / denominator
        output[:, head_start:head_end] = result.permute(1, 0, 2)
    return output


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("quant_type", [0, 1])
@pytest.mark.parametrize("kv_layout", ["nhd", "hnd"])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("page_size", [32, 64])
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend(
    quant_type: int,
    kv_layout: str,
    masked: bool,
    page_size: int,
) -> None:
    panel = _make_inputs(quant_type, kv_layout, masked, page_size)
    output = _implementation()(*panel.args)
    torch.npu.synchronize()
    reference = _reference(panel)
    assert torch.isfinite(reference).all()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output.cpu().float(), reference.float(), atol=0.1, rtol=0.1
    )


def _page_distinct_panel(page_size: int, permute: bool) -> tuple[_Panel, torch.Tensor]:
    """A panel whose KV pages hold distinct constants and whose table may be permuted.

    Every other case in this file hands the kernel `torch.arange(pages)`.  That
    cannot tell a kernel which resolves `block_ids` apart from one which indexes
    the dequantised cache by logical page: the pages hold random noise either
    way, so the two hypotheses differ only by the fp8 tolerance.  Here page `p`
    holds `(p + 1) * 0.25`, so the hypotheses differ by ~1.2, far outside it.
    """
    q_len, kv_len = 129, 257
    panel = _make_inputs(1, "nhd", False, page_size, q_len=q_len, kv_len=kv_len)
    pages = int(panel.args[7].shape[1])
    key = torch.empty(pages, page_size, KV_HEADS, HEAD_DIM)
    value = torch.empty(pages, page_size, KV_HEADS, HEAD_DIM)
    for page in range(pages):
        key[page] = (page + 1) * 0.25
        value[page] = (page + 1) * 0.5
    panel.k_fp8 = key.to(torch.float8_e4m3fn)
    panel.v_fp8 = value.to(torch.float8_e4m3fn)
    ids = torch.arange(pages, dtype=torch.int32)
    if permute:
        ids = ids.flip(0).contiguous()
    args = list(panel.args)
    args[1] = panel.k_fp8.view(torch.uint8).npu()
    args[2] = panel.v_fp8.view(torch.uint8).npu()
    args[7] = ids.view(1, -1).npu()
    panel.args = tuple(args)
    return panel, ids


def _panel_with_ids(panel: _Panel, ids: torch.Tensor) -> _Panel:
    args = list(panel.args)
    args[7] = ids.view(1, -1).npu()
    return _Panel(
        args=tuple(args),
        q_fp8=panel.q_fp8,
        k_fp8=panel.k_fp8,
        v_fp8=panel.v_fp8,
        k_scale_fp32=panel.k_scale_fp32,
        block_mask_cpu=panel.block_mask_cpu,
    )


def _qk_scores(panel: _Panel, q_len: int, kv_len: int) -> torch.Tensor:
    workspace = _prepare_workspace()(*panel.args[:11])
    _implementation()(*panel.args, workspace=workspace)
    torch.npu.synchronize()
    return workspace.scores[0, 0, :, :q_len, :kv_len].clone()


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("page_size", [32, 64])
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_resolves_page_table(
    page_size: int,
) -> None:
    """`block_ids` is a logical-to-physical page table, not a page number.

    The dequantised KV buffer holds the cache in place and the indirection is
    applied by the qk/pv kernels, so a kernel that reads that buffer at the
    logical page still passes every identity-table case above.  Two checks pin
    it down: permuting the table has to move the scores that qk writes, and the
    final output still has to match the reference for the permuted table.
    """
    q_len, kv_len = 129, 257
    permuted, permuted_ids = _page_distinct_panel(page_size, permute=True)
    identity, _ = _page_distinct_panel(page_size, permute=False)

    scores_permuted = _qk_scores(permuted, q_len, kv_len)
    scores_identity = _qk_scores(identity, q_len, kv_len)
    finite = torch.isfinite(scores_permuted)
    assert finite.any(), "qk wrote no finite scores"
    scale = scores_permuted[finite].abs().max().item()
    moved = (scores_permuted - scores_identity).abs()[finite].max().item()
    assert moved > 0.25 * scale, (
        f"permuting block_ids moved the qk scores by only {moved:.4e} against a "
        f"scale of {scale:.4e}; the page table is not being resolved"
    )

    output = _implementation()(*permuted.args).cpu().float()
    torch.npu.synchronize()
    expected = _reference(_panel_with_ids(permuted, permuted_ids)).float()
    assert torch.isfinite(expected).all()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, atol=0.1, rtol=0.1)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("kv_len", [384, 1152])
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_masks_tile_overhang(
    kv_len: int,
) -> None:
    """A tile can be a 128-multiple without being a power of two.

    `_prefill_tokens_per_split` rounds the tile up to a multiple of 128, so
    `tokens_per_split` is 384 at kv=384 and 1152 at kv=1152, while the softmax
    kernel runs with `VEC = next_power_of_2(tile)` = 512 / 2048.  The unmasked
    fast path carries no `valid_n`, so those extra lanes are read as the *next*
    q row's scores and corrupt the row max and the row sum: 8.9e-02 instead of
    2.0e-03 at kv=384, which the atol=0.1 used above would have hidden.
    """
    q_len = 128
    panel = _make_inputs(1, "nhd", False, 64, q_len=q_len, kv_len=kv_len)
    workspace = _prepare_workspace()(*panel.args[:11])
    assert workspace.tokens_per_split == kv_len
    assert workspace.tokens_per_split != triton.next_power_of_2(
        workspace.tokens_per_split
    )
    assert not workspace.unmasked, "overhanging tiles must take the masked path"

    output = _implementation()(*panel.args, workspace=workspace)
    torch.npu.synchronize()
    reference = _reference(panel).float()
    finite = ~torch.isnan(reference)
    assert finite.any()
    error = (output.cpu().float() - reference)[finite].abs().max().item()
    assert error < 0.01, f"tile overhang leaked into the row reduction ({error:.3e})"


def _benchmark_sparse_mask(q_len: int, kv_len: int) -> torch.Tensor:
    """The benchmark's block mask: 75% random skip inside the causal band.

    `_make_mask` above keeps nearly every cell once kv_len >> q_len, so it never
    exercises the qk block-sparsity skip.  This mirrors
    `benchmark/bench_hpc_ops_prefill_fp8.py::_make_block_mask` (MASK_SKIP_RATIO
    0.75, SEED 10086) including the frontier and first-q-tile fixups that keep
    every row non-empty.
    """
    q_tiles = math.ceil(q_len / BLOCK)
    kv_tiles = math.ceil(kv_len / BLOCK)
    rows = torch.arange(q_tiles).view(q_tiles, 1)
    cols = torch.arange(kv_tiles).view(1, kv_tiles)
    causal = cols <= rows + (kv_tiles - q_tiles)
    generator = torch.Generator()
    generator.manual_seed(10086 + 17)
    mask = torch.rand((1, Q_HEADS, q_tiles, kv_tiles), generator=generator) >= 0.75
    mask &= causal.view(1, 1, q_tiles, kv_tiles)
    frontier = torch.clamp(rows + (kv_tiles - q_tiles), max=kv_tiles - 1)
    mask |= (cols == frontier).view(1, 1, q_tiles, kv_tiles)
    first = torch.div(
        kv_len - q_len + rows * BLOCK, BLOCK, rounding_mode="floor"
    ).clamp(min=0, max=kv_tiles - 1)
    mask |= (cols == first).view(1, 1, q_tiles, kv_tiles)
    return mask.to(dtype=torch.uint8)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_skips_dead_blocks() -> None:
    """Dropping a dead (q tile, kv tile) block must not change anything.

    qk loads no K and runs no dot for a block whose mask is clear, and stores
    -inf there instead; that is only sound because the block contributes nothing
    to softmax, which relies on the -inf.  Compare the skipping path against the
    path that computes every block, and against the fp32 reference, on the
    benchmark's sparse mask rather than the dense causal one.
    """
    q_len, kv_len = 257, 1024
    panel = _make_inputs(1, "nhd", True, 64, q_len=q_len, kv_len=kv_len)
    mask = _benchmark_sparse_mask(q_len, kv_len)
    assert 0 < int(mask.sum()) < mask.numel()
    args = list(panel.args)
    args[11] = mask.npu()
    panel.args = tuple(args)
    panel.block_mask_cpu = mask

    module = importlib.import_module(
        "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill."
        "attention_blocksparse_prefill_fp8"
    )
    previous = module._QK_SKIP_OVERRIDE
    try:
        module._QK_SKIP_OVERRIDE = 2
        skipped = _implementation()(*panel.args)
        module._QK_SKIP_OVERRIDE = 0
        dense = _implementation()(*panel.args)
    finally:
        module._QK_SKIP_OVERRIDE = previous
    torch.npu.synchronize()
    assert torch.equal(skipped, dense), "skipping dead blocks changed the output"

    reference = _reference(panel).float()
    finite = ~torch.isnan(reference)
    assert finite.any()
    torch.testing.assert_close(
        skipped.cpu().float()[finite], reference[finite], atol=0.1, rtol=0.1
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_lists_active_blocks() -> None:
    """A declared-sparse mask lets qk walk only the blocks the mask keeps.

    The backend then fills SCORES with -inf once per pass and has qk iterate a
    compacted list, so the scores store stays unconditional (a data-dependent
    store predicate costs +38% to +159% on this backend).  The list has to
    reproduce the dense path bit for bit, and the sparsity valve has to keep
    every other caller on the dense path.
    """
    q_len, kv_len = 257, 1024
    panel = _make_inputs(1, "nhd", True, 64, q_len=q_len, kv_len=kv_len)
    mask = _benchmark_sparse_mask(q_len, kv_len)
    args = list(panel.args)
    args[11] = mask.npu()
    panel.args = tuple(args)
    panel.block_mask_cpu = mask

    module = importlib.import_module(
        "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill."
        "attention_blocksparse_prefill_fp8"
    )
    implementation = _implementation()
    previous = module._QK_LIST_OVERRIDE
    try:
        module._QK_LIST_OVERRIDE = True
        listed = implementation(*panel.args, sparsity_bucket=2)
        module._QK_LIST_OVERRIDE = False
        dense = implementation(*panel.args, sparsity_bucket=2)
        module._QK_LIST_OVERRIDE = None
        auto_sparse = implementation(*panel.args, sparsity_bucket=2)
        auto_bucket3 = implementation(*panel.args, sparsity_bucket=3)
        auto_dense_bucket = implementation(*panel.args, sparsity_bucket=0)
        auto_none = implementation(*panel.args)
    finally:
        module._QK_LIST_OVERRIDE = previous
    torch.npu.synchronize()

    assert torch.equal(listed, dense), "the active-block list changed the output"
    assert torch.equal(auto_sparse, listed), "bucket 2 must take the listed path"
    assert torch.equal(auto_bucket3, listed), "bucket 3 must take the listed path"
    assert torch.equal(auto_dense_bucket, dense), "bucket 0 must stay dense"
    assert torch.equal(auto_none, dense), "an undeclared mask must stay dense"

    reference = _reference(panel).float()
    finite = ~torch.isnan(reference)
    assert finite.any()
    torch.testing.assert_close(
        listed.cpu().float()[finite], reference[finite], atol=0.1, rtol=0.1
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_reuses_workspace() -> None:
    panel = _make_inputs(1, "nhd", True, 64)
    workspace = _prepare_workspace()(*panel.args[:10], panel.args[10])
    implementation = _implementation()
    first = implementation(*panel.args, workspace=workspace).clone()
    second = implementation(*panel.args, workspace=workspace).clone()
    torch.npu.synchronize()
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_multi_split() -> None:
    # kv_len 要 > 4096（tile 上限），否则会并成单个 split，覆盖不到多 split 路径
    panel = _make_inputs(
        1, "nhd", False, 64, q_len=17, kv_len=8193
    )
    workspace = _prepare_workspace()(*panel.args[:10], panel.args[10])
    output = _implementation()(*panel.args, workspace=workspace)
    torch.npu.synchronize()
    torch.testing.assert_close(
        output.cpu().float(), _reference(panel).float(), atol=0.1, rtol=0.1
    )


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_official_gqa8() -> None:
    q_len, kv_len, page_size = 9, 129, 64
    hq, hkv = 32, 4
    pages = math.ceil(kv_len / page_size)
    q_fp8 = torch.randn(q_len, hq, HEAD_DIM, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k_fp8 = torch.randn(
        pages, page_size, hkv, HEAD_DIM, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    v_fp8 = torch.randn(
        pages, page_size, hkv, HEAD_DIM, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    q = q_fp8.view(torch.uint8).npu()
    k = k_fp8.view(torch.uint8).npu()
    v = v_fp8.view(torch.uint8).npu()
    qscale = torch.ones(1, hq, q_len, dtype=torch.float32, device="npu")
    kscale = torch.ones(1, dtype=torch.float32, device="npu")
    vscale = torch.ones(1, dtype=torch.float32, device="npu")
    cuq = torch.tensor([0, q_len], dtype=torch.int32, device="npu")
    block_ids = torch.arange(pages, dtype=torch.int32, device="npu").view(1, -1)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device="npu")
    prepare = _prepare_workspace()
    workspace = prepare(
        q, k, v, qscale, kscale, vscale, cuq, block_ids, kv_lens, q_len, 1
    )
    assert workspace.q_tokens == 128
    output = _implementation()(
        q,
        k,
        v,
        qscale,
        kscale,
        vscale,
        cuq,
        block_ids,
        kv_lens,
        q_len,
        1,
        workspace=workspace,
    )
    torch.npu.synchronize()
    assert torch.isfinite(output).all()
    expected = torch.empty(q_len, hq, HEAD_DIM, dtype=torch.float32)
    causal = torch.arange(kv_len)[None, :] <= (
        torch.arange(q_len)[:, None] + kv_len - q_len
    )
    k_tokens = k_fp8.reshape(-1, hkv, HEAD_DIM)[:kv_len]
    v_tokens = v_fp8.reshape(-1, hkv, HEAD_DIM)[:kv_len]
    for kv_head in range(hkv):
        first = kv_head * (hq // hkv)
        last = first + hq // hkv
        scores = torch.einsum(
            "qhd,kd->hqk",
            q_fp8[:, first:last].float(),
            k_tokens[:, kv_head].float(),
        ) / math.sqrt(HEAD_DIM)
        scores.masked_fill_(~causal[None], -float("inf"))
        probabilities = torch.exp(scores - scores.amax(-1, keepdim=True))
        denominator = probabilities.sum(-1, keepdim=True)
        probabilities = (
            probabilities * 256.0
        ).to(torch.float8_e4m3fn).float()
        result = probabilities @ v_tokens[:, kv_head].float()
        expected[:, first:last] = (
            result / 256.0 / denominator
        ).permute(1, 0, 2)
    torch.testing.assert_close(
        output.cpu().float(), expected, atol=0.1, rtol=0.1
    )


def test_attention_blocksparse_prefill_fp8_ascend_partitions_launch_grid() -> None:
    module = importlib.import_module(
        "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill."
        "attention_blocksparse_prefill_fp8"
    )
    chunks = module._prefill_launch_chunks(256, 4, 64)
    assert chunks == tuple((offset, 32) for offset in range(0, 256, 32))
    assert sum(size for _, size in chunks) == 256
    assert all(size * 4 * 64 <= 65535 for _, size in chunks)


def test_attention_blocksparse_prefill_fp8_ascend_buckets_short_kv() -> None:
    module = importlib.import_module(
        "flag_attn.runtime.backend._ascend.hpc_ops_attention.prefill."
        "attention_blocksparse_prefill_fp8"
    )
    # tile = ceil_to_128(ceil(tokens / splits))，splits 保证 tile <= 2048：
    # 细粒度让最后一个 tile 基本填满（2305 -> 2x1280，而不是 2x2048 只填 12%）。
    assert tuple(
        module._prefill_tokens_per_split(tokens)
        for tokens in (64, 128, 192, 256, 320, 512, 640, 1024, 1088, 2304, 4096)
    ) == (128, 128, 256, 256, 384, 512, 640, 1024, 1152, 2304, 4096)


@pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")
@torch.no_grad()
def test_attention_blocksparse_prefill_fp8_ascend_variable_batch() -> None:
    """Packed Q and page-table offsets must remain batch-local."""
    torch.manual_seed(2026)
    q_lens = (65, 129)
    kv_lens = (129, 257)
    page_size = 64
    page_counts = tuple(math.ceil(length / page_size) for length in kv_lens)
    total_pages = sum(page_counts)
    q_fp8 = (
        torch.randn(sum(q_lens), Q_HEADS, HEAD_DIM) / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    k_fp8 = (
        torch.randn(total_pages, page_size, KV_HEADS, HEAD_DIM)
        / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    v_fp8 = torch.randn(
        total_pages, page_size, KV_HEADS, HEAD_DIM
    ).to(torch.float8_e4m3fn)
    q_scale = torch.rand(2, Q_HEADS, max(q_lens), dtype=torch.float32)
    k_scale = torch.tensor([0.75], dtype=torch.float32)
    v_scale = torch.tensor([1.25], dtype=torch.float32)
    cu_q = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32
    )
    block_ids = torch.zeros(2, max(page_counts), dtype=torch.int32)
    block_ids[0, :page_counts[0]] = torch.arange(page_counts[0])
    block_ids[1, :page_counts[1]] = torch.arange(
        page_counts[0], total_pages
    )
    implementation = _implementation()
    output_storage = torch.empty(
        q_fp8.shape, dtype=torch.bfloat16, device="npu"
    )
    output = implementation(
        q_fp8.view(torch.uint8).npu(),
        k_fp8.view(torch.uint8).npu(),
        v_fp8.view(torch.uint8).npu(),
        q_scale.npu(),
        k_scale.npu(),
        v_scale.npu(),
        cu_q.npu(),
        block_ids.npu(),
        torch.tensor(kv_lens, dtype=torch.int32).npu(),
        max(q_lens),
        1,
        output=output_storage,
    )
    torch.npu.synchronize()
    assert output.data_ptr() == output_storage.data_ptr()

    expected_batches = []
    q_cursor = 0
    group_size = Q_HEADS // KV_HEADS
    for batch, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        q_batch = q_fp8[q_cursor:q_cursor + q_len]
        q_cursor += q_len
        ids = block_ids[batch, :page_counts[batch]]
        k_batch = k_fp8[ids].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len]
        v_batch = v_fp8[ids].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len]
        batch_output = torch.empty_like(q_batch, dtype=torch.bfloat16)
        causal = torch.arange(kv_len)[None, :] <= (
            torch.arange(q_len)[:, None] + kv_len - q_len
        )
        for kv_head in range(KV_HEADS):
            first = kv_head * group_size
            last = first + group_size
            q_group = q_batch[:, first:last].permute(1, 0, 2).float()
            scores = torch.stack(
                [
                    q_group[index] @ k_batch[:, kv_head].float().T
                    for index in range(group_size)
                ]
            )
            scores *= (
                q_scale[batch, first:last, :q_len, None]
                * k_scale[0]
                / math.sqrt(HEAD_DIM)
            )
            scores.masked_fill_(~causal[None], -float("inf"))
            probabilities = torch.exp(
                scores - scores.amax(-1, keepdim=True)
            )
            denominator = probabilities.sum(-1, keepdim=True)
            probabilities = (
                probabilities * 256.0
            ).to(torch.float8_e4m3fn).float()
            result = torch.stack(
                [
                    probabilities[index] @ v_batch[:, kv_head].float()
                    for index in range(group_size)
                ]
            )
            result *= (v_scale[0] / 256.0) / denominator
            batch_output[:, first:last] = result.permute(1, 0, 2)
        expected_batches.append(batch_output)
    expected = torch.cat(expected_batches)
    torch.testing.assert_close(
        output.cpu().float(), expected.float(), atol=0.1, rtol=0.1
    )
