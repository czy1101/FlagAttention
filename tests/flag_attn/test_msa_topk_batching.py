"""Boundary regressions for the batched (multi-row) MSA prefill TopK path.

The reference below is a pure-torch reimplementation of the documented
selection semantics (visible blocks, local/init promotion, lowest-id tie-break,
-1 padding).  Cases cover the batched rows path, the streaming path and the
one-row-per-program fallback.
"""
from __future__ import annotations

import pytest
import torch

from flag_attn import runtime
from flag_attn.minimax_sparse_attention import SPARSE_BLOCK_SIZE, minimax_m3_index_topk
from flag_attn.testing import backend as test_backend


def _torch_topk_reference(score, cu_q, prefix_lens, topk, init_blocks, local_blocks):
    """Reference selection, mirroring the kernel's promotion/tie/padding rules."""
    heads, total_q, max_blocks = score.shape
    out = torch.full((heads, total_q, topk), -1, dtype=torch.int32, device=score.device)
    values = score.float()
    cu = cu_q.tolist()
    prefixes = prefix_lens.tolist()
    ids = torch.arange(max_blocks, device=score.device)
    for request_id in range(len(cu) - 1):
        q_start, q_end = cu[request_id], cu[request_id + 1]
        prefix = prefixes[request_id]
        for qid in range(q_end - q_start):
            valid = min(
                max((prefix + qid + SPARSE_BLOCK_SIZE) // SPARSE_BLOCK_SIZE, 0),
                max_blocks,
            )
            if valid <= 0:
                continue
            for head in range(heads):
                row = values[head, q_start + qid, :max_blocks].clone()
                row = torch.where(row == row, row, torch.full_like(row, float("-inf")))
                local_start = max(valid - local_blocks, 0)
                is_local = (ids >= local_start) & (ids < valid)
                row = torch.where(is_local, torch.full_like(row, 1.0e29), row)
                is_init = (ids < init_blocks) & (ids < valid) & (row < 1.0e29)
                row = torch.where(is_init, torch.full_like(row, 1.0e30), row)
                order = sorted(range(valid), key=lambda index: (-float(row[index]), index))
                chosen = order[: min(valid, topk)]
                out[head, q_start + qid, : len(chosen)] = torch.tensor(
                    chosen, dtype=torch.int32, device=score.device
                )
    return out


def _score_case(heads, max_blocks, q_len, prefix, kind="random"):
    torch.manual_seed(0)
    if kind == "zeros":
        score = torch.zeros((heads, q_len, max_blocks), device="gcu", dtype=torch.float32)
    elif kind == "neg_inf":
        score = torch.full((heads, q_len, max_blocks), float("-inf"), device="gcu", dtype=torch.float32)
    else:
        score = torch.randn((heads, q_len, max_blocks), device="gcu", dtype=torch.float32)
        if kind == "nan":
            score[0, 0, 1] = float("nan")
            score[0, q_len - 1, max_blocks - 1] = float("nan")
    cu_q = torch.tensor([0, q_len], device="gcu", dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device="gcu", dtype=torch.int32)
    return score, cu_q, prefix_lens


CASES = [
    # name, kind, max_blocks, q_len, prefix, topk, init, local
    ("batched_tail129_top4", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 4, 1, 2),
    ("batched_tail129_top1", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 1, 1, 2),
    ("batched_tail129_top2", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 2, 1, 2),
    ("batched_tail129_top3", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 3, 2, 3),
    ("batched_ties", "zeros", 64, 129, 9 * SPARSE_BLOCK_SIZE, 4, 1, 2),
    ("batched_nan", "nan", 64, 129, 9 * SPARSE_BLOCK_SIZE, 4, 1, 2),
    ("batched_all_neg_inf", "neg_inf", 64, 129, 9 * SPARSE_BLOCK_SIZE, 4, 1, 2),
    # prefix 0 => the first 128 queries see a single visible block (< topk)
    ("batched_valid_lt_topk", "random", 64, 129, 0, 4, 1, 2),
    ("batched_no_init_local", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 4, 0, 0),
    # topk=8 is not batched: it stays on the streaming path
    ("streaming_top8", "random", 64, 129, 9 * SPARSE_BLOCK_SIZE, 8, 1, 2),
    # max_blocks > 512 is not batched: it stays on the per-row fallback
    ("fallback_wide_rows", "random", 640, 9, 0, 4, 1, 2),
]


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.minimax_m3_sparse_attn
def test_prefill_topk_boundaries(case: tuple) -> None:
    if not test_backend.supports_operator("minimax_m3_sparse_attn"):
        pytest.skip(test_backend.skip_reason("minimax_m3_sparse_attn"))
    name, kind, max_blocks, q_len, prefix, topk, init_blocks, local_blocks = case
    score, cu_q, prefix_lens = _score_case(2, max_blocks, q_len, prefix, kind)
    actual = minimax_m3_index_topk(
        score, cu_q, prefix_lens, q_len, topk, init_blocks, local_blocks
    )
    expected = _torch_topk_reference(score, cu_q, prefix_lens, topk, init_blocks, local_blocks)
    assert torch.equal(actual, expected), f"{name}: selection/padding mismatch"


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_topk_noncontiguous_score_stride() -> None:
    if not test_backend.supports_operator("minimax_m3_sparse_attn"):
        pytest.skip(test_backend.skip_reason("minimax_m3_sparse_attn"))
    max_blocks, q_len, topk = 64, 129, 4
    _, cu_q, prefix_lens = _score_case(1, max_blocks, q_len, 9 * SPARSE_BLOCK_SIZE)
    torch.manual_seed(3)
    padded = torch.randn((1, q_len, max_blocks + 13), device="gcu", dtype=torch.float32)
    score = padded[:, :, :max_blocks]
    assert score.stride(2) == 1 and score.stride(1) == max_blocks + 13
    actual = minimax_m3_index_topk(score, cu_q, prefix_lens, q_len, topk, 1, 2)
    expected = _torch_topk_reference(score, cu_q, prefix_lens, topk, 1, 2)
    assert torch.equal(actual, expected)


@pytest.mark.minimax_m3_sparse_attn
def test_prefill_topk_dispatch_paths() -> None:
    """The batched rows path and the fallbacks are selected by shape only."""
    if runtime.device.vendor_name != "enflame":
        pytest.skip("Enflame-specific dispatch predicate")
    from flag_attn.runtime.backend._enflame.msa.index_topk import _batched_reduction_rows

    # batched rows: small topk, narrow padded score
    assert _batched_reduction_rows(4, 16, 1, 2)
    assert _batched_reduction_rows(1, 64, 4, 8)
    # streaming: topk above the reduction bound
    assert not _batched_reduction_rows(8, 64, 1, 2)
    # fallback: wide padded score rows
    assert not _batched_reduction_rows(4, 1024, 1, 2)
    # fallback: grid dimensions beyond the 2-D launch limits
    assert not _batched_reduction_rows(4, 64, 256, 2)
    assert not _batched_reduction_rows(4, 64, 1, 256)
