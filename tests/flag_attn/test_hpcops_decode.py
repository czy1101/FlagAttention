"""Correctness and GPU task-map protocol tests for HPC-Ops FP8 decode."""

from __future__ import annotations

import math

import pytest
import torch
import triton

from flag_attn.hpcops_decode_attention import (  # noqa: E402
    DecodeInputs,
    fp8_kvpertensor_decode,
    prepare_decode_workspace,
)
from flag_attn.hpcops_decode_attention.task_scheduler_kernel import (  # noqa: E402
    TASK_SLOTS,
    TASK_STRIDE,
    allocate_cluster_task_map,
)


BLOCK_SIZE = 64
HEAD_DIM = 128


def _supports_sm90_fp8_tle() -> bool:
    return (
        torch.cuda.is_available()
        and hasattr(torch, "float8_e4m3fn")
        and torch.cuda.get_device_capability()[0] >= 9
    )


@pytest.fixture(autouse=True)
def _triton_allocator():
    if torch.cuda.is_available():
        triton.set_allocator(
            lambda size, _align, _stream: torch.empty(
                size, dtype=torch.int8, device="cuda"
            )
        )
    yield


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    """Keep logical [block, token, head, dim] axes with HND physical strides."""
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _build_inputs(
    num_batch: int,
    max_seq_kv: int,
    num_head_kv: int,
    num_head_q: int,
    layout: str,
) -> DecodeInputs:
    generator = torch.Generator(device="cuda").manual_seed(12345)
    history_lens = torch.randint(
        1,
        max_seq_kv,
        (num_batch,),
        generator=generator,
        dtype=torch.int32,
        device="cuda",
    )
    kv_lens = history_lens + 1
    nblocks = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(nblocks.sum().item())
    capacity_blocks = total_blocks + num_batch + 8

    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    q_bf16 = torch.randn(
        (num_batch, num_head_q, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(dim=-1).clamp_min(1e-6)
    q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)
    k_cache = (
        torch.randn(
            capacity_blocks,
            BLOCK_SIZE,
            num_head_kv,
            HEAD_DIM,
            dtype=torch.bfloat16,
            device="cuda",
        )
        / math.sqrt(HEAD_DIM)
    ).to(torch.float8_e4m3fn)
    v_cache = torch.randn_like(k_cache, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    if layout == "HND":
        k_cache = _as_hnd_view(k_cache)
        v_cache = _as_hnd_view(v_cache)

    packed = torch.randperm(capacity_blocks, device="cuda")[:total_blocks].to(torch.int32)
    block_ids = torch.empty(
        (num_batch, int(nblocks.max().item())), dtype=torch.int32, device="cuda"
    )
    offset = 0
    for batch, blocks in enumerate(nblocks.cpu().tolist()):
        block_ids[batch, :blocks] = packed[offset : offset + blocks]
        offset += blocks

    return DecodeInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        block_ids=block_ids,
        kv_lens=kv_lens,
        q_scale=q_scale,
        k_scale=torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6),
        v_scale=torch.rand((1,), dtype=torch.float32, device="cuda").clamp_min(1e-6),
    )


def _reference(inputs: DecodeInputs) -> torch.Tensor:
    num_head_q = inputs.q.shape[1]
    num_head_kv = inputs.k_cache.shape[2]
    heads_per_group = num_head_q // num_head_kv
    outputs = []
    for batch in range(inputs.num_batch):
        length = int(inputs.kv_lens[batch].item())
        blocks = inputs.block_ids[batch, : (length + BLOCK_SIZE - 1) // BLOCK_SIZE]
        k = inputs.k_cache[blocks].reshape(-1, num_head_kv, HEAD_DIM)[:length]
        v = inputs.v_cache[blocks].reshape(-1, num_head_kv, HEAD_DIM)[:length]
        k = k.transpose(0, 1).repeat_interleave(heads_per_group, dim=0).float()
        v = v.transpose(0, 1).repeat_interleave(heads_per_group, dim=0).float()
        q = inputs.q[batch].float()
        scores = torch.einsum("hd,hnd->hn", q, k)
        scores *= (
            inputs.q_scale[batch, :, None].float()
            * inputs.k_scale.float()
            / math.sqrt(HEAD_DIM)
        )
        shifted = torch.exp(scores - scores.amax(dim=-1, keepdim=True))
        denominator = shifted.sum(dim=-1, keepdim=True)
        quantized_weights = (shifted * 256.0).to(torch.float8_e4m3fn).float()
        out = torch.einsum("hn,hnd->hd", quantized_weights, v)
        out *= inputs.v_scale.float() / (256.0 * denominator)
        outputs.append(out.to(torch.bfloat16))
    return torch.stack(outputs)


@pytest.mark.skipif(not _supports_sm90_fp8_tle(), reason="requires SM90 FP8 TLE")
@pytest.mark.parametrize("num_batch", [1, 16, 200])
@pytest.mark.parametrize("max_seq_kv", [1024, 4096])
@pytest.mark.parametrize("kv_head_q_head", [(1, 8), (4, 32)])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_fp8_decode_matches_pytorch(
    num_batch, max_seq_kv, kv_head_q_head, layout
):
    num_head_kv, num_head_q = kv_head_q_head
    inputs = _build_inputs(
        num_batch, max_seq_kv, num_head_kv, num_head_q, layout
    )
    workspace = prepare_decode_workspace(inputs)
    actual = fp8_kvpertensor_decode(inputs, workspace)
    expected = _reference(inputs)
    torch.cuda.synchronize()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=0.2, rtol=0.2)
    assert not bool(torch.count_nonzero(workspace.completion).item())
    assert torch.isfinite(actual).all()
    if layout == "HND" and num_head_kv > 1:
        assert inputs.k_cache.stride(1) != num_head_kv * HEAD_DIM


def _cpu_task_map_reference(
    lengths: list[int], *, num_head_kv: int, cluster_size: int, chunk_tokens: int
) -> tuple[torch.Tensor, dict]:
    reduction_clusters = []
    direct_tasks = []
    effective_chunks = []
    compute_tasks = 0
    dummy_tasks = 0
    for hkv in range(num_head_kv):
        for batch, total_len in enumerate(lengths):
            num_chunks = (total_len + chunk_tokens - 1) // chunk_tokens
            compute_tasks += num_chunks
            if num_chunks == 1:
                direct_tasks.append((hkv, batch, total_len))
                effective_chunks.append(1)
                continue
            groups = (num_chunks + cluster_size - 1) // cluster_size
            effective_chunks.append(groups)
            dummy_tasks += groups * cluster_size - num_chunks
            for group in range(groups):
                records = []
                for rank in range(cluster_size):
                    chunk = group * cluster_size + rank
                    real = chunk < num_chunks
                    start = chunk * chunk_tokens if real else 0
                    seq_len = min(chunk_tokens, total_len - start) if real else 0
                    last = real and chunk == num_chunks - 1
                    seq_kvcache = max(seq_len - 1, 0) if last else seq_len
                    records.append([
                        hkv, batch, chunk, start, seq_len, seq_kvcache,
                        (seq_len + 63) // 64, seq_kvcache // 64 if last else seq_len // 64,
                        int(last), 1 if real else 2, group, groups,
                    ])
                reduction_clusters.append(records)

    direct_clusters = []
    for index in range(0, len(direct_tasks), cluster_size):
        cluster = []
        for offset in range(cluster_size):
            position = index + offset
            if position >= len(direct_tasks):
                cluster.append(None)
                continue
            hkv, batch, total_len = direct_tasks[position]
            cluster.append([
                hkv, batch, 0, 0, total_len, max(total_len - 1, 0),
                (total_len + 63) // 64, max(total_len - 1, 0) // 64,
                1, 0, 0, 1,
            ])
        direct_clusters.append(cluster)

    clusters = reduction_clusters + direct_clusters
    physical_ctas = len(clusters) * cluster_size
    num_chunks_base = (TASK_SLOTS * physical_ctas + 1) * TASK_STRIDE
    num_sequences = len(lengths) * num_head_kv
    sched_ints = num_chunks_base + (
        (num_sequences + TASK_STRIDE - 1) // TASK_STRIDE * TASK_STRIDE
    )
    result = torch.full((sched_ints,), -1, dtype=torch.int32)
    result[:5] = torch.tensor(
        [chunk_tokens // 64 + 1, physical_ctas, num_head_kv, len(lengths), sched_ints * 4],
        dtype=torch.int32,
    )
    for cluster_id, cluster in enumerate(clusters):
        for rank, record in enumerate(cluster):
            cta = cluster_id * cluster_size + rank
            base = (cta * TASK_SLOTS + 1) * TASK_STRIDE
            if record is not None:
                result[base : base + TASK_STRIDE] = torch.tensor(record)
    result[num_chunks_base : num_chunks_base + num_sequences] = torch.tensor(
        effective_chunks
    )
    stats = {
        "num_clusters": len(clusters),
        "physical_ctas": physical_ctas,
        "compute_tasks": compute_tasks,
        "dummy_tasks": dummy_tasks,
    }
    return result, stats


@pytest.mark.skipif(not _supports_sm90_fp8_tle(), reason="requires SM90 FP8 TLE")
@pytest.mark.parametrize(
    "cluster_size,chunk_tokens", [(2, 1024), (4, 512), (8, 1024)]
)
@pytest.mark.parametrize("num_head_kv", [1, 4])
def test_gpu_task_schedule_matches_cpu_oracle(
    cluster_size, chunk_tokens, num_head_kv
):
    lengths = [64, 513, 4096, 65536]
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    expected, stats = _cpu_task_map_reference(
        lengths,
        num_head_kv=num_head_kv,
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
    )
    schedule = allocate_cluster_task_map(
        kv_lens,
        num_head_kv=num_head_kv,
        max_seq_kv=max(lengths),
        cluster_size=cluster_size,
        chunk_tokens=chunk_tokens,
    )
    torch.testing.assert_close(
        schedule.task_map[: schedule.sched_ints].cpu(), expected, rtol=0, atol=0
    )
    for name, value in stats.items():
        assert schedule.stats[name] == value
