"""HPC-Ops FP8 decode benchmark for the policy-selected Triton + TLE path.

Task maps are prepared before timing by default. Pass ``--include-taskmap`` to
time each provider's matching assign kernel immediately before decode.
"""

from __future__ import annotations

import argparse
import math
import statistics

import torch
import triton

import hpc  # noqa: E402
from flag_attn.hpcops_decode_attention import (  # noqa: E402
    DecodeInputs,
    fp8_kvpertensor_decode,
    prepare_decode_workspace,
)


BLOCK_SIZE = 64
HEAD_DIM = 128
CASES = {
    "uniform_512": [512] * 64,
    "uniform_4096": [4096] * 64,
    "skewed_mix": [128] * 32 + [4096] * 32,
    "skewed_extreme": [64] * 15 + [16 * 1024],
    "one_64k_7x4k": [64 * 1024] + [4096] * 7,
    "one_64k_15x4k": [64 * 1024] + [4096] * 15,
    "one_64k_31x4k": [64 * 1024] + [4096] * 31,
    "one_128k_31x4k": [128 * 1024] + [4096] * 31,
    "two_32k_30x4k": [32 * 1024] * 2 + [4096] * 30,
}


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def make_inputs(
    lengths: list[int], num_head_kv: int, num_head_q: int, layout: str
) -> DecodeInputs:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    nblocks = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(nblocks.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8
    q_bf16 = torch.randn(
        (len(lengths), num_head_q, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(dim=-1).clamp_min(1e-6)
    q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)
    packed = torch.randperm(capacity, device="cuda")[:total_blocks].to(torch.int32)
    block_ids = torch.empty(
        (len(lengths), int(nblocks.max().item())), dtype=torch.int32, device="cuda"
    )
    offset = 0
    for batch, blocks in enumerate(nblocks.cpu().tolist()):
        block_ids[batch, :blocks] = packed[offset : offset + blocks]
        offset += blocks

    k_cache = (
        torch.randn(
            capacity,
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


def assign_cuda_task_map(
    inputs: DecodeInputs,
    task_map: torch.Tensor,
    *,
    min_process_len: int,
) -> None:
    """Populate the comparison map through the official GPU assigner."""
    num_head_kv = int(inputs.k_cache.shape[2])
    hpc.assign_attention_decode_task(
        inputs.kv_lens,
        task_map,
        num_head_kv,
        1,
        True,
        min_process_len=min_process_len,
    )


def prepare_cuda_task_map(
    inputs: DecodeInputs,
    *,
    min_process_len: int,
) -> torch.Tensor:
    """Allocate and populate the official CUDA comparison task map."""
    task_map = hpc.get_attention_decode_task_workspace(
        inputs.num_batch,
        inputs.max_seq_kv,
        int(inputs.k_cache.shape[2]),
        min_process_len=min_process_len,
    )
    assign_cuda_task_map(inputs, task_map, min_process_len=min_process_len)
    return task_map


def run_cuda(
    inputs: DecodeInputs,
    task_map: torch.Tensor,
    output: torch.Tensor,
    *,
    include_assign: bool = False,
    min_process_len: int,
) -> torch.Tensor:
    if include_assign:
        assign_cuda_task_map(inputs, task_map, min_process_len=min_process_len)
    return hpc.attention_decode_fp8(
        inputs.q,
        inputs.k_cache,
        inputs.v_cache,
        inputs.block_ids,
        inputs.kv_lens,
        inputs.q_scale,
        inputs.k_scale,
        inputs.v_scale,
        mtp=0,
        new_kv_included=True,
        quant_type=hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR,
        splitk=True,
        task_map=task_map,
        output=output,
    )


def bench_us(call, warmup: int, iters: int, use_graph: bool) -> float:
    """Match the CUDA-event/CUDA-Graph timing used by the original sweep."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    if use_graph:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                call()
        torch.cuda.current_stream().wait_stream(capture_stream)
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        call = graph.replay

    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(iters)
    ]
    for start, end in events:
        start.record()
        call()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return times[len(times) // 2]


def measure(
    call,
    warmup: int,
    iters: int,
    repeat: int,
    use_graph: bool,
) -> tuple[float, float, float]:
    """Return median/min/max latency in microseconds."""
    samples = [bench_us(call, warmup, iters, use_graph) for _ in range(repeat)]
    return statistics.median(samples), min(samples), max(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--layout", nargs="+", choices=("NHD", "HND"), default=["NHD", "HND"])
    parser.add_argument("--num-head-kv", type=int, default=1)
    parser.add_argument("--num-head-q", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-process-len", type=int, default=512)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--no-graph",
        dest="graph",
        action="store_false",
        help="Use eager CUDA-event timing instead of CUDA Graph replay.",
    )
    parser.add_argument(
        "--include-taskmap",
        "--include-assign",
        dest="include_assign",
        action="store_true",
        help="Time assign kernel + compute for both CUDA and TLE providers.",
    )
    parser.set_defaults(graph=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    triton.set_allocator(
        lambda size, _align, _stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )
    mode = "assign + compute" if args.include_assign else "compute only (prebuilt task maps)"
    header = (
        f"{'case':>18} | {'layout':>6} | {'config':>18} | {'cuda':>10} | "
        f"{'triton+tle':>10} | {'min':>10} | {'max':>10} | {'vs cuda':>9} | "
        f"{'n':>3} | {'check':>7}"
    )
    width = len(header)
    print("=" * width)
    print(f"Attention Decode FP8 | runtime policy | {mode} | latency in us")
    print("-" * width)
    print(header)
    print("-" * width)
    for layout in args.layout:
        for case, lengths in CASES.items():
            if case not in args.cases:
                continue
            inputs = make_inputs(lengths, args.num_head_kv, args.num_head_q, layout)
            cuda_task_map = prepare_cuda_task_map(
                inputs, min_process_len=args.min_process_len
            )
            cuda_output = torch.empty_like(inputs.q, dtype=torch.bfloat16)
            workspace = prepare_decode_workspace(inputs)
            cuda_call = lambda: run_cuda(
                inputs,
                cuda_task_map,
                cuda_output,
                include_assign=args.include_assign,
                min_process_len=args.min_process_len,
            )
            tle_call = lambda: fp8_kvpertensor_decode(
                inputs,
                workspace,
                refresh_schedule="full" if args.include_assign else None,
            )
            check = ""
            if args.check:
                expected = cuda_call()
                actual = tle_call()
                torch.cuda.synchronize()
                check = str(bool(torch.allclose(actual, expected, atol=0.2, rtol=0.2)))
                if check != "True":
                    raise AssertionError(f"{case}/{layout}: output mismatch")
            cuda_us, _, _ = measure(
                cuda_call, args.warmup, args.iters, args.repeat, args.graph
            )
            tle_us, tle_min_us, tle_max_us = measure(
                tle_call, args.warmup, args.iters, args.repeat, args.graph
            )
            config = (
                f"cluster{workspace.config.cluster_size}"
                f"token{workspace.config.chunk_tokens}"
            )
            print(
                f"{case:>18} | {layout:>6} | {config:>18} | {cuda_us:10.2f} | "
                f"{tle_us:10.2f} | {tle_min_us:10.2f} | "
                f"{tle_max_us:10.2f} | {cuda_us / tle_us:8.2f}x | "
                f"{args.repeat:3d} | {check:>7}"
            )
    print("=" * width)


if __name__ == "__main__":
    main()
