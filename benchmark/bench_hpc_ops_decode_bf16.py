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

"""Benchmark the final BF16 static and dynamic policies against official CUDA."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode.dynamic.bf16_dynamic import (  # noqa: E402
    DynamicBF16Inputs,
    attention_decode_bf16_dynamic,
    bf16_dynamic_workspace_is_reset,
    prepare_dynamic_bf16_workspace,
)
from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.static.bf16_static import (  # noqa: E402
    BLOCK_SIZE,
    HEAD_DIM,
    OFFICIAL_CASES,
    StaticBF16Inputs,
    attention_decode_bf16_tle,
    prepare_static_bf16_workspace,
)


@dataclass
class Panel:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor


def _load_hpc():
    try:
        import hpc
        return hpc
    except (ImportError, OSError, AssertionError):
        # A source-tree ``hpc`` package may shadow the built extension and
        # assert because no local _C*.so lives beside its __init__.py.
        sys.modules.pop("hpc", None)
    build_roots = (REPO_ROOT / "build", REPO_ROOT.parent / "build")
    build_libs = sorted(
        path
        for build_root in build_roots
        for path in build_root.glob("lib.*")
        if list((path / "hpc").glob("_C*.so"))
    )
    if not build_libs:
        searched = ", ".join(str(path) for path in build_roots)
        print(f"hpc CUDA baseline unavailable ({searched}); omitting CUDA columns",
              file=sys.stderr)
        return None
    sys.path.insert(0, str(build_libs[0]))
    try:
        import hpc
    except (ImportError, OSError, AssertionError) as error:
        sys.modules.pop("hpc", None)
        print(f"hpc CUDA baseline unavailable ({error}); omitting CUDA columns",
              file=sys.stderr)
        return None
    return hpc


def _make_cuda_task_map(
    hpc,
    kv_lens: torch.Tensor,
    num_head_kv: int,
    num_seq_q: int,
    min_process_len: int,
) -> torch.Tensor:
    """Build the official CUDA dynamic schedule outside timed launches."""
    task_map = hpc.get_attention_decode_task_workspace(
        len(kv_lens), int(kv_lens.max().item()), num_head_kv,
        min_process_len=min_process_len,
    )
    hpc.assign_attention_decode_task(
        kv_lens,
        task_map,
        num_head_kv,
        num_seq_q,
        True,
        min_process_len=min_process_len,
    )
    return task_map


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def make_inputs(
    lengths, mtp: int, num_head_kv: int, num_head_q: int, layout: str,
) -> Panel:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    kv_lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    history = kv_lens - mtp
    if bool(torch.any(history < 0).item()):
        raise ValueError("every final KV length must be at least MTP")
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8
    q = torch.randn(
        (len(lengths) * mtp, num_head_q, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    k = torch.randn(
        (capacity, BLOCK_SIZE, num_head_kv, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    ) / math.sqrt(HEAD_DIM)
    v = torch.randn(
        (capacity, BLOCK_SIZE, num_head_kv, HEAD_DIM),
        device="cuda", dtype=torch.bfloat16,
    )
    packed = torch.randperm(capacity, device="cuda")[:total_blocks].int()
    block_ids = torch.empty(
        (len(lengths), int(block_counts.max().item())),
        device="cuda", dtype=torch.int32,
    )
    cursor = 0
    for batch, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch, :count] = packed[cursor:cursor + count]
        cursor += count
    if layout == "HND":
        k, v = _as_hnd_view(k), _as_hnd_view(v)
    return Panel(q, k, v, block_ids, kv_lens)


def pytorch_reference(panel: Panel, mtp: int) -> torch.Tensor:
    batch = panel.kv_lens.numel()
    hq, hkv = panel.q.shape[1], panel.k.shape[2]
    heads_per_group = hq // hkv
    q = panel.q.reshape(batch, mtp, hq, HEAD_DIM)
    outputs = []
    for batch_idx in range(batch):
        length = int(panel.kv_lens[batch_idx])
        pages = triton.cdiv(length, BLOCK_SIZE)
        ids = panel.block_ids[batch_idx, :pages]
        k = panel.k[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        v = panel.v[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        k = k.transpose(0, 1).repeat_interleave(heads_per_group, 0).float()
        v = v.transpose(0, 1).repeat_interleave(heads_per_group, 0).float()
        scores = q[batch_idx].transpose(0, 1).float() @ k.transpose(-1, -2)
        scores /= math.sqrt(HEAD_DIM)
        history = length - mtp
        causal = torch.cat((
            torch.ones((mtp, history), dtype=torch.bool, device=panel.q.device),
            torch.tril(torch.ones(
                (mtp, mtp), dtype=torch.bool, device=panel.q.device,
            )),
        ), dim=-1)
        scores.masked_fill_(~causal[None], -float("inf"))
        outputs.append((F.softmax(scores, -1) @ v).transpose(0, 1))
    return torch.stack(outputs).reshape(batch * mtp, hq, HEAD_DIM).bfloat16()


def _bench_us(call, warmup: int, iters: int, graph_mode: bool) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    if graph_mode:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                call()
        torch.cuda.current_stream().wait_stream(stream)
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
    samples = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return samples[len(samples) // 2]


def _measure(calls, warmup, iters, repeat, graph_mode):
    names = list(calls)
    samples = {name: [] for name in names}
    for repeat_idx in range(repeat):
        order = names[repeat_idx % len(names):] + names[:repeat_idx % len(names)]
        if repeat_idx & 1:
            order.reverse()
        for name in order:
            samples[name].append(
                _bench_us(calls[name], warmup, iters, graph_mode)
            )
    return {
        name: (statistics.median(values), min(values), max(values))
        for name, values in samples.items()
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtp", nargs="+", type=int, choices=(1, 2, 3),
                        default=[1, 2, 3])
    parser.add_argument("--cases", nargs="+", choices=OFFICIAL_CASES,
                        default=list(OFFICIAL_CASES))
    parser.add_argument("--methods", nargs="+", choices=("static", "dynamic"),
                        default=["static", "dynamic"])
    parser.add_argument("--layout", nargs="+", choices=("NHD", "HND"),
                        default=["NHD", "HND"])
    parser.add_argument("--num-head-kv", type=int, default=1)
    parser.add_argument("--num-head-q", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-process-len", type=int, default=64)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-graph", dest="graph", action="store_false")
    parser.set_defaults(graph=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.num_head_kv, args.num_head_q) not in ((1, 8), (4, 32)):
        raise ValueError("validated BF16 policy requires official GQA8 heads")
    triton.set_allocator(lambda size, _align, _stream: torch.empty(
        size, dtype=torch.int8, device="cuda",
    ))
    hpc = _load_hpc()
    if not HAS_TLE:
        skipped = [mtp for mtp in args.mtp if mtp != 1]
        if skipped:
            print("TLE unavailable; pure Triton fallback supports MTP=1 only; "
                  f"skipping MTP={skipped}", file=sys.stderr)
        args.mtp = [mtp for mtp in args.mtp if mtp == 1]
        if not args.mtp:
            raise ValueError("pure Triton fallback supports only --mtp 1")
    ratios = {
        (method, mtp, layout): []
        for method in args.methods
        for mtp in args.mtp
        for layout in args.layout
    }
    backend_label = "triton+tle" if HAS_TLE else "pure triton"
    if hpc is None:
        header = (f"{'case':>20} | {'method':>7} | {'mtp':>3} | {'layout':>6} | "
                  f"{backend_label:>12} | {'min':>9} | {'max':>9}")
    else:
        header = (f"{'case':>20} | {'method':>7} | {'mtp':>3} | {'layout':>6} | {'cuda':>9} | "
                  f"{backend_label:>12} | {'min':>9} | {'max':>9} | {'vs cuda':>8}")
    runtime = "Triton+TLE" if HAS_TLE else "Pure Triton"
    comparison = (
        f"HPC CUDA vs {runtime}"
        if hpc is not None
        else f"{runtime} | HPC CUDA unavailable"
    )
    title = f"Attention Decode BF16 | {comparison} | latency in us"
    if hpc is not None:
        title += (
            f"; x = CUDA / {runtime}; CUDA dynamic "
            f"min_process_len={args.min_process_len}"
        )
    width = max(len(header), len(title))
    print("=" * width, flush=True)
    print(title, flush=True)
    print("-" * width, flush=True)
    print(header, flush=True)
    print("-" * width, flush=True)

    for mtp in args.mtp:
        for case in args.cases:
            for layout in args.layout:
                panel = make_inputs(
                    OFFICIAL_CASES[case], mtp, args.num_head_kv,
                    args.num_head_q, layout,
                )
                reference = pytorch_reference(panel, mtp) if args.check else None
                for method in args.methods:
                    cuda_out = torch.empty_like(panel.q) if hpc is not None else None
                    if method == "static":
                        inputs = StaticBF16Inputs(
                            panel.q, panel.k, panel.v, panel.block_ids,
                            panel.kv_lens, layout,
                        )
                        workspace = prepare_static_bf16_workspace(inputs)
                        cuda_task_map = None
                        cuda_call = None if hpc is None else lambda: hpc.attention_decode_bf16(
                            panel.q, panel.k, panel.v, panel.block_ids,
                            panel.kv_lens, mtp=mtp - 1,
                            new_kv_included=True, splitk=True,
                            output=cuda_out,
                        )
                        tle_call = lambda: attention_decode_bf16_tle(
                            inputs, workspace,
                        )
                    else:
                        inputs = DynamicBF16Inputs(
                            panel.q, panel.k, panel.v, panel.block_ids,
                            panel.kv_lens, layout,
                        )
                        workspace = prepare_dynamic_bf16_workspace(inputs)
                        if HAS_TLE and getattr(workspace, "mtp", None) != mtp:
                            raise AssertionError(
                                f"official {case}/MTP{mtp} did not select a fixed route"
                            )
                        if HAS_TLE and not getattr(workspace, "route", ""):
                            raise AssertionError(
                                f"official {case}/MTP{mtp} has no production route label"
                            )
                        cuda_task_map = (_make_cuda_task_map(
                            hpc, panel.kv_lens, args.num_head_kv, mtp,
                            args.min_process_len,
                        ) if hpc is not None else None)
                        cuda_call = None if hpc is None else lambda: hpc.attention_decode_bf16(
                            panel.q, panel.k, panel.v, panel.block_ids,
                            panel.kv_lens, mtp=mtp - 1,
                            new_kv_included=True, splitk=True,
                            task_map=cuda_task_map, output=cuda_out,
                        )
                        tle_call = lambda: attention_decode_bf16_dynamic(
                            inputs, workspace,
                        )

                    if args.check:
                        actual = tle_call().detach().clone()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            actual, reference, atol=0.016, rtol=1e-5,
                        )
                        if cuda_call is not None:
                            expected = cuda_call().detach().clone()
                            torch.testing.assert_close(expected, reference, atol=0.016, rtol=1e-5)
                            torch.testing.assert_close(actual, expected, atol=0.032, rtol=1e-5)
                        if not torch.isfinite(actual).all():
                            raise AssertionError("non-finite BF16 output")
                        if (
                            method == "dynamic"
                            and not bf16_dynamic_workspace_is_reset(workspace)
                        ):
                            raise AssertionError(
                                "cooperative finalization counter was not reset"
                            )

                    calls = {"triton+tle": tle_call}
                    if cuda_call is not None:
                        calls["cuda"] = cuda_call
                    timing = _measure(
                        calls,
                        args.warmup, args.iters, args.repeat, args.graph,
                    )
                    tle_us, minimum, maximum = timing["triton+tle"]
                    if cuda_call is None:
                        print(f"{case:>20} | {method:>7} | {mtp:3d} | {layout:>6} | "
                              f"{tle_us:12.2f} | {minimum:9.2f} | {maximum:9.2f}", flush=True)
                    else:
                        cuda_us = timing["cuda"][0]
                        ratio = cuda_us / tle_us
                        ratios[method, mtp, layout].append(ratio)
                        print(f"{case:>20} | {method:>7} | {mtp:3d} | {layout:>6} | "
                              f"{cuda_us:9.2f} | {tle_us:12.2f} | {minimum:9.2f} | "
                              f"{maximum:9.2f} | {ratio:8.3f}x", flush=True)
                    del inputs, workspace, cuda_out
                    if cuda_task_map is not None:
                        del cuda_task_map
                del panel, reference
                gc.collect()
                torch.cuda.empty_cache()

    print("=" * width, flush=True)
    if hpc is None:
        print("BF16 benchmark complete (CUDA baseline unavailable)", flush=True)
        print("=" * width, flush=True)
        return
    print("Attention Decode BF16 summary", flush=True)
    for method in args.methods:
        for mtp in args.mtp:
            for layout in args.layout:
                values = ratios[method, mtp, layout]
                print(
                    f"{method} | mtp={mtp} | {layout} | bf16 | "
                    f"{min(values):.3f}x-{max(values):.3f}x",
                    flush=True,
                )
    for method in args.methods:
        for layout in args.layout:
            values = [
                ratio
                for mtp in args.mtp
                for ratio in ratios[method, mtp, layout]
            ]
            print(
                f"overall | {method} | {layout} | bf16 | "
                f"{min(values):.3f}x-{max(values):.3f}x",
                flush=True,
            )
    print("=" * width, flush=True)


if __name__ == "__main__":
    main()
