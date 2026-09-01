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

"""Benchmark FP8 static and dynamic decode against the official CUDA op."""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import triton


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from flag_attn.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_dynamic as fp8_qk_dynamic,
    fp8_qpertoken_perhead_kvpertensor_dynamic as fp8_kv_dynamic,
)
from flag_attn.hpc_ops_attention.decode import HAS_TLE  # noqa: E402
from flag_attn.hpc_ops_attention.decode.static import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_static as fp8_qk_static,
    fp8_qpertoken_perhead_kvpertensor_static as fp8_kv_static,
)


BLOCK_SIZE = 64
HEAD_DIM = 128
SUPPORTED_MTP = (1, 2, 4)
QUANT_TYPES = {
    "qkpertoken_perhead_vperhead": 0,
    "qpertoken_perhead_kvpertensor": 1,
}
OFFICIAL_CASES = fp8_qk_static.OFFICIAL_CASES
FP8DecodeInputs = fp8_qk_static.FP8DecodeInputs
_IMPLEMENTATIONS = {
    ("static", "qkpertoken_perhead_vperhead"): fp8_qk_static,
    ("static", "qpertoken_perhead_kvpertensor"): fp8_kv_static,
    ("dynamic", "qkpertoken_perhead_vperhead"): fp8_qk_dynamic,
    ("dynamic", "qpertoken_perhead_kvpertensor"): fp8_kv_dynamic,
}


def _implementation(schedule: str, quant_type: str):
    return _IMPLEMENTATIONS[(schedule, quant_type)]


@dataclass
class Panel:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    block_ids: torch.Tensor
    kv_lens: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor


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


def _as_hnd_view(cache: torch.Tensor) -> torch.Tensor:
    return cache.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def _quantize_k_per_token(storage: torch.Tensor) -> torch.Tensor:
    blocks, _, heads, dim = storage.shape
    scale = (
        storage[:, :BLOCK_SIZE].float().abs().amax(-1).clamp_min(1e-6)
        / 448.0
    )
    result = torch.empty_like(storage, dtype=torch.float8_e4m3fn)
    result[:, :BLOCK_SIZE] = (
        storage[:, :BLOCK_SIZE] / scale[..., None]
    ).to(torch.float8_e4m3fn)
    packed = (
        scale.permute(0, 2, 1).contiguous().view(torch.float8_e4m3fn)
        .reshape(blocks, heads, -1, dim).permute(0, 2, 1, 3).contiguous()
    )
    result[:, BLOCK_SIZE:] = packed
    return result


def _quantize_v_per_head(
    storage: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = storage.shape[2]
    scale = (
        storage[:, :BLOCK_SIZE].float().abs().permute(2, 0, 1, 3)
        .reshape(heads, -1).amax(-1).clamp_min(1e-6) / 448.0
    )
    quantized = (
        storage.float() / scale[None, None, :, None]
    ).to(torch.float8_e4m3fn)
    return quantized, scale


def make_inputs(
    lengths, mtp: int, hkv: int, hq: int, layout: str, quant_type: str,
) -> Panel:
    torch.manual_seed(41)
    torch.cuda.manual_seed(41)
    kv_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8

    # Match the official benchmark's seeded RNG order exactly:
    # Q -> K/V -> scales -> block IDs.
    q_bf16 = torch.randn(
        (len(lengths) * mtp, hq, HEAD_DIM),
        dtype=torch.bfloat16, device="cuda",
    ) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.float().abs().amax(-1).clamp_min(1e-6)
    q = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)

    if QUANT_TYPES[quant_type] == 0:
        raw_k = torch.randn(
            (capacity, BLOCK_SIZE + 2, hkv, HEAD_DIM),
            dtype=torch.bfloat16, device="cuda",
        )
        raw_v = torch.randn_like(raw_k)
        k_storage = _quantize_k_per_token(raw_k)
        v_storage, v_scale = _quantize_v_per_head(raw_v)
        if layout == "HND":
            k_storage = _as_hnd_view(k_storage)
            v_storage = _as_hnd_view(v_storage)
        k_cache = k_storage[:, :BLOCK_SIZE]
        v_cache = v_storage[:, :BLOCK_SIZE]
        k_scale = k_storage[:, BLOCK_SIZE:]
    else:
        k_cache = (
            torch.randn(
                (capacity, BLOCK_SIZE, hkv, HEAD_DIM),
                dtype=torch.bfloat16, device="cuda",
            ) / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        v_cache = torch.randn(
            (capacity, BLOCK_SIZE, hkv, HEAD_DIM),
            dtype=torch.bfloat16, device="cuda",
        ).to(torch.float8_e4m3fn)
        if layout == "HND":
            k_cache = _as_hnd_view(k_cache)
            v_cache = _as_hnd_view(v_cache)
        k_scale = torch.rand(
            (1,), dtype=torch.float32, device="cuda"
        ).clamp_min(1e-6)
        v_scale = torch.rand(
            (1,), dtype=torch.float32, device="cuda"
        ).clamp_min(1e-6)

    packed_ids = torch.randperm(capacity, device="cuda")[:total_blocks].int()
    block_ids = torch.empty(
        (len(lengths), int(block_counts.max().item())),
        dtype=torch.int32, device="cuda",
    )
    offset = 0
    for batch, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch, :count] = packed_ids[offset:offset + count]
        offset += count

    return Panel(
        q, k_cache, v_cache, block_ids, kv_lens,
        q_scale, k_scale, v_scale,
    )


def _inputs(panel: Panel) -> FP8DecodeInputs:
    return FP8DecodeInputs(
        panel.q, panel.k_cache, panel.v_cache, panel.block_ids,
        panel.kv_lens, panel.q_scale, panel.k_scale, panel.v_scale,
    )


def _cuda_quant_type(hpc, quant_type: str):
    return (
        hpc.QuantType.QPERTOKEN_PERHEAD_KPERTOKEN_PERHEAD_VPERHEAD
        if QUANT_TYPES[quant_type] == 0 else
        hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR
    )


def _cuda_task_map(hpc, panel: Panel, mtp: int, min_process_len: int):
    task_map = hpc.get_attention_decode_task_workspace(
        len(panel.kv_lens), int(panel.kv_lens.max()),
        panel.k_cache.shape[2], min_process_len=min_process_len,
    )
    hpc.assign_attention_decode_task(
        panel.kv_lens, task_map, panel.k_cache.shape[2], mtp, True,
        min_process_len=min_process_len,
    )
    return task_map


def _run_cuda(
    hpc, panel: Panel, output: torch.Tensor, mtp: int,
    quant_type: str, task_map,
):
    return hpc.attention_decode_fp8(
        panel.q, panel.k_cache, panel.v_cache, panel.block_ids,
        panel.kv_lens, panel.q_scale, panel.k_scale, panel.v_scale,
        mtp=mtp - 1, new_kv_included=True,
        quant_type=_cuda_quant_type(hpc, quant_type), splitk=True,
        task_map=task_map, output=output,
    )


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
    values = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return values[len(values) // 2]


def _measure(calls, warmup, iters, repeat, graph_mode):
    names = list(calls)
    samples = {name: [] for name in names}
    for index in range(repeat):
        order = names[index % len(names):] + names[:index % len(names)]
        if index & 1:
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
    parser.add_argument("--mtp", nargs="+", type=int, choices=SUPPORTED_MTP,
                        default=list(SUPPORTED_MTP))
    parser.add_argument("--quant-types", nargs="+", choices=QUANT_TYPES,
                        default=list(QUANT_TYPES))
    parser.add_argument("--schedules", nargs="+", choices=("static", "dynamic"),
                        default=["static", "dynamic"])
    parser.add_argument("--cases", nargs="+", choices=OFFICIAL_CASES,
                        default=list(OFFICIAL_CASES))
    parser.add_argument("--layout", nargs="+", choices=("NHD", "HND"),
                        default=["NHD", "HND"])
    parser.add_argument("--num-head-kv", type=int, default=1)
    parser.add_argument("--num-head-q", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--min-process-len", type=int, default=512)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-graph", dest="graph", action="store_false")
    parser.set_defaults(graph=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.num_head_q, args.num_head_kv) != (8, 1):
        raise ValueError("validated final FP8 policies require Hq=8 and Hkv=1")
    triton.set_allocator(lambda size, _align, _stream: torch.empty(
        size, dtype=torch.int8, device="cuda"
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
    if hpc is None and args.check:
        print("CUDA baseline unavailable; FP8 --check is limited to finite-output "
              "and workspace-reset checks", file=sys.stderr)
    ratios = {
        (schedule, mtp, quant_type, layout): []
        for schedule in args.schedules
        for mtp in args.mtp
        for quant_type in args.quant_types
        for layout in args.layout
    }
    backend_label = "triton+tle" if HAS_TLE else "pure triton"
    if hpc is None:
        header = (f"{'case':>18} | {'sched':>7} | {'mtp':>3} | {'layout':>6} | "
                  f"{'quant_type':>34} | {backend_label:>10} | {'min':>9} | {'max':>9}")
    else:
        header = (f"{'case':>18} | {'sched':>7} | {'mtp':>3} | {'layout':>6} | "
                  f"{'quant_type':>34} | {'cuda':>10} | {backend_label:>10} | "
                  f"{'min':>9} | {'max':>9} | {'vs cuda':>8}")
    runtime = "Triton+TLE" if HAS_TLE else "Pure Triton"
    comparison = (
        f"HPC CUDA vs {runtime}"
        if hpc is not None
        else f"{runtime} | HPC CUDA unavailable"
    )
    title = f"Attention Decode FP8 | {comparison} | latency in us"
    if hpc is not None:
        title += f"; x = CUDA / {runtime}"
    width = max(len(header), len(title))
    print("=" * width, flush=True)
    print(title, flush=True)
    print("-" * width, flush=True)
    print(header, flush=True)
    print("-" * width, flush=True)

    for mtp in args.mtp:
        for quant_type in args.quant_types:
            for layout in args.layout:
                for case in args.cases:
                    panel = make_inputs(
                        OFFICIAL_CASES[case], mtp, args.num_head_kv,
                        args.num_head_q, layout, quant_type,
                    )
                    inputs = _inputs(panel)
                    cuda_output = (torch.empty_like(panel.q, dtype=torch.bfloat16)
                                   if hpc is not None else None)
                    for schedule in args.schedules:
                        implementation = _implementation(schedule, quant_type)
                        workspace = implementation.prepare_decode_workspace(
                            inputs,
                        )
                        task_map = (
                            _cuda_task_map(
                                hpc, panel, mtp, args.min_process_len,
                            )
                            if hpc is not None and schedule == "dynamic" else None
                        )
                        tle_call = lambda: implementation.attention_decode_fp8(
                            inputs, workspace,
                        )
                        cuda_call = (None if hpc is None else lambda: _run_cuda(
                            hpc, panel, cuda_output, mtp, quant_type, task_map,
                        ))
                        if args.check:
                            actual = tle_call().detach().clone()
                            torch.cuda.synchronize()
                            reset = implementation.workspace_is_reset(workspace)
                            close = True
                            if cuda_call is not None:
                                expected = cuda_call().detach().clone()
                                close = bool(torch.allclose(actual, expected, atol=0.2, rtol=0.2))
                            if not bool(torch.isfinite(actual).all()) or not close or not reset:
                                expected = actual if cuda_call is None else expected
                                diff = (actual.float() - expected.float()).abs()
                                raise AssertionError(
                                    f"{schedule}/mtp={mtp}/{quant_type}/"
                                    f"{case}/{layout}: close={close}, reset={reset}, "
                                    f"mae={diff.mean().item():.6f}, "
                                    f"max={diff.max().item():.6f}"
                                )
                        calls = {"tle": tle_call}
                        if cuda_call is not None:
                            calls["cuda"] = cuda_call
                        timing = _measure(
                            calls,
                            args.warmup, args.iters, args.repeat, args.graph,
                        )
                        tle_us, minimum, maximum = timing["tle"]
                        if cuda_call is None:
                            print(f"{case:>18} | {schedule:>7} | {mtp:3d} | "
                                  f"{layout:>6} | {quant_type:>34} | {tle_us:10.2f} | "
                                  f"{minimum:9.2f} | {maximum:9.2f}", flush=True)
                        else:
                            cuda_us = timing["cuda"][0]
                            ratio = cuda_us / tle_us
                            ratios[(schedule, mtp, quant_type, layout)].append(ratio)
                            print(f"{case:>18} | {schedule:>7} | {mtp:3d} | "
                                  f"{layout:>6} | {quant_type:>34} | {cuda_us:10.2f} | "
                                  f"{tle_us:10.2f} | {minimum:9.2f} | {maximum:9.2f} | "
                                  f"{ratio:7.3f}x", flush=True)
                        del workspace, tle_call, cuda_call, task_map
                    del panel, inputs, cuda_output
                    gc.collect()
                    torch.cuda.empty_cache()

    print("=" * width, flush=True)
    if hpc is None:
        print("FP8 benchmark complete (CUDA baseline unavailable)", flush=True)
        print("=" * width, flush=True)
        return
    print("Attention Decode FP8 summary", flush=True)
    for schedule in args.schedules:
        for mtp in args.mtp:
            for quant_type in args.quant_types:
                for layout in args.layout:
                    values = ratios[(schedule, mtp, quant_type, layout)]
                    print(
                        f"{schedule} | mtp={mtp} | {layout} | {quant_type} | "
                        f"{min(values):.3f}x-{max(values):.3f}x",
                        flush=True,
                    )
    for layout in args.layout:
        values = [
            ratio
            for schedule in args.schedules
            for mtp in args.mtp
            for quant_type in args.quant_types
            for ratio in ratios[(schedule, mtp, quant_type, layout)]
        ]
        print(
            f"overall | {layout} | fp8 | "
            f"{min(values):.3f}x-{max(values):.3f}x",
            flush=True,
        )
    print("=" * width, flush=True)


if __name__ == "__main__":
    main()
