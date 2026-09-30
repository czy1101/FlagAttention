# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend FP8 decode benchmark for the nine HY3 workloads.

The CUDA benchmark in this directory covers the Hopper path; this script covers
the Ascend FP8 decode runtime backend in
``flag_attn.runtime.backend._ascend.hpc_ops_attention.decode`` with the same
nine workloads, the same official GQA splits and both layouts.

The FP8 runtime used to perform host-side ``.item()`` synchronisation inside its
entry point, which forbade NPUGraph capture.  That synchronisation is gone, so
``--graph`` captures the whole ``attention_decode_fp8`` call and times
``replay``.  The default stays eager plus ``torch.npu.Event``: every sample is
the end-to-end latency of one ``attention_decode_fp8`` call, measured with a
fresh event pair, and the reported number is the median of per-repeat medians.
``--graph`` measures eager and graph back to back in the same process and prints
one ``RESULT`` line per mode, so the two are directly comparable.  Unless
``--warmup``/``--iters`` pin them, the counts follow the measured per-call
latency and are echoed in every ``RESULT`` line.

    source /workspace/env.sh
    export ASCEND_VISIBLE_DEVICES=6 ASCEND_RT_VISIBLE_DEVICES=6
    TRITON_CACHE_DIR=/tmp/tc_dec_warm /workspace/FlagAttention/.venv/bin/python \
        benchmark/bench_hpc_ops_decode_fp8_ascend.py [options] [case ...]

Options:
    --heads HxKV          official GQA split, 8x1 (default) or 32x4
    --quant TYPE          qpertoken_perhead_kvpertensor (default) or
                          qkpertoken_perhead_vperhead
    --schedule static|dynamic|both   (default both)
    --layout NHD|HND|both (default both)
    --mtp 1|2|4           (default 1)
    --repeat N            event-timed repeats, median reported (default 3)
    --iters N             event samples per repeat (default: latency-driven)
    --warmup N            untimed calls before timing (default: latency-driven)
    --inner N             calls per event pair, timed together then divided out
                          (default: latency-driven, 1 when iters/warmup are pinned)
    --graph               also capture the call in an NPUGraph and time replay
    --check               compare against the E4M3 reference on CHECK_CASES
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.dynamic import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_dynamic as qk_dynamic,
    fp8_qpertoken_perhead_kvpertensor_dynamic as qt_dynamic,
)
from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.static import (  # noqa: E402
    fp8_qkpertoken_perhead_vperhead_static as qk_static,
    fp8_qpertoken_perhead_kvpertensor_static as qt_static,
)

BLOCK_SIZE = 64
HEAD_DIM = 128
CHECK_CASES = ("uniform_512", "skewed_extreme")

# Latency-driven timing budget.  The official FIA baseline spends 1600 calls per
# case because it targets <=600us shapes; at millisecond shapes that is minutes
# per case, so the counts are derived from one probe instead.  Every repeat aims
# at REPEAT_BUDGET_US of timed device work; ``inner`` packs several calls into
# one event pair (divided out afterwards) so the event overhead stays negligible
# on the sub-100us cases, and the sample count then fills the remaining budget.
REPEAT_BUDGET_US = 60_000.0
INNER_TARGET_US = 1_500.0
INNER_MAX = 64
ITER_MIN = 10
ITER_MAX = 50
WARMUP_BUDGET_US = 5_000.0
WARMUP_MIN = 3
WARMUP_MAX = 25

OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}

_MODULES = {
    ("qpertoken_perhead_kvpertensor", "dynamic"): qt_dynamic,
    ("qpertoken_perhead_kvpertensor", "static"): qt_static,
    ("qkpertoken_perhead_vperhead", "dynamic"): qk_dynamic,
    ("qkpertoken_perhead_vperhead", "static"): qk_static,
}


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.uint8)


def _as_hnd(value: torch.Tensor) -> torch.Tensor:
    return value.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)


def make_inputs(lengths, layout, hq, hkv, mtp, quant_type):
    """One shared paged pool, same random-fill order as the bf16 Ascend bench."""
    torch.manual_seed(41)
    torch.npu.manual_seed(41)
    hq = int(hq)
    hkv = int(hkv)
    batch = len(lengths)
    kv_lens = torch.tensor(lengths, dtype=torch.int32)
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + batch + 8

    q_bf16 = torch.randn(batch * mtp, hq, HEAD_DIM) / math.sqrt(HEAD_DIM)
    q_scale = q_bf16.abs().amax(-1) / 10.0
    q_fp8 = (q_bf16 / q_scale[..., None]).to(torch.float8_e4m3fn)

    if quant_type == "qpertoken_perhead_kvpertensor":
        k_fp8 = (
            torch.randn(capacity, BLOCK_SIZE, hkv, HEAD_DIM) / math.sqrt(HEAD_DIM)
        ).to(torch.float8_e4m3fn)
        v_fp8 = torch.randn(capacity, BLOCK_SIZE, hkv, HEAD_DIM).to(
            torch.float8_e4m3fn
        )
        k_scale = torch.tensor([0.7], dtype=torch.float32)
        v_scale = torch.tensor([0.4], dtype=torch.float32)
        k = _bits(k_fp8)
        v = _bits(v_fp8)
    else:
        raw_k = torch.randn(capacity, BLOCK_SIZE, hkv, HEAD_DIM)
        raw_v = torch.randn(capacity, BLOCK_SIZE, hkv, HEAD_DIM)
        token_scale = raw_k.abs().amax(-1) / 448.0
        k_fp8 = (raw_k / token_scale[..., None]).to(torch.float8_e4m3fn)
        packed_scale = (
            token_scale.permute(0, 2, 1)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(capacity, hkv, 2, HEAD_DIM)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k_storage = torch.empty(
            capacity, BLOCK_SIZE + 2, hkv, HEAD_DIM, dtype=torch.uint8
        )
        k_storage[:, :BLOCK_SIZE] = _bits(k_fp8)
        k_storage[:, BLOCK_SIZE:] = _bits(packed_scale)
        head_scale = (
            raw_v.abs().permute(2, 0, 1, 3).reshape(hkv, -1).amax(-1) / 448.0
        )
        v_fp8 = (raw_v / head_scale[None, None, :, None]).to(torch.float8_e4m3fn)
        k = k_storage[:, :BLOCK_SIZE]
        v = _bits(v_fp8)
        k_scale = k_storage[:, BLOCK_SIZE:]
        v_scale = head_scale

    packed = torch.randperm(capacity)[:total_blocks].to(torch.int32)
    block_ids = torch.empty(batch, int(block_counts.max().item()), dtype=torch.int32)
    cursor = 0
    for index, count in enumerate(block_counts.tolist()):
        block_ids[index, :count] = packed[cursor:cursor + count]
        cursor += count

    if layout == "HND":
        k = _as_hnd(k)
        v = _as_hnd(v)
        if quant_type == "qkpertoken_perhead_vperhead":
            k_scale = _as_hnd(k_scale)

    return dict(
        q=_bits(q_fp8).npu(),
        k_cache=k.npu(),
        v_cache=v.npu(),
        block_ids=block_ids.npu(),
        kv_lens=kv_lens.npu(),
        q_scale=q_scale.npu(),
        k_scale=k_scale.npu(),
        v_scale=v_scale.npu(),
    )


def bench_ms(call, warmup, iters, repeat, inner=1):
    """Eager timing: ``inner`` back-to-back calls per event pair, divided out.

    ``inner == 1`` is the original methodology and is bit-for-bit unchanged.
    """
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    medians = []
    for _ in range(repeat):
        events = [
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for start, end in events:
            start.record()
            for _ in range(inner):
                call()
            end.record()
        torch.npu.synchronize()
        samples = sorted(
            start.elapsed_time(end) / inner for start, end in events
        )
        medians.append(samples[len(samples) // 2] * 1000.0)
    return statistics.median(medians), min(medians), medians


def bench_graph_ms(call, warmup, iters, repeat, inner=1):
    """Capture ``inner`` calls in one NPUGraph, then time ``replay``.

    Returns the same ``(median, min, repeats)`` triple as :func:`bench_ms` plus
    the live graph so the caller can replay it once more for a numerics check.
    """
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(inner):
                call()
    torch.npu.current_stream().wait_stream(stream)
    for _ in range(warmup):
        graph.replay()
    torch.npu.synchronize()
    medians = []
    for _ in range(repeat):
        events = [
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
        torch.npu.synchronize()
        samples = sorted(
            start.elapsed_time(end) / inner for start, end in events
        )
        medians.append(samples[len(samples) // 2] * 1000.0)
    return statistics.median(medians), min(medians), medians, graph


def _probe_us(call, warm=3, samples=5):
    """One batched event pair around ``samples`` calls, in microseconds/call."""
    for _ in range(warm):
        call()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(samples):
        call()
    end.record()
    torch.npu.synchronize()
    return max(start.elapsed_time(end) * 1000.0 / samples, 1.0)


def plan_timing(call, warmup=None, iters=None, inner=None):
    """Derive ``(warmup, iters, inner)`` from the measured per-call latency.

    Short cases take many samples; millisecond shapes take the minimum.  Any
    explicit value wins, and pinning both ``warmup`` and ``iters`` also pins
    ``inner`` to one and skips the probe, so that invocation is exactly the
    original eager methodology.
    """
    if warmup is not None and iters is not None:
        return warmup, iters, 1 if inner is None else inner
    probe_us = _probe_us(call)
    if inner is None:
        inner = int(min(INNER_MAX, max(1, round(INNER_TARGET_US / probe_us))))
    if warmup is None:
        warmup = int(
            min(WARMUP_MAX, max(WARMUP_MIN, round(WARMUP_BUDGET_US / probe_us)))
        )
    if iters is None:
        window_us = max(inner * probe_us, 1.0)
        iters = int(
            min(ITER_MAX, max(ITER_MIN, round(REPEAT_BUDGET_US / window_us)))
        )
    return warmup, iters, inner


def _load_test_helpers():
    path = REPO_ROOT / "tests" / "flag_attn" / "test_hpc_ops_decode_fp8_ascend.py"
    spec = importlib.util.spec_from_file_location("fp8_decode_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=None)
    parser.add_argument("--heads", default="8x1", help="Hq x Hkv, e.g. 8x1 or 32x4")
    parser.add_argument(
        "--quant",
        default="qpertoken_perhead_kvpertensor",
        choices=("qpertoken_perhead_kvpertensor", "qkpertoken_perhead_vperhead"),
    )
    parser.add_argument(
        "--schedule", default="both", choices=("static", "dynamic", "both")
    )
    parser.add_argument("--mtp", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--layout", default="both", choices=("NHD", "HND", "both"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="event samples per repeat (default: derived from the latency)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="untimed calls before timing (default: derived from the latency)",
    )
    parser.add_argument(
        "--inner",
        type=int,
        default=None,
        help="calls packed per event pair, divided out (default: derived)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also capture the call in an NPUGraph and time replay, same process",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--pps",
        type=int,
        default=None,
        help="force pages_per_split for every workspace (matrix knob)",
    )
    args = parser.parse_args()
    if args.pps:
        from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode import (
            fp8 as _fp8,
        )

        _fp8.PAGES_PER_SPLIT = args.pps
        _fp8.PAGES_PER_SPLIT_CHOICES = (args.pps,)

    hq, hkv = (int(x) for x in args.heads.lower().split("x"))
    selected = args.cases or list(OFFICIAL_CASES)
    schedules = ("dynamic", "static") if args.schedule == "both" else (args.schedule,)
    layouts = ("NHD", "HND") if args.layout == "both" else (args.layout,)
    test_module = _load_test_helpers() if args.check else None

    print(
        f"device={torch.npu.get_device_name(0)} heads={hq}q/{hkv}kv "
        f"quant={args.quant} mtp={args.mtp} repeat={args.repeat} "
        f"warmup={args.warmup if args.warmup is not None else 'auto'} "
        f"iters={args.iters if args.iters is not None else 'auto'} "
        f"inner={args.inner if args.inner is not None else 'auto'} "
        f"timing={'eager+graph' if args.graph else 'eager+event'}"
    )
    for case in selected:
        lengths = OFFICIAL_CASES[case]
        for layout in layouts:
            attrs = make_inputs(lengths, layout, hq, hkv, args.mtp, args.quant)
            for schedule in schedules:
                module = _MODULES[(args.quant, schedule)]
                inputs = module.FP8DecodeInputs(**attrs)
                workspace = module.prepare_decode_workspace(inputs)

                def call(module=module, inputs=inputs, workspace=workspace):
                    return module.attention_decode_fp8(inputs, workspace)

                err = None
                ref_out = None
                if args.check and case in CHECK_CASES:
                    ref_out = call().detach().clone()
                    torch.npu.synchronize()
                    panel = test_module._make_panel(
                        args.quant, args.mtp, max(lengths), hkv, layout
                    )
                    expected = test_module._reference(
                        panel, args.mtp, max(lengths), hkv
                    )
                    err = (
                        (ref_out[:1].float().cpu() - expected.float()).abs().max().item()
                    )

                warmup, iters, inner = plan_timing(
                    call, args.warmup, args.iters, args.inner
                )
                counts = f"warmup={warmup} iters={iters} inner={inner}"
                shape = (
                    f"splits={workspace.max_splits} "
                    f"pps={workspace.pages_per_split} "
                    f"prod_tasks={workspace.num_producer_tasks} "
                    f"compact={int(workspace.compact_producer)} "
                    f"full={int(workspace.full_producer_splits)} "
                    f"hier={int(workspace.hierarchical_reduction)}"
                )
                extra = f" err={err:.2e}" if err is not None else ""

                def emit(mode, median_us, min_us, repeats, tail=""):
                    print(
                        f"RESULT {case} {layout} {args.quant} {schedule} "
                        f"heads={hq}x{hkv} mtp={args.mtp} mode={mode} "
                        f"median_us={median_us:.2f} "
                        f"min_us={min_us:.2f} {counts} {shape} "
                        f"repeats={','.join(f'{x:.1f}' for x in repeats)}"
                        f"{extra}{tail}",
                        flush=True,
                    )

                median_us, min_us, repeats = bench_ms(
                    call, warmup, iters, args.repeat, inner
                )
                emit("eager", median_us, min_us, repeats)

                if args.graph:
                    try:
                        g_med, g_min, g_rep, graph = bench_graph_ms(
                            call, warmup, iters, args.repeat, inner
                        )
                    except Exception as exc:  # noqa: BLE001
                        text = str(exc) or type(exc).__name__
                        message = text.splitlines()[0]
                        print(
                            f"RESULT {case} {layout} {args.quant} {schedule} "
                            f"heads={hq}x{hkv} mtp={args.mtp} mode=graph "
                            f"FAILED: {message[:150]}",
                            flush=True,
                        )
                    else:
                        tail = ""
                        if ref_out is not None:
                            # Capture replays the same kernels into the same
                            # workspace buffer; verify that buffer against the
                            # eager output instead of pretending the eager
                            # reference check covers the graph.
                            graph.replay()
                            torch.npu.synchronize()
                            delta = (
                                workspace.out.detach().float().cpu()
                                - ref_out.float().cpu()
                            ).abs().max().item()
                            tail = f" graph_delta={delta:.2e}"
                        emit("graph", g_med, g_min, g_rep, tail)

                del inputs, workspace, call
                torch.npu.empty_cache()
            del attrs
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
