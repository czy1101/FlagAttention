"""Official ACP vs V7.6 TLE. Full-call CUDA Event latency, never CUDA Graph.

Run with pytest -m parallel_forgetting_attn. Latencies are in ms.
The original reference adapters and input generation live in the single
correctness test file. Operator marker: parallel_forgetting_attn.
"""
from __future__ import annotations

import math
import statistics


import pytest
import torch

from flag_attn.forgetting_attention import has_tle
from test_forgetting_attention import (
    DEFAULT_CASES, case_id, make_inputs,
    call_optimized, call_official, assert_bitwise,
)

CUDA_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = [
    pytest.mark.parallel_forgetting_attn,
    pytest.mark.skipif(not CUDA_SM90, reason="ACP benchmark requires Hopper SM90 CUDA"),
    pytest.mark.skipif(not has_tle(), reason="ACP benchmark requires Triton 3.6 TLE"),
]


def event_samples(fn, warmup, rep):
    if warmup < 1 or rep < 1:
        raise ValueError("warmup and rep must both be positive iteration counts")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = []
    for _ in range(rep):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        pairs.append((begin, end))
    torch.cuda.synchronize()
    samples = [a.elapsed_time(b) for a, b in pairs]
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError("invalid CUDA Event timing")
    return samples


@torch.inference_mode()
def benchmark_case(case, warmup=40, rep=400, rounds=3, seed=0):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    inputs = make_inputs(case, seed)
    expected = call_official(inputs, case)
    assert_bitwise(call_optimized(inputs, case), expected)
    assert_bitwise(call_optimized(inputs, case), expected)
    del expected
    providers = {"official": lambda: call_official(inputs, case),
                 "tle_v76": lambda: call_optimized(inputs, case)}
    measurements = []
    for index in range(rounds):
        order = list(providers)
        if (index + int(case["id"][1:])) % 2:
            order.reverse()
        result = {"round": index, "provider_order": order}
        for name in order:
            values = event_samples(providers[name], warmup, rep)
            result[name] = {"median_ms": statistics.median(values), "samples_ms": values}
        measurements.append(result)
    baseline = statistics.median(r["official"]["median_ms"] for r in measurements)
    candidate = statistics.median(r["tle_v76"]["median_ms"] for r in measurements)
    return {"case": case, "baseline": "official_acp" if case["reference_kind"] == "native" else "official_acp_adapted",
            "latency_base": baseline, "latency": candidate, "speedup": baseline / candidate,
            "accuracy": "bitwise_pass", "rounds": measurements}


def print_row(row):
    case = row["case"]
    shape = ",".join(str(case[k]) for k in ("B", "M", "N", "HQ", "H", "D"))
    print(f'{case["id"]:<5} {shape:<28} {case["dtype"]:<9} {case["reference_kind"]:<7} '
          f'{row["latency_base"]:>10.6f} {row["latency"]:>10.6f} {row["speedup"]:>8.3f}x', flush=True)


def detail_for(row):
    case = row["case"]
    return {"op_name": "forgetting_attention", "dtype": case["dtype"], "mode": "forward", "level": "end_to_end",
            "result": [{"shape_detail": [[case[k] for k in ("B", "M", "N", "HQ", "H", "D")],
                                         {"scale": case["scale"], "threshold": -10.0,
                                          "baseline_kind": row["baseline"]}],
                        "latency_base": row["latency_base"], "latency": row["latency"],
                        "speedup": row["speedup"], "accuracy": row["accuracy"]}]}


@pytest.mark.parallel_forgetting_attn
@pytest.mark.parametrize("case", DEFAULT_CASES, ids=case_id)
def test_forgetting_attention_benchmark(case, record_property):
    row = benchmark_case(case)
    print_row(row)
    # Optional forward compatibility with PR #70+ recorders; PR #69 ignores this property.
    record_property("flag_attn_benchmark_result", detail_for(row))
