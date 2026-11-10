"""Compare Forgetting Attention with ACP and FLA using full-call CUDA Event latency.

FLA has no adaptive pruning; its rows use an inactive finite threshold and require M == N.
Run with pytest -m parallel_forgetting_attn. Latencies are in ms.
"""
from __future__ import annotations


import math
import statistics


import pytest
import torch

from flag_attn.testing import backend as test_backend
from test_forgetting_attention import (
    DEFAULT_CASES, case_id, make_inputs,
    call_optimized, call_official, assert_bitwise,
)

pytestmark = [
    pytest.mark.parallel_forgetting_attn,
    pytest.mark.skipif(not test_backend.supports_operator("parallel_forgetting_attn"), reason="selected Forgetting Attention implementation is unavailable"),
]


def event_samples(fn, warmup, rep):
    if warmup < 1 or rep < 1:
        raise ValueError("warmup and rep must both be positive iteration counts")
    if not hasattr(test_backend.device_fn, "Event"):
        return [test_backend.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")]
    for _ in range(warmup):
        fn()
    test_backend.device_fn.synchronize()
    pairs = []
    for _ in range(rep):
        begin = test_backend.device_fn.Event(enable_timing=True)
        end = test_backend.device_fn.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        pairs.append((begin, end))
    test_backend.device_fn.synchronize()
    samples = [a.elapsed_time(b) for a, b in pairs]
    if not all(math.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError("invalid CUDA Event timing")
    return samples


@torch.inference_mode()
def benchmark_case(case, baseline_kind="official", warmup=40, rep=400, rounds=3, seed=0):
    if rounds < 1:
        raise ValueError("rounds must be positive")
    if baseline_kind not in ("official", "fla", None):
        raise ValueError(f"unknown baseline: {baseline_kind}")

    inputs = make_inputs(case, seed)
    pruning = "adaptive(-10)" if baseline_kind != "fla" else "inactive(-1e9)"
    threshold = -10.0 if baseline_kind != "fla" else -1e9
    providers = {"flag_attn": lambda: call_optimized(inputs, case, threshold=threshold)}
    baseline_key = None
    baseline_name = None
    accuracy = "not_checked"

    if baseline_kind == "official" and test_backend.is_nvidia():
        baseline_key = "official"
        baseline_name = (
            "official_acp" if case["reference_kind"] == "native" else "official_acp_adapted"
        )
        baseline_fn = lambda: call_official(inputs, case, threshold=threshold)
        expected = baseline_fn()
        assert_bitwise(providers["flag_attn"](), expected)
        assert_bitwise(providers["flag_attn"](), expected)
        accuracy = "bitwise_pass"
        providers[baseline_key] = baseline_fn
        del expected
    elif baseline_kind == "fla":
        if not test_backend.is_nvidia():
            pytest.skip("FLA comparison requires NVIDIA CUDA")
        if case["M"] != case["N"]:
            pytest.skip("FLA parallel forgetting attention requires M == N")
        try:
            from fla.ops.forgetting_attn import parallel_forgetting_attn
        except ImportError as exc:
            pytest.skip(f"FLA comparison is unavailable: {exc}")
        # Every gate is finite and its worst possible N-token decay exceeds the threshold.
        assert inputs[3].amin().item() * case["N"] > threshold
        baseline_key = "fla"
        baseline_name = "fla_parallel_forgetting_attn"
        baseline_fn = lambda: parallel_forgetting_attn(*inputs, scale=case["scale"])
        expected = baseline_fn()
        torch.testing.assert_close(providers["flag_attn"](), expected, atol=0.02, rtol=0.02)
        accuracy = "close_pass"
        providers[baseline_key] = baseline_fn
        del expected

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

    baseline = (
        statistics.median(result[baseline_key]["median_ms"] for result in measurements)
        if baseline_key is not None else None
    )
    candidate = statistics.median(result["flag_attn"]["median_ms"] for result in measurements)
    return {
        "case": case,
        "baseline": baseline_name,
        "pruning": pruning,
        "threshold": threshold,
        "reference_kind": "direct" if baseline_kind == "fla" else case["reference_kind"],
        "latency_base": baseline,
        "latency": candidate,
        "speedup": baseline / candidate if baseline is not None else None,
        "accuracy": accuracy,
        "rounds": measurements,
    }


def format_table(rows):
    headers = (
        "Case", "Shape (B,M,N,Hq,Hkv,D)", "Dtype", "Type", "Pruning",
        "Baseline", "Baseline (ms)", "FlagAttention (ms)", "Speedup",
    )
    values = []
    for row in rows:
        case = row["case"]
        values.append((
            case["id"],
            ",".join(str(case[k]) for k in ("B", "M", "N", "HQ", "H", "D")),
            case["dtype"],
            row["reference_kind"],
            row["pruning"],
            row["baseline"] or "N/A",
            "N/A" if row["latency_base"] is None else f'{row["latency_base"]:.6f}',
            f'{row["latency"]:.6f}',
            "N/A" if row["speedup"] is None else f'{row["speedup"]:.3f}x',
        ))

    if any(row["baseline"] is not None for row in rows):
        numeric_columns = {6, 7, 8}
    else:
        # Without a baseline, display only the measured implementation latency.
        columns = (0, 1, 2, 3, 4, 7)
        headers = tuple(headers[i] for i in columns)
        values = [tuple(value[i] for i in columns) for value in values]
        numeric_columns = {5}
    widths = [max(len(value) for value in column) for column in zip(headers, *values)]

    def line(cells):
        return "| " + " | ".join(
            f"{value:>{widths[index]}}" if index in numeric_columns else f"{value:<{widths[index]}}"
            for index, value in enumerate(cells)
        ) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join((line(headers), separator, *(line(row) for row in values)))


@pytest.fixture(scope="module")
def benchmark_rows():
    rows = []
    yield rows
    if rows:
        print("\n" + format_table(rows), flush=True)


def detail_for(row):
    case = row["case"]
    return {"op_name": "parallel_forgetting_attn", "dtype": case["dtype"], "mode": "forward", "level": "end_to_end",
            "result": [{"shape_detail": [[case[k] for k in ("B", "M", "N", "HQ", "H", "D")],
                                         {"scale": case["scale"], "threshold": row["threshold"],
                                          "baseline_kind": row["baseline"],
                                          "pruning": row["pruning"]}],
                        "latency_base": row["latency_base"], "latency": row["latency"],
                        "speedup": row["speedup"], "accuracy": row["accuracy"]}]}


BENCHMARK_CASES = [
    pytest.param(case, "official", id=f"{case_id(case)}-official")
    for case in DEFAULT_CASES
] + [
    pytest.param(case, "fla", id=f"{case_id(case)}-fla")
    for case in DEFAULT_CASES if case["M"] == case["N"]
]


if not test_backend.is_nvidia():
    BENCHMARK_CASES = [
        pytest.param(case, None, id=f"{case_id(case)}-flag-attn")
        for case in DEFAULT_CASES
    ]


@pytest.mark.parametrize(("case", "baseline_kind"), BENCHMARK_CASES)
def test_forgetting_attention_benchmark(case, baseline_kind, record_property, benchmark_rows):
    row = benchmark_case(case, baseline_kind=baseline_kind)
    benchmark_rows.append(row)
    record_property("flag_attn_benchmark_result", detail_for(row))