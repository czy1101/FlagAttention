"""Aggregated multi-run performance test (pytest entry point).

Each case is measured ``PERF_RUNS`` times and the per-backend latency is
averaged across runs. Backends that are unavailable on the machine (typically
``official_cute`` on CI, or ``tle`` on a Triton build without
``triton.experimental.tle``) are skipped instead of failing the run.
"""

from __future__ import annotations

import importlib.util
import os
import statistics
import sys
from pathlib import Path

import pytest
import torch

# Load the local benchmark utilities by explicit path so that a top-level
# ``benchmarks`` package from another entry on PYTHONPATH (e.g. vllm-fa4-test)
# cannot shadow our own module.
_UTILS_PATH = (
    Path(__file__).parent.parent.parent
    / "benchmark" / "inkling_fa4" / "benchmark_utils.py"
)
_spec = importlib.util.spec_from_file_location("inkling_benchmark_utils", _UTILS_PATH)
_inkling_benchmark_utils = importlib.util.module_from_spec(_spec)
sys.modules["inkling_benchmark_utils"] = _inkling_benchmark_utils
_spec.loader.exec_module(_inkling_benchmark_utils)
append_csv = _inkling_benchmark_utils.append_csv
performance_cases = _inkling_benchmark_utils.performance_cases
resolve_backends = _inkling_benchmark_utils.resolve_backends
run_benchmark = _inkling_benchmark_utils.run_benchmark

pytestmark = [
    pytest.mark.inkling_fa4_rel_attention,
    pytest.mark.gpu,
    pytest.mark.performance,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="performance tests require a CUDA-capable GPU",
    ),
]



def _configured_cases() -> tuple[tuple[str, int], ...]:
    raw_splits = os.getenv("PERF_SPLITS")
    if raw_splits is None:
        return performance_cases()
    try:
        splits = tuple(int(value.strip()) for value in raw_splits.split(","))
    except ValueError as exc:
        raise ValueError("PERF_SPLITS must be comma-separated integers") from exc
    return performance_cases(split_config=splits)


CASES = _configured_cases()

RUNS = int(os.getenv("PERF_RUNS", "3"))
WARMUP = int(os.getenv("PERF_WARMUP", "100"))
ITERS = int(os.getenv("PERF_ITERS", "500"))
FLASH_ROOT = os.getenv("FLASH_ATTN_ROOT") or None

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SUMMARY_COLUMNS = ("case", "split", "backend", "mean_ms", "p50_ms", "p95_ms")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


@pytest.mark.parametrize("case_name,num_splits", CASES, ids=[c[0] for c in CASES])
def test_performance(case_name: str, num_splits: int) -> None:
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("performance tests target H100 / SM90+")

    backends = resolve_backends(flash_root=FLASH_ROOT)
    usable = {name: b for name, b in backends.items() if b.available}
    if not usable:
        pytest.skip("no benchmark backend available")

    samples: dict[str, dict[str, list[float]]] = {
        name: {"mean": [], "p50": [], "p95": []} for name in usable
    }
    for run_id in range(RUNS):
        rows = run_benchmark(
            cases=[case_name],
            backends=usable,
            rel="real",
            warmup=WARMUP,
            iters=ITERS,
            seed=run_id,
            num_splits=num_splits,
            use_graph=True,
            verbose=False,
        )
        for row in rows:
            if row["status"] != "ok":
                # ``official_cute`` is an optional third-party baseline, not the
                # operator under test, so its failures must not break CI.
                if row["backend"] == "official_cute":
                    print(
                        f"\n[warn] baseline official_cute failed on "
                        f"{case_name}: {row['status']}"
                    )
                    # CuTe failures are deterministic; drop it so the remaining
                    # runs do not pay its (slow) JIT compile again.
                    usable.pop("official_cute", None)
                    samples.pop("official_cute", None)
                    continue
                raise AssertionError(f"{case_name}/{row['backend']}: {row['status']}")
            for metric in ("mean", "p50", "p95"):
                samples[row["backend"]][metric].append(float(row[f"{metric}_ms"]))

    summary_rows = []
    means: dict[str, float] = {}
    for name, metrics in samples.items():
        if not metrics["mean"]:
            print(f"[warn] backend {name} produced no measurement; skipped")
            continue
        mean = _mean(metrics["mean"])
        means[name] = mean
        summary_rows.append(
            {
                "case": case_name,
                "split": num_splits,
                "backend": name,
                "mean_ms": f"{mean:.4f}",
                "p50_ms": f"{_mean(metrics['p50']):.4f}",
                "p95_ms": f"{_mean(metrics['p95']):.4f}",
            }
        )
    append_csv(
        summary_rows, OUT / "pytest_performance_summary.csv", SUMMARY_COLUMNS
    )

    ratios = []
    if "tle" in means:
        for name in ("official_cute", "triton"):
            if name in means:
                ratios.append(f"{name}/TLE={means[name] / means['tle']:.4f}x")

    print(
        f"\n{case_name:16s} split={num_splits} | "
        + " | ".join(f"{name}={value:.4f} ms" for name, value in means.items())
        + ((" | " + " | ".join(ratios)) if ratios else "")
    )
