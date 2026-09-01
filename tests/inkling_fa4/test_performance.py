import csv
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.performance,
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="performance tests require a CUDA-capable GPU",
    ),
]


CASES = [
    ("full_prefill", 1),
    ("ragged_prefill", 1),
    ("long_prefill", 2),
    ("chunked_prefill", 8),
    ("sliding_window", 8),
    ("decode_1k", 32),
    ("decode_8k", 64),
]

RUNS = int(os.getenv("PERF_RUNS", "3"))
WARMUP = int(os.getenv("PERF_WARMUP", "100"))
ITERS = int(os.getenv("PERF_ITERS", "500"))

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

PATTERN = re.compile(
    r"^\s*(official_cute|triton|tle)\s+"
    r"mean=([0-9.]+)\s+ms\s+"
    r"p50=([0-9.]+)\s+"
    r"p95=([0-9.]+)",
    re.MULTILINE,
)


@pytest.mark.parametrize("case_name,num_splits", CASES)
def test_performance(case_name, num_splits):
    OUT.mkdir(exist_ok=True)
    measurements = []

    for run_id in range(1, RUNS + 1):
        csv_path = OUT / f"pytest_perf_{case_name}_run{run_id}.csv"

        command = [
            sys.executable,
            "-u",
            str(ROOT / "benchmark" / "inkling_fa4" / "benchmark_Cute_Triton_tle.py"),
            "--case",
            case_name,
            "--num-splits",
            str(num_splits),
            "--warmup",
            str(WARMUP),
            "--iters",
            str(ITERS),
            "--rel",
            "real",
            "--seed",
            "0",
            "--out",
            str(csv_path),
        ]

        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

        log_path = OUT / f"pytest_perf_{case_name}_run{run_id}.log"
        log_path.write_text(result.stdout, encoding="utf-8")

        assert result.returncode == 0, (
            f"{case_name} run {run_id} failed.\n"
            f"See {log_path}\n\n{result.stdout}"
        )

        rows = {
            backend: {
                "mean": float(mean),
                "p50": float(p50),
                "p95": float(p95),
            }
            for backend, mean, p50, p95 in PATTERN.findall(result.stdout)
        }

        missing = {"official_cute", "triton", "tle"} - rows.keys()
        assert not missing, (
            f"{case_name} run {run_id} missing backends: {sorted(missing)}\n"
            f"See {log_path}"
        )

        measurements.append(rows)

    result_row = {"case": case_name, "split": num_splits}

    for backend in ("official_cute", "triton", "tle"):
        for metric in ("mean", "p50", "p95"):
            value = sum(row[backend][metric] for row in measurements) / RUNS
            result_row[f"{backend}_{metric}"] = value

    cute_mean = result_row["official_cute_mean"]
    triton_mean = result_row["triton_mean"]
    tle_mean = result_row["tle_mean"]

    result_row["cute_tle"] = cute_mean / tle_mean
    result_row["triton_tle"] = triton_mean / tle_mean

    summary_path = OUT / "pytest_performance_summary.csv"
    write_header = not summary_path.exists()

    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result_row)

    print(
        f"\n{case_name:18s} split={num_splits} | "
        f"CuTe={cute_mean:.4f} ms | "
        f"Triton={triton_mean:.4f} ms | "
        f"TLE={tle_mean:.4f} ms | "
        f"CuTe/TLE={cute_mean / tle_mean:.4f}x | "
        f"Triton/TLE={triton_mean / tle_mean:.4f}x"
    )
