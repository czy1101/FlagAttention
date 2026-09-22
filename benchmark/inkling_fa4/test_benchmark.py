"""Pytest entry point for the Inkling FA4 relative-attention benchmark.

Only the operator run time is measured. ``official_cute`` is probed at runtime
and skipped when the FlashAttention / CuTe DSL sources are unavailable, so the
suite runs unchanged on a CI box that has neither CuTe nor vLLM installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import cache
from pathlib import Path

import pytest
import torch

_UTILS_PATH = Path(__file__).parent / "benchmark_utils.py"
_spec = importlib.util.spec_from_file_location(
    "inkling_benchmark_utils_impl", _UTILS_PATH
)
_impl = importlib.util.module_from_spec(_spec)
sys.modules["inkling_benchmark_utils_impl"] = _impl
_spec.loader.exec_module(_impl)

BACKEND_NAMES = _impl.BACKEND_NAMES
Backend = _impl.Backend
append_csv = _impl.append_csv
case_names = _impl.case_names
resolve_backends = _impl.resolve_backends
run_benchmark = _impl.run_benchmark

pytestmark = [
    pytest.mark.inkling_fa4_rel_attention,
    pytest.mark.gpu,
    pytest.mark.benchmark,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"

WARMUP = int(os.getenv("BENCH_WARMUP", "20"))
ITERS = int(os.getenv("BENCH_ITERS", "100"))
REL = os.getenv("BENCH_REL", "zeros")
FLASH_ROOT = os.getenv("FLASH_ATTN_ROOT") or None


@cache
def available_backends() -> dict[str, Backend]:
    return resolve_backends(BACKEND_NAMES, FLASH_ROOT)


def _require_sm90() -> None:
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("benchmark targets H100 / SM90+")


@pytest.mark.parametrize("case_name", case_names())
@pytest.mark.parametrize("backend", BACKEND_NAMES)
def test_benchmark_backend(backend: str, case_name: str) -> None:
    _require_sm90()
    resolved = available_backends()[backend]
    if not resolved.available:
        pytest.skip(f"backend {backend!r} unavailable: {resolved.reason}")

    rows = run_benchmark(
        cases=[case_name],
        backends={backend: resolved},
        rel=REL,
        warmup=WARMUP,
        iters=ITERS,
        use_graph=True,
    )
    assert rows, f"no measurement produced for {case_name}/{backend}"

    for row in rows:
        assert row["status"] == "ok", f"{backend}/{case_name}: {row['status']}"
        assert float(row["mean_ms"]) > 0.0

    append_csv(rows, OUT / "pytest_benchmark_rows.csv")

    row = rows[0]
    print(
        f"\n{case_name:16s} {backend:14s} split={row['num_splits']} "
        f"mean={float(row['mean_ms']):.4f} ms p50={float(row['p50_ms']):.4f} "
        f"p95={float(row['p95_ms']):.4f} [{row['timing']}]"
    )


def test_official_cute_is_probed() -> None:
    """Record whether the optional CuTe backend is present on this machine."""
    resolved = available_backends()["official_cute"]
    print(
        f"\nofficial_cute: "
        f"{'available' if resolved.available else 'skipped (' + resolved.reason + ')'}"
    )
