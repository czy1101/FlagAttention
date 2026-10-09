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

"""Pytest Log-Linear Attention benchmark with an optional TileLang baseline."""

import pytest

import argparse
import csv
import importlib.util
from pathlib import Path


pytestmark = pytest.mark.log_linear_attn




def _load_benchmark_module():
    # Reuse the baseline kept in the corresponding test script, rather than
    # duplicating a TileLang kernel or introducing another baselines directory.
    path = Path(__file__).resolve().parents[1] / "tests/flag_attn/test_log_linear_attn.py"
    spec = importlib.util.spec_from_file_location("log_linear_attn_benchmark_support", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args, module=None):
    if module is None:
        module = _load_benchmark_module()
    if not module._tle_available():
        raise RuntimeError("requires CUDA and a FlagTree/Triton build with triton.experimental.tle")
    if args.provider == "both" and module.tilelang is None:
        raise RuntimeError("--provider both requires TileLang (validated with 0.1.13)")
    compare = args.provider != "tle" and module.tilelang is not None
    if args.provider == "auto" and not compare:
        print("TileLang is unavailable; measuring TLE only. Install tilelang for comparison.")
    shapes = args.shape or module.BENCHMARK_SHAPES
    print(f"GPU={module.torch.cuda.get_device_name()} dtype=BF16 provider={'both' if compare else 'tle'}")
    print(f"Kernel-only timings: rounds={args.rounds}, warmup={args.warmup} ms, rep={args.rep} ms")
    print("shape\ttilelang_ms\ttle_ms\tspeedup_vs_tilelang")
    results = []
    for shape in shapes:
        result = module.benchmark_shape(
            *shape, compare_tilelang=compare, rounds=args.rounds,
            warmup=args.warmup, rep=args.rep,
        )
        results.append(result)
        baseline = f"{result['tilelang_ms']:.6f}" if compare else "N/A"
        speedup = f"{result['speedup_vs_tilelang']:.3f}x" if compare else "N/A"
        print(f"{result['shape']}\t{baseline}\t{result['tle_ms']:.6f}\t{speedup}", flush=True)
    if args.csv:
        with args.csv.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved results to {args.csv}")


@pytest.mark.log_linear_attn
def test_log_linear_attn_benchmark():
    module = _load_benchmark_module()
    if not module._tle_available():
        pytest.skip("requires CUDA and FlagTree/TLE")
    args = argparse.Namespace(
        provider="auto", shape=None, rounds=7, warmup=1000, rep=100, csv=None,
    )
    run(args, module)
