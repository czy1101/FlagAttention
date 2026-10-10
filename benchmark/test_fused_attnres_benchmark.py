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

"""AttnRes public-API latency with an optional FLA public-API comparison.

Timings include output allocation and API dispatch, rather than measuring a
preallocated internal kernel. Both providers use the same timing protocol.
"""

import importlib
import statistics

import pytest
import torch

import flag_attn
from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import benchmark_metric, record_benchmark_result
except ModuleNotFoundError:  # Execution from the benchmark directory.
    from recording import benchmark_metric, record_benchmark_result


pytestmark = pytest.mark.fused_attnres

BENCHMARK_SHAPES = [(2, 1), (5, 128), (5, 1024), (9, 128), (9, 8192)]
BENCHMARK_ROUNDS = 7
BENCHMARK_WARMUP_MS = 1000
BENCHMARK_REP_MS = 100
BENCHMARK_SCOPE = "public_api_forward"


def _load_fla_reference():
    if test_backend.device != "cuda":
        return None, "FLA AttnRes requires CUDA tensors"
    try:
        return importlib.import_module("fla.ops.attnres.fused").fused_attnres, None
    except (ImportError, AttributeError) as exc:
        return None, str(exc)


def _bench(fn):
    return test_backend.do_bench(
        fn,
        warmup=BENCHMARK_WARMUP_MS,
        rep=BENCHMARK_REP_MS,
        return_mode="median",
    )


@pytest.mark.parametrize("output_norm", [False, True])
@pytest.mark.parametrize(("num_sources", "num_rows"), BENCHMARK_SHAPES)
@torch.inference_mode()
def test_fused_attnres_benchmark(num_sources, num_rows, output_norm, record_property):
    if not test_backend.supports_operator("fused_attnres", tle=True):
        pytest.skip(test_backend.skip_reason("fused_attnres"))

    torch.manual_seed(0)
    hidden_size = 7168
    dtype = torch.bfloat16
    residuals = [
        torch.randn(num_rows, hidden_size, device=test_backend.device, dtype=dtype) for _ in range(num_sources)
    ]
    query = torch.randn(hidden_size, device=test_backend.device, dtype=dtype)
    rms_weight = torch.randn(hidden_size, device=test_backend.device, dtype=dtype)
    output_rms_weight = torch.randn(hidden_size, device=test_backend.device, dtype=dtype) if output_norm else None
    kwargs = {"scale": hidden_size**-0.5, "return_weights": False}

    def run_flag():
        return flag_attn.fused_attnres(query, residuals, rms_weight, output_rms_weight, **kwargs)

    reference = flag_attn.testing.fused_attnres(query, residuals, rms_weight, output_rms_weight, **kwargs)
    actual = run_flag()
    torch.testing.assert_close(actual.float(), reference.float(), atol=5e-3, rtol=1e-2)

    fla_fused_attnres, fla_error = _load_fla_reference()

    def run_fla():
        return fla_fused_attnres(query, residuals, rms_weight, output_rms_weight, **kwargs)

    if fla_fused_attnres is not None:
        baseline_output = run_fla()
        torch.testing.assert_close(actual.float(), baseline_output.float(), atol=5e-3, rtol=1e-2)
    else:
        print(f"\n[baseline] FLA unavailable; measuring FlagAttention only: {fla_error}")

    test_backend.device_fn.synchronize()
    flag_rounds = []
    fla_rounds = []
    for round_index in range(BENCHMARK_ROUNDS):
        if fla_fused_attnres is not None and round_index % 2 == 0:
            fla_rounds.append(_bench(run_fla))
        flag_rounds.append(_bench(run_flag))
        if fla_fused_attnres is not None and round_index % 2 == 1:
            fla_rounds.append(_bench(run_fla))

    flag_ms = statistics.median(flag_rounds)
    fla_ms = statistics.median(fla_rounds) if fla_rounds else None
    speedup = fla_ms / flag_ms if fla_ms is not None else None
    record_benchmark_result(
        record_property,
        op_name="fused_attnres",
        dtype=str(dtype),
        mode="api",
        benchmark_scope=BENCHMARK_SCOPE,
        device=test_backend.device,
        vendor=test_backend.runtime.device.vendor_name,
        baseline="FLA" if fla_ms is not None else None,
        result=[
            benchmark_metric(
                shape_detail={
                    "num_sources": num_sources,
                    "num_rows": num_rows,
                    "hidden_size": hidden_size,
                    "output_norm": output_norm,
                },
                latency=flag_ms,
                latency_base=fla_ms,
                speedup=speedup,
                flag_rounds_ms=flag_rounds,
                fla_rounds_ms=fla_rounds,
            )
        ],
    )
    baseline_text = f"{fla_ms:.6f} ms" if fla_ms is not None else "N/A"
    speedup_text = f"{speedup:.3f}x" if speedup is not None else "N/A"
    print(
        f"\nL{num_sources}_N{num_rows}_D{hidden_size} output_norm={output_norm} "
        f"device={test_backend.device} scope={BENCHMARK_SCOPE}: "
        f"FlagAttention={flag_ms:.6f} ms, FLA={baseline_text}, speedup={speedup_text}",
        flush=True,
    )
