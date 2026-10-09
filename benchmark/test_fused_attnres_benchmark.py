"""Pytest entry for the existing fused AttnRes kernel benchmark."""

import pytest

import test_attnres as _attnres


@pytest.mark.fused_attnres
@pytest.mark.skipif(
    not _attnres._tle_available(), reason="AttnRes external benchmark requires CUDA/TLE",
)
@pytest.mark.skipif(
    _attnres.os.environ.get("FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS", "0") != "1",
    reason="set FLAG_ATTN_RUN_EXTERNAL_BENCHMARKS=1 to run AttnRes benchmarks",
)
@pytest.mark.parametrize("output_norm", [False, True])
@pytest.mark.parametrize(
    ("num_sources", "num_rows"), _attnres.ATTNRES_EXTERNAL_BENCHMARK_SHAPES,
)
def test_fused_attnres_benchmark(num_sources, num_rows, output_norm, record_property):
    _attnres.test_fused_attnres_kernel_benchmark(
        num_sources, num_rows, output_norm, record_property,
    )
