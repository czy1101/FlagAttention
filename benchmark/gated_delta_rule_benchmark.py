import os
from contextlib import contextmanager

import torch
import triton

from flag_attn.gated_delta_rule import chunk_gated_delta_rule_fwd


FULL_TLE_ENV = "FLAG_ATTN_CHUNK_GATED_DELTA_RULE_TLE"
RECOMPUTE_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_RECOMPUTE_TLE"
TWO_KERNEL_TLE_ENV = "FLAG_ATTN_CHUNK_GDR_TWO_KERNEL_TLE"
SHAPES = [
    (2, 16384, 16, 128, 128),
    (4, 2048, 16, 128, 128),
    (4, 4096, 64, 128, 128),
]


@contextmanager
def _set_tle(*, full_tle: bool, recompute_tle: bool, two_kernel_tle: bool):
    old_full = os.environ.get(FULL_TLE_ENV)
    old_recompute = os.environ.get(RECOMPUTE_TLE_ENV)
    old_two_kernel = os.environ.get(TWO_KERNEL_TLE_ENV)
    os.environ[FULL_TLE_ENV] = "1" if full_tle else "0"
    os.environ[RECOMPUTE_TLE_ENV] = "1" if recompute_tle else "0"
    os.environ[TWO_KERNEL_TLE_ENV] = "1" if two_kernel_tle else "0"
    try:
        yield
    finally:
        if old_full is None:
            os.environ.pop(FULL_TLE_ENV, None)
        else:
            os.environ[FULL_TLE_ENV] = old_full
        if old_recompute is None:
            os.environ.pop(RECOMPUTE_TLE_ENV, None)
        else:
            os.environ[RECOMPUTE_TLE_ENV] = old_recompute
        if old_two_kernel is None:
            os.environ.pop(TWO_KERNEL_TLE_ENV, None)
        else:
            os.environ[TWO_KERNEL_TLE_ENV] = old_two_kernel


def _make_inputs(shape: tuple[int, int, int, int, int], dtype: torch.dtype):
    B, T, H, K, V = shape
    device = torch.device("cuda")
    q = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype) / (K**0.5)
    v = torch.randn(B, T, H, V, device=device, dtype=dtype)
    g = (-torch.rand(B, T, H, device=device, dtype=torch.float32) * 0.1).to(dtype)
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    return q, k, v, g, beta, K**-0.5, None, True, None


@torch.inference_mode()
def benchmark() -> None:
    print("dtype,shape,baseline_ms,optimized_ms,speedup")
    for dtype in (torch.float16, torch.bfloat16):
        for shape in SHAPES:
            torch.manual_seed(42)
            args = _make_inputs(shape, dtype)
            with _set_tle(
                full_tle=False,
                recompute_tle=False,
                two_kernel_tle=False,
            ):
                baseline_ms = triton.testing.do_bench(
                    lambda: chunk_gated_delta_rule_fwd(*args), warmup=25, rep=100
                )
            with _set_tle(
                full_tle=False,
                recompute_tle=False,
                two_kernel_tle=True,
            ):
                optimized_ms = triton.testing.do_bench(
                    lambda: chunk_gated_delta_rule_fwd(*args), warmup=25, rep=100
                )
            shape_name = f"B{shape[0]}_T{shape[1]}_H{shape[2]}_K{shape[3]}_V{shape[4]}"
            print(
                f"{dtype},{shape_name},{baseline_ms:.6f},{optimized_ms:.6f},"
                f"{baseline_ms / optimized_ms:.3f}x"
            )


if __name__ == "__main__":
    benchmark()
