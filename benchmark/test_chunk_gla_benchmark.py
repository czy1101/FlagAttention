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


import pytest
import torch
import torch.nn.functional as F
import triton

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:  # Direct script execution.
    from recording import BenchmarkRecorder

from flag_attn.testing import backend as test_backend
from flag_attn import chunk_gla as flag_attn_chunk_gla, runtime

DEFAULT_WARMUP = 10
DEFAULT_REP = 20

IS_MTHREADS = runtime.device.vendor_name == "mthreads"
if IS_MTHREADS:
    from tileops.kernels.linear_attention.gla.gla_fwd import GLAFwdKernel
    from tileops.kernels.linear_attention.gla.gla_bwd import GLABwdKernel

    class _TileOpsGLAFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, g, fwd, bwd):
            out, _ = fwd.forward(q, k, v, g)
            ctx.save_for_backward(q, k, v, g, fwd._h_out)
            ctx.bwd = bwd
            return out

        @staticmethod
        def backward(ctx, do):
            q, k, v, g, h = ctx.saved_tensors
            dht = torch.zeros(
                q.shape[0], q.shape[2], q.shape[3], v.shape[3],
                device=q.device, dtype=torch.float32,
            )
            grads = ctx.bwd.forward(q, k, v, g, h, do.contiguous(), dht)
            return *grads, None, None


def _report_tileops_unavailable(shape, phase, exc):
    lines = str(exc).splitlines()
    reason = next(
        (line.strip() for line in lines if "Failed to set the allowed dynamic shared memory size" in line
         or line.lstrip().startswith("fatal error:")),
        next((line.strip() for line in reversed(lines) if line.strip()), type(exc).__name__),
    )
    print(f"[baseline] TileOPs unavailable for {shape}, {phase}: {reason[:240]}")


def _bench_tileops_ms(timer, fn, *args, shape, phase):
    try:
        return timer(fn, *args)
    except Exception as exc:
        # This failure occurs before launch when the driver rejects the
        # official kernel's shared-memory requirement. Other failures propagate.
        if "Failed to set the allowed dynamic shared memory size" not in str(exc):
            raise
        _report_tileops_unavailable(shape, phase, exc)
        return None


def _make_tileops_baseline(B, T, H, D, dtype, scale, backward=False):
    # Compile outside timing; retain FlagAttention cases if the official
    # default kernels cannot be compiled for a shape on this device.
    try:
        fwd = GLAFwdKernel(
            batch=B, seq_len=T, heads=H, dim_k=D, dim_v=D,
            scale=scale, output_final_state=False, dtype=dtype, tune=False,
        )
        bwd = GLABwdKernel(
            batch=B, seq_len=T, heads=H, dim_k=D, dim_v=D,
            scale=scale, dtype=dtype, tune=False,
        ) if backward else None
    except Exception as exc:
        phase = "forward_backward" if backward else "forward"
        _report_tileops_unavailable((B, T, H, D), phase, exc)
        return None

    def call(q, k, v, g, **kwargs):
        if backward:
            return _TileOpsGLAFunction.apply(q, k, v, g, fwd, bwd)
        return fwd.forward(q, k, v, g, initial_state=kwargs.get("initial_state"))

    return call


_fla_chunk_gla = None
if test_backend.is_nvidia():
    try:
        from fla.ops.gla import chunk_gla as _fla_chunk_gla
    except ImportError:
        _fla_chunk_gla = None


def _fla_chunk_wrapper(q, k, v, g, **kwargs):
    return _fla_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        scale=kwargs.get("scale", None),
        initial_state=kwargs.get("initial_state", None),
        output_final_state=kwargs.get("output_final_state", False),
        state_v_first=kwargs.get("state_v_first", False),
        cu_seqlens=kwargs.get("cu_seqlens", None),
    )


def _bench_ms(fn, warmup: int, rep: int) -> float:
    # do_bench performs an untimed first call and warmup before recording events,
    # so JIT/autotune work is excluded from the reported steady-state latency.
    return test_backend.do_bench(
        fn,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


def _build_inputs(B, T, H, D, dtype, requires_grad=False):
    device = torch.device(test_backend.device)
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    g_logit = torch.randn(
        B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )

    kwargs = {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }
    if requires_grad:
        return q, k, v, g_logit, kwargs
    else:
        g = F.logsigmoid(g_logit)
        return q, k, v, g, kwargs


# ---------------------------
# fwd+bwd benchmark helper
# ---------------------------
def _bench_fwd_bwd_ms(fn, q, k, v, g_logit, kwargs, warmup: int, rep: int) -> float:
    """Measure forward + backward pass time for a chunk_gla function.

    ``g_logit`` is the pre-logsigmoid raw tensor (a leaf).  F.logsigmoid is
    called *inside* the timed closure so each backward iteration gets a fresh
    computation graph.
    """

    params = [q, k, v, g_logit]

    def _fwd_bwd():
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
        g = F.logsigmoid(g_logit)
        out = fn(q, k, v, g, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()

    return test_backend.do_bench(
        _fwd_bwd,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


_SHAPES = [
    (1, 4096, 32, 512),
    (2, 2048, 16, 512),
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]

_DTYPES = [
    torch.bfloat16,
    # torch.float32,
    # torch.float16,
]


def _print_header(title: str, subtitle: str, warmup: int, rep: int) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  {subtitle}")
    print(f"  warmup={warmup}ms  rep={rep}ms")
    print(f"{'=' * 70}")

    if IS_MTHREADS:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'tileops(ms)':>10} {'flag_attn(ms)':>14} {'tileops/flag_attn':>16}"
        )
    elif _fla_chunk_gla is not None:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'fla(ms)':>10} {'flag_attn(ms)':>14} {'fla/flag_attn':>14}"
        )
    else:
        print(
            f"{'B':>3} {'T':>6} {'H':>4} {'D':>4} {'dtype':>8} "
            f"{'flag_attn(ms)':>14}"
        )


def _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn) -> None:
    dtype_str = str(dtype).split(".")[-1]
    if IS_MTHREADS:
        baseline = f"{ms_fla:.3f}" if ms_fla is not None else "N/A"
        speedup = (
            f"{ms_fla / ms_flag_attn:.2f}x" if ms_fla is not None else "N/A"
        )
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{baseline:>10} {ms_flag_attn:>14.3f} {speedup:>16}"
        )
    elif _fla_chunk_gla is not None:
        speedup = ms_fla / ms_flag_attn if ms_flag_attn > 0 else float("inf")
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{ms_fla:>10.3f} {ms_flag_attn:>14.3f} {speedup:>14.2f}x"
        )
    else:
        print(
            f"{B:>3} {T:>6} {H:>4} {D:>4} {dtype_str:>8} "
            f"{ms_flag_attn:>14.3f}"
        )

def run_benchmark(
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REP,
    record_property=None,
) -> None:
    if not test_backend.supports_operator("chunk_gla"):
        raise RuntimeError("chunk_gla benchmark requires an available accelerator")

    # ============================================================
    # Part 1: forward only
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD ONLY",
        (
            "provider: TileOPs vs FlagAttention"
            if IS_MTHREADS else
            "provider: FLA vs FlagAttention"
            if _fla_chunk_gla is not None
            else "provider: FlagAttention only (FLA unavailable)"
        ),
        warmup,
        rep,
    )

    for dtype in _DTYPES:
        recorder = BenchmarkRecorder(
            record_property,
            op_name="chunk_gla",
            dtype=str(dtype),
            baseline="TileOPs" if IS_MTHREADS else "FLA" if _fla_chunk_gla is not None else None,
            phase="forward",
        )
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g, kwargs = _build_inputs(B, T, H, D, dtype)

            ms_fla = None
            if IS_MTHREADS:
                baseline = _make_tileops_baseline(B, T, H, D, dtype, kwargs["scale"])
                if baseline is not None:
                    ms_fla = _bench_tileops_ms(
                        _bench_ms, lambda: baseline(q, k, v, g, **kwargs), warmup, rep,
                        shape=(B, T, H, D), phase="forward",
                    )
                del baseline
            elif _fla_chunk_gla is not None:
                ms_fla = _bench_ms(
                    lambda: _fla_chunk_wrapper(q, k, v, g, **kwargs), warmup, rep
                )
            ms_flag_attn = _bench_ms(
                lambda: flag_attn_chunk_gla(q, k, v, g, **kwargs), warmup, rep
            )
            _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn)
            recorder.add(
                shape_detail=(B, T, H, D),
                latency_base=ms_fla,
                latency=ms_flag_attn,
            )
        recorder.record()

    # ============================================================
    # Part 2: forward + backward
    # ============================================================
    _print_header(
        "chunk_gla benchmark — FWD + BWD",
        (
            "provider: TileOPs vs FlagAttention (forward + backward)"
            if IS_MTHREADS else
            "provider: FLA vs FlagAttention (forward + backward)"
            if _fla_chunk_gla is not None
            else "provider: FlagAttention only (forward + backward; FLA unavailable)"
        ),
        warmup,
        rep,
    )

    for dtype in _DTYPES:
        recorder = BenchmarkRecorder(
            record_property,
            op_name="chunk_gla",
            dtype=str(dtype),
            baseline="TileOPs" if IS_MTHREADS else "FLA" if _fla_chunk_gla is not None else None,
            phase="forward_backward",
        )
        print("\ndtype:", dtype)
        for B, T, H, D in _SHAPES:
            q, k, v, g_logit, kwargs = _build_inputs(
                B, T, H, D, dtype, requires_grad=True
            )

            ms_fla = None
            if IS_MTHREADS:
                baseline = _make_tileops_baseline(
                    B, T, H, D, dtype, kwargs["scale"], backward=True
                )
                if baseline is not None:
                    ms_fla = _bench_tileops_ms(
                        _bench_fwd_bwd_ms, baseline, q, k, v, g_logit, kwargs, warmup, rep,
                        shape=(B, T, H, D), phase="forward_backward",
                    )
                del baseline
            elif _fla_chunk_gla is not None:
                ms_fla = _bench_fwd_bwd_ms(
                    _fla_chunk_wrapper,
                    q,
                    k,
                    v,
                    g_logit,
                    kwargs,
                    warmup,
                    rep,
                )
            ms_flag_attn = _bench_fwd_bwd_ms(
                flag_attn_chunk_gla,
                q,
                k,
                v,
                g_logit,
                kwargs,
                warmup,
                rep,
            )
            _print_row(B, T, H, D, dtype, ms_fla, ms_flag_attn)
            recorder.add(
                shape_detail=(B, T, H, D),
                latency_base=ms_fla,
                latency=ms_flag_attn,
            )
        recorder.record()

    print(f"\n{'=' * 70}")
    print("  All done. ")
    print(f"{'=' * 70}\n")


@pytest.mark.chunk_gla
@pytest.mark.skipif(
    not test_backend.supports_operator("chunk_gla"), reason=test_backend.skip_reason("chunk_gla")
)
def test_chunk_gla_benchmark(record_property) -> None:
    run_benchmark(record_property=record_property)
