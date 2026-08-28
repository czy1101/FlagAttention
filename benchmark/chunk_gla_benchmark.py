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

import torch
import torch.nn.functional as F
import triton

from flag_attn.gated_linear_attention import chunk_gla as flag_attn_chunk_gla


# optional FLA reference
_HAS_FLA_CHUNK = False
_fla_chunk_gla = None

try:
    from fla.ops.gla import chunk_gla as _fla_chunk_gla

    _HAS_FLA_CHUNK = True
except Exception:
    _HAS_FLA_CHUNK = False


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


def _make_kwargs(D: int) -> dict:
    return {
        "scale": D**-0.5,
        "initial_state": None,
        "output_final_state": False,
        "state_v_first": False,
        "cu_seqlens": None,
        "cu_seqlens_cpu": None,
    }


def _build_inputs(B, T, H, D, dtype, requires_grad=False):
    device = torch.device("cuda")
    q = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad)
    g_logit = torch.randn(
        B, T, H, D, device=device, dtype=dtype, requires_grad=requires_grad
    )

    if requires_grad:
        return q, k, v, g_logit
    else:
        g = F.logsigmoid(g_logit)
        return q, k, v, g


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

    return triton.testing.do_bench(
        _fwd_bwd,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )


DEFAULT_WARMUP = 100
DEFAULT_REP = 200

_SHAPES = [
    # (B, T, H, D)
    # (1, 4096, 32, 512),
    # (2, 2048, 16, 512),
    (1, 8192, 96, 128),
    (2, 16384, 16, 128),
    (4, 2048, 16, 128),
    (4, 4096, 64, 128),
    (8, 2048, 32, 256),
    (2, 2048, 16, 512),
    (4, 1024, 8, 512),
    (8, 1024, 8, 64),
]

_T_DTYPES = [
    torch.bfloat16,
    # torch.float32,
    # torch.float16,
]

configs = [
    triton.testing.Benchmark(
        x_names=["B", "T", "H", "D"],
        x_vals=_SHAPES,
        line_arg="provider",
        line_vals=["flag_attn"] + (["fla"] if _HAS_FLA_CHUNK else []),
        line_names=["flag_attn"] + (["fla"] if _HAS_FLA_CHUNK else []),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"chunk_gla-mode-{mode}-dtype-{dtype}",
        args={"mode": mode, "dtype": dtype},
    )
    for mode in ["fwd", "bwd"]
    for dtype in _T_DTYPES
]


@triton.testing.perf_report(configs)
def bench_chunk_gla(B, T, H, D, mode, provider, dtype=torch.bfloat16, device="cuda"):
    assert mode in ["fwd", "bwd"]

    is_bwd = mode == "bwd"
    kwargs = _make_kwargs(D)

    if provider == "flag_attn":
        fn = flag_attn_chunk_gla
    elif provider == "fla":
        fn = _fla_chunk_wrapper
    else:
        raise ValueError(f"unknown provider: {provider}")

    if is_bwd:
        q, k, v, g_logit = _build_inputs(B, T, H, D, dtype, requires_grad=True)
        ms = _bench_fwd_bwd_ms(fn, q, k, v, g_logit, kwargs, DEFAULT_WARMUP, DEFAULT_REP)
    else:
        q, k, v, g = _build_inputs(B, T, H, D, dtype, requires_grad=False)
        ms = triton.testing.do_bench(
            lambda: fn(q, k, v, g, **kwargs), warmup=DEFAULT_WARMUP, rep=DEFAULT_REP
        )

    # Return raw latency in ms, matching the original benchmark semantics.
    return ms


# only works on post-Ampere GPUs right now
bench_chunk_gla.run(print_data=True)
