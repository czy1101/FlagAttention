"""Parallax 2x256 GQA cold compile, correctness, and backward check."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch_gcu  # noqa: F401

from flag_attn import parallel_parallax

sys.path.insert(0, str(Path("tests/flag_attn").resolve()))
from test_parallax import _reference_parallax  # noqa: E402


def diff(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    a, r = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = a - r
    return {
        "relative_rms": (delta.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-8)).item(),
        "max_abs": delta.abs().max().item(),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(r).all()),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
args = parser.parse_args()
torch.gcu.set_device(1)
dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
B, T, H, HQ, D = 2, 256, 2, 8, 64
torch.manual_seed(42)
q, r = [torch.randn(B, T, HQ, D, dtype=dtype).to("gcu:1").requires_grad_() for _ in range(2)]
k, v = [torch.randn(B, T, H, D, dtype=dtype).to("gcu:1").requires_grad_() for _ in range(2)]
inputs = (q, r, k, v)
grad_output = torch.randn(B, T, HQ, D, dtype=dtype).to("gcu:1")
scale = D**-0.5
print(json.dumps({"stage": "forward_start", "dtype": args.dtype}), flush=True)
started = time.perf_counter()
out = parallel_parallax(*inputs, scale=scale)
torch.gcu.synchronize()
forward_s = time.perf_counter() - started
print(json.dumps({"stage": "backward_start", "forward_cold_s": forward_s}), flush=True)
started = time.perf_counter()
grads = torch.autograd.grad(out, inputs, grad_output)
torch.gcu.synchronize()
backward_s = time.perf_counter() - started
print(json.dumps({"stage": "reference_start", "backward_cold_s": backward_s}), flush=True)
reference = _reference_parallax(*inputs, scale=scale, window_size=None)
ref_grads = torch.autograd.grad(reference, inputs, grad_output)
torch.gcu.synchronize()
print(json.dumps({"dtype": args.dtype, "shape": [B, T, H, HQ, D],
                  "forward_cold_s": forward_s, "backward_cold_s": backward_s,
                  "output": diff(out, reference),
                  "grads": {name: diff(actual, expected) for name, actual, expected in
                            zip(("q", "r", "k", "v"), grads, ref_grads)}}), flush=True)
