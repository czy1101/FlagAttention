"""Repeat target Parallax forward/backward after compilation on GCU 1."""

import json
import time

import torch
import torch_gcu  # noqa: F401

from flag_attn import parallel_parallax


torch.gcu.set_device(1)
torch.manual_seed(42)
dtype = torch.bfloat16
B, T, H, HQ, D = 2, 256, 2, 8, 64
q, r = [torch.randn(B, T, HQ, D, dtype=dtype).to("gcu:1").requires_grad_()
        for _ in range(2)]
k, v = [torch.randn(B, T, H, D, dtype=dtype).to("gcu:1").requires_grad_()
        for _ in range(2)]
grad_output = torch.randn(B, T, HQ, D, dtype=dtype).to("gcu:1")
inputs = (q, r, k, v)

for repeat in range(5):
    started = time.perf_counter()
    output = parallel_parallax(*inputs, scale=D**-0.5)
    torch.gcu.synchronize()
    forward = time.perf_counter() - started
    started = time.perf_counter()
    grads = torch.autograd.grad(output, inputs, grad_output)
    torch.gcu.synchronize()
    backward = time.perf_counter() - started
    print(json.dumps({"repeat": repeat, "forward_s": forward,
                      "backward_s": backward,
                      "finite": all(bool(torch.isfinite(g).all()) for g in grads)}),
          flush=True)
