"""Interleaved GLA FP16 forward plus backward latency comparison."""

import importlib
import json
import statistics

import torch
import torch.nn.functional as F
import torch_gcu  # noqa: F401

from flag_attn import chunk_gla
from flag_attn.testing import backend as test_backend

torch.gcu.set_device(1)
torch.manual_seed(0)
shape = (1, 512, 4, 64)
q, k, v, raw_g = [torch.randn(shape, dtype=torch.float16).to("gcu:1").requires_grad_()
                  for _ in range(4)]
state = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")
original = state.PARALLEL_STATE_ENABLED
samples = {"fallback": [], "parallel_fp16": []}

try:
    for repeat in range(3):
        for enabled, name in ((False, "fallback"), (True, "parallel_fp16")):
            def fwd_bwd():
                state.PARALLEL_STATE_ENABLED = enabled
                for tensor in (q, k, v, raw_g):
                    tensor.grad = None
                g = F.logsigmoid(raw_g)
                out = chunk_gla(q, k, v, g, scale=64**-0.5)
                if isinstance(out, tuple):
                    out = out[0]
                out.sum().backward()

            ms = test_backend.do_bench(fwd_bwd, warmup=10, rep=200, return_mode="median")
            samples[name].append(ms)
            print(json.dumps({"repeat": repeat, "path": name, "latency_ms": ms}), flush=True)
finally:
    state.PARALLEL_STATE_ENABLED = original

print(json.dumps({"shape": shape, "dtype": "fp16", "device": "gcu:1",
                  "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                  "samples_ms": samples}), flush=True)
