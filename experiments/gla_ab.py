"""Interleaved GLA FP16 fallback/parallel-state latency comparison."""

import argparse
import importlib
import json
import statistics

import torch
import torch.nn.functional as F
import torch_gcu  # noqa: F401

from flag_attn import chunk_gla
from flag_attn.testing import backend as test_backend

parser = argparse.ArgumentParser()
parser.add_argument("--T", type=int, default=512)
parser.add_argument("--H", type=int, default=4)
args = parser.parse_args()

torch.gcu.set_device(1)
torch.manual_seed(0)
shape = (1, args.T, args.H, 64)
q, k, v = [torch.randn(shape, dtype=torch.float16).to("gcu:1") for _ in range(3)]
g = F.logsigmoid(torch.randn(shape, dtype=torch.float16).to("gcu:1"))
state = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")
original = state.PARALLEL_STATE_ENABLED
samples = {"fallback": [], "parallel_fp16": []}

try:
    for repeat in range(3):
        for enabled, name in ((False, "fallback"), (True, "parallel_fp16")):
            def full():
                state.PARALLEL_STATE_ENABLED = enabled
                return chunk_gla(q, k, v, g, scale=64**-0.5)

            ms = test_backend.do_bench(full, warmup=10, rep=50, return_mode="median")
            samples[name].append(ms)
            print(json.dumps({"repeat": repeat, "path": name, "latency_ms": ms}), flush=True)
finally:
    state.PARALLEL_STATE_ENABLED = original

print(json.dumps({"shape": shape, "dtype": "fp16", "device": "gcu:1",
                  "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                  "samples_ms": samples}), flush=True)
