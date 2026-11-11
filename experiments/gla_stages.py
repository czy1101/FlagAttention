"""Exploratory GLA stage timing on Enflame GCU.

Run from the FlagAttention checkout with
PYTHONPATH="$PWD/src:$PYTHONPATH" python experiments/gla_stages.py --device 1.
Each number is a whole Python stage call, including allocation and launch.
"""

import argparse
import importlib
import json
import time

import torch
import torch.nn.functional as F
import torch_gcu  # noqa: F401  # Registers the GCU backend.

from flag_attn import chunk_gla
from flag_attn.testing import backend as test_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.gcu.set_device(args.device)
    device = f"gcu:{args.device}"
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    mod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.gla.chunk_gla")
    B, T, H, D = 1, 512, 4, 64
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D, dtype=dtype).to(device)
    k = torch.randn(B, T, H, D, dtype=dtype).to(device)
    v = torch.randn(B, T, H, D, dtype=dtype).to(device)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=dtype).to(device))
    scale = D**-0.5

    def cumsum():
        return mod.chunk_local_cumsum(g, 64, scale=mod.RCP_LN2)

    gs = cumsum()

    def state():
        return mod.chunk_fwd_h(
            k=k, v=v, g=None, gk=gs, gv=None, h0=None,
            output_final_state=False, cu_seqlens=None, chunk_size=64,
            states_in_fp32=False, state_v_first=False,
        )

    h, _ = state()

    def matrix():
        return mod.chunk_gla_fwd_intra_gk(q, k, gs, scale)

    A = matrix()

    def output():
        return mod.chunk_gla_fwd_o_gk(q, v, gs, A, h, scale)

    def full():
        return chunk_gla(q, k, v, g, scale=scale)

    stages = ("cumsum", cumsum), ("state", state), ("matrix", matrix), ("output", output), ("full", full)
    print(json.dumps({"device": device, "dtype": args.dtype, "shape": [B, T, H, D],
                      "matrix_block": mod._select_fwd_a_block_size(D, 64, dtype),
                      "tle": mod.HAS_TLE}), flush=True)
    for name, fn in stages:
        started = time.perf_counter()
        fn()
        torch.gcu.synchronize()
        first_s = time.perf_counter() - started
        samples = [test_backend.do_bench(fn, warmup=10, rep=50, return_mode="median")
                   for _ in range(args.repeats)]
        print(json.dumps({"stage": name, "first_call_s": first_s, "samples_ms": samples}), flush=True)


if __name__ == "__main__":
    main()
