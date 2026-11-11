"""Compare GLA FP16 parallel-state candidate with the existing fallback."""

import argparse
import importlib
import json

import torch
import torch.nn.functional as F
import torch_gcu  # noqa: F401

from flag_attn import chunk_gla


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not finite.all():
        return {
            "nan_actual": int(torch.isnan(a).sum()),
            "nan_expected": int(torch.isnan(b).sum()),
            "inf_actual": int(torch.isinf(a).sum()),
            "inf_expected": int(torch.isinf(b).sum()),
            "finite_elements": int(finite.sum()),
        }
    diff = a - b
    return {
        "relative_l2": (diff.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-8)).item(),
        "max_abs": diff.abs().max().item(),
        "nan": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=512)
    parser.add_argument("--gate-normalizer", type=float, default=1.0)
    parser.add_argument("--signed-gate", action="store_true")
    parser.add_argument("--grad", action="store_true")
    args = parser.parse_args()
    torch.gcu.set_device(1)
    device = "gcu:1"
    torch.manual_seed(42)
    shape = (1, args.T, 4, 64)
    data = [torch.rand(shape, dtype=torch.float16).to(device) for _ in range(3)]
    gate_data = torch.randn(shape, dtype=torch.float16) if args.signed_gate else torch.rand(shape, dtype=torch.float16)
    data.append(gate_data.to(device))
    do = torch.randn(shape, dtype=torch.float16).to(device)
    state_mod = importlib.import_module("flag_attn.runtime.backend._enflame.FLA.chunk_h")
    original_enabled = state_mod.PARALLEL_STATE_ENABLED

    def run(enabled: bool):
        state_mod.PARALLEL_STATE_ENABLED = enabled
        q, k, v, raw_g = [x.detach().clone().requires_grad_(args.grad) for x in data]
        g = F.logsigmoid(raw_g / args.gate_normalizer)
        o = chunk_gla(q, k, v, g, scale=64**-0.5)
        if isinstance(o, tuple):
            o = o[0]
        torch.gcu.synchronize()
        grads = None
        if args.grad:
            (o.float() * do.float()).sum().backward()
            torch.gcu.synchronize()
            grads = [x.grad for x in (q, k, v, raw_g)]
        return o, grads

    try:
        base_o, base_grads = run(False)
        fast_o, fast_grads = run(True)
    finally:
        state_mod.PARALLEL_STATE_ENABLED = original_enabled

    result = {"T": args.T, "gate_normalizer": args.gate_normalizer,
              "signed_gate": args.signed_gate, "grad": args.grad,
              "output": compare(fast_o, base_o)}
    if args.grad:
        result["grads"] = {name: compare(a, b) for name, a, b in
                           zip(("q", "k", "v", "g"), fast_grads, base_grads)}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
