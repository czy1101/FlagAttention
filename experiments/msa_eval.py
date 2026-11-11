"""MSA correctness and representative shape measurements on GCU 1."""

import argparse

import torch
import torch_gcu  # noqa: F401


def bridge_cpu_generators() -> None:
    # The benchmark supplies a CPU Generator to GCU random constructors.
    # Only input construction is changed; operator calls remain untouched.
    for name in ("randn", "rand", "randperm", "randint"):
        original = getattr(torch, name)

        def wrapped(*args, _original=original, **kwargs):
            generator = kwargs.get("generator")
            device = kwargs.get("device")
            if (generator is not None and device is not None
                    and str(generator.device) == "cpu" and str(device).startswith("gcu")):
                kwargs["device"] = "cpu"
                return _original(*args, **kwargs).to(device)
            return _original(*args, **kwargs)

        setattr(torch, name, wrapped)


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("tests", "prefill", "decode"))
args = parser.parse_args()
torch.gcu.set_device(1)
bridge_cpu_generators()

if args.mode == "tests":
    import pytest

    raise SystemExit(pytest.main([
        "-q", "--disable-warnings", "tests/flag_attn/test_minimax_sparse_attention.py",
        "-k", "prefill_bf16 and (selection or long_ragged_gqa8) or decode_bf16 and (selection or long_gqa)",
    ]))

from benchmark import test_minimax_sparse_attention_benchmark as bench

bench.PREFILL_SHAPES = [(1, 512, 4096, 1, 8)]
bench.DECODE_SHAPES = [(1, 1, 16384, 1, 8)]
config = bench.MSABenchmarkArgs(
    dtype="bf16", stage=args.mode, warmup=10, rep=50,
    per_step=True, no_vllm=True,
)
for repeat in range(3):
    print(f"REPEAT {repeat} STAGE {args.mode}", flush=True)
    bench.run_benchmark(config)
