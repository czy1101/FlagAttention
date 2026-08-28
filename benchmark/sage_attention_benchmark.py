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

import argparse

import torch
import triton

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None


MUSA_AVAILABLE = hasattr(torch, "musa") and torch.musa.is_available()

if MUSA_AVAILABLE:
    from flag_attn.runtime.backend._mthreads.sage_attention import forward
else:
    from flag_attn.sage_attention import forward


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SageAttention QK INT8 / PV FP16 forward")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=(1024, 2048, 4096, 8192, 16384, 32768))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--maxnreg", type=int)
    return parser.parse_args()


def _print_table_header():
    print(
        f"{'B':>4} {'T':>8} {'H':>4} {'D':>4} {'dtype':>10} "
        f"{'latency_ms':>14} {'tflops':>12}"
    )
    print("-" * 72)


def _print_result(result):
    print(
        f"{result['batch_size']:>4d} "
        f"{result['seq_len']:>8d} "
        f"{result['num_heads']:>4d} "
        f"{result['head_dim']:>4d} "
        f"{result['dtype']:>10} "
        f"{result['latency_ms']:>14.4f} "
        f"{result['tflops']:>12.2f}"
    )


def print_results(results):
    _print_table_header()
    for result in results:
        _print_result(result)


def benchmark(args, print_output=True):
    dtype = getattr(torch, args.dtype)
    if not MUSA_AVAILABLE and not torch.cuda.is_available():
        raise RuntimeError("No available MUSA or CUDA accelerator was found")
    device = torch.device("musa" if MUSA_AVAILABLE else "cuda")
    results = []

    if print_output:
        print(f"device={device}")
        _print_table_header()

    for seq_len in args.seq_lens:
        shape = (args.batch_size, args.num_heads, seq_len, args.head_dim)
        q = torch.randint(-100, 100, shape, device=device, dtype=torch.int8)
        # Match the K layout produced by the MUSA per-block quantizer.  K keeps
        # the public/logical [B, H, N, D] shape, while its [B, H, D, N]
        # backing storage makes N contiguous for the logical [D, N] QK
        # operand.  Storage creation happens outside run(), so the benchmark
        # continues to time only the attention core.

        # k = torch.randint(
        #     -100,
        #     100,
        #     (args.batch_size, args.num_heads, seq_len, args.head_dim),
        #     device=device,
        #     dtype=torch.int8,
        # )

        k_storage = torch.randint(
            -100,
            100,
            (
                args.batch_size,
                args.num_heads,
                args.head_dim,
                seq_len,
            ),
            device=device,
            dtype=torch.int8,
        )
        k = k_storage.transpose(-2, -1)

        v = torch.randn(shape, device=device, dtype=torch.float16)
        q_scale = torch.rand(
            (args.batch_size, args.num_heads, triton.cdiv(seq_len, 128)),
            device=device,
            dtype=torch.float32,
        )
        k_scale = torch.rand(
            (args.batch_size, args.num_heads, triton.cdiv(seq_len, 64)),
            device=device,
            dtype=torch.float32,
        )

        def run():
            return forward(
                q,
                k,
                v,
                q_scale,
                k_scale,
                output_dtype=dtype,
                maxnreg=args.maxnreg if device.type == "cuda" else None,
            )

        latency_ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
        flops = 4 * args.batch_size * args.num_heads * seq_len * seq_len * args.head_dim
        tflops = flops / latency_ms * 1e-9
        result = {
            "device": str(device),
            "batch_size": args.batch_size,
            "seq_len": seq_len,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "latency_ms": latency_ms,
            "tflops": tflops,
        }
        results.append(result)
        if print_output:
            _print_result(result)

    return results


if __name__ == "__main__":
    benchmark(parse_args())
