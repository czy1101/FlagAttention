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
import inspect
from pathlib import Path

import torch
import triton

from flag_attn.runtime.backend._metax.sage_attention import forward, per_block_int8


DEFAULT_SHAPES = (
    (1, 1024, 32, 128),
    (4, 1024, 32, 128),
    (1, 4096, 32, 128),
    (1, 8192, 32, 128),
    (1, 16384, 32, 128),
)


def _parse_shape(value):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must use B,T,H,D integers") from exc
    if len(shape) != 4 or any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape must contain four positive integers: B,T,H,D")
    return shape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the MetaX SageAttention FP16/BF16 end-to-end path"
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        dest="shapes",
        help="B,T,H,D; repeat the option to benchmark multiple shapes",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--tensor-layout", choices=("NHD", "HND"), default="NHD")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--maxnreg", type=int)
    return parser.parse_args()


def _build_inputs(shape, dtype, tensor_layout):
    batch, sequence, heads, head_dim = shape
    if tensor_layout == "NHD":
        tensor_shape = (batch, sequence, heads, head_dim)
    else:
        tensor_shape = (batch, heads, sequence, head_dim)
    return tuple(
        torch.randn(tensor_shape, device="cuda", dtype=dtype) for _ in range(3)
    )


def _run_e2e(q, k, v, tensor_layout, maxnreg):
    sequence_dim = 1 if tensor_layout == "NHD" else 2
    k_mean = k.mean(dim=sequence_dim, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q,
        k,
        km=k_mean,
        tensor_layout=tensor_layout,
    )
    output, _ = forward(
        q_int8,
        k_int8,
        v,
        q_scale,
        k_scale,
        tensor_layout=tensor_layout,
        output_dtype=q.dtype,
        return_lse=False,
        maxnreg=maxnreg,
    )
    return output


def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA-compatible MetaX device is required")
    device_name = torch.cuda.get_device_name(0)
    if "metax" not in device_name.lower():
        raise RuntimeError(f"a MetaX device is required, got {device_name}")
    if args.warmup < 0 or args.rep <= 0:
        raise ValueError("warmup must be non-negative and rep must be positive")

    shapes = args.shapes or DEFAULT_SHAPES
    dtype = getattr(torch, args.dtype)
    source_dir = Path(inspect.getsourcefile(forward)).resolve().parent
    if "/runtime/backend/_metax/sage_attention" not in source_dir.as_posix():
        raise RuntimeError(f"unexpected SageAttention route: {source_dir}")

    print("[IDENTITY]")
    print(f"device={device_name}")
    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    print(f"source={source_dir}")
    print(f"dtype={args.dtype} layout={args.tensor_layout}")
    print("scope=K smoothing + Q/K INT8 quantization + attention + output allocations")
    print(f"warmup={args.warmup} rep={args.rep}")
    print("B\tT\tH\tD\tlatency_ms\teffective_tflops\tfinite\tinputs_unchanged")

    for index, shape in enumerate(shapes):
        batch, sequence, heads, head_dim = shape
        torch.manual_seed(20260828 + index)
        q, k, v = _build_inputs(shape, dtype, args.tensor_layout)
        q_before, k_before, v_before = q.clone(), k.clone(), v.clone()

        def run():
            return _run_e2e(q, k, v, args.tensor_layout, args.maxnreg)

        output = run()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(output).all().item())
        inputs_unchanged = bool(
            torch.equal(q, q_before)
            and torch.equal(k, k_before)
            and torch.equal(v, v_before)
        )
        if output.shape != q.shape or output.dtype != dtype or not finite:
            raise RuntimeError(
                f"invalid output for {shape}: shape={output.shape} dtype={output.dtype} finite={finite}"
            )
        if not inputs_unchanged:
            raise RuntimeError(f"input mutation detected for {shape}")

        latency_ms = float(
            triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
        )
        operations = 4 * batch * heads * sequence * sequence * head_dim
        effective_tflops = operations / latency_ms * 1e-9
        print(
            f"{batch}\t{sequence}\t{heads}\t{head_dim}\t{latency_ms:.6f}\t"
            f"{effective_tflops:.4f}\t{finite}\t{inputs_unchanged}"
        )

    print("METAX_SAGE_SELF_E2E_BENCHMARK: PASSED")


if __name__ == "__main__":
    benchmark(parse_args())
