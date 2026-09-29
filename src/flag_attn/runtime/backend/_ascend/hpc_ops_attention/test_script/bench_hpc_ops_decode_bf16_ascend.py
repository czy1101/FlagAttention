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

"""Ascend kernel-only benchmark for the HY3 BF16 MTP1 decode workloads.

The CUDA benchmark in this directory covers the Hopper path; this script covers
the Ascend backend with the same nine workloads and the same measurement
口径 (fresh NPUGraph capture per repeat, 100 warmup / 300 event samples /
5 repeats, median of the per-repeat medians).

    source /workspace/env.sh
    export ASCEND_VISIBLE_DEVICES=7 ASCEND_RT_VISIBLE_DEVICES=7
    uv run --no-sync python benchmark/bench_hpc_ops_decode_bf16_ascend.py [options] [case ...]

Options:
    --heads HxKV   official GQA split, either 8x1 (default) or 32x4
    --check        also compare against an FP32 reference on a subset
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.dynamic.bf16_dynamic import (  # noqa: E402
    DynamicBF16Inputs,
    attention_decode_bf16_dynamic,
    prepare_dynamic_bf16_workspace,
)
from flag_attn.runtime.backend._ascend.hpc_ops_attention.decode.static.bf16_static import (  # noqa: E402
    StaticBF16Inputs,
    attention_decode_bf16_tle,
    prepare_static_bf16_workspace,
)

BLOCK_SIZE = 64
HEAD_DIM = 128
WARMUP = 100
ITERS = 300
REPEAT = 5
CHECK_CASES = ("uniform_512", "skewed_extreme")

OFFICIAL_CASES = {
    "uniform_512": (512,) * 64,
    "uniform_4096": (4096,) * 64,
    "skewed_mix": (128,) * 32 + (4096,) * 32,
    "skewed_extreme": (64,) * 15 + (16 * 1024,),
    "one_64k_7x4k": (64 * 1024,) + (4096,) * 7,
    "one_64k_15x4k": (64 * 1024,) + (4096,) * 15,
    "one_64k_31x4k": (64 * 1024,) + (4096,) * 31,
    "one_128k_31x4k": (128 * 1024,) + (4096,) * 31,
    "two_32k_30x4k": (32 * 1024,) * 2 + (4096,) * 30,
}


def make_inputs(lengths, layout, hq, hkv, mtp=1):
    """Same random-fill order as the CUDA benchmark's ``make_inputs``."""
    torch.manual_seed(41)
    torch.npu.manual_seed(41)
    kv_lens = torch.tensor(lengths, device="npu", dtype=torch.int32)
    block_counts = (kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_blocks = int(block_counts.sum().item())
    capacity = int(total_blocks * 1.2) + len(lengths) + 8
    q = torch.randn(
        len(lengths) * mtp, hq, HEAD_DIM,
        device="npu", dtype=torch.bfloat16,
    )
    q = q / math.sqrt(HEAD_DIM)
    k = torch.randn(
        capacity, BLOCK_SIZE, hkv, HEAD_DIM, device="npu", dtype=torch.bfloat16
    ) / math.sqrt(HEAD_DIM)
    v = torch.randn(capacity, BLOCK_SIZE, hkv, HEAD_DIM, device="npu", dtype=torch.bfloat16)
    packed = torch.randperm(capacity, device="npu")[:total_blocks].to(torch.int32)
    block_ids = torch.empty(
        len(lengths), int(block_counts.max().item()), device="npu", dtype=torch.int32
    )
    cursor = 0
    for batch, count in enumerate(block_counts.cpu().tolist()):
        block_ids[batch, :count] = packed[cursor:cursor + count]
        cursor += count
    k_nhd, v_nhd = k, v
    if layout == "HND":
        # BnNBsD strides; the reference always uses the NHD copy.
        k = k.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
    return q, k, v, block_ids, kv_lens, k_nhd, v_nhd


def reference(q, k, v, block_ids, kv_lens, hq, hkv):
    out = torch.empty_like(q)
    mtp = q.shape[0] // kv_lens.numel()
    for batch in range(kv_lens.numel()):
        length = int(kv_lens[batch])
        pages = (length + BLOCK_SIZE - 1) // BLOCK_SIZE
        ids = block_ids[batch, :pages]
        kb = k[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        vb = v[ids].reshape(-1, hkv, HEAD_DIM)[:length]
        kb = kb.transpose(0, 1).repeat_interleave(hq // hkv, 0).float()
        vb = vb.transpose(0, 1).repeat_interleave(hq // hkv, 0).float()
        qb = q[batch * mtp:(batch + 1) * mtp].transpose(0, 1).float()
        scores = qb @ kb.transpose(-1, -2)
        history = length - mtp
        causal = torch.cat(
            (
                torch.ones((mtp, history), dtype=torch.bool, device=q.device),
                torch.tril(torch.ones((mtp, mtp), dtype=torch.bool, device=q.device)),
            ),
            dim=-1,
        )
        scores = scores.masked_fill(~causal[None], -float("inf"))
        result = F.softmax(scores / math.sqrt(HEAD_DIM), -1) @ vb
        out[batch * mtp:(batch + 1) * mtp] = result.transpose(0, 1)
    return out


def bench_ms(call):
    for _ in range(WARMUP):
        call()
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            call()
    torch.npu.current_stream().wait_stream(stream)
    for _ in range(WARMUP):
        graph.replay()
    torch.npu.synchronize()
    events = [
        (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
        for _ in range(ITERS)
    ]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.npu.synchronize()
    samples = sorted(start.elapsed_time(end) for start, end in events)
    return samples[len(samples) // 2] * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", default=None)
    parser.add_argument("--heads", default="8x1", help="Hq x Hkv, e.g. 8x1 or 32x4")
    parser.add_argument("--mtp", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--schedule", choices=("static", "dynamic"), default="dynamic")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    hq, hkv = (int(x) for x in args.heads.lower().split("x"))
    selected = args.cases or list(OFFICIAL_CASES)
    print(f"device={torch.npu.get_device_name(0)} heads={hq}q/{hkv}kv mtp={args.mtp} "
          f"schedule={args.schedule} "
          f"warmup={WARMUP} iters={ITERS} repeat={REPEAT}")
    if args.schedule == "static":
        inputs_type = StaticBF16Inputs
        prepare_workspace = prepare_static_bf16_workspace
        attention_decode = attention_decode_bf16_tle
    else:
        inputs_type = DynamicBF16Inputs
        prepare_workspace = prepare_dynamic_bf16_workspace
        attention_decode = attention_decode_bf16_dynamic
    for case in selected:
        lengths = OFFICIAL_CASES[case]
        for layout in ("NHD", "HND"):
            q, k, v, block_ids, kv_lens, k_ref, v_ref = make_inputs(
                lengths, layout, hq, hkv, args.mtp
            )
            inputs = inputs_type(q, k, v, block_ids, kv_lens, layout)
            workspace = prepare_workspace(inputs)
            call = lambda: attention_decode(inputs, workspace)
            err = None
            if args.check and case in CHECK_CASES:
                got = call().detach().clone()
                torch.npu.synchronize()
                err = (
                    got.float()
                    - reference(q, k_ref, v_ref, block_ids, kv_lens, hq, hkv).float()
                ).abs().max().item()
            values = [bench_ms(call) for _ in range(REPEAT)]
            extra = f" err={err:.2e}" if err is not None else ""
            print(
                f"RESULT {case} {layout} median_us={statistics.median(values):.2f} "
                f"min_us={min(values):.2f} max_us={max(values):.2f} "
                f"splits={workspace.max_splits} compact={int(workspace.compact_producer)} "
                f"repeats={','.join(f'{x:.2f}' for x in values)}{extra}",
                flush=True,
            )
            del inputs, workspace, q, k, v, block_ids, kv_lens, k_ref, v_ref
            torch.npu.empty_cache()


if __name__ == "__main__":
    main()
