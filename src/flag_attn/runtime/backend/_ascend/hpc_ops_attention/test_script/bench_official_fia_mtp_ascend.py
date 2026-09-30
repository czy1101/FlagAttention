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

"""Official AscendC FIA baseline for Q_S>1 (MTP2/MTP3-style decode).

    uv run --no-sync python bench_official_fia_mtp.py [mtp ...] [case ...]

MTP semantics: query m of the block attends to KV positions <= kv_len - MTP + m,
which is exactly the right-down-causal mask (sparse_mode=3) over a query block
aligned to the end of the sequence.
"""
from __future__ import annotations
import math, statistics, sys
from pathlib import Path
import torch, torch_npu  # noqa: F401
import torch.nn.functional as F

BLOCK, D_HEAD, HQ, HKV = 64, 128, 8, 1
WARMUP, ITERS, REPEAT = 100, 300, 5
CASES = {
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


def make_inputs(lengths, mtp):
    torch.manual_seed(41)
    torch.npu.manual_seed(41)
    kv = torch.tensor(lengths, device="npu", dtype=torch.int32)
    counts = (kv + BLOCK - 1) // BLOCK
    total = int(counts.sum().item())
    cap = int(total * 1.2) + len(lengths) + 8
    q = torch.randn(len(lengths), mtp, HQ * D_HEAD, device="npu", dtype=torch.bfloat16)
    q = q / math.sqrt(D_HEAD)
    k = torch.randn(cap, BLOCK, HKV * D_HEAD, device="npu", dtype=torch.bfloat16)
    k = k / math.sqrt(D_HEAD)
    v = torch.randn(cap, BLOCK, HKV * D_HEAD, device="npu", dtype=torch.bfloat16)
    packed = torch.randperm(cap, device="npu")[:total].to(torch.int32)
    bt = torch.empty(len(lengths), int(counts.max().item()), device="npu", dtype=torch.int32)
    cur = 0
    for b, c in enumerate(counts.cpu().tolist()):
        bt[b, :c] = packed[cur:cur + c]
        cur += c
    return q, k, v, bt, kv


def reference(q, k, v, bt, kv_lens, mtp):
    batch = kv_lens.numel()
    out = torch.empty_like(q)
    for b in range(batch):
        length = int(kv_lens[b])
        pages = (length + BLOCK - 1) // BLOCK
        ids = bt[b, :pages]
        kb = k[ids].reshape(-1, HKV, D_HEAD)[:length].transpose(0, 1).float()
        vb = v[ids].reshape(-1, HKV, D_HEAD)[:length].transpose(0, 1).float()
        qb = q[b].reshape(mtp, HQ, D_HEAD).transpose(0, 1).float()
        scores = qb @ kb.transpose(-1, -2) / math.sqrt(D_HEAD)
        history = length - mtp
        causal = torch.cat(
            (torch.ones((mtp, history), dtype=torch.bool, device=q.device),
             torch.tril(torch.ones((mtp, mtp), dtype=torch.bool, device=q.device))),
            dim=-1,
        )
        scores = scores.masked_fill(~causal[None], -float("inf"))
        out[b] = (F.softmax(scores, -1) @ vb).transpose(0, 1).reshape(mtp, HQ * D_HEAD)
    return out


CAUSAL_MASK = None


def causal_mask():
    """Compressed 2048x2048 mask required by sparse_mode=3 (True = masked)."""
    global CAUSAL_MASK
    if CAUSAL_MASK is None:
        tri = torch.tril(torch.ones(2048, 2048, dtype=torch.bool))
        CAUSAL_MASK = (~tri).npu()
    return CAUSAL_MASK


def call_fia(q, k, v, bt, lengths, mtp):
    return torch_npu.npu_fused_infer_attention_score(
        q, k, v,
        atten_mask=causal_mask(),
        block_table=bt,
        num_heads=HQ,
        num_key_value_heads=HKV,
        input_layout="BSH",
        scale=1.0 / math.sqrt(D_HEAD),
        block_size=BLOCK,
        actual_seq_lengths=[mtp] * len(lengths),
        actual_seq_lengths_kv=list(lengths),
        sparse_mode=3,
        pre_tokens=2147483647,
        next_tokens=0,
    )


def bench(call):
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
    events = [(torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)) for _ in range(ITERS)]
    for s, e in events:
        s.record()
        graph.replay()
        e.record()
    torch.npu.synchronize()
    vals = sorted(s.elapsed_time(e) for s, e in events)
    return vals[len(vals) // 2] * 1000.0


mtps = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2, 3]
cases = [a for a in sys.argv[1:] if not a.isdigit()] or list(CASES)
for mtp in mtps:
    for case in cases:
        lengths = CASES[case]
        q, k, v, bt, kv = make_inputs(lengths, mtp)
        try:
            out = call_fia(q, k, v, bt, lengths, mtp)[0].detach().clone()
            torch.npu.synchronize()
            err = (out.float() - reference(q, k, v, bt, kv, mtp).float()).abs().max().item()
            vals = [bench(lambda: call_fia(q, k, v, bt, lengths, mtp)) for _ in range(REPEAT)]
            print(f"RESULT mtp{mtp} {case:16s} fia_median_us={statistics.median(vals):9.2f} "
                  f"err={err:.2e} repeats={','.join(f'{x:.2f}' for x in vals)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            line = next((l.strip() for l in msg.splitlines() if "error" in l.lower() or "Error" in l), msg.splitlines()[0])
            print(f"RESULT mtp{mtp} {case:16s} FAILED: {line[:150]}", flush=True)
        del q, k, v, bt, kv
        torch.npu.empty_cache()
