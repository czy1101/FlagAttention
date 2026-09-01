#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fair benchmark: Official Dao-AILab CuTe FlashAttention-4 (STANDARD FA)
            vs  Inkling FA4 Relative Attention (Triton and custom CuTe wrapper).

==============================================================================
FAIRNESS INVARIANTS (why this harness is trustworthy)
==============================================================================
1. Inputs are built ONCE per case, then reused by ALL backends.
   - Our kernel consumes a PAGED KV cache (key_cache/value_cache + block_table).
   All implementations consume the same PAGED KV cache. No unused dense copy
   is constructed; this avoids setup cost and removes a misleading claim from
   the former harness documentation.

2. num_splits is decided by ONE heuristic -- inkling_fa4_num_splits(...) -- and
   the SAME integer is handed to all backends. (This kills the v1=1 / v2=dynamic
   inconsistency that made your old CSVs non-comparable.)

3. Identical, GRAPH-FIRST timing for all backends. CUDA graph is ON by default
   -- it strips the per-launch CPU->GPU overhead so you measure the PURE
   operator time (essential for tiny decode kernels, cleanest for prefill too).
   Event timing is only a fallback when graph capture fails.
   >>> THE OPERATOR RUN TIME (mean/p50/p95/p99) IS THE ONLY METRIC. <<<

==============================================================================
USAGE
==============================================================================
  python benchmark_Cute_Triton.py                    # all backends, all 7 cases
  python bench_fa4_cute_vs_triton.py --quick         # only full_prefill (smoke)
  python bench_fa4_cute_vs_triton.py --no-triton     # official CuTe only
  python bench_fa4_cute_vs_triton.py --no-cute       # our Triton only
  python bench_fa4_cute_vs_triton.py --rel real      # our kernel uses real rel bias
  python bench_fa4_cute_vs_triton.py --no-cuda-graph # turn graph OFF (default ON)
  python bench_fa4_cute_vs_triton.py --out result.csv

Run on an SM90 (H100) box with `inkling_fa4` installed. The function under test
is `inkling_fa4.triton_kernel.inkling_fa4_rel_attention` (clean import, no CuTe
dispatch). The official CuTe backend needs `flash_attn.cute` (torch>=2.6 or a
compat shim); it auto-disables with a [warn] when unavailable.
"""

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (must match your kernel: BLOCK_SIZE is fixed by the paged layout)
# ---------------------------------------------------------------------------
HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Backend 1: official Dao-AILab CuTe FlashAttention (fully implemented)
# ---------------------------------------------------------------------------
HAS_CUTE = False
_cute_rel_attention = None
try:
    from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
        inkling_fa4_rel_attention as _cute_rel_attention,
    )
    HAS_CUTE = True
    print("[info] using vLLM Inkling CuTe backend")
except Exception as e:
    print(
        f"[warn] vLLM Inkling CuTe unavailable ({e}); "
        "official backend disabled.",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Backend 2: custom CuTe SM90 wrapper
# ---------------------------------------------------------------------------
HAS_CUSTOM_CUTE = False
_custom_cute_rel_attention = None
try:
    from inkling_fa4.cute_sm90_backend import (
        inkling_fa4_rel_attention_cute_sm90 as _custom_cute_rel_attention,
        is_cute_sm90_available,
    )
    HAS_CUSTOM_CUTE = bool(is_cute_sm90_available())
    if HAS_CUSTOM_CUTE:
        print("[info] using custom Inkling CuTe SM90 backend")
    else:
        print("[warn] custom CuTe SM90 backend reported unavailable",
              file=sys.stderr)
except Exception as e:
    print(f"[warn] custom CuTe SM90 backend unavailable ({e})",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# Backend 3: OUR Triton kernel (Inkling FA4 Relative Attention)
# The function under test is `inkling_fa4.triton_kernel.inkling_fa4_rel_attention`
# -- a clean import that does NOT pull in vLLM's CuTe dispatch, so it runs even
# when the CuTe compile path is broken on this box. The split heuristic is taken
# from vLLM's Inkling op when importable, else a local fallback is used.
# ---------------------------------------------------------------------------
def _fallback_num_splits(is_local, batch_size, max_query_len, num_heads,
                         num_kv_heads, max_kv_len):
    # Decode-like (query_len == 1): split long KV so the tiny query overlaps
    # with KV compute. Prefill: 1 split. Placeholder used only when the real
    # vLLM heuristic cannot be imported.
    if max_query_len == 1:
        return max(1, (max_kv_len + 1023) // 1024)
    return 1


def _load_inkling():
    """Load OUR Triton kernel + the split heuristic.

    Returns (num_splits_fn, inkling_rel_attention).
    No fake `quack` shim: the real quack (if present) imports naturally; the
    kernel under test does not need it at all.
    """
    import inkling_fa4.triton_kernel as tk
    inkling_rel_attention = tk.inkling_fa4_rel_attention

    num_splits_fn = None
    try:
        from vllm.models.inkling.nvidia.ops.fa4_rel_attention import (
            inkling_fa4_num_splits,
        )
        num_splits_fn = inkling_fa4_num_splits
        print("[info] using vLLM inkling_fa4_num_splits heuristic")
    except Exception as e:
        num_splits_fn = _fallback_num_splits
        print(f"[warn] inkling_fa4_num_splits unavailable ({e}); "
              f"using local fallback heuristic", file=sys.stderr)
    return num_splits_fn, inkling_rel_attention


# ---------------------------------------------------------------------------
# Benchmark case definitions (mirror your existing 7-case suite)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    seq_lens: tuple  # list of (query_len, kv_len) per sequence
    num_heads: int
    num_kv_heads: int
    rel_extent: int
    window_left: int | None = None


def build_cases(quick: bool) -> list:
    cases = [
        BenchmarkCase("full_prefill", ((64, 64),), 4, 4, 128),
        BenchmarkCase("ragged_prefill", ((64, 64), (33, 33), (17, 17)), 8, 2, 128),
        BenchmarkCase("long_prefill", ((512, 512),), 8, 2, 1024),
        BenchmarkCase("chunked_prefill", ((32, 512),), 8, 2, 1024),
        BenchmarkCase("sliding_window", ((64, 512),), 8, 2, 256, window_left=255),
        BenchmarkCase("decode_1k", ((1, 1024),), 8, 2, 1024),
        BenchmarkCase("decode_8k", ((1, 8192),), 8, 2, 1024),
    ]
    return cases[:1] if quick else cases


# ---------------------------------------------------------------------------
# Build inputs ONCE: paged KV (for our kernel) + dense K/V (for official)
# ---------------------------------------------------------------------------
def dense_from_paged(cache, block_table, kv_lens):
    """Reconstruct (total_kv, nkv, head_dim) dense tensor from a paged cache.

    cache:       (num_blocks, BLOCK_SIZE, nkv, head_dim)
    block_table: (num_seq, max_blocks) int32
    kv_lens:     (num_seq,) python ints
    """
    num_seq = block_table.shape[0]
    total_kv = int(sum(kv_lens))
    out = torch.empty(total_kv, cache.shape[2], cache.shape[3],
                      device=cache.device, dtype=cache.dtype)
    pos = 0
    for s in range(num_seq):
        L = int(kv_lens[s])
        for t in range(L):
            b = int(block_table[s, t // BLOCK_SIZE])
            off = t % BLOCK_SIZE
            out[pos] = cache[b, off]
            pos += 1
    return out


def prepare_case(config: BenchmarkCase, rel_mode: str, seed: int,
                 num_splits_fn, forced_num_splits: int | None = None):
    torch.manual_seed(seed)
    device = "cuda"

    q_lens = [ql for ql, _ in config.seq_lens]
    kv_lens = [kl for _, kl in config.seq_lens]
    total_q = sum(q_lens)
    num_seq = len(config.seq_lens)
    scale = 1.0 / HEAD_DIM

    # --- q (varlen, shared by both backends) ---
    q = torch.randn(total_q, config.num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = F.normalize(q.float(), dim=-1).to(DTYPE)

    # --- paged KV cache (for OUR kernel) ---
    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seq * max_blocks + 1
    key_cache = torch.randn(num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM,
                            device=device, dtype=DTYPE)
    key_cache = F.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM,
                              device=device, dtype=DTYPE)

    block_table = torch.zeros(num_seq, max_blocks, dtype=torch.int32, device=device)
    for s in range(num_seq):
        block_table[s] = torch.arange(1 + s * max_blocks,
                                      1 + (s + 1) * max_blocks,
                                      dtype=torch.int32, device=device)

    cum_q = [0]
    for ql in q_lens:
        cum_q.append(cum_q[-1] + ql)
    cu_seqlens_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    # --- rel_logits ---
    if rel_mode == "real":
        rel_logits = torch.randn(total_q, config.num_heads, config.rel_extent,
                                 device=device, dtype=DTYPE)
    else:  # "zeros": mathematically plain attention, but rel path still runs
        rel_logits = torch.zeros(total_q, config.num_heads, config.rel_extent,
                                 device=device, dtype=DTYPE)

    window_size = ((-1, -1) if config.window_left is None
                   else (config.window_left, 0))

    # --- SINGLE num_splits decision, handed to BOTH backends ---
    if forced_num_splits is not None:
        num_splits = forced_num_splits
    elif num_splits_fn is not None:
        
        num_splits = int(num_splits_fn(
            is_local=config.window_left is not None,
            batch_size=num_seq,
            max_query_len=max(q_lens),
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            max_kv_len=max(kv_lens)))
    else:
        num_splits = 1  # fallback; only happens if inkling not importable

    return dict(
        config=config,
        q=q, key_cache=key_cache, value_cache=value_cache,
        block_table=block_table, cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        rel_logits=rel_logits, window_size=window_size, scale=scale,
        num_splits=num_splits,
        total_q=total_q, total_kv=sum(kv_lens),
        max_q=max(q_lens), max_k=max(kv_lens),
        out_official_cute=torch.empty_like(q),
        out_custom_cute=torch.empty_like(q),
        out_triton=torch.empty_like(q),
    )


# ---------------------------------------------------------------------------
# (Correctness is covered by the official dedicated test suite; this harness
#  measures only the operator run time.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Timing: warmup + iters, event-based (CUDA graph optional)
# ---------------------------------------------------------------------------
def time_fn(fn, warmup, iters, use_graph):
    """Return (timing_method, latencies_ms_list).

    The PRIMARY metric is the operator's own run time. CUDA graph (default)
    isolates the pure kernel time by eliminating per-launch overhead; this is
    mandatory for tiny decode kernels and cleanest for prefill too. Falls back
    to event timing if graph capture fails for a backend.
    """
    # First call = shape-specific compile; then warmup.
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            torch.cuda.synchronize()
            for _ in range(warmup):
                g.replay()
            torch.cuda.synchronize()
            latencies = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                start.record()
                g.replay()
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            return "cuda_graph", latencies
        except Exception as e:
            print(f"[warn] CUDA graph timing failed ({e}); using event timing.",
                  file=sys.stderr)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()
    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return "event_e2e", latencies


def summarize_latency(latencies):
    """Core latency statistics -- the PRIMARY output of this harness."""
    ordered = sorted(latencies)
    n = len(ordered)

    def pct(p):
        pos = p * (n - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        w = pos - lo
        return ordered[lo] * (1 - w) + ordered[hi] * w

    return dict(
        mean_ms=statistics.fmean(latencies),
        p50_ms=pct(0.50),
        p95_ms=pct(0.95),
        p99_ms=pct(0.99),
    )


# ---------------------------------------------------------------------------
# Backend runners
# ---------------------------------------------------------------------------
def run_cute(prep):
    out = prep["out_official_cute"]

    result = _cute_rel_attention(
        prep["q"],
        prep["key_cache"],
        prep["value_cache"],
        block_table=prep["block_table"],
        cache_seqlens=prep["cache_seqlens"],
        cu_seqlens_q=prep["cu_seqlens_q"],
        max_seqlen_q=prep["max_q"],
        softmax_scale=prep["scale"],
        causal=True,
        window_size=prep["window_size"],
        rel_extent=prep["config"].rel_extent,
        rel_logits=prep["rel_logits"],
        num_splits=prep["num_splits"],
        out=out,
    )
    return result if result is not None else out


def run_custom_cute(prep):
    out = prep["out_custom_cute"]
    result = _custom_cute_rel_attention(
        prep["q"],
        prep["key_cache"],
        prep["value_cache"],
        block_table=prep["block_table"],
        cache_seqlens=prep["cache_seqlens"],
        cu_seqlens_q=prep["cu_seqlens_q"],
        max_seqlen_q=prep["max_q"],
        softmax_scale=prep["scale"],
        causal=True,
        window_size=prep["window_size"],
        rel_extent=prep["config"].rel_extent,
        rel_logits=prep["rel_logits"],
        num_splits=prep["num_splits"],
        out=out,
    )
    return result if result is not None else out


def run_inkling(prep, inkling_rel_attention):
    import inspect
    out = prep["out_triton"]
    kwargs = dict(
        q=prep["q"],
        key_cache=prep["key_cache"],
        value_cache=prep["value_cache"],
        block_table=prep["block_table"],
        cache_seqlens=prep["cache_seqlens"],
        cu_seqlens_q=prep["cu_seqlens_q"],
        max_seqlen_q=prep["max_q"],
        softmax_scale=prep["scale"],
        causal=True,
        window_size=prep["window_size"],
        rel_extent=prep["config"].rel_extent,
        rel_logits=prep["rel_logits"],
        num_splits=prep["num_splits"],
        out=out,
    )
    # Keep only the parameters the function actually accepts, so a small
    # signature difference (param naming, optional out=) does not crash.
    accepted = {k: v for k, v in kwargs.items()
                if k in inspect.signature(inkling_rel_attention).parameters}
    result = inkling_rel_attention(**accepted)
    return result if result is not None else out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only full_prefill")
    ap.add_argument(
        "--case",
        choices=(
            "full_prefill",
            "ragged_prefill",
            "long_prefill",
            "chunked_prefill",
            "sliding_window",
            "decode_1k",
            "decode_8k",
        ),
        default=None,
        help="Run only one benchmark case; useful for Nsight profiling.",
    )
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--no-cute", action="store_true")
    ap.add_argument("--no-custom-cute", action="store_true")
    ap.add_argument("--rel", choices=["zeros", "real"], default="zeros",
                    help="zeros=plain attention (rel path on, zero bias, DEFAULT); "
                         "real=our kernel uses real rel_logits")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cuda-graph", action="store_true",
                    help="Disable CUDA graph (default ON). Graph gives the pure "
                         "operator time by removing launch overhead; turn OFF only "
                         "if a backend cannot be captured.")
    ap.add_argument("--out", type=Path, default=Path("fa4_bench_cute_vs_triton.csv"))
    ap.add_argument(
        "--triton-split-sweep", action="store_true",
        help="Additionally benchmark explicit Triton splits 1,2,4,8,16,32. "
             "The ordinary rows remain strict same-num_splits comparisons.",
    )
    ap.add_argument(
        "--num-splits", type=int, default=None,
        help="Force one split count for both backends. Useful for the official "
             "Hopper split=1 baseline; omitted means use the imported heuristic.",
    )
    args = ap.parse_args()
    if args.num_splits is not None and args.num_splits < 1:
        ap.error("--num-splits must be positive")

    assert torch.cuda.is_available(), "CUDA unavailable"
    cap = torch.cuda.get_device_capability(0)
    assert cap >= (9, 0), f"FA4 needs SM90+, got {cap}"

    # which backends
    inkling_fn = None
    run_inkling_fn = None
    backends = []
    if not args.no_cute and HAS_CUTE:
        backends.append("official_cute")
    if not args.no_custom_cute and HAS_CUSTOM_CUTE:
        backends.append("custom_cute_sm90")
    if not args.no_triton:
        try:
            num_splits_fn, inkling_call = _load_inkling()
            inkling_fn = num_splits_fn
            run_inkling_fn = lambda prep: run_inkling(prep, inkling_call)
            backends.append("triton")
        except Exception as e:
            print(f"[warn] could not load our Triton kernel ({e}); "
                  f"triton backend disabled.", file=sys.stderr)

    if not backends:
        print("No backends available. Exit.", file=sys.stderr)
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}  cap={cap}")
    print(f"rel_mode={args.rel}  backends={backends}  "
          f"cuda_graph={not args.no_cuda_graph}")

    cases = build_cases(False if args.case is not None else args.quick)
    if args.case is not None:
        cases = [case for case in cases if case.name == args.case]
    if not cases:
        raise RuntimeError(f"Unknown benchmark case: {args.case!r}")

    rows = []
    for ci, config in enumerate(cases):
        prep = prepare_case(
            config, args.rel, args.seed + ci, inkling_fn, args.num_splits
        )

        use_graph = not args.no_cuda_graph

        print(f"\n=== {config.name}  num_splits={prep['num_splits']} "
              f"window={prep['window_size']} rel={args.rel} ===")

        for backend in backends:
            row = dict(case=config.name, backend=backend,
                       num_splits=prep["num_splits"],
                       window=str(prep["window_size"]), rel=args.rel,
                       total_q=prep["total_q"], total_kv=prep["total_kv"],
                       heads=config.num_heads, kv_heads=config.num_kv_heads,
                       mean_ms="", p50_ms="", p95_ms="", p99_ms="",
                       timing="", status="ok")
            try:
                if backend == "official_cute":
                    fn = lambda: run_cute(prep)
                elif backend == "custom_cute_sm90":
                    fn = lambda: run_custom_cute(prep)
                else:
                    fn = lambda: run_inkling_fn(prep)

                # --- THE ONLY METRIC: operator run time ---
                method, lats = time_fn(fn, args.warmup, args.iters, use_graph)
                stats = summarize_latency(lats)

                row.update(
                    mean_ms=f"{stats['mean_ms']:.4f}",
                    p50_ms=f"{stats['p50_ms']:.4f}",
                    p95_ms=f"{stats['p95_ms']:.4f}",
                    p99_ms=f"{stats['p99_ms']:.4f}",
                    timing=method,
                )
                print(f"  {backend:7s}  mean={stats['mean_ms']:.4f} ms  "
                      f"p50={stats['p50_ms']:.4f}  p95={stats['p95_ms']:.4f}  "
                      f"p99={stats['p99_ms']:.4f}  [{method}]")
            except Exception as e:
                row.update(status=f"ERROR: {type(e).__name__}: {e}")
                print(f"  {backend:7s} ERROR: {e}")
            rows.append(row)

        # Strict same-configuration rows answer implementation efficiency;
        # this optional sweep answers the separate deployment question: which
        # scheduling choice is fastest for the Triton backend on this H100?
        if args.triton_split_sweep and run_inkling_fn is not None:
            original_splits = prep["num_splits"]
            for split in (1, 2, 4, 8, 16, 32, 64, 128):
                prep["num_splits"] = split
                try:
                    method, lats = time_fn(
                        lambda: run_inkling_fn(prep),
                        args.warmup, args.iters, use_graph,
                    )
                    stats = summarize_latency(lats)
                    print(f"  triton split={split:2d} mean={stats['mean_ms']:.4f} ms")
                    rows.append(dict(
                        case=config.name, backend="triton_sweep",
                        num_splits=split, window=str(prep["window_size"]),
                        rel=args.rel, total_q=prep["total_q"],
                        total_kv=prep["total_kv"], heads=config.num_heads,
                        kv_heads=config.num_kv_heads,
                        mean_ms=f"{stats['mean_ms']:.4f}",
                        p50_ms=f"{stats['p50_ms']:.4f}",
                        p95_ms=f"{stats['p95_ms']:.4f}",
                        p99_ms=f"{stats['p99_ms']:.4f}", timing=method,
                        status="ok",
                    ))
                except Exception as e:
                    print(f"  triton split={split:2d} ERROR: {e}")
            prep["num_splits"] = original_splits

    cols = ["backend", "case", "num_splits", "window", "rel",
            "total_q", "total_kv", "heads", "kv_heads",
            "mean_ms", "p50_ms", "p95_ms", "p99_ms",
            "timing", "status"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
