# msa — S60 benchmark report

Date: 2026-11-16
Branch: `feat/enflame-msa` · optimized commit tested: `2f3abd10347e465548ce22897f7cd0c979020ee3`
Benchmark: `benchmark/test_minimax_sparse_attention_benchmark.py`

## Environment and method

- **Device**: Enflame S60, physical card **7** (`TOPS_VISIBLE_DEVICES=7`);
  every run re-checked `efsmi -pmon` first. A container process started with
  `TOPS_VISIBLE_DEVICES=7` was observed by host `efsmi` on device 7
  (container sees `device_count()==1`).
- **Image**: `harbor.baai.ac.cn/flagos-app/vllm0.24.0-enflame-tops1.9.10:2.3.0-0.3.0`
- **Stack**: Python 3.12.13, torch 2.10.0+cpu, torch_gcu 2.10.0+3.7.20260408,
  triton 3.6.0 (FlagTree, `/opt/flagtree`), flagtree 0.7.0+enflame3.6,
  triton-gcu 3.6.0+1.0.20260521.cc.1.9.10, efsmi/efml/topsruntime 1.9.10,
  topscc 1.9.10.
- **Baseline commit**: `3912bd75af73ce258267ecac3c508d918f63ee1f` (origin/main,
  unchanged during this round).
- **Method**: each version ran in its own worktree from its own `src`; the
  benchmark module was imported from `<worktree>/benchmark` so the driver code
  belongs to that commit. Same card, same seed, same shapes; the two versions
  were run back-to-back (serial, never concurrent).
- **Timing source**: the benchmark's own `test_backend.do_bench` (`warmup`/`rep`
  are **millisecond budgets**, not call counts), reported as the median of the
  event samples. No hand-written timing replaced the repo benchmark.


## Workload and benchmark contract

- MSA prefill: `B=1, Q=1024, KV=1024, KVH=2, QH=16, D=128`, BF16, `topk=4`,
  `init/local = 1/2`, random page table, seed 42, `no_vllm=True`.
- Inputs: FP16 Q/K/V source → INT8 Q/K + FP32 scales per block; index K cache is
  a separate paged tensor. Output BF16.
- Timing scope: `FlagAttn` = score → topk → sparse attention public chain,
  `warmup=25ms`, `rep=100ms`; `Score`/`TopK`/`Attn` columns time the three
  stages separately with the same budgets.
- Reference: vLLM prints `unavailable` in this container, so no vLLM number is
  claimed.


## Original vs optimized (repository benchmark)

| column | baseline (ms) | optimized (ms) | change |
|---|---|---|---|
| FlagAttn (full chain) | 2.3250 | **1.6861** | 1.379×, −27.5% |
| Index Score | 0.1512 | 0.1539 | flat |
| TopK | 1.0057 | **0.8609** | −14.4% |
| Sparse Attention | 0.7638 | 0.7676 | flat |

Public-call speedup: **1.379×** (reduction 27.5%).


## Reference provider

See the workload section: the benchmark's reference provider is
vLLM MSA (unavailable here). When it is unavailable or noisy this report keeps the raw log and does
not claim a reference latency.


## Preprocessing, hotspot kernels and full chain

- The earlier optimization report recorded a bimodal public-call distribution
  for the baseline at this shape (≈1.82 ms and ≈2.33 ms). This benchmark run
  landed in the slow mode (2.3250 ms); the raw per-sample values are in the log
  and the cause is still **not attributed**.
- Preprocessing: score/topk outputs are allocated per call (inside the chain);
  hot kernels are `_prefill_topk_reduction_enflame`/`_prefill_topk_rows_enflame`
  and `_gqa_sparse_fwd_grouped8`.
- Not measured: other default prefill shapes (T up to 8192, D up to 128),
  decode, packed/varlen.
- Logs: `bench_logs/msa_{base,opt}.log`.


## Optimization summary (what changed and why)

- Commit `262377f` (`perf(enflame): batch dense MSA TopK rows per program`) plus
  the dispatch helper from `2f3abd1`, `_enflame/msa/index_topk.py`:
  for dense prefill with `0 < topk <= 4`, `max_blocks <= 512`, `batch <= 255`
  and `heads <= 255`, the reduction TopK reuses the repository's existing
  multi-row kernel `_prefill_topk_rows_enflame` with `BLOCK_SIZE_Q = 8` instead
  of launching one program per query row.
- Why it can help: the reduction path is a tiny per-row scan (16 fp32 scores,
  four max/tie-break rounds). One program per row produced 2048 single-warp
  programs for the target shape; batching 8 rows per program cuts that to 256
  programs and amortises the per-program cost. It is a work-decomposition
  change, not a new selection algorithm, and the selection/tie/`-1` padding
  semantics are unchanged (verified with exact-equality replay on 10 boundary
  cases, including ties, NaN, all `-inf`, tail rows, `blocks == topk`,
  non-contiguous strides and `topk` 1..3).
- Measured effect: TopK kernel 1.0057 → 0.8609 ms in this benchmark (−14.4%);
  the earlier in-context profiler showed 948 → 810 µs for the kernel and
  1780 → 1642 µs for the three-kernel span. The public-call gain here (27.5%)
  is larger than the device-time gain; the extra part is **not attributed**
  (baseline public-call samples were bimodal at this shape).
- Scope: `topk > 4` keeps the existing streaming path (already multi-row),
  `max_blocks > 512` and oversized grids keep the one-row fallback; packed
  varlen bypasses the autotuner by design.

## Correctness and caveats

- Only the benchmark's own provider checks apply; no additional numerical gate
  was introduced in this round. Any pass/fail text is taken verbatim from the
  logs listed above.
- Timing values are medians of the benchmark's event samples; `warmup`/`rep`
  are millisecond budgets. No profiler kernel time is mixed into these numbers.
- Historical numbers from other cards/commits are not merged into these tables.


## Reproduction

```bash
# container, physical card 7, one version at a time
export TOPS_VISIBLE_DEVICES=7
export PYTHONPATH=/workspace/hjq/<worktree>/src:/opt/flagtree
python /workspace/hjq/bench_work/run_bench.py \
  --root /workspace/hjq/<worktree> --op msa --tag <tag>
```
