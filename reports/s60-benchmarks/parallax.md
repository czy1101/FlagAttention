# parallax — S60 benchmark report

Date: 2026-11-16
Branch: `feat/enflame-parallax` · optimized commit tested: `8dd9b0f690f8a5bd44757d2ab515ec790425b068`
Benchmark: `benchmark/test_parallax_benchmark.py`

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

- Dense causal prefill, BF16, `(B,T,H,HQ,D) = (2,2048,2,8,64)`, scale = D^-0.5.
- Mode: `PARALLAX_BENCH_MODE=kernel` (default), warm-up 100 ms / iteration
  100 ms budgets; 12 samples per provider (mirrored ordering is skipped because
  FLA is unavailable, so only the FlagAttention provider runs).
- Timing scope: full public call (`parallel_parallax`), including allocations
  and launch; forward and forward+backward tables are separate.
- Reference: FLA unavailable (`n/a` in the table); no reference latency claimed.
- The decode benchmark is **not** part of this report (the optimized target is
  the prefill forward).


## Original vs optimized (repository benchmark)

| phase | baseline (ms) | optimized (ms) | speedup | reduction |
|---|---|---|---|---|
| forward | 3.8468 | **2.1001** | 1.832× | −45.4% |
| forward+backward | 14.837 | **13.111** | 1.132× | −11.6% |


## Reference provider

See the workload section: the benchmark's reference provider is
FLA (unavailable here). When it is unavailable or noisy this report keeps the raw log and does
not claim a reference latency.


## Preprocessing, hotspot kernels and full chain

- The forward result matches the optimization report's paired A/B
  (3.8493 → 2.1008 ms) taken with explicit per-call events, so the benchmark
  and the earlier harness agree within 0.03%.
- Preprocessing: none; hot kernel is `parallel_parallax_fwd_kernel`
  (BT=128, BS=512, 1 warp). Two changes are behind this number: BT=64→128
  (query-row reuse) and forward autotune candidates BS=256/512.
- Not measured: other default cases (T up to 8192, D up to 128) and the FLA
  reference.
- Logs: `bench_logs/parallax_{base,opt}.log`.


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
  --root /workspace/hjq/<worktree> --op parallax --tag <tag>
```
