# chunk_gdn2 — S60 benchmark report

Date: 2026-11-16
Branch: `feat/enflame-gdn2` · optimized commit tested: `808e3c17e29fd07753de5d48df613f37dbdfbe0c`
Benchmark: `benchmark/test_gdn2_benchmark.py`

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

- `(B,T,H,K,V) = (1,4096,16,64,64)`, BF16, `chunk_size=64`,
  `output_final_state=True`; weights/gates built by the benchmark with seed 42.
- **The benchmark default uses a random non-zero `initial_state`.** A separate
  run with `initial_state=None` (the optimization target) is reported below and
  is not mixed with the default.
- Timing scope: public `chunk_gdn2` forward only, `warmup=20ms`, `rep=100ms`.
- Reference: the native Triton comparison is only printed when
  `HAS_TLE_GDN2`; on this backend it is `baseline: unavailable`, so no native
  reference number is claimed.


## Original vs optimized (repository benchmark)

Benchmark default (`initial_state` random; three runs of the optimized build to
expose variance — the first optimized run, 5.2020 ms, was a transient outlier
and is kept in the log):

| version | run 1 | run 2 | run 3 | median used |
|---|---|---|---|---|
| baseline | 4.2784 | 4.1707 | – | 4.17 |
| optimized | 5.2020 | 3.5723 | 3.4609 | **3.572** |

| phase | baseline (ms) | optimized (ms) | speedup | reduction |
|---|---|---|---|---|
| forward (default) | 4.1707 | **3.572** | 1.17× | −14.4% |
| forward (`initial_state=None`) | 4.1854 | **3.4624** | 1.21× | −17.3% |


## Reference provider

See the workload section: the benchmark's reference provider is
native Triton kernel (not enabled on this backend). When it is unavailable or noisy this report keeps the raw log and does
not claim a reference latency.


## Preprocessing, hotspot kernels and full chain

- The `initial_state=None` row comes from a supplemental driver that reuses
  the benchmark's input builder and `do_bench(20,100)`; it is a separate
  measurement, not the benchmark column.
- Preprocessing: none outside the operator (state init/tail handled internally).
  Hot kernels: intra-chunk solve, gate scan and inter-chunk recurrence.
- Not measured: the other `DEFAULT_SHAPES`/`ALL_SHAPES` entries and the native
  Triton reference (not enabled on this backend).
- Logs: `bench_logs/gdn2_{base,opt}*.log`, `bench_logs/gdn2_supp_*.json`.


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
  --root /workspace/hjq/<worktree> --op chunk_gdn2 --tag <tag>
```
