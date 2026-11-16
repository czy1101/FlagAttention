# chunk_gla — S60 benchmark report

Date: 2026-11-16
Branch: `feat/enflame-chunk-gla` · optimized commit tested: `ec9942a181e7ab2a9684e029d08e40176b2cd06d`
Benchmark: `benchmark/test_chunk_gla_benchmark.py`

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

- Shape: `(B,T,H,D) = (4,2048,16,128)`, BF16; masked/gated causal GLA.
- Layout: HND; `g` is the log-space gate, converted inside the operator.
- Timing scope: the benchmark's `flag_attn(ms)` covers the forward call only
  (inputs pre-built outside the timed region); a second table times
  forward+backward.
- Reference: FLA is unavailable in this container, so the table prints
  `provider: FlagAttention only (FLA unavailable)`; no reference latency is
  claimed.


## Original vs optimized (repository benchmark)

| phase | baseline (ms) | optimized (ms) | speedup | reduction |
|---|---|---|---|---|
| forward | 6.0006 | **4.6236** | 1.298× | −22.9% |
| forward+backward | 386.020 | **339.201** | 1.138× | −12.1% |


## Reference provider

See the workload section: the benchmark's reference provider is
FLA (unavailable here). When it is unavailable or noisy this report keeps the raw log and does
not claim a reference latency.


## Preprocessing, hotspot kernels and full chain

- Preprocessing measured by the benchmark: none outside the operator (gate
  prefix scan is internal). Hot kernels: the packed gate-scan tile kernel and
  the intra/inter chunk attention kernels; the optimization packs two heads per
  gate-scan tile.
- Not measured here: other default shapes in `_SHAPES` (the full list includes
  T up to 16384) and the FLA reference (unavailable).
- Logs: `/workspace/hjq/bench_logs/gla_{base,opt}.log` (host
  `/data/hjq/bench_logs/`).


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
  --root /workspace/hjq/<worktree> --op chunk_gla --tag <tag>
```
