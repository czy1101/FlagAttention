# sage_attention — S60 benchmark report

Date: 2026-11-16
Branch: `feat/enflame-sage-attention` · optimized commit tested: `eba192c8e1b3b72dbb399253c85e239541f4b7c1`
Benchmark: `benchmark/test_sage_attention_benchmark.py`

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

- Shapes `(B,H,T,D) = (4,32,{1024,2048,4096},128)`, HND, no mask, seed 42.
- dtypes: source Q/K/V **FP16**; quantized Q/K **INT8** with **FP32** per-block
  scales (Q block 128 incl. `sm_scale*log2(e)`, K block 64 scale 1.0);
  accumulation FP32; output **BF16**.
- Timing scope: `flag_attn(ms)` times the attention forward only — the
  benchmark performs Q/K quantization **before** timing
  (`BENCHMARK_SCOPE = attention_kernel_only`).
- Reference: `torch_sdpa_plus_bfloat16_cast` executed on the same S60
  (PyTorch SDPA is not NVIDIA data). Its own timing is noisy between runs
  (T1024: 75.9 vs 95.3 ms), so only the direction is reported, not a ratio.


## Original vs optimized (repository benchmark)

| T | baseline flag_attn (ms) | optimized flag_attn (ms) | speedup | reduction |
|---|---|---|---|---|
| 1024 | 3.909 | **3.429** | 1.140× | −12.3% |
| 2048 | 14.391 | **11.754** | 1.224× | −18.3% |
| 4096 | 55.209 | **42.798** | 1.290× | −22.5% |

Correctness (repo thresholds): cosine = 0.999850, relative L2 = 0.012424 at
T1024 (same values for both versions; T2048/4096 within 1e-5).

Quantization + full chain (not part of the repo benchmark; separate driver on
the same card and inputs, `do_bench(25,100)`):

| T | quant baseline | quant optimized | full chain baseline | full chain optimized |
|---|---|---|---|---|
| 1024 | 8.226 | **1.494** (5.51×) | 12.045 | **4.871** (2.47×) |
| 2048 | 16.411 | **2.919** (5.62×) | 30.802 | **14.629** (2.11×) |
| 4096 | 32.691 | **5.762** (5.67×) | 87.683 | **48.734** (1.80×) |


## Reference provider

See the workload section: the benchmark's reference provider is
torch SDPA on the same S60 (runs, but timing is noisy). When it is unavailable or noisy this report keeps the raw log and does
not claim a reference latency.


## Preprocessing, hotspot kernels and full chain

- The attention forward may also run an internal K pre-dequantisation kernel
  (`_dequant_k_fp16`) once the reuse threshold is met; at these shapes the
  optimized build enables it (D128, ≥8 reuse blocks), so the timed forward can
  contain two kernels. The baseline build ran the inline-conversion path only.
- Preprocessing: Q/K quantization (measured separately above, excluded by the
  repo benchmark), layout strides, and output cast.
- Not measured: masked paths, NHD timing, `km`, FP8, decode, T>4096.
- Logs: `bench_logs/sage_{base,opt}.log`; quant/full-chain raw data in
  `sage_logs/ab_*.json`.


## Optimization summary (what changed and why)

Two commits on this branch:

1. `428a9bc` — `_enflame/sage_attention/quant_per_block.py` now launches both
   Q/K quantisation kernels with `num_warps=1` instead of the default 4.
   Mechanism: each program performs a **block-wide max** over the whole
   `[BLK, D]` tile to derive the per-block scale. With 4 warps the compiled
   module gains a cross-warp reduction barrier and 512 B more shared memory
   (`syncthreads` 2 vs 1, 66048 vs 65536 B); measured 8.23 → 1.49 ms (5.5×) at
   T=1024, and the INT8 tensors plus both scale tensors are **bitwise identical**
   (SHA-256 checked), so the quantisation semantics do not change.
   Rejected in the same round: chunked two-pass tiles (slower at 1 warp),
   reciprocal-multiply (no gain, 773/4670 INT8 values changed by 1 LSB), and
   INT8 `tl.dot` (compiles and is exact, but emits **no `convgen` matrix
   intrinsic** on this backend, so it is not a faster matrix path).
2. `eba192c` — `_enflame/sage_attention/attn_qk_int8_per_block.py` lowers the
   D128 K pre-dequantisation threshold from 64 to 8 reuse blocks. The kernel
   otherwise re-converts every K tile from INT8 to FP16 once per query program;
   pre-dequantising once turns that into one linear pass plus a cheaper QK loop.
   Measured attention-only: −11.3% / −18.6% / −22.4% at T=1024/2048/4096. This
   extends the repository's existing mechanism (D64 keeps threshold 32) and adds
   an FP16 K temporary of 2 B/element.

Evidence level: both effects are measured on card 7 with the same inputs; the
warp-count explanation is supported by the compiled metadata/IR but not by
hardware counters. Scope: quantisation change applies to all shapes; the
pre-dequantisation change needs `attn_mask is None` and D128.

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
  --root /workspace/hjq/<worktree> --op sage_attention --tag <tag>
```
