# [Operator] Add and optimize Log-Linear Attention Triton support for Hygon BW1000

## PR Category

Operator Porting / Performance Optimization / Backend Integration

## Type of Change

Add a Hygon BW1000 backend for Log-Linear Attention in FlagAttention, preserving the existing five-argument public interface, causal constraints, and hierarchical weighting semantics.

## Description

This delivery implements `flag_attn.log_linear_attn(q, k, v, g, level_scales)` as a dense causal, BF16, forward-only Hygon operator. It also exports the `chunk_log_linear_attn` alias.

The implementation uses 64-token chunks. Within each chunk, it computes QK, gating, and PV. Across chunks, it accumulates historical contributions through hierarchical KV states. The level index is determined by the most significant differing bit in the XOR of the query and key token positions, with level 0 used on the diagonal. Per-token, per-head log gates control temporal decay, while `level_scales` provides per-token, per-head, per-level weights.

The core kernels use Triton. Inspection of compiled gfx936 artifacts confirms BF16 `v_mmac_f32_16x16x16_bf16` matrix instructions. The BW1000 path uses standard Triton loads/stores and Hygon backend scheduling. Mathematical validation reuses the repository's `flag_attn.testing.log_linear_attn` reference.

### Scope and Interfaces

| Item | Description |
| --- | --- |
| Target device | Hygon BW1000, gfx936, wave64 |
| Public entry points | `flag_attn.log_linear_attn` and `flag_attn.chunk_log_linear_attn` |
| Q/K | BF16, both shaped `[B, T, 1, K]`; all V heads share one Q/K head |
| V | BF16, shaped `[B, T, H, V]` |
| Gate | FP32, shaped `[B, T, H]`; natural-log gates contribute to accumulated decay |
| Level weights | BF16, shaped `[B, T, H, L]`, where `L = ceil(log2(T)) + 1` |
| Sequence length | T must be a power of two and divisible by 64 |
| Dimensions | K and V independently take values 64, 128, or 256, including unequal K/V dimensions |
| Layout | The public entry point makes noncontiguous inputs contiguous |
| Device | All five inputs must reside on the same device |
| Output | BF16, shaped `[B, T, H, V]`; each ordinary invocation allocates an independent output |
| Computation | Causal forward computation with hierarchical weights and gate decay |
| Gradient constraint | The entry point is forward-only and explicitly rejects inputs with `requires_grad=True` |
| Performance coverage | Six official shapes covering D64/D128/D256, B1/B2/B4/B8, and H8/H16/H32/H64/H96 |

With `FLAG_ATTN_BACKEND=hygon`, the public entry point resolves to `flag_attn.runtime.backend._hygon.log_linear_attn.chunk`. This delivery is scoped to the five-argument interface above. Variable-length sequences, initial/final recurrent states, a separate decode interface, and backward computation are outside the current support scope.

### Code Organization

```text
src/flag_attn/
├── FLA/log_linear_attn/
│   ├── __init__.py
│   └── chunk_tle.py
├── runtime/backend/_hygon/
│   ├── __init__.py
│   └── log_linear_attn/
│       ├── __init__.py
│       └── chunk.py
└── testing/log_linear_attn.py

tests/flag_attn/test_log_linear_attn.py
benchmark/test_log_linear_attn_benchmark.py
```

| File | Role |
| --- | --- |
| `runtime/backend/_hygon/__init__.py` | Registers both public entry points for Hygon |
| `runtime/backend/_hygon/log_linear_attn/__init__.py` | Exports the Hygon chunk interfaces |
| `runtime/backend/_hygon/log_linear_attn/chunk.py` | Input validation, local and historical contribution kernels, state workspaces, shape dispatch, and forward execution |
| `FLA/log_linear_attn/chunk_tle.py` | The repository's existing generic TLE implementation |
| `testing/log_linear_attn.py` | The repository's existing mathematical reference |
| Official tests / benchmark | Retain the original files, cases, assertions, error thresholds, and statistical methods |

The Hygon directory follows the FLA operator's division into package exports and a chunk implementation. Final cleanup renamed the Hygon `forward.py` to `chunk.py` and removed an unused legacy output-reduction kernel. The remaining code was checked through AST comparison and runtime validation. Experimental scripts, historical candidates, profiles, frozen baselines, and raw performance records are archived outside the production implementation directory.

### Main Optimizations

1. **One gating and PV operation after full-K reduction.** Eligible paths complete FP32 QK accumulation before gating, BF16 conversion, and PV, reducing repeated local computation and output partials across K tiles. The inter-chunk kernel directly combines historical contributions with the local result.
2. **Full-QK sharing for D128 with many heads.** Eligible H64/H96 shapes reuse complete FP32 scores across heads. Each head applies its own gates and level weights, while a separate kernel produces KV leaf states. An unshared full-K control variant distinguishes the gains from computational decomposition and QK sharing.
3. **Head-first scheduling for D128 leaf states.** The grid prioritizes heads belonging to the same chunk while retaining the leaf kernel's matrix tiles, computation, and state storage format. In the controlled long-sequence H16 comparison, the leaf kernel decreases from approximately 6.05ms to 0.88ms, and public API latency decreases from approximately 11.20ms to 6.02ms.
4. **Matching D256 state production and consumption formats.** Two-level merge performs FP32 combination on chip and writes nodes required by their consumers. Query-only intermediate levels, the top level, and eligible level-2 copies are supplied to inter-chunk computation in BF16. States required for subsequent merges retain FP32. State workspaces are compactly allocated by the actual nodes in each level.
5. **Reusing inter-chunk preparation.** Eligible full-K paths prepare gate decay and row scales outside the K-tile loop, reducing repeated reductions, scale loads, and exponentials. Row scaling remains after the dot product to preserve the required numerical accuracy.
6. **Two-level merge and single-kernel short sequences for D64.** Eligible T512–T2048 paths merge adjacent levels, reducing the official T1024 shape from five kernel dispatches to four. T64/T128 complete output computation inside the local kernel, reducing two dispatches to one. For T128, the second chunk computes its sole historical leaf state and contribution within the same kernel.

Dispatch uses actual shapes and configuration to select kernels within validated ranges. Each public invocation allocates its own output and workspace. Shared caches contain input-value-independent data, such as the device's 64×64 level-index LUT.

### Validation

Run from the repository root in a configured Hygon HIP/PyTorch environment. The caller selects the physical device through environment variables. This acceptance run used physical GPU 5.

```bash
source /opt/dtk/env.sh
export CUDA_VISIBLE_DEVICES=5
export HIP_VISIBLE_DEVICES=5
unset ROCR_VISIBLE_DEVICES

export FLAG_ATTN_BACKEND=hygon
export FLAG_ATTN_TEST_DEVICE=cuda:0
export PYTHONPATH="$PWD/src:$PWD"

python -c 'import torch,triton; print(torch.__version__,torch.version.hip); print(triton.__version__,triton.__file__)'

python -m pytest -q -ra tests/flag_attn/test_log_linear_attn.py
# 4 passed, 0 skipped; acceptance records include 1 PyTorch TF32-setting deprecation warning

python -m pytest -q -s benchmark/test_log_linear_attn_benchmark.py
# 1 passed; seven rounds for each of six shapes

python - <<'PY'
import flag_attn
assert flag_attn.log_linear_attn is flag_attn.chunk_log_linear_attn
assert flag_attn.log_linear_attn.__module__ == (
    "flag_attn.runtime.backend._hygon.log_linear_attn.chunk"
)
print(flag_attn.log_linear_attn.__module__)
PY
```

**Results rerun after cleanup on 2026-11-19**

- Official tests report **4 passed, 0 skipped**, and the official benchmark reports **1 passed**.
- All **33 extended reference checks pass**, covering K/V dimension combinations, noncontiguous inputs, gate/scale boundaries, mathematical references for all six official shapes, and rejection of invalid inputs.
- All **54 invocations compared with the final implementation before cleanup are bitwise identical**.
- The maximum RMS error ratio in the extended reference checks is **0.0037598**, satisfying the original `RMS < 0.004` threshold.
- The public API was confirmed to execute the cleaned-up Hygon `chunk.py`, and source hashes were checked between the local and container copies.
- File-integrity checks pass for the official tests, benchmark, and generic TLE implementation.

**Specialized validation records retained from optimization stages**

- The D128 high-head path passes six sampled-row stress checks covering mixed gates and isolated local/historical levels, with a maximum RMS error ratio of 0.0031251.
- D64 short-sequence and merge-fusion paths pass 12 gate/boundary comparisons. Dual-stream checks for T64/T128/T1024 confirm independent outputs and bitwise agreement with the previous version.
- D64 fixed-workspace Graph comparisons verify input updates and output preservation, with performance recorded separately.

These specialized records come from their respective optimization stages. The complete post-cleanup performance results were collected again using the final source.

### Known Limitations

- The current implementation provides forward computation with BF16 Q/K/V and level weights, and FP32 gates.
- The full-K path changes floating-point accumulation and rounding boundaries relative to the initial K-partial implementation. Acceptance uses the original mathematical reference and error threshold. Bitwise-equality claims apply to the versions and comparison scopes explicitly identified in this report.
- Validation uses the HCU FlagTree/Triton distribution and its compilation toolchain. Other chips, drivers, or compiler versions require revalidation.
- Graph comparisons use caller-managed fixed inputs and workspaces and include the corresponding input-copy and output-preservation costs. Production performance below uses the ordinary public API.
- The native baseline was deferred by agreement for this task. The comparison below uses the frozen FLA_baseline and user-provided H100 TLE results.
- The submitted source was validated in a clean checkout based on `main` at `e2b9528f442c32eb6acffa501c64a40d391c9069`. The submission branch contains only this operator, its two backend registrations, and this delivery note.

## Performance

### Environment and Methodology

- Device: Hygon BW1000, gfx936, 80 CUs, wave64, 64KiB LDS/CU, 8MiB L2.
- Validated environment: `xiaodan-dev`, Python 3.10, PyTorch 2.9.0, HIP 6.3.26045, HCU FlagTree 0.7.0+hcu3.6, and the loaded Triton runtime reporting version 3.6.0.
- Inputs use BF16 Q/K/V and level weights, FP32 gates, one Q/K head, and 64-token chunks. Each shape uses the official input construction with seed 0.
- Both the final Hygon version and FLA_baseline reran the unmodified official benchmark on **2026-11-19**, with seven rounds per shape, warmup=1000ms, rep=100ms, and median latency.
- Both Hygon measurements cover the **public API forward invocation**, including ordinary preparation and output/workspace allocation. The two implementations were measured sequentially as complete benchmark suites.
- All 42 raw rounds and the corresponding source identity are retained for each implementation. Graph and profiler data are archived separately.

### Three-Way Performance Comparison

**FLA_baseline** is the frozen `FLA_baseline-v1`, established through minimal Hygon compatibility adaptation of the generic implementation and executed on BW1000.

**H100 TLE** uses values from the user-provided screenshot. The device is recorded as H100 based on the user's confirmation, and the screenshot's TLE column is used here. The complete software versions, implementation commit, warmup, repeat settings, and actual timing scope have not been provided. These values serve as a cross-device performance reference.

**This implementation** in the reviewed comparison below is the pre-submission cleaned implementation, SHA256 `1649b8e1b39fb646155dfb29749af2333d71cd5229741ddc98fccc59b67a43c1`. The final submission received import ordering, formatting, an equivalent `functools.cache` substitution, and redundant-expression cleanup. Its independent acceptance results are recorded below.

Latencies are in ms; lower is better. Speedup is **comparison implementation latency / this implementation latency**. Values above 1 indicate that the final Hygon implementation is faster. Values below 1 in the H100 column indicate lower latency for the corresponding H100 TLE implementation.

| Shape | FLA_baseline | H100 TLE | This implementation | FLA_baseline / This | H100 TLE / This |
|---|---:|---:|---:|---:|---:|
| `B1_T8192_H96_D128` | 20.073429 | 3.650080 | **7.763198** | **2.586×** | 0.470× |
| `B2_T16384_H16_D128` | 17.823990 | 2.448288 | **6.059678** | **2.941×** | 0.404× |
| `B4_T2048_H16_D128` | 2.653119 | 0.576064 | **1.492639** | **1.777×** | 0.386× |
| `B4_T4096_H64_D128` | 21.817108 | 4.530208 | **9.631195** | **2.265×** | 0.470× |
| `B8_T2048_H32_D256` | 38.495979 | 7.514592 | **13.395113** | **2.874×** | 0.561× |
| `B8_T1024_H8_D64` | 0.507360 | 0.063104 | **0.461920** | **1.098×** | 0.137× |

All six official shapes are faster than the initial FLA_baseline, with speedups of **1.098–2.941×**. All six supplied H100 TLE results have lower latency than the final Hygon implementation, with the largest gap at D64.

The submitted `chunk.py` SHA256 is `9cc5a3ccf27e266391b30d5bcd62b553538906d7a38551bb4cce3bf6c2898205`. The frozen baseline SHA256 is `9efd211d1d580b4ac76879ecb6b197bdc5c160822cd1c141d9b99bae5b2ebbc9`.

### Final Submission Acceptance on Current main

The final source was independently tested on 2026-11-19 in `xiaodan-dev`, physical GPU 5 (PCI `0000:b5:00.0`), from a clean source tree based on `main` commit `e2b9528f442c32eb6acffa501c64a40d391c9069`. The public API's resolved file was asserted to belong to that tree. Source hashes matched the local submission checkout.

- Unmodified current-main official tests: **4 passed, 0 skipped**.
- Unmodified current-main official benchmark: **1 passed, 0 skipped**, six shapes, seven rounds each. A separate final run invoked `python -m pytest` directly, without the external timer observer; the latencies below are from this uninstrumented run.
- Extended external mathematical checks: **33 passed**, maximum RMS error ratio **0.0037598**, below the unchanged 0.004 threshold.
- Compared with the accepted implementation before submission cleanup: **54 forward calls were bitwise identical**.
- Submitted operator Python files pass Ruff lint and format checks. Syntax, whitespace, and file-integrity checks pass. The complete pre-commit hook suite was not run.
- Current-main tests, benchmark, generic TLE implementation, and unrelated operators remain unchanged. The current-main benchmark includes an optional NVIDIA FLA baseline; it reports that baseline as unavailable on this Hygon environment. This does not skip execution of the Hygon implementation.

| Shape | Submitted implementation (ms) |
| --- | ---: |
| `B1_T8192_H96_D128` | 7.739357 |
| `B2_T16384_H16_D128` | 6.033918 |
| `B4_T2048_H16_D128` | 1.474080 |
| `B4_T4096_H64_D128` | 9.625115 |
| `B8_T2048_H32_D256` | 13.350313 |
| `B8_T1024_H8_D64` | 0.433760 |

The three-way table above preserves the reviewed paired baseline comparison. These submission results are a separate verification run; they do not replace the frozen baseline or establish a new optimization gain. In particular, differences between the two D64 measurements must not be attributed to a new kernel optimization. Raw logs, per-round observations from the preceding observed run, XML results, source manifests, and earlier profiles remain archived outside the submitted implementation directory.
