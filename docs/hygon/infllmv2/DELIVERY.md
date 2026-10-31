# [Operator] Add and optimize InfLLM-V2 Triton support for Hygon BW1000

## PR Category

Operator Porting / Performance Optimization / Backend Integration

## Type of Change

Add a Hygon backend implementation while preserving the existing public interfaces and operator semantics.

## Description

This change adds a Hygon BW1000 backend for InfLLM-V2 in FlagAttention, covering Prefill, Chunk Prefill, Decode, and sparse attention backward computation.

The implementation preserves the two-stage sparse attention pipeline. Stage 1 compresses K, computes scores using two levels of compressed context, and selects context blocks through pooling and exact Top-K. Stage 2 performs causal GQA attention over the selected blocks. Top-K selection is discrete: gradients flow through Stage 2 Q/K/V, without differentiating through block selection.

Compression, scoring, block selection, sparse attention, and gradient kernels are implemented in Triton. On BW1000, `tl.dot` maps to Hygon MMAC matrix instructions. The core computation does not call the separately developed native HIP/MMAC baseline or use TLE kernels. The dense path reuses the library's FlashAttention implementation, while configuration and mathematical references reuse the common InfLLM-V2 modules.

### Scope and Interfaces

| Item | Description |
| --- | --- |
| Target device | Hygon BW1000, gfx936, wave64 |
| Validated precision | BF16 and FP16 correctness; all performance measurements use BF16 |
| Prefill / Chunk inputs | Q/K/V use contiguous packed `[tokens, heads, head_dim]` layouts; the public entry points make noncontiguous inputs contiguous |
| Variable-length sequences | Q/KV sequences are described by one-dimensional int32 cumulative-length arrays on the same device |
| GQA | The number of query heads must be an integer multiple of the number of KV heads; the performance suite uses Hq=32 and Hkv=2 |
| Causal positions | Each Q sequence is the suffix of its KV sequence; query i has position `Lkv - Lq + i` |
| Decode | Q has shape `[batch, Hq, D]`, and KV uses a packed cache; the caller appends the current token, and the operator does not update the cache in place |
| Dense / Sparse dispatch | The default is `dense_len=8192`; a declared maximum KV length below the threshold takes the dense path, while equality takes the sparse path |
| Backward | Computes sparse attention gradients for Q/K/V; does not differentiate through discrete Top-K selection |
| Performance coverage | 17 cases covering D64/D128, Full/Chunk Prefill, forward plus backward, and B1/B8 Decode |

The public entry points remain `flag_attn.infllmv2_attention` and `flag_attn.infllmv2_decode`, with the existing `InfLLMV2Config`. Set `FLAG_ATTN_BACKEND=hygon` to use the Hygon implementation. Production registration does not depend on historical experiment selectors.

This validation scope does not imply unconditional support for arbitrary dtypes, shapes, software versions, or hardware. The implementation preserves causal packed attention semantics; it does not introduce paged KV, in-place cache append, dropout, or noncausal modes.

### Code Organization

```text
src/flag_attn/runtime/backend/_hygon/
├── __init__.py
└── infllmv2/
    ├── __init__.py
    ├── operator.py
    ├── forward.py
    ├── backward.py
    └── reference.py
```

| File | Role |
| --- | --- |
| `_hygon/__init__.py` | Registers the Hygon attention and decode entry points |
| `infllmv2/__init__.py` | Exports the public interfaces and cache-clearing entry point |
| `operator.py` | Public interfaces, dense/sparse pipeline dispatch, and dense/decode execution plans |
| `forward.py` | K compression, scoring, pooling, exact Top-K, sparse forward computation, and runtime guards |
| `backward.py` | Q/K/V gradient computation and shape-dependent task dispatch |
| `reference.py` | Reuses the common mathematical references for validation |

### Main Optimizations

1. **Stage 1 causal trimming and score storage.** Trims loops over invalid future context. Intermediate scores are stored after the prescribed BF16 rounding, avoiding the additional traffic of storing already-rounded values in FP32.
2. **Stage 2 GQA and MMAC tiling.** Each workgroup for a query token and KV head processes the complete GQA group, reusing K/V with a matrix layout suited to wave64. Guarded D128 Full Prefill uses N16 subtiles, while D64 retains N32.
3. **Single-wave exact Top-K merging.** Validated Full Prefill shapes use streaming merges with 64 keys and one wave, preserving the original packed score/index keys, NaN handling, and tie-breaking. At 64K, the Top-K kernel decreases from approximately 10.17ms to 2.48ms, and explicitly allocated LDS decreases from 4KiB to zero.
4. **Shape-dependent pooling configurations.** 8K Full Prefill uses two waves; 32K/64K retain the configurations that performed better in measurement.
5. **Decode parallelism and invocation overhead.** Split-KV and a stable merge improve parallelism for small query counts. Eligible calls reuse guarded compiled execution plans and length metadata, while each invocation still produces independent outputs and workspaces.
6. **Chunk and backward scheduling.** Reuses versioned metadata and launch information. Validated training shapes use rectangular owner/split task assignment to reduce task-construction and synchronization overhead.

Shape specializations are guarded by device, precision, actual cumulative lengths, and configuration. Calls that do not meet these conditions retain the existing paths. The submitted implementation does not hardcode a physical GPU index, PCI address, Python path, or repository path.

### Validation

Run from the repository root in a configured HIP environment:

```bash
export FLAG_ATTN_BACKEND=hygon
export PYTHONPATH="$PWD/src:$PWD"

python -m pytest -q -ra tests/flag_attn/test_infllmv2_attention.py
# 19 passed, 4 skipped, 1 warning

python -m pytest -q -s benchmark/test_infllmv2_attention_benchmark.py
# 17 passed

python tools/infllmv2_release/check_contract.py
# HYGON_CONTRACT_ACCEPTANCE_PASS
```

Validation results:

- The original official tests and the same suite with component entry points bound to the Hygon implementation both report **19 passed, 4 skipped, 1 warning**. The binding plugin changes only component entry points, not cases, assertions, or tolerances; production execution does not depend on the plugin.
- The four skipped cases concern Hopper/TLE or the common NVIDIA implementation contract and are not counted as Hygon passes.
- All 17 performance shapes were numerically compared against the previous production Triton implementation. External forward comparisons retain `atol=rtol=0.04`, and gradient comparisons retain `0.03`; official tests continue to use their original assertions.
- Six full-row exact Top-K stress checks cover equal scores, random scores, NaNs, and positive/negative infinities, with bitwise-identical output indices.
- The existing 15 dense and 20 decode checks pass, covering cache invalidation, layouts, stream isolation, hook fallback, and related boundary conditions. Actual decode Graph fallback has been validated.
- Checks pass for Stage 2 Top-K sizes, the LSE path, and actual-length guards.
- A separate source copy with historical InfLLMv2/DSL directories removed passes imports, tests, and benchmarks.
- The five implementation files pass Ruff lint and format checks. Integrity checks pass for the official files, common implementation, and frozen DSL baseline.

### Known Limitations

- For finite zero or negative scales, the Hygon dense path reuses the existing all-block attention kernels and gradient path, avoiding the common Flash path's NaN issue. Default and positive scales retain their existing paths; nonfinite scales are rejected explicitly. The common implementation remains unchanged.
- Call the Hygon implementation's `clear_cache()` after untracked modifications to cumulative lengths through `.data`, raw pointers, or similar mechanisms.
- Execution plans depend on the HCU FlagTree/Triton runtime ABI validated here. Other distributions require revalidation. No additional support for actual dense Graph capture is claimed.

## Performance

### Environment and Methodology

- Device: Hygon BW1000, gfx936, 80 CUs, wave64, 64KiB LDS/CU, 8MiB L2.
- Container: `xiaodan-dev`; PyTorch 2.9.0, HIP 6.3.26045, Triton 3.6.0, HCU FlagTree 0.7.0+hcu3.6.
- Versions refer to the modules actually loaded. The Triton distribution metadata in this environment differs from `triton.__version__`; both module paths and versions should be checked when reproducing the environment.
- Inputs: BF16, Hq=32, Hkv=2, D64/D128; default Top-K=64, block size=64, and dense threshold=8192.
- Uses the official benchmark with warmup=20ms and rep=100ms, reporting `median_ms`. CUDA Graph or profiler timings are not substituted for ordinary invocation latency.
- The caller selects the experimental device through the environment; the submitted code does not bind execution to a GPU index.

### Four-Way Comparison

- **Initial FLA port:** the initial Triton implementation integrated with the Hygon backend.
- **DSL baseline:** the independently developed and frozen native HIP/MMAC Native v3 implementation. It is not an official Hygon release of InfLLM-V2 and is not a runtime dependency of this submission.
- **H100:** user-provided H100 benchmark results. The screenshot does not provide the complete implementation version, commit, or software environment; these results serve as a supplementary cross-device comparison.
- **This implementation:** the Hygon Triton implementation delivered here, using all 17 results from the latest official benchmark run.

The first three columns retain previously recorded measurements; the final latency column is from the current run. All columns report BF16, warmup=20ms, and rep=100ms. This aggregation across runs is not treated as a same-run, interleaved controlled experiment, and the gains are not attributed solely to the compiler.

Latencies are in ms; lower is better. Speedup = comparison implementation latency / this implementation latency. Values above 1 indicate that this implementation is faster.

| Case | Initial FLA port | DSL baseline | H100 | This implementation | FLA / This | DSL / This | H100 / This |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense-1x1024-d64 | 0.6507 | 0.4562 | 0.2267 | 0.4083 | 1.594× | 1.117× | 0.555× |
| full-1x8192-d64 | 40.0835 | 11.2939 | 2.4219 | 6.3329 | 6.329× | 1.783× | 0.382× |
| full-1x8192-d128 | 97.8922 | 21.4517 | 3.5432 | 10.3530 | 9.455× | 2.072× | 0.342× |
| full-1x32768-d64 | 201.5493 | 71.5136 | 11.4853 | 32.0156 | 6.295× | 2.234× | 0.359× |
| full-1x32768-d128 | 498.7588 | 136.5460 | 18.1480 | 52.8883 | 9.430× | 2.582× | 0.343× |
| full-1x65536-d64 | 441.0247 | 208.6489 | 27.8027 | 78.8033 | 5.597× | 2.648× | 0.353× |
| full-1x65536-d128 | 1070.3525 | 383.4950 | 43.4036 | 124.1421 | 8.622× | 3.089× | 0.350× |
| chunk-1x128x16384-d64 | 2.7618 | 1.3003 | 0.6805 | 0.9181 | 3.008× | 1.416× | 0.741× |
| chunk-1x128x65536-d128 | 4.6894 | 3.4272 | 0.7243 | 1.4846 | 3.159× | 2.309× | 0.488× |
| backward-1x128x8192-d64 | 4.4434 | 2.7992 | 1.3318 | 2.1726 | 2.045× | 1.288× | 0.613× |
| backward-1x128x8192-d128 | 6.5874 | 5.1145 | 1.3329 | 3.0701 | 2.146× | 1.666× | 0.434× |
| decode-1x8192-d64 | 2.2472 | 0.3118 | 0.6919 | 0.3245 | 6.925× | 0.961× | 2.132× |
| decode-1x8192-d128 | 2.6760 | 0.3829 | 0.6883 | 0.3510 | 7.624× | 1.091× | 1.961× |
| decode-1x65536-d64 | 2.2489 | 1.1064 | 0.6940 | 0.7853 | 2.864× | 1.409× | 0.884× |
| decode-1x65536-d128 | 2.6869 | 1.6282 | 0.6856 | 0.7837 | 3.428× | 2.078× | 0.875× |
| decode-8x32768-d64 | 2.3901 | 1.0290 | 0.7233 | 0.7718 | 3.097× | 1.333× | 0.937× |
| decode-8x32768-d128 | 3.1883 | 1.6615 | 0.8483 | 1.1149 | 2.860× | 1.490× | 0.761× |

This implementation is faster than the initial FLA port in 17/17 cases and faster than the DSL baseline in 16/17 cases; `decode-1x8192-d64` remains slightly slower. In the cross-device comparison, the two B1/KV8K decode cases are faster than the supplied H100 results, while the remaining cases are slower than H100.

The `backward-*` cases use mode `forward_backward`: their latencies include both forward and backward execution and must not be interpreted as standalone backward kernel timings.
