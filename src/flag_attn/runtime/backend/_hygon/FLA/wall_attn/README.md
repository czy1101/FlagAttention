# [Operator] Add and optimize Wall Attention Triton support for Hygon BW1000

## PR Category

Operator Porting / Performance Optimization / Backend Integration

## Type of Change

Add a Hygon BW1000 backend implementation providing Wall Attention forward, backward, single-step decode, and KV-cache construction through the existing interfaces.

## Description

This change adds a Hygon BW1000 backend for the Wall Attention operator in FlagAttention. The implementation follows causal vector-gated attention semantics. It accumulates per-channel gates, combines queries and keys to compute attention scores, applies softmax, and computes the weighted sum of values. The training path provides gradients for Q, K, V, vector gates, and optional scalar gates and sink bias.

The core computation uses Triton. Sampled BW1000 compilation artifacts contain precision-specific `v_mmac_*` matrix instructions. Softmax states and output accumulation use FP32. Pre-rescaled historical keys have numerical validity flags, and the main kernel selects the computation path using safety conditions for the current queries and cached blocks.

The release follows the five-file structure of the project's official `FLA/wall_attn` package. Relative to the official snapshot used for this adaptation, `866c86a40b79c7f616cac8e4921d08385a58fe52`, the source diff consists of five Hygon implementation files and one backend registration entry. Integrity checks confirm matching hashes for the official tests, benchmark, and common Wall Attention files.

### Scope and Interfaces

| Item | Description |
| --- | --- |
| Target device | Hygon BW1000, gfx936, wave64 |
| Validated precision | BF16 and FP16, with additional small FP32 cases for the default forward path and optional arguments |
| Prefill inputs | Q and G use `[B, T, HQ, K]`, K uses `[B, T, H, K]`, and V uses `[B, T, H, V]` |
| MHA / GQA | The number of query heads HQ must be an integer multiple of the number of KV heads H; performance shapes use HQ=8, with H=8 for MHA and H=2 for GQA |
| Causality and variable lengths | Forward computation uses causal masking and supports sliding windows, scalar gates, and sink bias; variable-length inputs use `cu_seqlens` with B=1 packed sequences |
| Training | Validated full forward and backward computation, including multiple batches, variable lengths, sliding windows, D128, and different value dimensions |
| Decode | Queries use `[B, 1, HQ, K]`; computation reads pre-rescaled keys, block anchors, and values, and returns the output and log2 LSE |
| Cache construction | `build_wall_kv_cache` constructs pre-rescaled keys and block anchors from the gate prefixes associated with each query head |
| Performance coverage | 16 dense causal Prefill shapes with B=1, HQ=8, K=V=64, T=512/1024/2048/4096, BF16/FP16, and MHA/GQA |

The public entry point is `flag_attn.parallel_wall_attn`. Set `FLAG_ATTN_BACKEND=hygon` when explicit backend selection is needed. The Hygon component package also exports `parallel_wall_attn_decode`, `build_wall_kv_cache`, and `naive_wall_attn`.

### Code Organization

```text
src/flag_attn/runtime/backend/_hygon/
├── __init__.py
└── FLA/wall_attn/
    ├── __init__.py
    ├── parallel.py
    ├── parallel_fwd.py
    ├── decode.py
    └── naive.py
```

| File | Role |
| --- | --- |
| `_hygon/__init__.py` | Registers `parallel_wall_attn` with the Hygon component package |
| `wall_attn/__init__.py` | Exports Prefill, decode, cache construction, reference, and route-reporting interfaces |
| `parallel.py` | Public entry point, shape dispatch, full forward and backward computation, autograd, and compatibility paths |
| `parallel_fwd.py` | Gate scans, historical key rescaling, attention main kernels, specialized short-sequence forward computation, and compiled launcher reuse |
| `decode.py` | Single-step decode and pre-rescaled KV-cache construction |
| `naive.py` | Mathematical reference implementation for correctness validation |

### Main Optimizations

1. **Pre-rescaled historical keys and reuse.** Builds pre-rescaled keys and validity flags for 256-token history blocks, reducing repeated gate-prefix reads and exponential rescaling across query workgroups. Query safety, block validity, and anchor conditions jointly determine cache use.
2. **Historical softmax and PV layout.** Processes complete history in 64-token subtiles to reduce online softmax updates. PV accumulation uses a layout suited to wave64 while preserving FP32 softmax, accumulation, and local-reference handling in the diagonal region.
3. **Two-pass FP32 gate prefixes.** For medium and long sequences, first computes summaries for 256-token blocks, then combines preceding summaries with each block's local scan to increase prefix-scan parallelism.
4. **Long-history query scheduling.** Assigns earlier workgroup IDs to later query tiles and places the heads of each tile next to one another, reducing execution tails caused by different causal history lengths.
5. **Separate short-sequence configurations.** T512 uses a single-pass gate scan with 8 channels and 8 waves per workgroup. T1024 uses 4 channels and 8 waves. T1024 attention uses BT64 and long-history-query-first scheduling while retaining numerical anchors at 128-token logical boundaries.
6. **Lower public-call submission overhead.** Validated inference paths allocate each call's outputs and workspaces before submission, reuse compiled launchers, and enqueue computation consecutively on the current stream. T2048/T4096 execute four kernels in order for gate summaries, gate prefixes, cache preparation, and attention.

Specialized dispatch checks actual shapes, dtypes, layouts, devices, scales, and runtime state. Each invocation allocates independent outputs and workspaces. Compiled launchers are cached by device and the required specialization parameters. The caller configures the execution device.

### Validation

Run from the repository root in a configured HIP environment after selecting the visible device.

```bash
export FLAG_ATTN_BACKEND=hygon
export PYTHONPATH="$PWD/src:$PWD"

python -m pytest -q -rs tests/flag_attn/test_parallel_wall_attn.py
# 25 passed, 12 skipped, 1 warning

python -m pytest -q -s benchmark/test_parallel_wall_attn_benchmark.py
# 2 passed, 2 warnings; covers 16 performance shapes
```

Validation was performed on 2026-11-19 against the release commit `b0f84400ac5add60a5cdf2217fd5c073149be6c0`.

- The original official tests report **25 passed and 12 skipped**. An external tool binds the test module's provider entry point to the Hygon implementation and reuses the same assertions, obtaining the same results. The two runs are recorded separately.
- The 12 skipped cases require H100/TLE or conditions specific to the common NVIDIA implementation.
- Both dtype tests in the official benchmark pass, covering all 16 Prefill shapes. The previously accepted implementation was also rerun on the same card to check end-to-end performance for each shape after code organization.
- All 56 targeted input cases and 224 FP64 reference-row checks for T4096/T1024 pass. Inputs cover ordinary gates, zero gates, strong decay, alternating signs, small positive gates, and partial or complete cache invalidation.
- For the targeted T4096 cases, the maximum absolute differences from the accepted implementation are zero for both the output and log2 LSE. The corresponding T1024 values are 0.015625 and 0.00051117, satisfying the original combined absolute and relative tolerances.
- Independent reference checks pass for multiple batches, FP16/BF16, variable lengths, sliding windows, wider value dimensions, D128, outputs, six gradient types, decode, and cache construction. These external reference checks fix forward and backward configurations to BT64/BS32/4 waves/2 stages and decode to BT16. The official benchmark uses normal public dispatch.
- Checks pass for T4095/T4097, noncontiguous inputs, other scales, optional-argument fallback, Graph capture, multiple streams, and output independence.
- AST checks confirm matching function bodies for eight relocated kernels. All 62 tracked test and benchmark files are byte-identical, and the local and remote release source identities match.

The official tests retain their original assertions and tolerances. External binding and reference checks are included in the acceptance evidence. Production dispatch uses the Hygon backend registration.

### Known Limitations

- The performance table covers 16 Prefill shapes with B=1, HQ=8, and K=V=64. Training, decode, variable-length, wider-value, and D128 results apply to the correctness checks described above.
- Launcher reuse for short, medium, and long sequences depends on the validated HCU FlagTree / Triton runtime. Changes to the distribution or ABI require revalidation.
- Calls enter compatibility paths when Graph capture or runtime checks do not satisfy the specialized dispatch conditions. Existing guards for Graph capture, hooks, and special execution contexts remain active.
- GQA gate prefixes depend on the query head, so pre-rescaled keys are stored per query head. Workspace requirements grow with T, HQ, and head dimension.
- As agreed for this task, the Triton `FLA_baseline` was established and frozen. Performance comparisons use this baseline, the official H100 TLE results, and the final BW1000 implementation.

## Performance

### Environment and Methodology

The Hygon test device is BW1000, gfx936, with 80 CUs, wave64, 64 KiB LDS/CU, and 8 MiB L2. The environment uses the `xiaodan-dev` container, Python 3.10, PyTorch 2.9.0, HIP 6.3.26045, HCU FlagTree 0.7.0+hcu3.6, and a loaded Triton runtime reporting version 3.6.0. The Triton distribution metadata reports `3.5.1+das.opt1.dtk2604.torch290`. Reproduction should verify both the loaded module version and its path.

Performance inputs use B=1, HQ=8, K=V=64, H=8 for MHA, H=2 for GQA, seed=0, and scale=0.125. Vector gates are generated from negative absolute random values multiplied by 0.05. All BW1000 performance measurements use physical card 5, selected through the environment outside the repository.

BW1000 latency comes from ordinary public API calls in the original official benchmark, including allocation, gate-prefix computation, cache preparation, and attention. Each shape receives 30 warmup calls and seven measurements of 200 ms each, with the median reported. GPU-event, Graph, and profiler diagnostic results retain their respective timing scopes.

The H100 comparison uses the final A42 BF16 / A43 FP16 TLE versions from PR #88 on H100 80GB HBM3, with PyTorch 2.7.0+cu126 and Triton/FlagTree 3.6.0. Shapes and input configurations match. The official record describes the public API timing scope, warmup, and repetition settings. [Official H100 TLE performance record](https://github.com/flagos-ai/FlagAttention/pull/88#performance)

### Three-Way Performance Comparison

| Implementation | Data source |
| --- | --- |
| `FLA_baseline` | The minimally adapted and frozen official Triton implementation on BW1000, using its latest official benchmark rerun from 2026-11-18 |
| H100 TLE | The Final Optimized Version in FlagAttention PR #88, matched by dtype, MHA/GQA, and T |
| Final BW1000 | Official benchmark results from the organized release commit `b0f84400ac5add60a5cdf2217fd5c073149be6c0`, measured on 2026-11-19 |

The three columns use the recorded runs described above. The BW1000 baseline and final implementation use the same official benchmark. The H100 comparison also involves differences in hardware and software environments. Latencies are in ms, with lower values indicating faster execution. All ratios are calculated per shape from unrounded latencies and displayed to three decimal places.

- **Speedup versus FLA_baseline = FLA_baseline latency / final BW1000 latency.**
- **Speedup versus H100 TLE = H100 TLE latency / final BW1000 latency.**

| Precision | Family | T | FLA_baseline ms | H100 TLE ms | Final BW1000 ms | FLA_baseline / Final BW1000 | H100 TLE / Final BW1000 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | MHA | 512 | 0.461919 | 0.032160 | 0.168320 | 2.744× | 0.191× |
| BF16 | MHA | 1024 | 0.585440 | 0.051232 | 0.218080 | 2.685× | 0.235× |
| BF16 | MHA | 2048 | 0.888320 | 0.088736 | 0.335040 | 2.651× | 0.265× |
| BF16 | MHA | 4096 | 1.798559 | 0.211328 | 0.670240 | 2.683× | 0.315× |
| BF16 | GQA | 512 | 0.460640 | 0.031936 | 0.173280 | 2.658× | 0.184× |
| BF16 | GQA | 1024 | 0.579039 | 0.050976 | 0.216960 | 2.669× | 0.235× |
| BF16 | GQA | 2048 | 0.883840 | 0.088480 | 0.333600 | 2.649× | 0.265× |
| BF16 | GQA | 4096 | 1.787039 | 0.209152 | 0.669120 | 2.671× | 0.313× |
| FP16 | MHA | 512 | 0.470720 | 0.034016 | 0.166880 | 2.821× | 0.204× |
| FP16 | MHA | 1024 | 0.561440 | 0.054752 | 0.201760 | 2.783× | 0.271× |
| FP16 | MHA | 2048 | 0.808000 | 0.095904 | 0.325439 | 2.483× | 0.295× |
| FP16 | MHA | 4096 | 1.587359 | 0.210720 | 0.644320 | 2.464× | 0.327× |
| FP16 | GQA | 512 | 0.473120 | 0.033312 | 0.164960 | 2.868× | 0.202× |
| FP16 | GQA | 1024 | 0.562240 | 0.053568 | 0.199680 | 2.816× | 0.268× |
| FP16 | GQA | 2048 | 0.810079 | 0.093712 | 0.329120 | 2.461× | 0.285× |
| FP16 | GQA | 4096 | 1.578879 | 0.209376 | 0.639520 | 2.469× | 0.327× |

The final BW1000 implementation achieves **2.461–2.868×** speedup over `FLA_baseline` across all 16 shapes. H100 TLE latency divided by final BW1000 latency is **0.184–0.327×**. H100 TLE has lower latency than the final BW1000 implementation for all 16 shapes.
