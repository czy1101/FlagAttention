# Forgetting Attention / ACP: V7.6 TLE

This integration adds the selected V7.6 TLE inference implementation for Hopper
SM90. It follows the pytest/benchmark organization of FlagAttention PR #69,
without merging unrelated operators or replacing the cyc branch's test framework.

## Files

- `src/flag_attn/forgetting_attention/`: complete production implementation.
  - `tle.py`: public callable.
  - `h100_tle_kernel.py`: explicit TMA + WGMMA for M > 1.
  - `tle_decode.py`: explicit TMA + vector path for M = 1.
  - `host.py`, `fast_launch.py`: input preparation and bounded compiled-launch plans.
  - `prepare.py`, `h100_prepare.py`, `exact_prepare.py`, `starts.py`,
    `threshold.py`: exact prefix/boundary preparation and scalar threshold cache.
- `tests/flag_attn/test_forgetting_attention.py`: correctness and contract tests.
- `tests/flag_attn/forgetting_attention_support/`: official baseline and 90-case catalog.
- `benchmark/test_forgetting_attention_benchmark.py`: pytest and standalone benchmark.

Production code depends only on PyTorch and compatible Triton/TLE, not on
test support, previous experiment directories, or private local paths.
Preserve the entire production subpackage, not just the entry file.

## Requirements and API

Use a compatible Triton 3.6 / FlagTree TLE compiler and Hopper SM90. The verified
environment is H100 80GB HBM3, PyTorch 2.10.0+cu128 and the existing FlagTree
`1b3a7b85` `python_cgfix1` build. Compiled-launch reuse uses Triton runtime
interfaces and must be revalidated after compiler upgrades.

```python
import torch
from flag_attn.forgetting_attention import forgetting_attention

with torch.inference_mode():
    output = forgetting_attention(
        q, k, v, log_fgate,
        head_first=False,
        seq_start=None,
        sm_scale=0.125,
        adaptive_threshold=-10.0,
    )
```

- Q: BF16/FP16 `[B,M,Hq,D]`; K/V: `[B,N,Hkv,D]`.
- Q/K/V must be finite, contiguous, 16-byte aligned and on the same CUDA device.
- Gate: contiguous FP32 `[B,N,Hq]`, finite and nonpositive.
- `0 < M <= N`, `Hq % Hkv == 0`, D in `{16,32,60,64,100,128}`.
- Finite scale and finite nonpositive scalar threshold.
- Inference only, `head_first=False`, `seq_start=None`.
- No backward, packed-varlen, automatic threshold or threshold=None dense mode.
- D60/D100 padding and output cropping remain inside every call.
- No ordinary-attention fallback, no CUDA Graph and no output/tensor caching.
- The Torch 2.10 exact-prefix specialization preserves its accumulation tree;
  other Torch versions use Torch cumsum and require fresh validation.

M=1 intentionally uses TMA plus vector reductions instead of WGMMA to preserve
the official low-precision product rounding. For M>1, TLE explicitly issues TMA,
barriers and WGMMA. The GPU kernel bodies are unchanged from the validated V7.6
dependency chain; this integration packages them with relative imports.

## Tests and benchmarks

Use an environment that already provides the compatible dependencies. For an
uninstalled checkout, run from the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD/tests/flag_attn:${PYTHONPATH:-}"

# Only Forgetting Attention / ACP correctness, including all 90 configurations.
python -m pytest tests/flag_attn/test_forgetting_attention.py \
  -m forgetting_attention -v --strict-markers

# PR #69-style pytest benchmark: 7 representative configurations.
python -m pytest benchmark/test_forgetting_attention_benchmark.py \
  -m forgetting_attention -v -s --strict-markers

# Full 90-configuration comparison with raw Event samples.
python benchmark/test_forgetting_attention_benchmark.py \
  --full --warmup 40 --rep 400 --rounds 3 --output event_full_new.json

# A selected subset.
python benchmark/test_forgetting_attention_benchmark.py \
  --ids S011,S024 --output event_selected_new.json
```

Both test modules have `forgetting_attention` and `acp` markers, registered in
`pyproject.toml`. CUDA/SM90/TLE availability is checked explicitly.
Correctness stays in `tests/`; benchmark is not included in the default testpaths.
The standalone CLI works without adding a new test runner to this older branch.

`--warmup` and `--rep` are iteration counts, not milliseconds.
`--output` is the standalone benchmark's option, not a pytest option introduced
by PR #69. Existing reports are not overwritten. If exporting pytest benchmark
results to JUnit, use `-o junit_family=xunit1` for custom properties.
The optional `flag_attn_benchmark_result` property is compatible with later
FlagAttention recorders; those recorders are not bundled here.

## Baseline and numerical contract

The reference is the original official Forgetting Transformer ACP:
https://github.com/zhixuan-lin/forgetting-transformer

- Commit: `883f260c636e87971339749b0d8310004794a5cc`.
- File: `src/forgetting_transformer/ops/forgetting_attention.py`.
- SHA256: `6a71b9ebef4dbfaf489ca32fb4ad32ed51a3dc474b6d681c21ddf026eef1b5d2`.

The copied source retains its original license and is byte-identical to that
commit. This baseline is not our optimized V5.8h/V6.6.

There are 86 native-reference configurations and 4 explicit adapted references
(S002-S005, D padding and/or repeated KV heads). Adapter overhead is inside the
reference's timed call; adapted results are never called native upstream support.

The 206 correctness tests cover two seeds for all 90 configurations, bitwise
cold/hot output equality, historical seed-0 fingerprints under Torch 2.10,
prefix/boundary checks, threshold/scale changes, new and mutated tensors,
gate alignment, nondefault streams, tensor lifetime, unsupported APIs, generated
TMA/WGMMA instructions and actual launch-cache hits. No tolerance is relaxed.

## Verified performance snapshot

The 2026-09-24 full Event run uses the same H100 / isolated FlagTree environment
for official ACP and V7.6 TLE, not timings mixed across compiler environments.
Gate = logsigmoid(U[0,10]), threshold=-10, seed=0.
Per provider/case: 40 warmups, 400 samples, 3 rounds; median of round medians,
with alternating provider order. Full-operator time includes prefix/starts,
allocations, descriptors, padding/crop and Event-visible CPU launch gaps.

| Reference category | Cases | Faster cases | Geometric mean speedup |
|---|---:|---:|---:|
| Native official ACP | 86 | 86 | 2.376x |
| Adapted official ACP | 4 | 4 | 2.073x |

All benchmark cases passed bitwise checks before timing. All 216,000 Event
samples were validated. See [the complete per-shape table](forgetting_attention_event_comparison.md).
These are strong-pruning results on this environment, not a universal speed
guarantee. CUDA V8.13 has a different tolerance contract and is not mixed in.
