# Hygon InfLLM-V2

The Hygon backend implements packed causal InfLLM-V2 attention for BW1000
(gfx936, wave64). Compression, scoring, pooling, exact Top-K, sparse attention,
and sparse attention gradients use Triton. The implementation does not load
the experimental native HIP baseline or historical InfLLMv2 directories.

## Runtime requirements

The validated environment is Python 3.10, PyTorch 2.9.0 / HIP 6.3.26045,
HCU FlagTree 0.7.0+hcu3.6, and the HCU Triton runtime reporting version 3.6.0.
Use the vendor HCU distribution; installing a generic Triton wheel is not a
substitute. Private launch-plan APIs are distribution-specific. The backend
reports an actionable ImportError when required private APIs or the validated
Triton version are unavailable. Other backends retain their own requirements.

Check the modules actually loaded, rather than relying only on pip metadata:

```bash
python -c 'import torch,triton; print(torch.__version__,torch.version.hip); print(triton.__version__,triton.__file__)'
```

The test environment retains older Triton distribution metadata even though
the loaded runtime is 3.6.0; acceptance records both the metadata and module
locations. Device selection belongs to the caller. The implementation and
validation scripts do not select a physical GPU or hardcode a PCI address.

## Interfaces and layout

Set `FLAG_ATTN_BACKEND=hygon` when explicit selection is needed. Production
registration always selects this implementation; the historical
`FLAG_ATTN_INFLLMV2_IMPL` experiment selector is ignored. No versioned
experiment environment variables are required for the release defaults.
Legacy launch-tuning switches remain private diagnostic controls, not public
API requirements.

The public entry points are `flag_attn.infllmv2_attention` and
`flag_attn.infllmv2_decode`, configured with `InfLLMV2Config`.

- Q/K/V use packed `[tokens, heads, head_dim]` layouts. The public interface
  makes noncontiguous inputs contiguous.
- Q and KV cumulative lengths are matching one-dimensional int32 tensors on
  the same device. Each Q sequence is the suffix of its KV sequence.
- Decode Q has `[batch, Hq, D]` layout. KV includes the current token; the caller
  owns cache append and storage. There is no in-place cache update or paged KV.
- Query heads must be divisible by KV heads. Noncausal configurations are
  rejected. Performance coverage uses BF16, Hq32/Hkv2, D64/D128; correctness
  additionally covers FP16 and variable-length inputs.
- Declared maximum KV length below `dense_len` selects dense attention;
  equality selects sparse attention. The default threshold is 8192.
- Gradients flow through Stage 2 Q/K/V, not discrete Top-K selection.

Finite zero and negative softmax scales are supported by the Hygon dense
path. It reuses the existing sparse attention/autograd kernels with all blocks
selected, avoiding the shared Flash implementation's nonpositive-scale issue.
Default and positive scales retain the existing dense path. Nonfinite scales
are rejected explicitly. The common implementation is unchanged.

## Implementation structure

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

`operator.py` owns dispatch and dense/decode plans. `forward.py` contains
forward kernels, metadata guards and launch specialization. `backward.py`
contains gradient kernels and task dispatch. `reference.py` reexports the
common mathematical references. Dense attention normally reuses library
FlashAttention; the dense decode fallback also uses library split-KV.

The measured full-prefill specialization checks actual cumulative lengths,
BF16, gfx936/wave64 and the validated runtime. It uses a wave-local 64-key
exact Top-K merge, N16 subtiles for eligible D128 inference, and two-wave
pooling at 8K. Other cases retain existing configurations. Each invocation
allocates independent output/workspace buffers.

## Validation

From the repository root in the configured HIP environment:

```bash
export FLAG_ATTN_BACKEND=hygon
export PYTHONPATH="$PWD/src:$PWD"
python -m pytest -q -ra tests/flag_attn/test_infllmv2_attention.py
python -m pytest -q -s benchmark/test_infllmv2_attention_benchmark.py
python tools/infllmv2_release/check_contract.py --output results/hygon-contract.json
```

`PYTHON=/path/to/python bash tools/infllmv2_release/run_checks.sh` also runs
the original official suite with its component imports bound to Hygon by the
external `release_plugin`. This changes neither official assertions nor
tolerances. Production execution does not depend on that plugin. The original
test file also imports common component implementations, so both runs matter.

The standalone contract checks cover nonpositive-scale outputs/gradients,
exact Top-K ordering, metadata mutation, independent outputs, specialization
guards, stale experiment selectors and import/ABI isolation. They do not add
cases to the official test collection. Acceptance additionally builds a wheel
from the recorded Git snapshot and tests that installation outside the source
tree with PYTHONPATH removed; the report records the tested commit and wheel
hash separately from the multi-run performance comparison.

## Cache and capture boundaries

Normal tensor writes invalidate versioned cumulative-length metadata.
Untracked `.data` or raw-pointer updates to cumulative lengths require
`flag_attn.runtime.backend._hygon.infllmv2.clear_cache()`. Actual decode Graph
fallback has been tested; additional dense Graph support is not claimed.

## Integration into the zjx branch

This branch predates the common InfLLM-V2 operator. Its five common source
files and the official test/benchmark are copied verbatim from upstream
commit `48e1e05ac1fc3be09ad61aa75088d56909c0f98f`, alongside the three
public API exports. The branch's existing Hy3 implementation, runtime,
device finder, shared Flash/split-KV kernels, test helpers and benchmark
recording remain unchanged. On this older branch, set
`FLAG_ATTN_BACKEND=hygon` explicitly; automatic gfx936 detection from newer
upstream revisions is not included in this change.
