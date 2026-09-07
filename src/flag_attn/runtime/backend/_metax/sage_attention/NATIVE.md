# Optional MetaX C550 SageAttention native path

The public `forward` return signature and quantization are unchanged. Triton
remains available for all previously supported inputs. Installing/importing
FlagAttention does not compile or automatically download native code.

## Explicit build and enable

Use your trusted MetaX Python environment and existing MACA SDK (tested with
Python 3.12, MetaX Torch 2.8.0+metax3.7.2.0, MACA 3.7.2, C550). No SDK headers,
third-party binaries or prebuilt `.so` files are shipped with this package.

From a checkout, first select that checkout's source:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m flag_attn.runtime.backend._metax.sage_attention.build_native \
  --maca-home /opt/maca-3.7.2 --output-dir /absolute/new/sage-native-build
```

The output parent must already exist. The output directory must be new and must
not traverse symlinks; it is never overwritten. Build failures retain `build.log`
and do not create a ready manifest. Building does not constitute correctness or
performance acceptance; validate the built artifact before deployment.

To enable a validated build in a fresh process:

```bash
export FLAG_ATTN_SAGE_NATIVE_DIR=/absolute/new/sage-native-build
```

Unset that variable to retain Triton-only dispatch. Only configure directories
and binaries you trust, and keep them immutable for the lifetime of all worker
processes. The loader checks the packaged source, binary hash, Python ABI,
platform, exact Torch version and launch contract. These hashes detect accidental
mismatches, not malicious native code. Rebuild and revalidate after changing the
runtime or native source. Missing/unloadable/incompatible configured artifacts
warn once per cached configuration and fall back to Triton; there is no online
compilation. Loaded modules and failed loads are cached; restart workers after
changing an artifact. Actual native execution errors propagate, not fall back.

## Accelerated domain

Eager inference on C550, packed contiguous **HND**, equal batch/head count for
Q/K/V, head dimension128, INT8 Q/K, FP16 V/output, FP32 block scales, Q length a
positive multiple of128 and KV length a positive multiple of64. Scales must be
generated for BLKQ128/BLKK64 with the existing `per_block_int8` convention (Q scale
already includes softmax scale and log2(e)). Shape/alignment/grid/device guards
run before native launch. No implicit layout conversions are inserted.

NHD, masks, GQA, tails, other dtypes/head dimensions, LSE, maxnreg, graph capture
and gradient-requiring inputs remain on Triton. Falling back preserves the old
implementation; it is not a new support claim for inputs that Triton itself does
not support. Callers retain responsibility for cross-stream input readiness;
native dispatch uses the current stream and records allocator lifetime.

## Source and evidence

The Apache-2.0 EXP10A C++/MCTlass source is preserved byte-for-byte, SHA256
`08cc8e4b4f15f914ddc12c52c74c311501a5da81150bcefc044ec380c0b89843`.
Its inherited `build_info` source/kernel hash strings are historical and are
not identity authorities. The builder hashes the actual packaged source and
new artifact. GPU kernels, quantization and the formal benchmark are unchanged.

The prior isolated validation adapter passed23 correctness cases and measured
approximately1.912x geometric-mean HND end-to-end speedup over formal Triton on
five FP16 shapes. This is evidence for the kernel/validated adapter, **not a
claim that this production loader, build command, or all deployment environments
have passed**. Run the formal integration gates before merging this packaging.
