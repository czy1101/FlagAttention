"""SageAttention versus SDPA on the assigned Enflame GCU, with timing scopes explicit."""

import json

import torch
import torch_gcu  # noqa: F401

from benchmark import test_sage_attention_benchmark as sage
from flag_attn.testing import backend


torch.gcu.set_device(1)

for length in (128, 512, 1024, 2048, 4096):
    shape = (1, 2, length, 64)
    q, k, v, qi, qs, ki, ks = sage._make_inputs(shape)
    baseline = sage._run_baseline(q, k, v)
    actual = sage._run_flagattention(qi, ki, v, qs, ks,
                                     output_dtype=torch.float16, maxnreg=None)
    cosine, relative_l2 = sage._accuracy_metrics(actual, baseline)

    def pure():
        return sage._run_flagattention(qi, ki, v, qs, ks,
                                       output_dtype=torch.float16, maxnreg=None)

    def total():
        q8, qscale, k8, kscale = sage.per_block_int8(q, k)
        return sage._run_flagattention(q8, k8, v, qscale, kscale,
                                       output_dtype=torch.float16, maxnreg=None)

    for repeat in range(3):
        sdpa_ms = backend.do_bench(lambda: sage._run_baseline(q, k, v),
                                   warmup=10, rep=50, return_mode="median")
        pure_ms = backend.do_bench(pure, warmup=10, rep=50,
                                   return_mode="median")
        total_ms = backend.do_bench(total, warmup=10, rep=50,
                                    return_mode="median")
        print(json.dumps({"shape": shape, "repeat": repeat,
                          "cosine": cosine, "relative_l2": relative_l2,
                          "sdpa_ms": sdpa_ms, "sage_pure_ms": pure_ms,
                          "sage_total_ms": total_ms}), flush=True)
