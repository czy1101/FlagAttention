"""PR #69-style ACP tests: explicit marks, parametrization and capability skips."""
import importlib
import math
import weakref
import gc

import pytest
import torch

from flag_attn.forgetting_attention import has_tle
from forgetting_attention_support import (
    CASES, case_id, make_inputs, call_optimized, call_official, optimized_entry,
    fingerprint, assert_bitwise,
)

CUDA_SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)
pytestmark = [
    pytest.mark.forgetting_attention,
    pytest.mark.acp,
    pytest.mark.skipif(not CUDA_SM90, reason="ACP V7.6 requires Hopper SM90 CUDA"),
    pytest.mark.skipif(not has_tle(), reason="ACP V7.6 requires compatible Triton 3.6 TLE"),
]


@pytest.mark.parametrize("case", CASES, ids=case_id)
@pytest.mark.parametrize("seed", [0, 1])
@torch.inference_mode()
def test_forgetting_attention_matches_official(case, seed):
    inputs = make_inputs(case, seed)
    expected = call_official(inputs, case)
    cold = call_optimized(inputs, case)
    warm = call_optimized(inputs, case)
    assert_bitwise(cold, expected)
    assert_bitwise(warm, expected)
    if seed == 0 and torch.__version__.startswith("2.10."):
        assert fingerprint(warm) == case["v66_frozen_sha256"], "V7.6 frozen-output regression"


@pytest.mark.parametrize("case", [c for c in CASES if c["id"] in
                                  ("S001", "S003", "S011", "S024", "S031", "S034", "S064")], ids=case_id)
@torch.inference_mode()
def test_forgetting_attention_prefix_and_boundaries(case):
    from flag_attn.forgetting_attention.host import prepare, h100_config
    inputs = make_inputs(case)
    b, m, n, h, hk, d = (case[k] for k in ("B", "M", "N", "HQ", "H", "D"))
    prep = h100_config(b, m, n, h, hk, d)[2]
    prefix, starts, _, _, bn = prepare(*inputs, False, None, case["scale"], -10.0, prep, True)
    expected_prefix = torch.cumsum(inputs[3].transpose(1, 2), -1, dtype=torch.float32)
    assert torch.equal(prefix, expected_prefix)
    qm = 1 if m == 1 else 128
    qpos = n - m + torch.arange(0, m, qm, device="cuda")
    kpos = torch.arange(bn - 1, n + bn - 1, bn, device="cuda").clamp_max(n - 1)
    anchors = expected_prefix[..., qpos]
    ends = expected_prefix[..., kpos]
    expected_starts = ((anchors[..., :, None] - ends[..., None, :]) < -10.0).sum(-1) * bn
    assert torch.equal(starts, expected_starts)


def _reuse_case(dtype="bfloat16"):
    return dict(B=2, M=65, N=129, HQ=4, H=4, D=64, scale=0.125,
                dtype=dtype, reference_kind="native", id="reuse")


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@torch.inference_mode()
def test_forgetting_attention_launch_reuse(dtype):
    case = _reuse_case(dtype)
    inputs = make_inputs(case, 37)
    for threshold, scale in [(-10.0, 1), (-5.0, 1.0), (-10.0, 0.125), (0.0, 0.125)]:
        assert_bitwise(call_optimized(inputs, case, threshold, scale),
                       call_official(inputs, case, threshold, scale))
    fresh = tuple(t.clone() for t in inputs)
    assert_bitwise(call_optimized(fresh, case), call_official(fresh, case))
    fresh[0].mul_(0.5)
    fresh[2].neg_()
    fresh[3].mul_(1.125)
    assert_bitwise(call_optimized(fresh, case), call_official(fresh, case))


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
@torch.inference_mode()
def test_forgetting_attention_gate_alignment(offset):
    case = _reuse_case()
    q, k, v, gate = make_inputs(case, 5)
    storage = torch.empty(gate.numel() + offset, device="cuda", dtype=gate.dtype)
    shifted = storage[offset:].view_as(gate)
    shifted.copy_(gate)
    inputs = (q, k, v, shifted)
    assert_bitwise(call_optimized(inputs, case), call_official(inputs, case))


@torch.inference_mode()
def test_forgetting_attention_nondefault_stream_and_lifetime():
    case = _reuse_case()
    for _ in range(2):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            inputs = make_inputs(case, 8)
            expected = call_official(inputs, case)
            out = call_optimized(inputs, case)
            assert_bitwise(out, expected)
        stream.synchronize()
        refs = [weakref.ref(t) for t in (*inputs, out)]
        del inputs, expected, out
        gc.collect()
        assert all(ref() is None for ref in refs), "launch plan retains caller tensors"


@pytest.mark.parametrize("problem", ["head_first", "seq_start", "threshold_none", "threshold_positive",
                                    "threshold_nan", "scale_inf", "gate_dtype", "q_dtype", "noncontiguous"])
@torch.inference_mode()
def test_forgetting_attention_rejects_unsupported(problem):
    case = _reuse_case()
    q, k, v, gate = make_inputs(case)
    kwargs = dict(head_first=False, seq_start=None, sm_scale=0.125, adaptive_threshold=-10.0)
    if problem == "head_first":
        kwargs["head_first"] = True
    elif problem == "seq_start":
        kwargs["seq_start"] = torch.zeros(2, device="cuda", dtype=torch.int32)
    elif problem == "threshold_none":
        kwargs["adaptive_threshold"] = None
    elif problem == "threshold_positive":
        kwargs["adaptive_threshold"] = 1.0
    elif problem == "threshold_nan":
        kwargs["adaptive_threshold"] = float("nan")
    elif problem == "scale_inf":
        kwargs["sm_scale"] = float("inf")
    elif problem == "gate_dtype":
        gate = gate.half()
    elif problem == "q_dtype":
        q = q.float()
    else:
        q = torch.empty((*q.shape[:-1], q.shape[-1] * 2), device="cuda", dtype=q.dtype)[..., ::2]
    with pytest.raises((NotImplementedError, ValueError)):
        optimized_entry()(q, k, v, gate, **kwargs)


@pytest.mark.parametrize("id", ["S031", "S034", "S024"])
@torch.inference_mode()
def test_forgetting_attention_explicit_async_and_hot_launch(id):
    from flag_attn.forgetting_attention.fast_launch import capture
    case = next(c for c in CASES if c["id"] == id)
    inputs = make_inputs(case)
    call_optimized(inputs, case)
    with capture() as records:
        call_optimized(inputs, case)
    torch.cuda.synchronize()
    attention = [r for r in records if "cp.async.bulk.tensor" in r[1].asm["ptx"]]
    assert len(attention) == 1
    _, compiled, _, _, cache_hit = attention[0]
    assert cache_hit
    assert ("wgmma.mma_async" in compiled.asm["ptx"]) == (case["M"] > 1)
