"""Shared inputs and frozen official baseline for ACP tests and benchmarks.

This is test-only support: the production implementation never imports it.
"""
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CASES = json.loads(Path(__file__).with_name("cases.json").read_text())
DEFAULT_IDS = ("S001", "S003", "S011", "S024", "S031", "S034", "S064")
DEFAULT_CASES = [c for c in CASES if c["id"] in DEFAULT_IDS]
OFFICIAL_COMMIT = "883f260c636e87971339749b0d8310004794a5cc"


def case_id(case):
    return f'{case["id"]}-{case["dtype"]}-{case["reference_kind"]}'


def make_inputs(case, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    b, m, n, hq, hkv, d = (case[k] for k in ("B", "M", "N", "HQ", "H", "D"))
    dtype = getattr(torch, case["dtype"])
    q = torch.randn((b, m, hq, d), device="cuda", dtype=dtype, generator=generator)
    k = torch.randn((b, n, hkv, d), device="cuda", dtype=dtype, generator=generator)
    v = torch.randn((b, n, hkv, d), device="cuda", dtype=dtype, generator=generator)
    gate = F.logsigmoid(torch.empty((b, n, hq), device="cuda", dtype=torch.float32)
                       .uniform_(0.0, 10.0, generator=generator))
    return q, k, v, gate


@lru_cache(maxsize=1)
def optimized_entry():
    return importlib.import_module("flag_attn.forgetting_attention").forgetting_attention


@lru_cache(maxsize=1)
def official_entry():
    return importlib.import_module(".official", __name__).forgetting_attention


def call_optimized(inputs, case, threshold=-10.0, scale=None):
    return optimized_entry()(*inputs, head_first=False, seq_start=None,
                             sm_scale=case["scale"] if scale is None else scale,
                             adaptive_threshold=threshold)


def call_official(inputs, case, threshold=-10.0, scale=None):
    q, k, v, gate = inputs
    # Explicit adapter, NOT a claim that upstream natively supports these cases.
    # All padding/repetition/cropping remains inside the timed reference call.
    d = case["D"]
    padded = 1 << (d - 1).bit_length()
    if d != padded:
        q, k, v = (F.pad(t, (0, padded - d)) for t in (q, k, v))
    groups = case["HQ"] // case["H"]
    if groups != 1:
        k, v = (t.repeat_interleave(groups, dim=2) for t in (k, v))
    out = official_entry()(q, k, v, gate, head_first=False, seq_start=None,
                           sm_scale=case["scale"] if scale is None else scale,
                           adaptive_threshold=threshold)
    return out if d == padded else out[..., :d].contiguous()


def fingerprint(tensor):
    data = tensor.detach().contiguous().view(torch.int16).cpu().numpy()
    return hashlib.sha256(memoryview(data)).hexdigest()


def assert_bitwise(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all().item(), "non-finite optimized output"
    assert torch.isfinite(expected).all().item(), "non-finite reference output"
    assert torch.equal(actual.contiguous().view(torch.int16),
                       expected.contiguous().view(torch.int16)), (
        f"outputs differ; max_abs={(actual.float() - expected.float()).abs().max().item()}"
    )
