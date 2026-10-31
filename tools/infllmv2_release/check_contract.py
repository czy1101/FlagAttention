# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Standalone Hygon regressions; does not change official test collection.

Run after selecting the visible device. No physical device is selected here.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


def dense_reference(q, k, v, qlens, klens, scale):
    outputs = []
    qs = ks = 0
    for nq, nk in zip(qlens, klens):
        qq = q[qs : qs + nq].float().transpose(0, 1)
        kk = k[ks : ks + nk].float().transpose(0, 1).repeat_interleave(q.shape[1] // k.shape[1], 0)
        vv = v[ks : ks + nk].float().transpose(0, 1).repeat_interleave(q.shape[1] // k.shape[1], 0)
        mask = torch.arange(nk, device=q.device)[None, :] <= nk - nq + torch.arange(nq, device=q.device)[:, None]
        probabilities = torch.softmax(((qq @ kk.transpose(1, 2)) * scale).masked_fill(~mask, float("-inf")), -1)
        outputs.append((probabilities @ vv).transpose(0, 1))
        qs += nq
        ks += nk
    return torch.cat(outputs).to(q.dtype)


def check_scales(api):
    results = []
    cases = (
        ((32,), (64,), 8, 2, 64, True),
        ((13, 17), (64, 96), 8, 2, 128, True),
        ((128,), (256,), 32, 2, 128, True),
        ((1024,), (1024,), 32, 2, 64, False),
        ((1, 1), (64, 128), 8, 2, 64, True),
    )
    for dtype in (torch.bfloat16, torch.float16):
        for qlens, klens, hq, hkv, dim, training in cases:
            for scale in (0.0, -0.125):
                generator = torch.Generator(device="cuda").manual_seed(101)
                shapes = ((sum(qlens), hq, dim), (sum(klens), hkv, dim), (sum(klens), hkv, dim))
                q, k, v = [
                    (torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.2).requires_grad_(training)
                    for shape in shapes
                ]
                qr, kr, vr = [x.detach().clone().requires_grad_(training) for x in (q, k, v)]
                cq = torch.tensor([0, *torch.tensor(qlens).cumsum(0).tolist()], device=q.device, dtype=torch.int32)
                ck = torch.tensor([0, *torch.tensor(klens).cumsum(0).tolist()], device=q.device, dtype=torch.int32)
                if max(qlens) == 1:
                    actual = api.infllmv2_decode(q, k, v, ck, max(klens), softmax_scale=scale)
                else:
                    actual = api.infllmv2_attention(q, k, v, cq, ck, max(qlens), max(klens), softmax_scale=scale)
                expected = dense_reference(qr, kr, vr, qlens, klens, scale)
                assert torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                if training:
                    grad = torch.randn(actual.shape, device=q.device, dtype=dtype, generator=generator)
                    da = torch.autograd.grad(actual, (q, k, v), grad)
                    de = torch.autograd.grad(expected, (qr, kr, vr), grad)
                    for x, y in zip(da, de):
                        assert torch.isfinite(x).all()
                        torch.testing.assert_close(x, y, atol=0.03, rtol=0.03)
                    if scale == 0:
                        assert not torch.count_nonzero(da[0]) and not torch.count_nonzero(da[1])
                results.append({"dtype": str(dtype), "q": qlens, "kv": klens, "d": dim, "scale": scale})
    for invalid in (float("nan"), float("inf"), -float("inf")):
        try:
            api.infllmv2_attention(q, k, v, cq, ck, max(qlens), max(klens), softmax_scale=invalid)
        except ValueError:
            continue
        raise AssertionError("Nonfinite scale was accepted")
    return results


def check_topk(forward):
    results = []
    for length in (8192, 32768, 65536):
        blocks = length // 64
        cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        for ties in (False, True):
            generator = torch.Generator(device="cuda").manual_seed(2026)
            score = torch.randn((2, length, blocks), device="cuda", dtype=torch.bfloat16, generator=generator)
            if ties:
                score.zero_()
            else:
                score[:, :, ::13] = float("nan")
                score[:, :, ::17] = float("inf")
                score[:, :, ::19] = float("-inf")
            actual = forward.select_blocks(score, cu, 64, 64, cu)
            # Independent stable-sort reference. Future blocks are excluded;
            # invalid output slots follow all valid indices and contain -1.
            positions = torch.arange(blocks, device=score.device)[None, None, :]
            for start in range(0, length, 512):
                stop = min(start + 512, length)
                visible = ((torch.arange(start, stop, device=score.device) + 64) // 64)[None, :, None]
                values = score[:, start:stop].float()
                values = torch.where(torch.isnan(values), float("-inf"), values)
                values = values.masked_fill(positions >= visible, float("-inf"))
                ids = torch.argsort(values, dim=-1, descending=True, stable=True)[..., :64]
                ids = torch.where(ids < visible, ids, blocks).sort(dim=-1).values
                expected = torch.where(ids < blocks, ids, -1).to(torch.int32)
                torch.testing.assert_close(actual[:, start:stop], expected, atol=0, rtol=0)
            results.append({"length": length, "ties": ties, "exact": True})
    return results


def check_cache(api):
    generator = torch.Generator(device="cuda").manual_seed(2026)
    q = torch.randn((1, 32, 64), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.2
    k = torch.randn((8192, 2, 64), device=q.device, dtype=q.dtype, generator=generator) * 0.2
    v = torch.randn_like(k)
    cu = torch.tensor([0, 8192], device=q.device, dtype=torch.int32)
    first = api.infllmv2_decode(q, k, v, cu, 8192)
    second = api.infllmv2_decode(q, k, v, cu, 8192)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.data_ptr() != second.data_ptr()
    with torch.no_grad():
        q.add_(0.1)
        k.mul_(0.5)
        cu[1] = 4096
    actual = api.infllmv2_decode(q, k, v, cu, 8192)
    expected = dense_reference(q, k, v, (1,), (4096,), 64**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    api.clear_cache()
    torch.testing.assert_close(actual, api.infllmv2_decode(q, k, v, cu, 8192), atol=0, rtol=0)
    with torch.inference_mode():
        unversioned = torch.tensor([0, 8192], device=q.device, dtype=torch.int32)
        api.infllmv2_decode(q, k, v, unversioned, 8192)
        unversioned[1] = 4096
        torch.testing.assert_close(api.infllmv2_decode(q, k, v, unversioned, 8192), expected, atol=0.02, rtol=0.02)
    return {"tracked_cu_mutation": True, "inference_cu_mutation": True, "independent_outputs": True}


def check_guards(forward):
    original = forward._attention_launcher
    observed = []

    class Observe:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                observed.append(args[18])
                return original[grid](*args, **kwargs)

            return launch

    length = 8192
    q = torch.randn((length, 32, 128), device="cuda", dtype=torch.bfloat16) * 0.2
    k = torch.randn((length, 2, 128), device=q.device, dtype=q.dtype) * 0.2
    v = torch.randn_like(k)
    cu = torch.tensor([0, length], device=q.device, dtype=torch.int32)
    forward._attention_launcher = Observe()
    try:
        for topk, store_lse, tile in ((64, False, 16), (32, False, 32), (128, False, 32), (64, True, 32)):
            ids = (
                torch.arange(topk, device=q.device, dtype=torch.int32)[None, None, :]
                .expand(2, length, topk)
                .contiguous()
            )
            out = forward._forward(q, k, v, ids, cu, cu, length, store_lse=store_lse)[0]
            assert torch.isfinite(out).all() and observed[-1] == tile
        different = torch.tensor([0, 4096], device=q.device, dtype=torch.int32)
        assert not forward._full_prefill(length, 2, different, cu, length)
    finally:
        forward._attention_launcher = original
    return {"tiles": observed, "actual_length_guard": True}


def check_imports():
    results = []
    for selector in ("hcu", "native-v3", "opt-v7", "obsolete-value"):
        env = dict(os.environ, FLAG_ATTN_BACKEND="hygon", FLAG_ATTN_INFLLMV2_IMPL=selector)
        code = """
from flag_attn.runtime.backend import _hygon
from flag_attn.runtime.backend import resolve_operator
assert '_minimax_m3_sparse_attn_prefill' in _hygon._OPERATOR_EXPORTS
assert 'parallel_nsa' in _hygon._OPERATOR_EXPORTS
assert resolve_operator('infllmv2_attention', 'flag_attn.infllmv2.operator').__module__.endswith('infllmv2.operator')
"""
        subprocess.run([sys.executable, "-c", code], env=env, check=True)
        results.append(selector)
    code = """
import triton
triton.__version__ = '3.5.0'
try:
    import flag_attn.runtime.backend._hygon.infllmv2.forward
except ImportError as exc:
    assert 'validated HCU FlagTree' in str(exc), str(exc)
else:
    raise AssertionError('Unsupported ABI was accepted')
"""
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ, FLAG_ATTN_BACKEND="hygon"), check=True)
    for backend in ("cpu", "nvidia"):
        code = "import sys,flag_attn; assert not any('._hygon.infllmv2' in n for n in sys.modules)"
        subprocess.run([sys.executable, "-c", code], env=dict(os.environ, FLAG_ATTN_BACKEND=backend), check=True)
    return {"ignored_legacy_selectors": results, "unsupported_abi_error": True, "other_backend_imports": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Run these regressions in the documented HCU runtime on BW1000")
    forward = importlib.import_module("flag_attn.runtime.backend._hygon.infllmv2.forward")
    target = forward.driver.active.get_current_target()
    if target.arch != "gfx936":
        raise SystemExit("These specialization regressions require BW1000/gfx936")
    api = importlib.import_module("flag_attn.runtime.backend._hygon.infllmv2.operator")
    result = {}
    for name, check in (
        ("scales", lambda: check_scales(api)),
        ("topk", lambda: check_topk(forward)),
        ("cache", lambda: check_cache(api)),
        ("guards", lambda: check_guards(forward)),
        ("imports", check_imports),
    ):
        result[name] = check()
        print(f"CONTRACT_PASS {name}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("HYGON_CONTRACT_ACCEPTANCE_PASS", flush=True)


if __name__ == "__main__":
    main()
