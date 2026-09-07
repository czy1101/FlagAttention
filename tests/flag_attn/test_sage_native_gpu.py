# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public-entry native and fallback regression tests for the optional C550 build.

Set FLAG_ATTN_SAGE_NATIVE_DIR to a trusted built artifact to execute native
cases. Ordinary installations without the optional build retain fallback tests.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from test_sage_attention import _reference, _running_on_metax


_IS_METAX = _running_on_metax()
pytestmark = pytest.mark.skipif(not _IS_METAX, reason="MetaX Sage native GPU tests require a MetaX device")
_OBSERVED_ROUTES = {"native": 0, "triton": 0}
if _IS_METAX:
    sage = importlib.import_module("flag_attn.runtime.backend._metax.sage_attention")
    native = importlib.import_module("flag_attn.runtime.backend._metax.sage_attention.native")
    loader = importlib.import_module("flag_attn.runtime.backend._metax.sage_attention._native_loader")


@pytest.fixture(autouse=True)
def restore_tf32():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
        torch.cuda.synchronize()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.fixture
def required_native():
    directory = os.environ.get("FLAG_ATTN_SAGE_NATIVE_DIR", "")
    if not directory:
        pytest.skip("build and explicitly configure FLAG_ATTN_SAGE_NATIVE_DIR to test native execution")
    if torch.cuda.get_device_name(0) != "MetaX C550":
        pytest.skip("EXP10A native path is restricted to C550")
    extension = loader.load_extension(directory, torch.__version__)
    assert extension is not None, "configured native artifact failed to load; do not accept fallback as native"
    return extension


@pytest.fixture
def routes(monkeypatch):
    counts = {"native": 0, "triton": 0}
    original_load = loader.load_extension
    original_triton = native.triton_forward

    def load(*args):
        extension = original_load(*args)
        if extension is None:
            return None

        def launch(*arguments):
            result = extension.launch(*arguments)
            counts["native"] += 1
            return result

        return SimpleNamespace(launch=launch)

    def triton(*args, **kwargs):
        result = original_triton(*args, **kwargs)
        counts["triton"] += 1
        return result

    monkeypatch.setattr(loader, "load_extension", load)
    monkeypatch.setattr(native, "triton_forward", triton)
    yield counts
    for key in counts:
        _OBSERVED_ROUTES[key] += counts[key]


def inputs(b=1, h=2, nq=128, nk=128, d=128, hkv=None, layout="HND"):
    torch.manual_seed(901)
    hkv = h if hkv is None else hkv
    q = torch.randn((b, h, nq, d), device="cuda", dtype=torch.float16)
    k = torch.randn((b, hkv, nk, d), device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    if layout == "NHD":
        q, k, v = (value.transpose(1, 2).contiguous() for value in (q, k, v))
    seq_dim = 2 if layout == "HND" else 1
    q8, qs, k8, ks = sage.per_block_int8(q, k, km=k.mean(dim=seq_dim, keepdim=True), tensor_layout=layout)
    return q8, k8, v, qs, ks


def check_output(actual, expected):
    assert bool(torch.isfinite(actual).all())
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
    expected_norm = torch.linalg.vector_norm(expected.float())
    error_norm = torch.linalg.vector_norm(actual.float() - expected.float())
    if expected_norm.item() == 0:
        assert error_norm.item() == 0
    else:
        assert (error_norm / expected_norm).item() <= 0.02


@pytest.mark.parametrize("b,h,nq,nk,custom", [
    (1, 1, 128, 64, False), (1, 2, 128, 128, False),
    (1, 2, 256, 192, True), (2, 3, 256, 256, False),
])
@torch.inference_mode()
def test_native_matches_reference_repeat_and_stream(b, h, nq, nk, custom, required_native, routes):
    args = inputs(b=b, h=h, nq=nq, nk=nk)
    saved = [value.clone() for value in args]
    expected, _ = _reference(*args, "HND")
    stream = torch.cuda.Stream() if custom else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        outputs = [sage.forward(*args) for _ in range(3)]
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    assert routes == {"native": 3, "triton": 0}
    for output, lse in outputs:
        check_output(output, expected)
        assert lse.numel() == 0 and lse.device.type == "cpu" and lse.dtype == torch.float32
        assert torch.equal(output, outputs[0][0])
    assert all(torch.equal(a, b) for a, b in zip(args, saved))


@torch.inference_mode()
def test_native_zero_v_and_output_guard(monkeypatch, required_native, routes):
    args = list(inputs())
    args[2].zero_()
    q = args[0]
    storage = torch.full((q.numel() + 64,), 37, dtype=torch.float16, device=q.device)
    out = storage[32:-32].view(q.shape)
    monkeypatch.setattr(native, "_empty_output", lambda _: out)
    result, _ = sage.forward(*args)
    torch.cuda.synchronize()
    assert routes == {"native": 1, "triton": 0}
    assert result.data_ptr() == out.data_ptr()
    assert torch.count_nonzero(out).item() == 0
    assert bool((storage[:32] == 37).all()) and bool((storage[-32:] == 37).all())


@pytest.mark.parametrize("case", ["NHD", "GQA", "LSE", "D64", "tail", "mask"])
@torch.inference_mode()
def test_fallback_preserves_reference(case, routes):
    options, kwargs = {}, {}
    if case == "NHD":
        options["layout"] = "NHD"
        kwargs["tensor_layout"] = "NHD"
    elif case == "GQA":
        options["hkv"] = 1
    elif case == "LSE":
        kwargs["return_lse"] = True
    elif case == "D64":
        options["d"] = 64
    elif case == "tail":
        options.update(nq=129, nk=70)
    else:
        mask = torch.ones((1, 2, 128, 128), device="cuda", dtype=torch.bool)
        mask[..., ::3] = False
        kwargs["attn_mask"] = mask
    args = inputs(**options)
    saved = [value.clone() for value in args]
    output, lse = sage.forward(*args, **kwargs)
    expected, expected_lse = _reference(*args, kwargs.get("tensor_layout", "HND"), kwargs.get("attn_mask"))
    check_output(output, expected)
    if kwargs.get("return_lse"):
        torch.testing.assert_close(lse, expected_lse, atol=2e-2, rtol=2e-2)
    assert routes == {"native": 0, "triton": 1}
    assert all(torch.equal(a, b) for a, b in zip(args, saved))


@torch.inference_mode()
def test_missing_configured_extension_falls_back(monkeypatch, tmp_path, routes):
    args = inputs()
    monkeypatch.setenv("FLAG_ATTN_SAGE_NATIVE_DIR", str(tmp_path / "not-built"))
    with pytest.warns(RuntimeWarning, match="using Triton"):
        output, _ = sage.forward(*args)
    expected, _ = _reference(*args, "HND")
    check_output(output, expected)
    assert routes == {"native": 0, "triton": 1}


@torch.inference_mode()
def test_unconfigured_extension_falls_back(monkeypatch, routes):
    args = inputs()
    monkeypatch.delenv("FLAG_ATTN_SAGE_NATIVE_DIR", raising=False)
    output, _ = sage.forward(*args)
    expected, _ = _reference(*args, "HND")
    check_output(output, expected)
    assert routes == {"native": 0, "triton": 1}


@torch.inference_mode()
def test_bad_scale_and_storage_are_rejected_before_launch():
    args = list(inputs())
    args[3] = args[3][..., :0]
    assert native.unsupported_reason(*args) == "scale_shape"
    args = list(inputs())
    wide = torch.empty((*args[0].shape[:-1], 256), device="cuda", dtype=torch.int8)
    args[0] = wide[..., ::2]
    assert native.unsupported_reason(*args) == "strides"
    args = list(inputs())
    backing = torch.empty(args[0].numel() + 1, device="cuda", dtype=torch.int8)
    args[0] = backing[1:].view(args[0].shape)
    assert native.unsupported_reason(*args) == "alignment"


@torch.inference_mode()
def test_native_failure_is_not_hidden(monkeypatch):
    args = inputs()

    def broken_launch(*unused):
        raise RuntimeError("synthetic launch failure")

    def forbidden_fallback(*unused, **kwargs):
        raise AssertionError("must not hide native failure")

    monkeypatch.setenv("FLAG_ATTN_SAGE_NATIVE_DIR", "/synthetic-loader-replaced-in-this-test")
    monkeypatch.setattr(loader, "load_extension", lambda *unused: SimpleNamespace(launch=broken_launch))
    monkeypatch.setattr(native, "triton_forward", forbidden_fallback)
    with pytest.raises(RuntimeError, match="synthetic launch failure"):
        sage.forward(*args)


def test_public_function_is_production_dispatch():
    assert sage.forward is native.forward
