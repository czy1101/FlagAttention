# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only native-loader and dispatch contracts; no Torch import or SDK needed."""

import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import warnings


_NEAR = Path(__file__).with_name("_native_loader.py")
_LOADER = (_NEAR if _NEAR.is_file() else Path(__file__).resolve().parents[2]
           / "src/flag_attn/runtime/backend/_metax/sage_attention/_native_loader.py")


def standalone_loader():
    spec = importlib.util.spec_from_file_location("sage_loader_cpu_test", _LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.loader = standalone_loader()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.source = self.directory / "source.cu"
        self.source.write_bytes(b"fixture source")
        self.runtime = self.loader.runtime_identity("fixture-metax")
        # Cross-platform CPU tests are not real Python-extension import tests.
        self.runtime["ext_suffix"] = ".fixture.so"
        self.library = self.directory / (self.loader.MODULE_NAME + self.runtime["ext_suffix"])
        self.library.write_bytes(b"fixture binary; NEVER loaded")
        self.loader.SOURCE_SHA256 = self.loader.digest(self.source.read_bytes())
        self.manifest = {"schema_version": 1, "runtime": self.runtime, "contract": self.loader.CONTRACT,
                         "source_sha256": self.loader.SOURCE_SHA256, "library": self.library.name,
                         "library_sha256": self.loader.digest(self.library.read_bytes())}
        self.write_manifest()

    def write_manifest(self):
        (self.directory / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def verify(self):
        return self.loader.verify_artifact(str(self.directory), self.source, self.runtime)

    def test_matching_artifact_is_verified_without_import(self):
        self.assertEqual(self.verify(), self.library)

    def test_runtime_mismatch(self):
        self.manifest["runtime"] = dict(self.runtime, torch_version="different-metax")
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "runtime"):
            self.verify()

    def test_python_abi_mismatch(self):
        self.manifest["runtime"] = dict(self.runtime, soabi="different")
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.verify()

    def test_kernel_source_and_binary_mismatch(self):
        self.library.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "library identity"):
            self.verify()
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.verify()

    def test_filename_traversal_is_rejected(self):
        self.manifest["library"] = "../other.so"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "filename"):
            self.verify()

    def test_incomplete_build_is_rejected(self):
        (self.directory / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.verify()

    def test_malformed_manifest(self):
        for data in (b"[]", b"{broken"):
            with self.subTest(data=data):
                (self.directory / "manifest.json").write_bytes(data)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_relative_directory_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.loader.verify_artifact("relative", self.source, self.runtime)

    def test_unconfigured_does_not_load(self):
        with mock.patch.object(self.loader, "_import_library", side_effect=AssertionError("no import")):
            self.assertIsNone(self.loader.load_extension("", "fixture"))

    def test_unavailable_load_is_cached_and_warns_once(self):
        runtime = dict(self.runtime, platform="linux")
        with (mock.patch.object(self.loader, "runtime_identity", return_value=runtime),
              mock.patch.object(self.loader, "verify_artifact", side_effect=ValueError("fixture")) as verify,
              warnings.catch_warnings(record=True) as records):
            warnings.simplefilter("always")
            for _ in range(3):
                self.assertIsNone(self.loader.load_extension(str(self.directory), "fixture"))
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(len(records), 1)

    def test_successful_load_is_cached(self):
        extension = object()
        with (mock.patch.object(self.loader, "runtime_identity", return_value=dict(self.runtime, platform="linux")),
              mock.patch.object(self.loader, "verify_artifact", return_value=self.library),
              mock.patch.object(self.loader, "_import_library", return_value=extension) as load):
            self.assertIs(self.loader.load_extension(str(self.directory), "fixture"), extension)
            self.assertIs(self.loader.load_extension(str(self.directory), "fixture"), extension)
            self.assertEqual(load.call_count, 1)

    def test_import_error_falls_back(self):
        with (mock.patch.object(self.loader, "runtime_identity", return_value=dict(self.runtime, platform="linux")),
              mock.patch.object(self.loader, "verify_artifact", return_value=self.library),
              mock.patch.object(self.loader, "_import_library", side_effect=ImportError("missing dependency")),
              self.assertWarns(RuntimeWarning)):
            self.assertIsNone(self.loader.load_extension(str(self.directory), "fixture"))


class _Tensor:
    def __init__(self, shape, dtype="int8", device="cuda:0", pointer=4096):
        self.shape, self.ndim = tuple(shape), len(shape)
        self.dtype, self.device = dtype, device
        self.is_cuda = device.startswith("cuda")
        self.requires_grad = False
        self.pointer = pointer
        self.contiguous = True
        self.streams = []

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.pointer

    def record_stream(self, stream):
        self.streams.append(stream)


class NativeDispatchTests(unittest.TestCase):
    def setUp(self):
        self.native_calls, self.fallback_calls = [], []
        self.stream = SimpleNamespace(cuda_stream=1234)
        self.torch = SimpleNamespace(
            Tensor=_Tensor, int8="int8", float16="float16", float32="float32", __version__="fixture-metax",
            cuda=SimpleNamespace(get_device_name=lambda _: "MetaX C550", device=lambda _: nullcontext(),
                                 current_stream=lambda _: self.stream, is_current_stream_capturing=lambda: False),
            empty=lambda shape, dtype, device: _Tensor(shape, dtype, device),
        )
        self.extension = SimpleNamespace(launch=lambda *args: self.native_calls.append(args))
        self.loader = SimpleNamespace(load_extension=mock.Mock(return_value=self.extension))
        self.environment = {"FLAG_ATTN_SAGE_NATIVE_DIR": "/fixture/trusted/build"}

        def fallback(*arguments):
            self.fallback_calls.append(arguments)
            return "formal_output", "formal_lse"

        tree = ast.parse(_LOADER.with_name("native.py").read_bytes())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        self.code = {"torch": self.torch, "_native_loader": self.loader, "triton_forward": fallback,
                     "os": SimpleNamespace(environ=self.environment)}
        exec(compile(tree, "<native-dispatch-cpu-contract>", "exec"), self.code)
        self.arguments = [_Tensor((2, 3, 256, 128)), _Tensor((2, 3, 192, 128)),
                          _Tensor((2, 3, 192, 128), "float16"), _Tensor((2, 3, 2), "float32"),
                          _Tensor((2, 3, 3), "float32")]

    def test_native_contract_stream_and_return(self):
        out, lse = self.code["forward"](*self.arguments)
        self.assertEqual(self.native_calls[0][6:], (6, 256, 192, 1234))
        self.assertEqual((lse.shape, lse.dtype, lse.device), ((0,), "float32", "cpu"))
        self.assertTrue(all(t.streams == [self.stream] for t in (*self.arguments, out)))
        self.assertFalse(self.fallback_calls)

    def test_unconfigured_no_loader(self):
        self.environment.clear()
        self.assertEqual(self.code["forward"](*self.arguments), ("formal_output", "formal_lse"))
        self.loader.load_extension.assert_not_called()

    def test_unsupported_does_not_load_and_preserves_arguments(self):
        for keyword, value in (("tensor_layout", "NHD"), ("attn_mask", object()),
                               ("return_lse", True), ("maxnreg", 168), ("output_dtype", "float32")):
            with self.subTest(keyword=keyword):
                self.assertEqual(self.code["forward"](*self.arguments, **{keyword: value}),
                                 ("formal_output", "formal_lse"))
        self.loader.load_extension.assert_not_called()
        self.assertEqual(self.fallback_calls[-1][7], "float32")

    def test_unavailable_extension_falls_back(self):
        self.loader.load_extension.return_value = None
        self.assertEqual(self.code["forward"](*self.arguments), ("formal_output", "formal_lse"))
        self.assertFalse(self.native_calls)

    def test_native_execution_failure_propagates(self):
        self.extension.launch = mock.Mock(side_effect=RuntimeError("launch failed"))
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            self.code["forward"](*self.arguments)
        self.assertFalse(self.fallback_calls)

    def test_metadata_rejections(self):
        changes = [(0, "contiguous", False, "strides"), (0, "pointer", 4097, "alignment"),
                   (0, "pointer", 0, "alignment"), (1, "device", "cuda:1", "device"),
                   (2, "dtype", "bfloat16", "dtype"), (3, "shape", (2, 3, 1), "scale_shape"),
                   (1, "shape", (2, 1, 192, 128), "batch_or_heads"),
                   (0, "shape", (2, 3, 129, 128), "length"), (0, "shape", (2, 3, 256, 64), "head_dim"),
                   (0, "ndim", 3, "rank"), (2, "requires_grad", True, "autograd")]
        for index, name, value, reason in changes:
            with self.subTest(name=name, value=value):
                tensor = self.arguments[index]
                old = getattr(tensor, name)
                setattr(tensor, name, value)
                self.assertEqual(self.code["unsupported_reason"](*self.arguments), reason)
                setattr(tensor, name, old)
        self.torch.cuda.is_current_stream_capturing = lambda: True
        self.assertEqual(self.code["unsupported_reason"](*self.arguments), "graph_capture")
        self.torch.cuda.get_device_name = lambda _: "other"
        self.assertEqual(self.code["unsupported_reason"](*self.arguments), "hardware")
        self.assertFalse(self.native_calls)


if __name__ == "__main__":
    unittest.main()
