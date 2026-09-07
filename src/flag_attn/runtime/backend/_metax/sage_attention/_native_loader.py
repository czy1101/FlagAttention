# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Load an explicitly configured, locally built Sage extension; never compile.

FLAG_ATTN_SAGE_NATIVE_DIR must name a trusted, immutable build directory.
Hashes detect mismatches, not malicious binaries signed by a trusted party.
"""

from functools import lru_cache
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import struct
import sys
import sysconfig
from threading import Lock
import warnings


MODULE_NAME = "_sage_native_attention_cta128m2"
SOURCE_NAME = "sage_native_attention_cta128m2.cu"
SOURCE_SHA256 = "08cc8e4b4f15f914ddc12c52c74c311501a5da81150bcefc044ec380c0b89843"
CONTRACT = {"architecture": "xcore1000", "head_dim": 128, "q_block": 128, "kv_block": 64}
_load_lock = Lock()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def runtime_identity(torch_version):
    return {
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:2]),
        "soabi": sysconfig.get_config_var("SOABI"),
        "ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "platform": sys.platform,
        "machine": platform.machine(),
        "pointer_bits": struct.calcsize("P") * 8,
        "torch_version": str(torch_version),
    }


def _regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked native artifact")
    return path.read_bytes()


def verify_artifact(directory, source, runtime):
    directory = Path(directory)
    if not directory.is_absolute() or directory.resolve(strict=True) != directory:
        raise ValueError("native directory must be absolute and must not traverse symlinks")
    manifest = json.loads(_regular(directory / "manifest.json"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or manifest.get("runtime") != runtime or manifest.get("contract") != CONTRACT):
        raise ValueError("native artifact runtime or launch contract mismatch")
    if manifest.get("source_sha256") != SOURCE_SHA256 or digest(_regular(source)) != SOURCE_SHA256:
        raise ValueError("native source identity mismatch")
    filename = MODULE_NAME + runtime["ext_suffix"]
    if manifest.get("library") != filename:
        raise ValueError("native library filename mismatch")
    library = directory / filename
    if digest(_regular(library)) != manifest.get("library_sha256"):
        raise ValueError("native library identity mismatch")
    return library


def _import_library(path):
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("no loader for native library")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not all(callable(getattr(module, name, None)) for name in ("launch", "build_info", "kernel_resources")):
        raise ImportError("native module API mismatch")
    info = module.build_info()
    if any(info.get(key) != value for key, value in CONTRACT.items()):
        raise ImportError("native module launch contract mismatch")
    # The experiment's inherited build_info hash strings are stale; the actual
    # packaged source and compiled library above are the identity authorities.
    return module


@lru_cache(maxsize=4)
def _load_once(directory, torch_version):
    try:
        runtime = runtime_identity(torch_version)
        if runtime["platform"] != "linux" or not runtime["ext_suffix"]:
            raise ValueError("native extension requires a compatible Linux Python ABI")
        library = verify_artifact(directory, Path(__file__).with_name("csrc") / SOURCE_NAME, runtime)
        return _import_library(library)
    except (ImportError, OSError, ValueError) as error:
        warnings.warn(
            "Configured Sage native extension is unavailable (" + type(error).__name__
            + "); using Triton. Rebuild for this environment or unset FLAG_ATTN_SAGE_NATIVE_DIR.",
            RuntimeWarning, stacklevel=3,
        )
        return None


def load_extension(directory, torch_version):
    if not directory:
        return None
    # Serialize cold loads, including negative caching and its one-time warning.
    with _load_lock:
        return _load_once(directory, str(torch_version))
