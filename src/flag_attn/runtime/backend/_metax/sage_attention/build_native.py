# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicitly build the optional MetaX Sage extension into a new directory.

This command does not install dependencies, edit the repository, import the
new extension, run GPU kernels, or enable native dispatch in the calling shell.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sysconfig

from ._native_loader import CONTRACT, MODULE_NAME, SOURCE_NAME, SOURCE_SHA256, digest, runtime_identity


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compile_command(command, output_dir, timeout):
    process = None
    try:
        with (output_dir / "build.log").open("xb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait(timeout=timeout)
        require(code == 0, "cucc failed; inspect build.log (no artifact enabled)")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="new absolute artifact directory; never overwritten")
    parser.add_argument("--maca-home", type=Path, default=Path(os.environ.get("MACA_HOME", "/opt/maca")))
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    import torch
    runtime = runtime_identity(torch.__version__)
    require(runtime["platform"] == "linux" and "metax" in runtime["torch_version"].lower(),
            "requires Linux and a MetaX Torch build")
    require(args.timeout > 0, "timeout must be positive")
    destination = args.output_dir
    require(destination.is_absolute() and not destination.exists() and not destination.is_symlink(),
            "output-dir must be a new absolute directory")
    require(destination.parent.resolve(strict=True) == destination.parent,
            "output parent must exist and must not traverse symlinks")
    sdk = args.maca_home.resolve(strict=True)
    compiler = sdk / "tools/cu-bridge/bin/cucc"
    source = Path(__file__).with_name("csrc") / SOURCE_NAME
    require(digest(source.read_bytes()) == SOURCE_SHA256, "packaged EXP10A source hash mismatch")
    includes = [sdk / "tools/cu-bridge/include", sdk / "include",
                Path(sysconfig.get_paths()["include"]), Path(torch.__file__).resolve().parent / "include"]
    headers = [includes[0] / "cuda_runtime.h", includes[1] / "mctlass/arch/mma_sm80.h",
               includes[1] / "mctlass/half.h", includes[2] / "Python.h", includes[3] / "pybind11/pybind11.h"]
    require(compiler.is_file() and os.access(compiler, os.X_OK) and all(p.is_file() for p in headers),
            "existing cucc/SDK/Python/pybind11 headers missing; no dependencies installed")
    require(bool(runtime["ext_suffix"]), "Python extension suffix is unavailable")
    version = subprocess.run([str(compiler), "--version"], check=True, capture_output=True,
                             text=True, timeout=30, stdin=subprocess.DEVNULL).stdout
    filename = MODULE_NAME + runtime["ext_suffix"]
    command = [str(compiler), "-O3", "-std=c++17", "-shared", "-fPIC", "-DUSE_MACA",
               "--offload-arch=xcore1000", *["-I" + str(p) for p in includes], str(source), "-o", str(destination / filename)]
    header_hashes = {str(path): digest(path.read_bytes()) for path in headers}
    destination.mkdir(mode=0o700)
    with (destination / "command.json").open("x") as output:
        json.dump({"argv": command, "compiler_version": version, "header_sha256": header_hashes}, output, indent=2)
    print("native_build_log=" + str(destination / "build.log"), flush=True)
    compile_command(command, destination, args.timeout)
    require(digest(source.read_bytes()) == SOURCE_SHA256
            and all(digest(Path(path).read_bytes()) == value for path, value in header_hashes.items()),
            "source or headers changed during compilation")
    manifest = {"schema_version": 1, "runtime": runtime, "contract": CONTRACT,
                "source_sha256": SOURCE_SHA256, "library": filename,
                "library_sha256": digest((destination / filename).read_bytes()),
                "compiler_version": version, "header_sha256": header_hashes}
    # Written last. An interrupted build is never eligible for the loader.
    with (destination / "manifest.json").open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print("native_build=COMPLETE; GPU_validation=NOT_PERFORMED; automatically_enabled=NO", flush=True)
    print("artifact_directory=" + str(destination), flush=True)


if __name__ == "__main__":
    main()
