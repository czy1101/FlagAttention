# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter for the optional FlashAttention-3 CUDA extension.

The FA3 kernel is installed in the active Python environment as
``fa3_runtime/_vllm_fa3_C.abi3.so``.  This module loads that binary directly
with ``torch.ops.load_library`` and exposes the paged-cache runner used by the
DiffKV benchmark.  No vLLM Python package is required.

Install the matching FA3 revision in the active Conda environment before
running the benchmark (the build is memory-limited intentionally)::

    conda activate yu
    git clone https://github.com/vllm-project/flash-attention.git
    cd flash-attention
    git checkout f3e1a4f74c99145c0717709860bf765de1703779
    export CUDA_HOME=/home/yuli/cuda-tle-12.8
    export CUDACXX=/root/miniconda3/envs/tle/bin/nvcc
    export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 NINJAFLAGS=-j1
    python -m pip install --no-build-isolation .
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    mkdir -p "$SITE_PACKAGES/fa3_runtime"
    install -m 644 /path/to/_vllm_fa3_C.abi3.so "$SITE_PACKAGES/fa3_runtime/"

The resulting ``_vllm_fa3_C.abi3.so`` must be placed under the active
environment's ``site-packages/fa3_runtime`` directory.  Verify with::

    python -c "from flag_attn.diffkv_attention.FA3 import load_fa3_provider; print(load_fa3_provider('on'))"
"""

from __future__ import annotations

import os
import pathlib
import site
import sys

import torch


def fa3_op_available() -> bool:
    """Return whether the FA3 custom op is registered in this process."""
    try:
        namespace = getattr(torch.ops, "_vllm_fa3_C", None)
        return namespace is not None and hasattr(namespace, "fwd")
    except Exception:
        return False


def _environment_roots() -> list[pathlib.Path]:
    """Return FA3 search roots inside the active Python environment only."""
    environment = pathlib.Path(sys.prefix).resolve()
    roots: list[pathlib.Path] = []
    for directory in site.getsitepackages():
        site_packages = pathlib.Path(directory).resolve()
        try:
            site_packages.relative_to(environment)
        except ValueError:
            continue
        roots.extend((site_packages / "fa3_runtime", site_packages))
    return roots


def extension_candidates(path: str | None = None):
    """Yield FA3 shared libraries installed in the active environment."""
    configured = path or os.environ.get("VLLM_FLASH_ATTN_EXTENSION_DIR")
    if configured:
        roots = [pathlib.Path(configured).expanduser()]
    else:
        roots = _environment_roots()

    environment = pathlib.Path(sys.prefix).resolve()

    seen: set[pathlib.Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        try:
            root.relative_to(environment)
        except ValueError:
            continue
        if root in seen:
            continue
        seen.add(root)
        if root.is_file():
            yield root
            continue
        if not root.is_dir():
            continue
        exact = root / "_vllm_fa3_C.abi3.so"
        if exact.is_file():
            yield exact
        for candidate in sorted(root.glob("*_vllm_fa3_C*.so")):
            if candidate != exact:
                yield candidate


def load_fa3_provider(requested: str = "auto", extension_path: str | None = None):
    """Load FA3 and return ``available/source/error`` status metadata."""
    if requested not in {"auto", "on", "off"}:
        raise ValueError(f"unsupported FA3 mode: {requested!r}")
    if requested == "off":
        return {"available": False, "source": "disabled", "error": None}
    if fa3_op_available():
        return {"available": True, "source": "already-registered", "error": None}

    errors: list[str] = []
    candidates = list(extension_candidates(extension_path))
    configured = extension_path or os.environ.get("VLLM_FLASH_ATTN_EXTENSION_DIR")
    if configured and not candidates:
        errors.append(f"no _vllm_fa3_C*.so found under {configured}")
    elif not configured and not candidates:
        errors.append(
            "no FA3 extension found in the active Python environment; "
            "install _vllm_fa3_C.abi3.so under site-packages/fa3_runtime"
        )
    for candidate in candidates:
        try:
            torch.ops.load_library(str(candidate))
        except Exception as exc:  # pragma: no cover - host ABI dependent
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if fa3_op_available():
            return {
                "available": True,
                "source": str(candidate),
                "error": None,
            }
        errors.append(f"{candidate}: loaded but _vllm_fa3_C.fwd was not registered")

    detail = "; ".join(errors) if errors else "FA3 extension unavailable"
    status = {"available": False, "source": "unavailable", "error": detail}
    if requested == "on":
        raise RuntimeError(
            "FA3 was requested but _vllm_fa3_C.fwd is unavailable: " + detail
        )
    return status


def build_fa3_runner(inputs: tuple[torch.Tensor, ...], window_size: int):
    """Build a direct FA3 paged-cache runner with preallocated buffers."""
    if not fa3_op_available():
        raise RuntimeError("_vllm_fa3_C.fwd is not registered")
    query, key_cache, value_cache, context_lens, block_tables = inputs
    batch, hq, dqk = query.shape
    dv = value_cache.shape[-1]
    seq_len = int(context_lens.max().item())
    cu_seqlens_q = torch.arange(batch + 1, device=query.device, dtype=torch.int32)
    out = torch.empty(batch, hq, dv, device=query.device, dtype=query.dtype)
    scale = dqk**-0.5
    window_left, window_right = (
        (window_size - 1, 0) if window_size > 0 else (-1, -1)
    )

    scheduler_metadata = None
    try:
        get_scheduler_metadata = getattr(
            torch.ops._vllm_fa3_C, "get_scheduler_metadata"
        )
        scheduler_metadata = get_scheduler_metadata(
            batch,
            1,
            seq_len,
            hq,
            key_cache.shape[2],
            dqk,
            dv,
            query.dtype,
            context_lens,
            cu_seqlens_q,
            None,
            None,
            None,
            None,
            key_cache.shape[1],
            0,
            True,
            window_left,
            window_right,
            False,
            0,
            None,
            0,
        )
    except (AttributeError, RuntimeError):
        # Older overlays may not expose scheduler metadata; fwd remains valid
        # and selects its fallback schedule.
        scheduler_metadata = None

    def run():
        torch.ops._vllm_fa3_C.fwd(
            query,
            key_cache,
            value_cache,
            None,
            None,
            None,
            out,
            cu_seqlens_q,
            None,
            None,
            None,
            context_lens,
            1,
            seq_len,
            block_tables,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            scale,
            True,
            window_left,
            window_right,
            0.0,
            True,
            scheduler_metadata,
            0,
            None,
            0,
            None,
            1,
            0,
            None,
        )

    return run


__all__ = [
    "fa3_op_available",
    "extension_candidates",
    "load_fa3_provider",
    "build_fa3_runner",
]
