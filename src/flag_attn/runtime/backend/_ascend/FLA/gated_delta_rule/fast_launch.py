# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead launcher for this backend's Triton kernels.

The GDN fast path is host-bound on small shapes: four ``kernel[grid](...)``
calls per forward, and every call re-runs Triton's Python side (option/kwargs
parsing, device/stream lookup, generated binder, cache-key build, global-val
check, grid canonicalization).  On 910B that is ~0.06-0.10 ms per launch while
the actual C launch is only ~0.02-0.03 ms.  This module caches the launch
decision per ``(JITFunction, grid, options, argument specialization)`` and then
calls the compiled launcher directly.

Scope: only kernels whose module lives under
``flag_attn.runtime.backend._ascend`` are accelerated; every other Triton
kernel, all warmup/compile calls and every call that has a pre-run hook or a
launch hook installed fall back to the regular ``JITFunction.run``.  Disable
the whole mechanism with ``FLAG_ATTN_ASCEND_GDN_FASTLAUNCH=0``.

The cache key is built from launch-relevant facts only -- tensor dtype and
``data_ptr() % 16`` alignment, non-specialized scalars, grid and launch
options -- so calls with *new* input tensors of the same shape/dtype hit the
cache and are launched with those tensors; no fixed-address assumption.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.runtime.jit as tj

_ORIG_RUN = tj.JITFunction.run
_PKG_PREFIX = "flag_attn.runtime.backend._ascend"
_OFF = {"0", "false", "off", "no"}
_CACHE: dict = {}
_SKIP_IDX: dict = {}
_INSTALLED = False

try:  # pragma: no cover - availability depends on the torch_npu build
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _raw_stream
    _HAS_RAW_STREAM = True
except Exception:  # noqa: BLE001
    _raw_stream = None
    _HAS_RAW_STREAM = False

try:  # pragma: no cover
    from torch_npu._C import _npu_getDevice as _get_device
except Exception:  # noqa: BLE001
    _get_device = None

_LAUNCH_HOOKS = (None, None)


def _enabled() -> bool:
    return os.environ.get(
        "FLAG_ATTN_ASCEND_GDN_FASTLAUNCH", "1"
    ).lower() not in _OFF


def _device() -> int:
    if _get_device is not None:
        try:
            return _get_device()
        except Exception:  # noqa: BLE001
            pass
    return torch.npu.current_device()


def _stream(dev: int) -> int:
    if _HAS_RAW_STREAM:
        return _raw_stream(dev)
    from triton.runtime.driver import driver
    return driver.active.get_current_stream(dev)


def _skip_indices(fn) -> tuple:
    idx = _SKIP_IDX.get(id(fn))
    if idx is None:
        params = getattr(fn, "params", ())
        idx = tuple(
            i for i, p in enumerate(params) if getattr(p, "do_not_specialize", False)
        )
        _SKIP_IDX[id(fn)] = idx
    return idx


def _hooks_active() -> bool:
    """True when a profiler/trace hook is installed (fast path is unsafe then).

    A cached low-level launch passes ``launch_metadata=None``; a real profiler
    hook expects the metadata object, so any registered hook must fall back to
    the regular ``JITFunction.run``.  ``HookChain.calls`` is the registered
    callback list (empty by default, so this is a cheap list check).
    """
    try:
        runtime = triton.knobs.runtime
        if getattr(runtime.launch_enter_hook, "calls", None):
            return True
        if getattr(runtime.launch_exit_hook, "calls", None):
            return True
        if getattr(runtime, "debug", False):
            return True
    except Exception:  # noqa: BLE001 - unknown knob layout: be conservative
        return True
    return False


def _cache_key(fn, vals, grid, opts, skip, dev):
    # The cached entry holds a device-specific function handle and stream, so
    # the device is part of the key (multi-card safety).  ``vals`` is the
    # canonical positional argument list (kwargs already folded in); tensors are
    # represented by dtype + 16-byte alignment, never by object identity.
    key = [id(fn), dev, len(vals), tuple(grid), tuple(opts.items())]
    is_tensor = torch.Tensor
    for i, a in enumerate(vals):
        if i in skip:
            continue
        if type(a) is is_tensor:
            key.append((a.dtype, a.data_ptr() & 15))
        else:
            key.append(a)
    return tuple(key)


def _canonicalize(self, args, kwargs):
    """Fold keyword kernel arguments into the canonical positional list.

    The cached low-level launcher always takes every kernel argument
    positionally, so call sites that pass some parameters by keyword (e.g.
    ``chunk_o_ascend``) must be canonicalized first; otherwise the launcher
    receives too few arguments.  Launch options (``num_warps`` ...) stay in a
    separate dict.
    """
    names = getattr(self, "arg_names", None)
    if not names:
        return None
    n = len(names)
    if len(args) > n:
        return None
    vals = list(args)
    for j in range(len(vals), n):
        name = names[j]
        if name in kwargs:
            vals.append(kwargs[name])
        else:
            # Missing trailing parameters would need Triton's own default
            # handling: let the regular path deal with those rare calls.
            return None
    opts = {k: v for k, v in kwargs.items() if k not in names}
    return vals, opts


def _fast_run(self, *args, grid, warmup, **kwargs):
    if warmup or not _enabled():
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    if getattr(self, "pre_run_hooks", None):
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    fn = getattr(self, "fn", None)
    if fn is None or not (fn.__module__ or "").startswith(_PKG_PREFIX):
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    if callable(grid) or getattr(self, "used_global_vals", None):
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    if _hooks_active():
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    canon = _canonicalize(self, args, kwargs)
    if canon is None:
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    vals, opts = canon
    dev = _device()
    try:
        key = _cache_key(self, vals, grid, opts, _skip_indices(self), dev)
    except TypeError:
        return _ORIG_RUN(self, *args, grid=grid, warmup=warmup, **kwargs)
    ent = _CACHE.get(key)
    if ent is None:
        kern = _ORIG_RUN(self, *args, grid=grid, warmup=False, **kwargs)
        if hasattr(kern, "result"):
            kern = kern.result()
        if kern is None:
            return None
        ent = (
            kern.run.launch,
            kern.function,
            kern.packed_metadata,
            grid[0],
            grid[1] if len(grid) > 1 else 1,
            grid[2] if len(grid) > 2 else 1,
            dev,
        )
        _CACHE[key] = ent
        return kern
    launch, function, packed, g0, g1, g2, dev = ent
    # No launch hooks can be registered here (checked above before the key
    # lookup), so pass None for the metadata/hook slots like the compiled
    # launcher does when nothing is profiling.
    launch(g0, g1, g2, _stream(dev), function, packed, None, None, None, *vals)
    return None


def install() -> bool:
    """Patch ``JITFunction.run`` for this backend's kernels (idempotent)."""
    global _INSTALLED
    if _INSTALLED:
        return True
    tj.JITFunction.run = _fast_run
    _INSTALLED = True
    return True


def uninstall() -> None:
    global _INSTALLED
    if _INSTALLED:
        tj.JITFunction.run = _ORIG_RUN
        _INSTALLED = False
        _CACHE.clear()




# ---------------------------------------------------------------------------
# Per-site prepared launchers (forward-scoped stream)
# ---------------------------------------------------------------------------
# ``_fast_run`` still rebuilds a launch signature on every call.  The GDN fast
# path knows its (shape, site) ahead of time, so the launch decision can be
# cached in the existing per-shape scratch entry instead: one dict lookup per
# launch, no kwargs tuple and no ``_device()`` call.  The current stream is
# looked up once per forward (``begin_forward``) instead of once per launch.
#
# The guards are the same as ``_fast_run``: a launch hook or ``debug`` falls
# back to the regular ``JITFunction.run`` per launch, ``FLAG_..._FASTLAUNCH=0``
# disables the whole mechanism, and calls outside a forward scope keep the old
# per-call behaviour.

_SCOPE = None


def begin_forward() -> None:
    """Cache the device + current stream once for the whole forward call."""
    global _SCOPE
    if not _enabled():
        _SCOPE = ()
        return
    _SCOPE = (_stream(_device()),)


def end_forward() -> None:
    global _SCOPE
    _SCOPE = None


class FastKernel:
    """Prepared low-level launch for one ``(JITFunction, grid, options)`` site."""

    __slots__ = ("fn", "grid", "options", "dev", "launch", "function",
                 "packed", "g0", "g1", "g2")

    def __init__(self, fn, grid, **options):
        self.fn = fn
        self.grid = grid
        self.options = options
        self.launch = None

    def prepare(self, *args):
        """Compile (or resolve) the kernel for this argument set and cache it."""
        kern = _ORIG_RUN(self.fn, *args, grid=self.grid, warmup=False,
                         **self.options)
        if hasattr(kern, "result"):
            kern = kern.result()
        if kern is None:
            return None
        grid = self.grid
        self.dev = _device()
        self.launch = kern.run.launch
        self.function = kern.function
        self.packed = kern.packed_metadata
        self.g0 = grid[0]
        self.g1 = grid[1] if len(grid) > 1 else 1
        self.g2 = grid[2] if len(grid) > 2 else 1
        return kern

    def __call__(self, *args):
        if self.launch is None:
            # First call: the regular path compiles *and* launches with these
            # arguments, so the kernel has already run.
            self.prepare(*args)
            return None
        if _hooks_active():
            return _ORIG_RUN(self.fn, *args, grid=self.grid, warmup=False,
                             **self.options)
        scope = _SCOPE
        if scope is None:
            if not _enabled():
                return _ORIG_RUN(self.fn, *args, grid=self.grid, warmup=False,
                                 **self.options)
            stream = _stream(self.dev)
        elif scope:
            stream = scope[0]
        else:  # inside a forward scope with the launcher disabled
            return _ORIG_RUN(self.fn, *args, grid=self.grid, warmup=False,
                             **self.options)
        self.launch(self.g0, self.g1, self.g2, stream, self.function,
                    self.packed, None, None, None, *args)
        return None


def site_launcher(cache: dict, key, fn, grid, **options) -> FastKernel:
    """Return the cached per-site ``FastKernel`` handle for one shape."""
    lk = cache.get(key)
    if lk is None:
        lk = FastKernel(fn, grid, **options)
        cache[key] = lk
    return lk

__all__ = ["install", "uninstall", "FastKernel", "site_launcher", "begin_forward", "end_forward"]
