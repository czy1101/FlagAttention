# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Stateless Hygon decode dispatch using the standard Triton DiffKV kernels."""

from collections import OrderedDict
from functools import lru_cache
from threading import RLock
from typing import Literal

import torch
import triton
import triton.language as tl
from triton import knobs
from triton.backends import BaseBackend
from triton._C.libtriton import native_specialize_impl
from triton.compiler.compiler import make_backend
from triton.runtime import jit as jit_module
from triton.runtime.driver import driver

# Support the upstream module rename without changing the public operator.
try:
    from flag_attn.diffkv import api
    from flag_attn.diffkv.diffkv import (
        _DIFFKV_AUTOTUNE,
        _device_num_sms,
        _fallback_kernel_reduce_segments_diffkv as _generic_reduce,
        _fallback_kernel_unified_attention_diffkv as _generic_attention,
    )
except ModuleNotFoundError as error:
    if error.name != "flag_attn.diffkv":
        raise
    from flag_attn.diffkv_attention import api
    from flag_attn.diffkv_attention.diffkv import (
        _DIFFKV_AUTOTUNE,
        _device_num_sms,
        _fallback_kernel_reduce_segments_diffkv as _generic_reduce,
        _fallback_kernel_unified_attention_diffkv as _generic_attention,
    )


class _LaunchCache:
    """Bounded code-only plans; resolve data pointers and streams per call."""
    def __init__(self, kernel, maxsize=128):
        self.kernel = kernel
        self.fn = kernel.fn
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.backends = {}
        self.lock = RLock()
        self.hits = self.misses = 0

    @staticmethod
    def _key(backend, device, grid, kwargs):
        descriptors, tensors = [], {}
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                spec = tensors.get(id(value))
                if spec is None:
                    spec = native_specialize_impl(backend, value, False, True, True)
                    tensors[id(value)] = spec
                descriptors.append((name, spec))
            else:
                descriptors.append((name, type(value), value))
        return device, grid, tuple(descriptors)

    def __getitem__(self, grid):
        grid = tuple(grid) + (1,) * (3 - len(grid))

        def run(**kwargs):
            prism = getattr(jit_module, '_flagprism', None)
            if prism is not None:
                prism.apply_compile_options(kwargs)
            jit = getattr(self.kernel, 'fn', self.kernel)
            compatible = not (getattr(jit, 'pre_run_hooks', ()) or getattr(jit, 'debug', False) or kwargs.get('debug')
                              or knobs.runtime.debug or knobs.compilation.instrumentation_mode)
            ctx_type = getattr(jit_module, 'DistributedRtContext', None)
            if ctx_type is not None and (ctx_type._init_count or ctx_type().is_lite_mode):
                compatible = False
            if not compatible:
                return self.kernel[grid](**kwargs)
            # Pointer type, alignment and range are compile-time attributes.
            # Actual pointers/data are always supplied by the current call.
            device = driver.active.get_current_device()
            backend = self.backends.get(device, BaseBackend)
            for name, heuristic in self.kernel.values.items():
                kwargs[name] = heuristic(kwargs)
            key = self._key(backend, device, grid, kwargs)
            with self.lock:
                plan = self.cache.get(key)
                if plan is not None:
                    self.cache.move_to_end(key)
            if plan is not None and compatible:
                self.hits += 1
                compiled, runner, names, constants = plan
                missing = object()
                for (name, _), (value, globals_dict) in jit.used_global_vals.items():
                    current = globals_dict.get(name, missing)
                    if current is not value and current != value:
                        raise RuntimeError(f'Kernel global {name} changed after compilation')
                runner(*(kwargs[name] if name in kwargs else constants[index] for index, name in enumerate(names)))
                return compiled
            self.misses += 1
            compiled = self.fn[grid](**kwargs)
            if compiled is None or not hasattr(compiled, 'src'):
                return compiled
            self.backends[device] = make_backend(compiled.metadata.target)
            key = self._key(self.backends[device], device, grid, kwargs)
            if getattr(compiled.metadata, 'tle_node_buffer_bindings', ''):
                return compiled
            names = tuple(compiled.src.fn.arg_names)
            constants = {index[0]: value for index, value in compiled.src.constants.items()}
            if any(isinstance(value, torch.Tensor) for value in constants.values()):
                return compiled
            runner = compiled[grid]
            with self.lock:
                self.cache[key] = (compiled, runner, names, constants)
                self.cache.move_to_end(key)
                while len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            return compiled

        return run


_generic_attention = _LaunchCache(_generic_attention)
_generic_reduce = _LaunchCache(_generic_reduce)


@lru_cache(maxsize=128)
def _segments(capacity: int, programs: int, num_sms: int) -> int:
    """Expose CU parallelism without splitting into tiny KV fragments."""
    if capacity <= api.WORKLOAD.short_k:
        return 1
    parallel = triton.next_power_of_2(triton.cdiv(num_sms, max(programs, 1)))
    useful = 1 << max((max(capacity // 512, 1)).bit_length() - 1, 0)
    return min(parallel, useful, 128)


@torch.no_grad()
def fallback_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    attn_scale: float | None = None,
    window_size: int = -1,
    path: Literal["2d", "3d", "auto"] = "2d",
    num_segments: int | None = None,
    backend: api.DiffKVBackend | None = None,
) -> torch.Tensor:
    """Paged decode with no host reads of device length metadata.

    Explicit 2D/3D requests are retained; a one-segment 3D request writes its
    final output directly. ``auto`` is opt-in and uses the same 3D policy.
    Page-table width is a scheduling upper bound, never the attention mask:
    every kernel reads the current context length from the input tensor.
    No input-dependent tensor or mutable workspace is cached between calls.
    """
    selected = api._resolve_backend(backend)
    normalized = path.strip().lower()
    if selected != "triton" or _DIFFKV_AUTOTUNE:
        if normalized == "auto":
            normalized = "2d"
        return api.diffkv_attention(
            query, key_cache, value_cache, context_lens, block_tables,
            attn_scale=attn_scale, window_size=window_size, path=normalized,
            num_segments=num_segments, backend=selected,
        )
    if normalized not in {"2d", "3d", "auto"}:
        raise ValueError("path must be 2d, 3d, or auto")
    batch, hq, dq, dv = api._validate_public_inputs(
        query, key_cache, value_cache, context_lens, block_tables, window_size
    )
    hkv, block_size = key_cache.shape[2], key_cache.shape[1]
    capacity = block_tables.shape[1] * block_size
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(dtype=torch.int32)
    if block_tables.dtype != torch.int32:
        block_tables = block_tables.to(dtype=torch.int32)
    out = torch.empty((batch, hq, dv), device=query.device, dtype=query.dtype)
    if batch == 0:
        return out
    segments = 1
    if normalized != "2d":
        if num_segments is None:
            # The shared kernel partitions the full context before window
            # masking. A truncated window would otherwise leave empty splits.
            segments = (
                1 if 0 < window_size < capacity
                else _segments(capacity, batch * hkv, _device_num_sms(query.device))
            )
        else:
            if not isinstance(num_segments, int) or num_segments < 1 or num_segments & (num_segments - 1):
                raise ValueError("num_segments must be a positive power of two")
            segments = num_segments
    use_3d = segments > 1
    medium = api.WORKLOAD.short_k < capacity <= api.WORKLOAD.medium_k
    tile = 32 if not use_3d or (medium and batch >= 8) else 16
    split_qk = use_3d and medium and batch <= 8 and dq == 192 and dv == 128
    group = hq // hkv
    attention_warps = 4
    dedup = split_qk and tile == 2 * block_size
    # The 192-wide Full/GQA16 layout benefits from two exact dot products
    # rather than padding QK to 256. Keep small/windowed/other layouts on
    # the existing path; their public-call cost is mostly host dispatch.
    if (
        window_size == -1 and dq == 192 and dv == 128 and group == 16
        and block_size == 16 and capacity > api.WORKLOAD.short_k
    ):
        tile = 32
        split_qk = True
        # A populated single-segment grid profits from page-id broadcast;
        # fewer waves avoid excessive LDS exchange in the remaining grids.
        populated = batch * hkv >= _device_num_sms(query.device)
        attention_warps = 4 if not use_3d and populated else 2
        dedup = attention_warps == 4
    block_m = max(16, triton.next_power_of_2(group))
    if use_3d:
        partial = torch.empty((batch, hq, segments, triton.next_power_of_2(dv)),
                              device=query.device, dtype=torch.float32)
        maxima = torch.empty((batch, hq, segments), device=query.device, dtype=torch.float32)
        sums = torch.empty_like(maxima)
    else:
        partial = maxima = sums = out
    _generic_attention[(batch, hkv, segments)](
        output_ptr=out, segm_output_ptr=partial, segm_max_ptr=maxima,
        segm_expsum_ptr=sums, query_ptr=query, key_cache_ptr=key_cache,
        value_cache_ptr=value_cache, sink_ptr=None, block_tables_ptr=block_tables,
        seq_lens_ptr=context_lens, alibi_slopes_ptr=None,
        scale=float(attn_scale) if attn_scale is not None else dq**-0.5,
        softcap=0.0, num_query_heads=hq, num_queries_per_kv=group,
        block_table_stride=block_tables.stride(0), query_stride_0=query.stride(0),
        query_stride_1=query.stride(1), output_stride_0=out.stride(0),
        output_stride_1=out.stride(1), BLOCK_SIZE=block_size, TILE_SIZE=tile,
        HEAD_SIZE_QK=dq, USE_SPLIT_QK=split_qk, HEAD_SIZE_V=dv,
        USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_SOFTCAP=False,
        USE_SINKS=False, SLIDING_WINDOW=max(window_size, 0),
        stride_k_cache_0=key_cache.stride(0), stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2), stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0), stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2), stride_v_cache_3=value_cache.stride(3),
        # IS_DECODE eliminates all loads through this otherwise required pointer.
        query_start_len_ptr=context_lens, BLOCK_Q=1, num_seqs=batch, BLOCK_M=block_m,
        NUM_SEGMENTS_PER_SEQ=segments, IS_3D=use_3d, IS_DECODE=True,
        SPLIT_HEADS=False, DEDUP_BLOCK_TABLE=dedup,
        num_warps=attention_warps, num_stages=2,
    )
    if use_3d:
        reduce_warps = 2 if medium and batch >= 8 else 1 if capacity <= 1024 and batch >= 8 else 4
        _generic_reduce[(batch, hq)](
            output_ptr=out, segm_output_ptr=partial, segm_max_ptr=maxima,
            segm_expsum_ptr=sums, seq_lens_ptr=context_lens, num_seqs=batch,
            num_query_heads=hq, output_stride_0=out.stride(0), output_stride_1=out.stride(1),
            TILE_SIZE=tile, HEAD_SIZE_V=dv, query_start_len_ptr=context_lens,
            BLOCK_Q=1, NUM_SEGMENTS_PER_SEQ=segments, IS_DECODE=True,
            num_warps=reduce_warps, num_stages=2,
        )
    return out


@triton.heuristics(values={"kpack": lambda args: 2})
@triton.jit
def wide_attention(Q, K, V, Lens, Table, Out, Partial, Maxima, Sums,
              STRIDES: tl.constexpr, HQ: tl.constexpr, GROUP: tl.constexpr,
              SCALE: tl.constexpr, WINDOW: tl.constexpr, SEGMENTS: tl.constexpr,
              TILE: tl.constexpr, STAGES: tl.constexpr, WAVES: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.program_id(2)
    qb: tl.constexpr = STRIDES[0]
    qh: tl.constexpr = STRIDES[1]
    kb: tl.constexpr = STRIDES[2]
    kt: tl.constexpr = STRIDES[3]
    kh: tl.constexpr = STRIDES[4]
    vb: tl.constexpr = STRIDES[5]
    vt: tl.constexpr = STRIDES[6]
    vh: tl.constexpr = STRIDES[7]
    ts: tl.constexpr = STRIDES[8]
    length = tl.load(Lens + b)
    begin_window = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    tiles = tl.cdiv(length, TILE)
    start = tl.maximum(tiles * seg // SEGMENTS, begin_window // TILE)
    stop = tiles * (seg + 1) // SEGMENTS
    w = tl.arange(0, WAVES)
    m = tl.arange(0, 16)
    d0 = tl.arange(0, 128)
    d1 = tl.arange(0, 64)
    n = tl.arange(0, TILE)
    heads = h * GROUP + m
    q0 = tl.load(Q + b * qb + heads[:, None] * qh + d0[None, :], m[:, None] < GROUP, 0)
    q1 = tl.load(Q + b * qb + heads[:, None] * qh + 128 + d1[None, :], m[:, None] < GROUP, 0)
    q0 = tl.broadcast_to(q0[None, :, :], (WAVES,16,128))
    q1 = tl.broadcast_to(q1[None, :, :], (WAVES,16,64))
    maximum = tl.full((WAVES,16), -float('inf'), tl.float32)
    denominator = tl.full((WAVES,16), 0., tl.float32)
    acc = tl.full((WAVES,16,128), 0., tl.float32)
    for offset in tl.range(start, stop, WAVES, num_stages=STAGES):
        tile = offset + w
        tokens = tile[:, None] * TILE + n[None, :]
        valid = (tokens < length) & (tokens >= begin_window) & (tile[:, None] < stop)
        pages = tile[:, None] * (TILE // 16) + tl.arange(0, TILE // 16)[None,:]
        ids = tl.load(Table + b * ts + pages, (pages < tl.cdiv(length, 16)) & (tile[:,None] < stop), 0)
        ids = tl.broadcast_to(ids[:,:,None], (WAVES,TILE // 16,16)).reshape(WAVES,TILE)
        base_k = ids.to(tl.int64) * kb + (n[None,:] % 16) * kt + h * kh
        k0 = tl.load(K + base_k[:,None,:] + d0[None,:,None], valid[:,None,:], 0)
        k1 = tl.load(K + base_k[:,None,:] + 128 + d1[None,:,None], valid[:,None,:], 0)
        scores = tl.dot(q0, k0)
        scores = tl.dot(q1, k1, scores) * (SCALE * 1.4426950408889634)
        mask = (m[None,:,None] < GROUP) & valid[:,None,:]
        scores = tl.where(mask, scores, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 2))
        alpha = tl.where(denominator > 0, tl.exp2(maximum - new_maximum), 0.)
        probabilities = tl.where(mask, tl.exp2(scores - new_maximum[:,:,None]), 0.)
        denominator = denominator * alpha + tl.sum(probabilities, 2)
        acc = acc * alpha[:,:,None]
        base_v = ids.to(tl.int64) * vb + (n[None,:] % 16) * vt + h * vh
        values = tl.load(V + base_v[:,:,None] + d0[None,None,:], valid[:,:,None], 0)
        acc = tl.dot(probabilities.to(values.dtype), values, acc)
        maximum = new_maximum
    merged_max = tl.max(maximum,0)
    weights = tl.where(denominator > 0, tl.exp2(maximum-merged_max[None,:]),0.)
    merged_sum = tl.sum(denominator*weights,0)
    merged_acc = tl.sum(acc*weights[:,:,None],0)
    bh = b * HQ + heads
    if SEGMENTS > 1:
        tl.store(Partial + (bh[:, None] * SEGMENTS + seg) * 128 + d0[None, :], merged_acc, m[:, None] < GROUP)
        tl.store(Maxima + bh * SEGMENTS + seg, merged_max, m < GROUP)
        tl.store(Sums + bh * SEGMENTS + seg, merged_sum, m < GROUP)
    else:
        result = merged_acc / tl.where(merged_sum > 0, merged_sum, 1.)[:, None]
        tl.store(Out + bh[:, None] * 128 + d0[None, :], result, m[:, None] < GROUP)


@triton.heuristics(values={})
@triton.jit
def attention(Q, K, V, Lens, Table, Out, Partial, Maxima, Sums,
              STRIDES: tl.constexpr, HQ: tl.constexpr, GROUP: tl.constexpr,
              SCALE: tl.constexpr, WINDOW: tl.constexpr, SEGMENTS: tl.constexpr,
              TILE: tl.constexpr, STAGES: tl.constexpr, WAVES: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seg = tl.program_id(2)
    qb: tl.constexpr = STRIDES[0]
    qh: tl.constexpr = STRIDES[1]
    kb: tl.constexpr = STRIDES[2]
    kt: tl.constexpr = STRIDES[3]
    kh: tl.constexpr = STRIDES[4]
    vb: tl.constexpr = STRIDES[5]
    vt: tl.constexpr = STRIDES[6]
    vh: tl.constexpr = STRIDES[7]
    ts: tl.constexpr = STRIDES[8]
    length = tl.load(Lens + b)
    begin_window = tl.maximum(0, length - WINDOW) if WINDOW > 0 else 0
    tiles = tl.cdiv(length, TILE)
    start = tl.maximum(tiles * seg // SEGMENTS, begin_window // TILE)
    stop = tiles * (seg + 1) // SEGMENTS
    m = tl.arange(0, 16)
    d0 = tl.arange(0, 128)
    d1 = tl.arange(0, 64)
    n = tl.arange(0, TILE)
    heads = h * GROUP + m
    q0 = tl.load(Q + b * qb + heads[:, None] * qh + d0[None, :], m[:, None] < GROUP, 0)
    q1 = tl.load(Q + b * qb + heads[:, None] * qh + 128 + d1[None, :], m[:, None] < GROUP, 0)
    maximum = tl.full((16,), -float('inf'), tl.float32)
    denominator = tl.full((16,), 0., tl.float32)
    acc = tl.full((16, 128), 0., tl.float32)
    for tile in tl.range(start, stop, num_stages=STAGES):
        tokens = tile * TILE + n
        valid = (tokens < length) & (tokens >= begin_window)
        pages = tile * (TILE // 16) + tl.arange(0, TILE // 16)
        ids = tl.load(Table + b * ts + pages, pages < tl.cdiv(length, 16), 0)
        ids = tl.broadcast_to(ids[:, None], (TILE // 16, 16)).reshape(TILE)
        base_k = ids.to(tl.int64) * kb + (n % 16) * kt + h * kh
        k0 = tl.load(K + base_k[None, :] + d0[:, None], valid[None, :], 0)
        k1 = tl.load(K + base_k[None, :] + 128 + d1[:, None], valid[None, :], 0)
        scores = tl.dot(q0, k0)
        scores = tl.dot(q1, k1, scores) * (SCALE * 1.4426950408889634)
        mask = (m[:, None] < GROUP) & valid[None, :]
        scores = tl.where(mask, scores, -float('inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.where(denominator > 0, tl.exp2(maximum - new_maximum), 0.)
        probabilities = tl.where(mask, tl.exp2(scores - new_maximum[:, None]), 0.)
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        acc = acc * alpha[:, None]
        base_v = ids.to(tl.int64) * vb + (n % 16) * vt + h * vh
        values = tl.load(V + base_v[:, None] + d0[None, :], valid[:, None], 0)
        acc = tl.dot(probabilities.to(values.dtype), values, acc)
        maximum = new_maximum
    bh = b * HQ + heads
    if SEGMENTS > 1:
        tl.store(Partial + (bh[:, None] * SEGMENTS + seg) * 128 + d0[None, :], acc, m[:, None] < GROUP)
        tl.store(Maxima + bh * SEGMENTS + seg, maximum, m < GROUP)
        tl.store(Sums + bh * SEGMENTS + seg, denominator, m < GROUP)
    else:
        result = acc / tl.where(denominator > 0, denominator, 1.)[:, None]
        tl.store(Out + bh[:, None] * 128 + d0[None, :], result, m[:, None] < GROUP)


@triton.heuristics(values={})
@triton.jit
def reduce(Out, Partial, Maxima, Sums, SEGMENTS: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.arange(0, SEGMENTS)
    d = tl.arange(0, 128)
    maxima = tl.load(Maxima + bh * SEGMENTS + s)
    sums = tl.load(Sums + bh * SEGMENTS + s)
    maximum = tl.max(maxima, 0)
    weights = tl.where(sums > 0, tl.exp2(maxima - maximum), 0.)
    denominator = tl.sum(sums * weights, 0)
    values = tl.load(Partial + (bh * SEGMENTS + s[:, None]) * 128 + d[None, :])
    result = tl.sum(values * weights[:, None], 0) / tl.where(denominator > 0, denominator, 1.)
    tl.store(Out + bh * 128 + d, result)


class _ExecutionPlan(_LaunchCache):
    """Code-only positional plans; share specialization only within one call."""

    @staticmethod
    @lru_cache(maxsize=128)
    def _constants_key(constants):
        return tuple((type(value), value) for value in constants)

    def launch(self, grid, tensors, constants, warps, stages, state):
        grid = tuple(grid) + (1,) * (3 - len(grid))
        jit, runtime_knobs = self.fn, knobs
        ctx_type = getattr(jit_module, 'DistributedRtContext', None)
        prism = getattr(jit_module, '_flagprism', None)
        enter, exit = runtime_knobs.runtime.launch_enter_hook, runtime_knobs.runtime.launch_exit_hook
        # The general launcher retains dynamic heuristics and instrumentation.
        compatible = not (self.kernel.values or jit.pre_run_hooks or jit.debug
                          or runtime_knobs.runtime.debug or runtime_knobs.compilation.instrumentation_mode
                          or (enter is not None and (type(enter) is not runtime_knobs.HookChain or enter.calls))
                          or (exit is not None and (type(exit) is not runtime_knobs.HookChain or exit.calls))
                          or (ctx_type is not None and (ctx_type._init_count or ctx_type().is_lite_mode))
                          or any(type(tensor) is not torch.Tensor for tensor in tensors)
                          or (prism is not None and
                              (not hasattr(prism, '_registered_callbacks')
                               or prism._registered_callbacks('apply_compile_options'))))
        args = tensors + constants
        if not compatible:
            kwargs = dict(zip(jit.arg_names, args))
            return self[grid](**kwargs, num_warps=warps, num_stages=stages)
        if not state:
            device = driver.active.get_current_device()
            state.extend((device, driver.active.get_current_stream(device), {}))
        device, stream, specializations = state
        backend = self.backends.get(device, BaseBackend)

        def key_for(backend):
            specs = []
            backend_type = backend if isinstance(backend, type) else type(backend)
            buffer_ops = runtime_knobs.hcu.use_buffer_ops
            for tensor in tensors:
                identity = (backend_type, buffer_ops, id(tensor))
                spec = specializations.get(identity)
                if spec is None:
                    spec = native_specialize_impl(backend, tensor, False, True, True)
                    specializations[identity] = spec
                specs.append(spec)
            return (device, grid, tuple(specs),
                    self._constants_key(constants), (warps, stages))

        key = key_for(backend)
        with self.lock:
            plan = self.cache.get(key)
            if plan is not None:
                self.cache.move_to_end(key)
        if plan is not None:
            self.hits += 1
            missing = object()
            for (name, _), (value, globals_dict) in jit.used_global_vals.items():
                current = globals_dict.get(name, missing)
                if current is not value and current != value:
                    raise RuntimeError(f'Kernel global {name} changed after compilation')
            plan[1](*args, stream=stream)
            return plan[0]
        self.misses += 1
        options = dict(num_warps=warps, num_stages=stages)
        compiled = self.fn[grid](**dict(zip(jit.arg_names, args)), **options)
        if compiled is None or not hasattr(compiled, 'src'):
            return compiled
        self.backends[device] = make_backend(compiled.metadata.target)
        if getattr(compiled.metadata, 'tle_node_buffer_bindings', ''):
            return compiled
        compiled_constants = {index[0]: value for index, value in compiled.src.constants.items()}
        if any(isinstance(value, torch.Tensor) for value in compiled_constants.values()):
            return compiled
        key = key_for(self.backends[device])
        plan = (compiled, compiled[grid], tuple(jit.arg_names), compiled_constants)
        with self.lock:
            self.cache[key] = plan
            self.cache.move_to_end(key)
            while len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)
        return compiled


_attention = _ExecutionPlan(attention)
_wide_attention = _LaunchCache(wide_attention)
_reduce = _ExecutionPlan(reduce)
