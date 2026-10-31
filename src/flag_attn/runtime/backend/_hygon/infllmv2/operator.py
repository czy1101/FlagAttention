# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Public packed attention/decode dispatch and host execution plans."""

from __future__ import annotations

import math
import os
import weakref
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import fields
from typing import TYPE_CHECKING

import torch
import triton
from triton.runtime.driver import driver

from flag_attn import flash, split_kv

from . import forward as _runtime
from .forward import (
    DistributedRtContext,
    _flagprism,
    compress_k,
    decode_cu,
    get_cache_invalidating_env_vars,
    knobs,
    pool_scores,
    select_blocks,
    sparse_attention,
    stage1,
)

if TYPE_CHECKING:
    from flag_attn.infllmv2.operator import InfLLMV2Config

_dense_fn = flash._fwd_kernel
_dense_get_config = flash.get_fwd_config
_dense_get_split_config = split_kv.get_fwd_config
_dense_get_splits = split_kv.num_splits_herustic
_dense_lengths = OrderedDict()
_dense_plans = OrderedDict()
_dense_targets = {}
_dense_field_names = {}
_dense_stats = {"length_hits": 0, "length_reads": 0, "builds": 0, "hits": 0, "metadata_only": 0}
_dense_missing = object()


def _dense_clear_caches():
    _dense_lengths.clear()
    _dense_plans.clear()


def _dense_runtime_stats():
    return dict(_dense_stats, plans=len(_dense_plans), hashes=sorted({p.compiled.hash for p in _dense_plans.values()}))


def _dense_has_hook(hook):
    return bool(hook.calls) if type(hook) is knobs.HookChain else hook is not None


def _dense_context(q, k, v, cuq, cuk):
    if triton.__version__ != "3.6.0":
        return None
    if any(type(t) is not torch.Tensor or not t.is_contiguous() for t in (q, k, v, cuq, cuk)):
        return None
    if cuq.shape != (2,) or cuk.shape != (2,):
        return None
    if _dense_has_hook(knobs.runtime.launch_enter_hook) or _dense_has_hook(knobs.runtime.launch_exit_hook):
        return None
    if _dense_fn.pre_run_hooks or _dense_fn.debug or knobs.runtime.debug or knobs.compilation.instrumentation_mode:
        return None
    if _flagprism._registered_components():
        return None
    if (
        flash._fwd_kernel is not _dense_fn
        or flash.get_fwd_config is not _dense_get_config
        or split_kv.get_fwd_config is not _dense_get_split_config
        or (split_kv.num_splits_herustic is not _dense_get_splits)
    ):
        return None
    if torch.cuda.current_device() != q.device.index or torch.cuda.is_current_stream_capturing():
        return None
    if DistributedRtContext().is_lite_mode:
        return None
    device = q.device.index
    target = _dense_targets.get(device)
    if target is None:
        target = driver.active.get_current_target()
        _dense_targets[device] = target
    if target.backend != "hip" or target.arch != "gfx936" or target.warp_size != 64:
        return None
    return (torch.cuda.current_stream(q.device).cuda_stream, tuple(sorted(get_cache_invalidating_env_vars().items())))


def _dense_length_values(cu, stream):
    try:
        version = cu._version
    except RuntimeError:
        version = None
    key = (id(cu), version, cu.device.index, stream)
    cached = _dense_lengths.get(key) if version is not None else None
    if cached is not None and cached[0]() is cu:
        _dense_lengths.move_to_end(key)
        _dense_stats["length_hits"] += 1
        return cached[1]
    values = tuple(cu.detach().cpu().tolist())
    _dense_stats["length_reads"] += 1
    if version is not None:
        _dense_lengths[key] = (weakref.ref(cu), values)
        while len(_dense_lengths) > 64:
            _dense_lengths.popitem(last=False)
    return values


def _dense_key(q, k, v, cfg, scale, stream, env):
    names = _dense_field_names.get(type(cfg))
    if names is None:
        names = tuple(f.name for f in fields(cfg))
        _dense_field_names[type(cfg)] = names
    values = tuple(getattr(cfg, n) for n in names)
    if any(type(x) not in (bool, int, float, str, bytes, type(None)) for x in values):
        return None
    if type(scale) not in (float, int, type(None)):
        return None
    traits = tuple(
        (t.dtype, t.device, tuple(t.shape), t.data_ptr() % 16 == 0, t.untyped_storage().size() <= 2**31 - 1)
        for t in (q, k, v)
    )
    return (traits, tuple((type(x), x) for x in values), (type(scale), scale), stream, env)


class _DensePlan:
    def __init__(self, q, k, v, scale, stream):
        bm, bn, stages, warps = _dense_get_config(1, 32, 1024, 1024, 64, True)
        sk = _dense_get_split_config(1, 32, 1024, 1024, 64, True)
        splits = _dense_get_splits(
            1, 32, 1024, 1024, sk[0], sk[1], torch.cuda.get_device_properties(q.device).multi_processor_count, 128
        )
        if splits != 1 or (bm, bn, stages, warps) != (32, 32, 1, 4):
            raise ValueError("unsupported Dense launch configuration")
        self.grid = (32, 32, 1)
        out, lse = self.allocate(q)
        qq, kk, vv, oo = [t.transpose(0, 1).unsqueeze(0) for t in (q, k, v, out)]
        positional = (
            qq,
            kk,
            vv,
            64 ** (-0.5) if scale is None else scale,
            0.0,
            0,
            0,
            lse,
            oo,
            *qq.stride(),
            *kk.stride(),
            *vv.stride(),
            *oo.stride(),
            1,
            32,
            1024,
            1024,
            0,
            16,
        )
        options = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_DMODEL": 64,
            "IS_CAUSAL": True,
            "IS_DROPOUT": False,
            "LARGER_M": False,
            "DIVISIBLE_M": True,
            "DIVISIBLE_N": True,
            "num_warps": warps,
            "num_stages": stages,
        }
        self.compiled = _dense_fn[self.grid](*positional, **options)
        self.compiled._init_handles()
        names = tuple(_dense_fn.signature.parameters)
        values = positional + tuple(options[n] for n in names[len(positional) :])
        template = list(values)
        for index in (0, 1, 2, 7, 8):
            template[index] = None
        self.template = tuple(template)
        self.launch = self.compiled.run
        self.function = self.compiled.function
        self.packed = self.compiled.packed_metadata
        self.first_result = out

    @staticmethod
    def allocate(q):
        return (
            torch.empty((1024, 32, 64), device=q.device, dtype=q.dtype),
            torch.empty((1, 32, 1024), device=q.device, dtype=torch.float32),
        )

    def execute(self, q, k, v, stream):
        for (name, _), (old, namespace) in _dense_fn.used_global_vals.items():
            if namespace.get(name, _dense_missing) != old:
                raise RuntimeError(f"Global variable {name} changed after kernel compilation")
        out, lse = self.allocate(q)
        args = list(self.template)
        args[0:3] = (q, k, v)
        args[7] = lse
        args[8] = out
        self.launch(
            *self.grid,
            stream,
            self.function,
            self.packed,
            None,
            knobs.runtime.launch_enter_hook,
            knobs.runtime.launch_exit_hook,
            *args,
        )
        return out


def _dense_run(q, k, v, cfg, scale, stream, env):
    if os.environ.get("FLAG_ATTN_INFLLMV2_V7_PLAN", "1") != "1":
        _dense_stats["metadata_only"] += 1
        return (
            flash.attention(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)), causal=True, sm_scale=scale)
            .squeeze(0)
            .transpose(0, 1)
        )
    key = _dense_key(q, k, v, cfg, scale, stream, env)
    if key is None:
        return (
            flash.attention(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)), causal=True, sm_scale=scale)
            .squeeze(0)
            .transpose(0, 1)
        )
    plan = _dense_plans.get(key)
    if plan is None:
        try:
            plan = _DensePlan(q, k, v, scale, stream)
        except ValueError:
            return (
                flash.attention(*(t.transpose(0, 1).unsqueeze(0) for t in (q, k, v)), causal=True, sm_scale=scale)
                .squeeze(0)
                .transpose(0, 1)
            )
        out = plan.first_result
        del plan.first_result
        _dense_plans[key] = plan
        _dense_stats["builds"] += 1
        while len(_dense_plans) > 32:
            _dense_plans.popitem(last=False)
        return out
    _dense_plans.move_to_end(key)
    _dense_stats["hits"] += 1
    return plan.execute(q, k, v, stream)


_decode_plans = OrderedDict()
_decode_stats = {"builds": 0, "hits": 0, "fallbacks": 0}
_decode_missing = object()
_decode_fields = {}


def _clear_decode_plans():
    _decode_plans.clear()


def _decode_plan_stats():
    return dict(
        _decode_stats,
        plans=len(_decode_plans),
        hashes=sorted({c.hash for p in _decode_plans.values() for _, c, _, _ in p.records}),
    )


def _decode_key(q, k, v, cu, maxk, config, scale, scope, lengths):
    if _flagprism._registered_components():
        return None
    if knobs.compilation.instrumentation_mode or knobs.runtime.debug:
        return None
    traits = []
    for t in (q, k, v, cu):
        if type(t) is not torch.Tensor or not t.is_contiguous():
            return None
        traits.append(
            (tuple(t.shape), t.dtype, t.device, t.data_ptr() % 16 == 0, t.untyped_storage().size() <= 2**31 - 1)
        )
    env = tuple(
        (n, os.environ.get(n))
        for n in (
            "FLAG_ATTN_INFLLMV2_V2_TILE",
            "FLAG_ATTN_INFLLMV2_V2_WARPS",
            "FLAG_ATTN_INFLLMV2_V2_STAGES",
            "FLAG_ATTN_INFLLMV2_V2_SPLITS",
        )
    )
    names = _decode_fields.get(type(config))
    if names is None:
        names = tuple(f.name for f in fields(config))
        _decode_fields[type(config)] = names
    values = tuple(getattr(config, n) for n in names)
    if any(type(x) not in (int, float, bool, str, bytes, type(None)) for x in values):
        return None
    return (
        tuple(traits),
        tuple(lengths),
        maxk,
        tuple((type(x), x) for x in values),
        (type(scale), scale),
        scope["stream"],
        scope["env"],
        env,
    )


class _DecodePlan:
    def __init__(self, records, inputs, out):
        names = [r[0].__name__ for r in records]
        if names != [
            "_compress_k_kernel",
            "_compress_k_kernel",
            "_stage1_kernel",
            "_pool_scores_kernel",
            "_select_blocks_streaming_kernel",
            "_sparse_attention_gqa_kernel",
            "_merge_splits",
        ]:
            raise ValueError("unsupported launch sequence")
        self.records = records
        produced = [
            records[0][3][1],
            records[1][3][1],
            records[2][3][3],
            records[3][3][1],
            records[4][3][1],
            records[5][3][4],
            records[5][3][5],
            out,
        ]
        tensors = list(inputs) + produced
        self.specs = []
        for i, t in enumerate(produced):
            if not t.is_contiguous() or t.data_ptr() % 16 or t.untyped_storage().size() > 2**31 - 1:
                raise ValueError("unsupported allocation trait")
            self.specs.append((tuple(t.shape), t.dtype, 0 if i == 2 else -1 if i == 4 else None))
        self.metadata = []
        for t in (records[0][3][3], records[1][3][3]):
            self.metadata.append((t, t._version))
            tensors.append(t)
        ids = {id(t): i for i, t in enumerate(tensors)}
        self.calls = []
        self.globals = []
        self.functions = []
        seen = set()
        for fn, compiled, grid, values in records:
            if fn.debug:
                raise ValueError("debug kernel")
            template = list(values)
            bindings = []
            for i, t in enumerate(values):
                if type(t) is torch.Tensor:
                    if id(t) not in ids:
                        raise ValueError("unrecognized tensor role")
                    bindings.append((i, ids[id(t)]))
                    template[i] = None
            compiled._init_handles()
            self.calls.append(
                (compiled.run, grid, compiled.function, compiled.packed_metadata, tuple(template), tuple(bindings))
            )
            self.functions.append(fn)
            for (name, _), (old, namespace) in fn.used_global_vals.items():
                marker = (name, id(namespace))
                if marker not in seen:
                    self.globals.append((name, old, namespace))
                    seen.add(marker)
        self.records = tuple(((fn, compiled, grid, ()) for fn, compiled, grid, _ in records))

    def valid(self):
        if any((t._version != version for t, version in self.metadata)):
            return False
        if any(fn.pre_run_hooks or fn.debug for fn in self.functions):
            return False
        for name, old, namespace in self.globals:
            if namespace.get(name, _decode_missing) != old:
                raise RuntimeError(f"Global variable {name} changed after kernel compilation")
        return True

    def execute(self, inputs, stream):
        tensors = list(inputs)
        device = inputs[0].device
        for shape, dtype, fill in self.specs:
            if fill is None:
                t = torch.empty(shape, device=device, dtype=dtype)
            elif fill == 0:
                t = torch.zeros(shape, device=device, dtype=dtype)
            else:
                t = torch.full(shape, fill, device=device, dtype=dtype)
            tensors.append(t)
        tensors.extend((t for t, _ in self.metadata))
        for run, grid, function, packed, template, bindings in self.calls:
            args = list(template)
            for pos, index in bindings:
                args[pos] = tensors[index]
            run(
                *grid,
                stream,
                function,
                packed,
                None,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                *args,
            )
        return tensors[len(inputs) + 7]


def _decode_lookup(key):
    plan = _decode_plans.get(key)
    if plan is not None and plan.valid():
        _decode_plans.move_to_end(key)
        _decode_stats["hits"] += 1
        return plan
    return None


def _decode_remember(key, records, inputs, out):
    try:
        plan = _DecodePlan(records, inputs, out)
    except (ValueError, AttributeError, RuntimeError):
        _decode_stats["fallbacks"] += 1
        return
    _decode_plans[key] = plan
    _decode_stats["builds"] += 1
    while len(_decode_plans) > 32:
        _decode_plans.popitem(last=False)


def _validate_scale(scale):
    if scale is not None:
        if not isinstance(scale, (int, float)):
            raise TypeError("softmax_scale must be a finite Python number or None")
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")


def _dense_packed(q, k, v, cuq, cuk, softmax_scale):
    # The shared Flash path rescales its initial -inf state and is undefined
    # for zero/negative scale. Dense attention is the all-block sparse case;
    # reuse its existing stable online softmax and autograd kernels instead.
    if softmax_scale is not None and softmax_scale <= 0:
        maxq = _runtime.max_length(cuq)
        blocks = triton.cdiv(_runtime.max_length(cuk), 64)
        selected = (
            torch.arange(blocks, device=q.device, dtype=torch.int32)
            .view(1, 1, blocks)
            .expand(k.shape[1], q.shape[0], blocks)
            .contiguous()
        )
        return sparse_attention(q, k, v, selected, cuq, cuk, maxq, 64, softmax_scale, True)
    from flag_attn.infllmv2.operator import _dense_packed as dense

    return dense(q, k, v, cuq, cuk, softmax_scale)


def _run_sparse_pipeline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
) -> torch.Tensor:
    with torch.no_grad():
        k1, cu_k1 = compress_k(k, cu_seqlens_k, config.k1_kernel_size, config.k1_stride)
        k2, cu_k2 = compress_k(k, cu_seqlens_k, config.k2_kernel_size, config.k2_stride)
        token_score = stage1(
            q,
            k1,
            k2,
            cu_seqlens_q,
            cu_k1,
            cu_k2,
            max_seqlen_q,
            config.k1_stride,
            config.k2_stride,
            softmax_scale,
            config.causal,
            cu_seqlens_k,
        )
        block_score = pool_scores(
            token_score,
            cu_seqlens_q,
            cu_k1,
            max_seqlen_q,
            config.k1_kernel_size,
            config.k1_stride,
            config.block_size,
            config.init_blocks,
            config.local_blocks,
            cu_seqlens_k,
            max_seqlen_k,
            _valid_only=True,
        )
        selected = select_blocks(block_score, cu_seqlens_q, config.topk, config.block_size, cu_seqlens_k)
    return sparse_attention(
        q, k, v, selected, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, config.block_size, softmax_scale, config.causal
    )


def _dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    config: InfLLMV2Config,
    softmax_scale: float | None,
) -> torch.Tensor:
    if max_seqlen_k < config.dense_len:
        return _dense_packed(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale)
    return _run_sparse_pipeline(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, config, softmax_scale)


def infllmv2_attention(
    q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *, config=None, softmax_scale=None
):
    from flag_attn.infllmv2 import operator as generic

    config = config or generic.InfLLMV2Config()
    config.validate()
    _validate_scale(softmax_scale)
    generic._validate_packed_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    dense_candidate = (
        type(q) is torch.Tensor
        and q.shape == (1024, 32, 64)
        and q.is_cuda
        and q.dtype == torch.bfloat16
        and k.shape == (1024, 2, 64)
        and max_seqlen_q == 1024
        and max_seqlen_k == 1024
        and max_seqlen_k < config.dense_len
        and (
            softmax_scale is None
            or (type(softmax_scale) in (float, int) and math.isfinite(softmax_scale) and softmax_scale > 0)
        )
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))
    )
    if dense_candidate:
        context = _dense_context(q, k, v, cu_seqlens_q, cu_seqlens_k)
        if context is not None:
            stream, env = context
            qvalues = _dense_length_values(cu_seqlens_q, stream)
            kvalues = qvalues if cu_seqlens_q is cu_seqlens_k else _dense_length_values(cu_seqlens_k, stream)
            if qvalues == kvalues == (0, 1024):
                return _dense_run(q, k, v, config, softmax_scale, stream, env)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cuq, cuk = cu_seqlens_q.contiguous(), cu_seqlens_k.contiguous()
    chunk = (
        max_seqlen_q == 128
        and q.dtype == torch.bfloat16
        and q.shape[1] == 32
        and k.shape[1] == 2
        and q.shape[-1] in (64, 128)
        and not (torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)))
    )
    scope = _runtime.chunk_scope(q, k, v, max_seqlen_q) if chunk else nullcontext()
    with scope:
        return _dispatch(q, k, v, cuq, cuk, max_seqlen_q, max_seqlen_k, config, softmax_scale)


def infllmv2_decode(q, k_cache, v_cache, cu_seqlens_k, max_seqlen_k, *, config=None, softmax_scale=None):
    from flag_attn.infllmv2 import operator as generic

    config = config or generic.InfLLMV2Config()
    config.validate()
    _validate_scale(softmax_scale)
    batch = cu_seqlens_k.numel() - 1
    if q.ndim != 3 or q.shape[0] != batch:
        raise ValueError("decode q must have [batch, Hq, D] layout")
    cuq = decode_cu(batch, q.device)
    generic._validate_packed_inputs(q, k_cache, v_cache, cuq, cu_seqlens_k, 1, max_seqlen_k)
    contiguous = all(t.is_contiguous() for t in (q, k_cache, v_cache, cu_seqlens_k))
    q, k_cache, v_cache = q.contiguous(), k_cache.contiguous(), v_cache.contiguous()
    cuk = cu_seqlens_k.contiguous()
    with _runtime.decode_scope(q, k_cache, v_cache, 1):
        scope = _runtime._scope()
        candidate = (
            contiguous
            and q.shape[0] == 1
            and k_cache.shape[:2] == (8192, 2)
            and max_seqlen_k == 8192
            and max_seqlen_k >= config.dense_len
            and scope["enabled"]
        )
        inputs = (q, k_cache, v_cache, cuq, cuk)
        key = None
        if candidate:
            key = _decode_key(
                q, k_cache, v_cache, cuk, max_seqlen_k, config, softmax_scale, scope, _runtime._entry(cuk)["values"]
            )
            if key is not None:
                cached = _decode_lookup(key)
                if cached is not None:
                    return cached.execute(inputs, scope["stream"])
        if key is not None:
            with _runtime.record() as records:
                out = _dispatch(*inputs, 1, max_seqlen_k, config, softmax_scale)
            _decode_remember(key, records, inputs, out)
            return out
        return _dispatch(*inputs, 1, max_seqlen_k, config, softmax_scale)


def clear_cache():
    """Invalidate host metadata and launch plans after untracked external cu writes."""
    _dense_clear_caches()
    _clear_decode_plans()
    _runtime.clear_metadata_cache()
    _runtime.clear_launch_cache()


def runtime_stats():
    """Return diagnostic cache counters without synchronizing the device."""
    return {"dense": _dense_runtime_stats(), "decode": _decode_plan_stats(), "launch": _runtime.launch_stats()}
