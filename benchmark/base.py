# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass
from functools import partial

import torch

from flag_attn.testing import backend as test_backend

try:
    from benchmark.recording import BenchmarkRecorder
except ModuleNotFoundError:
    from recording import BenchmarkRecorder


@dataclass(frozen=True)
class BenchmarkConfig:
    """Warmup and repetition budgets in milliseconds."""

    warmup_ms: float = 100
    rep_ms: float = 500

    def __post_init__(self):
        for name, minimum in (("warmup_ms", 0), ("rep_ms", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be an int or float, not {type(value).__name__}")
            if not math.isfinite(value) or value < minimum:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.rep_ms == 0:
            raise ValueError("rep_ms must be positive")

    @classmethod
    def from_env(cls):
        """Read shared benchmark timing budgets in milliseconds."""
        return cls(
            warmup_ms=float(os.getenv("FLAG_ATTN_BENCH_WARMUP_MS", "100")),
            rep_ms=float(os.getenv("FLAG_ATTN_BENCH_REP_MS", "500")),
        )


class Benchmark:
    """Measure a reference and a FlagAttention callable with shared arguments."""

    DEFAULT_DTYPES = (torch.bfloat16,)
    DEFAULT_METRICS = ("latency_base", "latency", "speedup")
    DEFAULT_SHAPES = ()

    def __init__(
        self, op_name, torch_op=None, *, flagattn_op=None, dtypes=None,
        shapes=None, config=None, metrics=None,
    ):
        if not isinstance(op_name, str):
            raise TypeError("op_name must be a string")
        if not op_name.strip():
            raise ValueError("op_name must be nonempty")
        for name, op in (("torch_op", torch_op), ("flagattn_op", flagattn_op)):
            if op is not None and not callable(op):
                raise TypeError(f"{name} must be callable or None")
        if config is not None and not isinstance(config, BenchmarkConfig):
            raise TypeError("config must be BenchmarkConfig or None")
        dtypes = self.DEFAULT_DTYPES if dtypes is None else dtypes
        shapes = self.DEFAULT_SHAPES if shapes is None else shapes
        if isinstance(dtypes, (str, bytes)) or isinstance(shapes, (str, bytes)):
            raise TypeError("dtypes and shapes must be iterable collections, not strings")
        self.op_name = op_name
        self.torch_op = torch_op
        self.flagattn_op = flagattn_op
        self.dtypes = list(dtypes)
        self.shapes = list(shapes)
        self.config = config if config is not None else BenchmarkConfig.from_env()
        self.set_dtypes()
        self.set_metrics(metrics)

    def set_flagattn(self, op):
        if not callable(op):
            raise TypeError("flagattn_op must be callable")
        self.flagattn_op = op

    def set_dtypes(self, dtypes=None):
        values = self.dtypes if dtypes is None else dtypes
        if isinstance(values, (str, bytes)):
            raise TypeError("dtypes must be an iterable collection of torch.dtype values")
        selected = list(values)
        if not selected:
            raise ValueError("selected dtypes must be nonempty")
        if any(not isinstance(dtype, torch.dtype) for dtype in selected):
            raise TypeError("dtypes must contain only torch.dtype values")
        if len(set(selected)) != len(selected):
            raise ValueError("selected dtypes must not contain duplicates")
        if any(dtype not in self.dtypes for dtype in selected):
            raise ValueError("selected dtypes must be supported")
        self.to_bench_dtypes = selected

    def set_metrics(self, metrics=None):
        values = self.DEFAULT_METRICS if metrics is None else metrics
        if isinstance(values, (str, bytes)):
            raise TypeError("metrics must be an iterable collection of metric names")
        selected = list(values)
        if not selected or any(metric not in self.DEFAULT_METRICS for metric in selected):
            raise ValueError("selected metrics must be nonempty and supported")
        if len(set(selected)) != len(selected):
            raise ValueError("selected metrics must not contain duplicates")
        self.display_metrics = selected

    def get_input_iter(self, dtype):
        """Yield argument tuples; dict items provide keyword arguments."""
        raise NotImplementedError

    @staticmethod
    def unpack_to_args_kwargs(inputs):
        args, kwargs = [], {}
        for item in inputs:
            if isinstance(item, dict):
                if any(not isinstance(key, str) for key in item):
                    raise TypeError("keyword argument names must be strings")
                overlap = kwargs.keys() & item.keys()
                if overlap:
                    raise ValueError(f"Duplicate keyword arguments: {sorted(overlap)}")
                kwargs.update(item)
            else:
                args.append(item)
        return args, kwargs

    @staticmethod
    def record_shapes(*args, **kwargs):
        def describe(value):
            if isinstance(value, torch.Tensor):
                return dict(shape=tuple(value.shape), dtype=str(value.dtype), device=str(value.device))
            if isinstance(value, dict):
                return {key: describe(item) for key, item in value.items()}
            if isinstance(value, list):
                return [describe(item) for item in value]
            if isinstance(value, tuple):
                return tuple(describe(item) for item in value)
            return value
        return dict(args=describe(list(args)), kwargs=describe(kwargs))

    def get_latency(self, op, *args, **kwargs):
        if not callable(op):
            raise TypeError(f"{self.op_name}: op must be callable")
        latency = float(test_backend.do_bench(
            partial(op, *args, **kwargs), warmup=self.config.warmup_ms,
            rep=self.config.rep_ms, return_mode="median",
        ))
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"invalid latency: {latency!r} ms")
        return latency

    def _run_case(self, recorder, index, inputs, dtype):
        args, kwargs = self.unpack_to_args_kwargs(inputs)
        shape = self.record_shapes(*args, **kwargs)
        baseline = None
        stage = "baseline" if self.torch_op is not None else "flagattn"
        try:
            if self.torch_op is not None:
                baseline = self.get_latency(self.torch_op, *args, **kwargs)
            stage = "flagattn"
            latency = self.get_latency(self.flagattn_op, *args, **kwargs)
        except Exception as exc:
            recorder.add(
                case_id=str(index), candidate_source="flagattn", shape_detail=shape,
                latency=None, latency_base=baseline, error_msg=f"{stage}: {exc}",
            )
            raise RuntimeError(f"{self.op_name} dtype={dtype} case={index} {stage}: {exc}") from exc
        metric = recorder.add(
            case_id=str(index), candidate_source="flagattn", shape_detail=shape,
            latency=latency, latency_base=baseline,
        )
        columns = [
            f"{name}={metric[name]:.6f}" if metric[name] is not None else f"{name}=N/A"
            for name in self.display_metrics
        ]
        print(f"{self.op_name} dtype={dtype}: " + ", ".join(columns) + f" inputs={shape}", flush=True)

    def run(self, record_property=None):
        if not callable(self.flagattn_op):
            raise TypeError("set a callable flagattn_op before running")
        for dtype in self.to_bench_dtypes:
            recorder = BenchmarkRecorder(
                record_property, op_name=self.op_name, dtype=str(dtype),
                provider="flagattn", baseline="torch_reference" if self.torch_op is not None else None,
                warmup_ms=self.config.warmup_ms, rep_ms=self.config.rep_ms,
            )
            try:
                index = 0
                for inputs in self.get_input_iter(dtype):
                    self._run_case(recorder, index, inputs, dtype)
                    del inputs
                    gc.collect()
                    if hasattr(torch, "gcu"):
                        torch.gcu.empty_cache()
                    index += 1
            finally:
                recorder.record()


class GenericBenchmark(Benchmark):
    """Adapt input_fn(shape, dtype, device) -> one argument tuple per shape."""

    def __init__(self, *args, input_fn, **kwargs):
        if not callable(input_fn):
            raise TypeError("input_fn must be callable")
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            inputs = self.input_fn(shape, dtype, test_backend.device)
            if not isinstance(inputs, tuple):
                raise TypeError("input_fn must return one argument tuple per shape")
            yield inputs
