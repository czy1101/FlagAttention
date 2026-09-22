"""Benchmark harness for the Inkling FA4 relative-attention operator.

Fairness invariants:

* Inputs are built once per case and shared by every backend.
* ``num_splits`` comes from a single heuristic and is handed to all backends.
* All backends are timed identically (CUDA graph first, CUDA events fallback).
* The operator run time (mean / p50 / p95 / p99) is the only metric.

Backends are resolved lazily and never raise. ``official_cute`` is probed at
runtime and simply reported unavailable when the FlashAttention / CuTe DSL
sources are absent, which is the expected situation on CI machines.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import math
import statistics
import sys
import types
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

HEAD_DIM = 128
BLOCK_SIZE = 16
DTYPE = torch.bfloat16

BACKEND_NAMES = ("official_cute", "triton", "tle")
_PERFORMANCE_NUM_SPLITS = (1, 1, 2, 8, 8, 32, 64)
CSV_COLUMNS = (
    "backend",
    "case",
    "num_splits",
    "rel",
    "total_q",
    "total_kv",
    "heads",
    "kv_heads",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "timing",
    "status",
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    seq_lens: tuple[tuple[int, int], ...]
    num_heads: int
    num_kv_heads: int
    rel_extent: int
    window_left: int | None = None


@dataclass(frozen=True)
class Backend:
    name: str
    fn: Callable[..., Any] | None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.fn is not None


@dataclass
class Prepared:
    config: BenchmarkCase
    q: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    rel_logits: torch.Tensor
    window_size: tuple[int, int]
    scale: float
    num_splits: int
    total_q: int
    total_kv: int
    max_q: int
    outputs: dict[str, torch.Tensor] = field(default_factory=dict)


def build_cases(quick: bool = False) -> list[BenchmarkCase]:
    cases = [
        BenchmarkCase("full_prefill", ((64, 64),), 4, 4, 128),
        BenchmarkCase("ragged_prefill", ((64, 64), (33, 33), (17, 17)), 8, 2, 128),
        BenchmarkCase("long_prefill", ((512, 512),), 8, 2, 1024),
        BenchmarkCase("chunked_prefill", ((32, 512),), 8, 2, 1024),
        BenchmarkCase("sliding_window", ((64, 512),), 8, 2, 256, window_left=255),
        BenchmarkCase("decode_1k", ((1, 1024),), 8, 2, 1024),
        BenchmarkCase("decode_8k", ((1, 8192),), 8, 2, 1024),
    ]
    return cases[:1] if quick else cases


def case_names() -> list[str]:
    return [case.name for case in build_cases()]


def performance_cases(
    split_config: tuple[int, ...] | None = None,
) -> tuple[tuple[str, int], ...]:
    """Return the shared performance cases and their configured split counts."""
    cases = build_cases()
    splits = _PERFORMANCE_NUM_SPLITS if split_config is None else split_config
    if len(cases) != len(splits):
        raise RuntimeError("performance split configuration is out of sync with cases")
    return tuple(
        (case.name, split_count)
        for case, split_count in zip(cases, splits, strict=True)
    )


def fallback_num_splits(
    is_local: bool,
    batch_size: int,
    max_query_len: int,
    num_heads: int,
    num_kv_heads: int,
    max_kv_len: int,
) -> int:
    """Split long KV on decode-like shapes; single split for prefill."""
    del is_local, batch_size, num_heads, num_kv_heads
    if max_query_len == 1:
        return max(1, (max_kv_len + 1023) // 1024)
    return 1


@cache
def load_num_splits_fn() -> Callable[..., int]:
    try:
        module = importlib.import_module(
            "vllm.models.inkling.nvidia.ops.fa4_rel_attention"
        )
        return module.inkling_fa4_num_splits
    except Exception:
        return fallback_num_splits


@cache
def make_relative_score_mod(rel_extent: int):
    """Build the CuTe DSL score-mod closure for the official FA4 operator."""
    import cutlass.cute as cute
    from cutlass.cute import Float32
    from flash_attn.cute.seqlen_info import SeqlenInfoQK

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        del b_idx
        rel_logits = aux_tensors[0]
        local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + local_offset) - kv_idx
        global_q = seqlen_info.offset_q + q_idx
        distance = rel_dist[0]
        rel_index = distance if distance >= 0 else 0
        rel_index = rel_index if rel_index < rel_extent else rel_extent - 1
        bias = rel_logits[global_q[0], h_idx[0], rel_index]
        bias = Float32(bias) if distance == rel_index else Float32(0.0)
        return scores + bias

    return score_mod_rel_bias


def _install_namespace_package(name: str, package_dir: Path) -> None:
    """Expose ``flash_attn.cute`` without executing ``flash_attn/__init__``.

    The upstream ``__init__`` eagerly imports the ``flash_attn_2_cuda``
    extension, which the CuTe DSL path does not need.
    """
    existing = sys.modules.get(name)
    if existing is not None and not hasattr(existing, "__path__"):
        del sys.modules[name]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(package_dir)]
        package.__package__ = name
        sys.modules[name] = package
    importlib.invalidate_caches()


def _load_official_cute(flash_root: str | Path | None) -> tuple[Any, str]:
    try:
        importlib.import_module("cutlass.cute")
    except Exception as exc:
        return None, f"cutlass.cute unavailable ({type(exc).__name__}: {exc})"

    if flash_root is not None:
        package_dir = Path(flash_root).expanduser() / "flash_attn"
        if not (package_dir / "cute" / "interface.py").is_file():
            return None, f"no flash_attn/cute/interface.py under {package_dir}"
        _install_namespace_package("flash_attn", package_dir)

    try:
        module = importlib.import_module("flash_attn.cute")
        operator = getattr(module, "flash_attn_varlen_func", None)
    except Exception as exc:
        return None, f"flash_attn.cute unavailable ({type(exc).__name__}: {exc})"
    if operator is None:
        return None, "flash_attn.cute has no flash_attn_varlen_func"
    return operator, ""


def _load_inkling_backend(name: str) -> tuple[Any, str]:
    try:
        from inkling_fa4.backend import get_backend

        return get_backend(name), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def resolve_backends(
    names: tuple[str, ...] = BACKEND_NAMES,
    flash_root: str | Path | None = None,
) -> dict[str, Backend]:
    """Probe every backend once. Missing sources yield an unavailable slot."""
    resolved: dict[str, Backend] = {}
    for name in names:
        if name == "official_cute":
            fn, reason = _load_official_cute(flash_root)
        elif name in ("triton", "tle"):
            fn, reason = _load_inkling_backend(name)
        else:
            fn, reason = None, f"unknown backend {name!r}"
        resolved[name] = Backend(name=name, fn=fn, reason=reason)
    return resolved


def prepare_case(
    config: BenchmarkCase,
    rel_mode: str = "zeros",
    seed: int = 0,
    forced_num_splits: int | None = None,
    backends: tuple[str, ...] = BACKEND_NAMES,
) -> Prepared:
    torch.manual_seed(seed)
    device = "cuda"

    q_lens = [ql for ql, _ in config.seq_lens]
    kv_lens = [kl for _, kl in config.seq_lens]
    total_q = sum(q_lens)
    num_seq = len(config.seq_lens)
    scale = 1.0 / HEAD_DIM

    q = torch.randn(total_q, config.num_heads, HEAD_DIM, device=device, dtype=DTYPE)
    q = F.normalize(q.float(), dim=-1).to(DTYPE)

    max_blocks = (max(kv_lens) + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = num_seq * max_blocks + 1
    key_cache = torch.randn(
        num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )
    key_cache = F.normalize(key_cache.float(), dim=-1).to(DTYPE)
    value_cache = torch.randn(
        num_blocks, BLOCK_SIZE, config.num_kv_heads, HEAD_DIM, device=device, dtype=DTYPE
    )

    block_table = torch.zeros(num_seq, max_blocks, dtype=torch.int32, device=device)
    for seq in range(num_seq):
        block_table[seq] = torch.arange(
            1 + seq * max_blocks,
            1 + (seq + 1) * max_blocks,
            dtype=torch.int32,
            device=device,
        )

    cum_q = [0]
    for ql in q_lens:
        cum_q.append(cum_q[-1] + ql)
    cu_seqlens_q = torch.tensor(cum_q, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    if rel_mode == "real":
        rel_logits = torch.randn(
            total_q, config.num_heads, config.rel_extent, device=device, dtype=DTYPE
        )
    else:
        rel_logits = torch.zeros(
            total_q, config.num_heads, config.rel_extent, device=device, dtype=DTYPE
        )

    window_size = (-1, -1) if config.window_left is None else (config.window_left, 0)

    if forced_num_splits is not None:
        num_splits = forced_num_splits
    else:
        num_splits = int(
            load_num_splits_fn()(
                is_local=config.window_left is not None,
                batch_size=num_seq,
                max_query_len=max(q_lens),
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                max_kv_len=max(kv_lens),
            )
        )

    return Prepared(
        config=config,
        q=q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        rel_logits=rel_logits,
        window_size=window_size,
        scale=scale,
        num_splits=num_splits,
        total_q=total_q,
        total_kv=sum(kv_lens),
        max_q=max(q_lens),
        outputs={name: torch.empty_like(q) for name in backends},
    )


def call_operator(operator: Callable[..., Any], prep: Prepared, backend: str) -> Any:
    """Invoke ``operator`` on ``prep``, tolerating small signature differences."""
    if backend == "official_cute":
        # FlashAttention CuTe expects a (left, right) tuple; (-1, -1) means
        # full causal attention. Passing None makes it subscript None.
        window = prep.window_size
        result = operator(
            q=prep.q,
            k=prep.key_cache,
            v=prep.value_cache,
            cu_seqlens_q=prep.cu_seqlens_q,
            seqused_k=prep.cache_seqlens,
            max_seqlen_q=prep.max_q,
            page_table=prep.block_table,
            softmax_scale=prep.scale,
            causal=True,
            window_size=window,
            num_splits=prep.num_splits,
            score_mod=make_relative_score_mod(prep.config.rel_extent),
            aux_tensors=[prep.rel_logits.contiguous()],
            return_lse=False,
            out=prep.outputs[backend],
        )
        return result[0] if isinstance(result, tuple) else result

    kwargs = dict(
        q=prep.q,
        key_cache=prep.key_cache,
        value_cache=prep.value_cache,
        block_table=prep.block_table,
        cache_seqlens=prep.cache_seqlens,
        cu_seqlens_q=prep.cu_seqlens_q,
        max_seqlen_q=prep.max_q,
        softmax_scale=prep.scale,
        causal=True,
        window_size=prep.window_size,
        rel_extent=prep.config.rel_extent,
        rel_logits=prep.rel_logits,
        num_splits=prep.num_splits,
        out=prep.outputs[backend],
    )
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(operator).parameters
    }
    result = operator(**accepted)
    return result if result is not None else prep.outputs[backend]


def time_fn(
    fn: Callable[[], Any], warmup: int, iters: int, use_graph: bool
) -> tuple[str, list[float]]:
    """Return ``(timing_method, latencies_ms)`` for the operator under test."""
    fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    if use_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            latencies = []
            for _ in range(iters):
                start.record()
                graph.replay()
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            return "cuda_graph", latencies
        except Exception:
            pass

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return "event_e2e", [s.elapsed_time(e) for s, e in zip(starts, ends)]


def summarize_latency(latencies: list[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    count = len(ordered)

    def percentile(p: float) -> float:
        pos = p * (count - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        weight = pos - lo
        return ordered[lo] * (1 - weight) + ordered[hi] * weight

    return {
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def run_benchmark(
    cases: list[str] | None = None,
    backends: dict[str, Backend] | None = None,
    *,
    rel: str = "zeros",
    warmup: int = 20,
    iters: int = 100,
    seed: int = 0,
    num_splits: int | None = None,
    use_graph: bool = True,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Benchmark every available backend on every requested case.

    Returns one row per (case, backend). Rows for backends that failed at
    runtime carry ``status="ERROR: ..."`` and empty latency fields.
    """
    assert torch.cuda.is_available(), "CUDA unavailable"

    if backends is None:
        backends = resolve_backends()
    selected = {
        name: backend for name, backend in backends.items() if backend.available
    }
    if not selected:
        return []

    all_cases = build_cases()
    if cases:
        all_cases = [case for case in all_cases if case.name in set(cases)]
    if not all_cases:
        raise ValueError(f"unknown benchmark case(s): {cases}")

    rows: list[dict[str, Any]] = []
    for index, config in enumerate(all_cases):
        prep = prepare_case(
            config, rel, seed + index, num_splits, tuple(selected)
        )
        if verbose:
            print(
                f"\n=== {config.name} splits={prep.num_splits} "
                f"window={prep.window_size} rel={rel} ==="
            )

        for name, backend in selected.items():
            row = dict.fromkeys(CSV_COLUMNS, "")
            row.update(
                backend=name,
                case=config.name,
                num_splits=prep.num_splits,
                rel=rel,
                total_q=prep.total_q,
                total_kv=prep.total_kv,
                heads=config.num_heads,
                kv_heads=config.num_kv_heads,
                status="ok",
            )
            try:
                method, latencies = time_fn(
                    lambda: call_operator(backend.fn, prep, name),
                    warmup,
                    iters,
                    use_graph,
                )
                stats = summarize_latency(latencies)
                row.update(
                    mean_ms=f"{stats['mean_ms']:.4f}",
                    p50_ms=f"{stats['p50_ms']:.4f}",
                    p95_ms=f"{stats['p95_ms']:.4f}",
                    p99_ms=f"{stats['p99_ms']:.4f}",
                    timing=method,
                )
                if verbose:
                    print(
                        f"  {name:14s} mean={stats['mean_ms']:.4f} ms "
                        f"p50={stats['p50_ms']:.4f} p95={stats['p95_ms']:.4f}"
                    )
            except Exception as exc:
                row["status"] = f"ERROR: {type(exc).__name__}: {exc}"
                if verbose:
                    print(f"  {name:14s} ERROR: {type(exc).__name__}: {exc}")
            rows.append(row)

    return rows


def append_csv(
    rows: list[dict[str, Any]],
    path: Path,
    columns: tuple[str, ...] = CSV_COLUMNS,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = [
    "BACKEND_NAMES",
    "BLOCK_SIZE",
    "CSV_COLUMNS",
    "DTYPE",
    "HEAD_DIM",
    "Backend",
    "BenchmarkCase",
    "Prepared",
    "append_csv",
    "build_cases",
    "call_operator",
    "case_names",
    "fallback_num_splits",
    "load_num_splits_fn",
    "make_relative_score_mod",
    "performance_cases",
    "prepare_case",
    "resolve_backends",
    "run_benchmark",
    "summarize_latency",
    "time_fn",
]
