#!/usr/bin/env python
"""Fair benchmark: official vLLM CuTe vs Triton vs TLE Inkling FA4."""

import argparse
import csv
import importlib
import sys
import types
from functools import cache
from pathlib import Path

import torch

from benchmark_Cute_Triton import (
    _fallback_num_splits,
    build_cases,
    prepare_case,
    run_inkling,
    summarize_latency,
    time_fn,
)
from inkling_fa4.triton_kernel import inkling_fa4_rel_attention as triton_op
from inkling_fa4.triton_tle_kernel import inkling_fa4_rel_attention_tle as tle_op


def add_flash_attn_source(flash_root: Path) -> None:
    """Expose ``flash_attn.cute`` without executing flash_attn/__init__.py.

    The repository root package eagerly imports ``flash_attn_2_cuda``.  FA4
    CuTe DSL does not need that extension, so install a namespace-style shell
    whose search path points at the source checkout.
    """
    root = flash_root.expanduser().resolve()
    package_dir = root / "flash_attn"
    if not (package_dir / "cute" / "interface.py").is_file():
        raise FileNotFoundError(
            f"{root} 不是有效的 flash-attention 仓库：缺少 "
            "flash_attn/cute/interface.py"
        )
    old = sys.modules.get("flash_attn")
    if old is not None and not hasattr(old, "__path__"):
        del sys.modules["flash_attn"]
    if "flash_attn" not in sys.modules:
        package = types.ModuleType("flash_attn")
        package.__path__ = [str(package_dir)]
        package.__package__ = "flash_attn"
        package.__file__ = str(package_dir / "__init__.py")
        sys.modules["flash_attn"] = package
    importlib.invalidate_caches()


@cache
def make_relative_score_mod(rel_extent: int):
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


def load_official_cute(flash_root: Path):
    add_flash_attn_source(flash_root)
    try:
        module = importlib.import_module("flash_attn.cute")
        return module.flash_attn_varlen_func
    except Exception as exc:
        print(
            f"[warn] official Dao-AILab CuTe disabled: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None


def run_official_cute(prep, operator):
    window = ((None, None) if prep["window_size"] == (-1, -1)
              else prep["window_size"])
    result = operator(
        q=prep["q"],
        k=prep["key_cache"],
        v=prep["value_cache"],
        cu_seqlens_q=prep["cu_seqlens_q"],
        seqused_k=prep["cache_seqlens"],
        max_seqlen_q=prep["max_q"],
        page_table=prep["block_table"],
        softmax_scale=prep["scale"],
        causal=True,
        window_size=window,
        num_splits=prep["num_splits"],
        score_mod=make_relative_score_mod(prep["config"].rel_extent),
        aux_tensors=[prep["rel_logits"].contiguous()],
        return_lse=False,
        out=prep["out_official_cute"],
    )
    return result[0] if isinstance(result, tuple) else result


def make_row(name, prep, config):
    return dict(
        backend=name,
        case=config.name,
        num_splits=prep["num_splits"],
        rel=prep["rel_mode"],
        total_q=prep["total_q"],
        total_kv=prep["total_kv"],
        heads=config.num_heads,
        kv_heads=config.num_kv_heads,
        mean_ms="",
        p50_ms="",
        p95_ms="",
        p99_ms="",
        timing="",
        status="ok",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash-root", type=Path,
                        default=Path("/home/xiaodan/flash-attention"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--case", default=None)
    parser.add_argument("--rel", choices=("zeros", "real"), default="zeros")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-splits", type=int, default=None)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--no-cute", action="store_true")
    parser.add_argument(
        "--only-backend",
        choices=("official_cute", "triton", "tle"),
        default=None,
        help="Run only one backend; default runs every available backend.",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("fa4_bench_cute_triton_tle.csv"))
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA unavailable"
    capability = torch.cuda.get_device_capability()
    assert capability >= (9, 0), f"requires SM90+, got {capability}"
    if args.num_splits is not None and args.num_splits < 1:
        parser.error("--num-splits must be positive")

    cute_op = None
    split_fn = _fallback_num_splits
    if not args.no_cute:
        cute_op = load_official_cute(args.flash_root)

    operators = []
    if cute_op is not None and args.only_backend in (None, "official_cute"):
        operators.append(("official_cute", cute_op, "out_official_cute"))
    if args.only_backend in (None, "triton"):
        operators.append(("triton", triton_op, "out_triton"))
    if args.only_backend in (None, "tle"):
        operators.append(("tle", tle_op, "out_tle"))

    cases = build_cases(False if args.case else args.quick)
    if args.case:
        cases = [case for case in cases if case.name == args.case]
    if not cases:
        parser.error(f"unknown case: {args.case}")

    names = [item[0] for item in operators]
    print(f"GPU: {torch.cuda.get_device_name(0)} cap={capability}")
    print(f"backends={names} rel={args.rel} graph={not args.no_cuda_graph}")
    rows = []

    for index, config in enumerate(cases):
        prep = prepare_case(config, args.rel, args.seed + index, split_fn,
                            args.num_splits)
        prep["rel_mode"] = args.rel
        prep["out_tle"] = torch.empty_like(prep["q"])
        print(f"\n=== {config.name} splits={prep['num_splits']} ===")

        for name, operator, output_name in operators:
            row = make_row(name, prep, config)
            original = prep["out_triton"]
            prep["out_triton"] = prep[output_name]
            try:
                if name == "official_cute":
                    call = lambda operator=operator: run_official_cute(
                        prep, operator
                    )
                else:
                    call = lambda operator=operator: run_inkling(prep, operator)
                method, samples = time_fn(
                    call, args.warmup, args.iters, not args.no_cuda_graph
                )
                stats = summarize_latency(samples)
                row.update(
                    mean_ms=f"{stats['mean_ms']:.4f}",
                    p50_ms=f"{stats['p50_ms']:.4f}",
                    p95_ms=f"{stats['p95_ms']:.4f}",
                    p99_ms=f"{stats['p99_ms']:.4f}",
                    timing=method,
                )
                print(
                    f"  {name:14s} mean={stats['mean_ms']:.4f} ms "
                    f"p50={stats['p50_ms']:.4f} p95={stats['p95_ms']:.4f}"
                )
            except Exception as exc:
                row["status"] = f"ERROR: {type(exc).__name__}: {exc}"
                print(f"  {name:14s} ERROR: {type(exc).__name__}: {exc}")
            finally:
                prep["out_triton"] = original
            rows.append(row)

    columns = ["backend", "case", "num_splits", "rel", "total_q",
               "total_kv", "heads", "kv_heads", "mean_ms", "p50_ms",
               "p95_ms", "p99_ms", "timing", "status"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
