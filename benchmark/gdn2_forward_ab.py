#!/usr/bin/env python
"""GDN2-003 reproducible full-forward A/B for the Enflame S60 GDN-2 operator.

Two variants are compared on the same GCU:

* ``legacy``  -- tile sizes hard-wired to 256 (the pre-GDN2-003 behaviour),
* ``adaptive`` -- per-call tiles clamped to the logical K/V extent.

The driver extracts both variants from immutable git commits with
``git archive``, then runs them alternately for several rounds.  Each child
process gets a private Triton cache directory, so the first call of every
(round, variant, shape) measures cold compile, and the following synchronized
event samples measure steady state.  Every record carries the launched kernel
names, their BK/BV tiles, peak device allocation, finite checks and the error
against an independent CPU FP64 recurrence.

Usage (inside the task clone):

    /flagos/bin/python benchmark/gdn2_forward_ab.py drive \
        --old-ref a16751119704c9b6688db3a537b9156bddaa1e4a \
        --new-ref HEAD --rounds 3 \
        --out agent-results/GDN2-003-ab.jsonl

    # one side only, e.g. to reproduce a single measurement
    /flagos/bin/python benchmark/gdn2_forward_ab.py run \
        --tree . --variant adaptive --round 1 \
        --out /tmp/one.jsonl

Only GCU 0 (PCI 23:00.0) and CPUs 0-31 are used.  Set
``FLAG_ATTN_GDN2_FIXED_TILES=1`` to force the legacy tiles in any tree.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEVICE = "gcu:0"
EXPECTED_PCI = "23:00.0"
CPU_LIST = "0-31"
SEED = 42
CHUNK_SIZE = 64
STEADY_SAMPLES = 15
WARMUP_ITERS = 5

# T/state axes: short and long T, two head counts, and K/V = 64, 128, 256.
# The 256 case is the control: its tiles are already 256, so it must not move.
BF16_SHAPES = [
    (2, 512, 8, 64, 64),
    (1, 2048, 8, 64, 64),
    (1, 4096, 16, 64, 64),
    (2, 2048, 4, 128, 128),
    (1, 1024, 4, 256, 256),
]
FP16_SHAPES = [
    (2, 512, 8, 64, 64),
    (2, 2048, 4, 128, 128),
]

GATE_ATOL = 1e-3
GATE_RTOL = 2e-2


def _jsonl_write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _loadavg() -> str:
    try:
        return Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError:
        return "unavailable"


def _efsmi_used_mib(index: int = 0) -> float | None:
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-i", str(index)],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Used Size\s*:\s*(\d+)\s*MiB", out)
    return float(match.group(1)) if match else None


def _efsmi_pci(index: int = 0) -> str | None:
    try:
        out = subprocess.run(
            ["efsmi", "-q", "-d", "PCIE", "--json-format"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    for item in data.get("devices", []):
        if str(item.get("id")) == str(index):
            pcie = item.get("PCIe Info", {})
            return f"{pcie.get('Bus')}:{pcie.get('Dev')}.{pcie.get('Func')}"
    return None


def _build_inputs(B, T, H, K, V, dtype, device):
    import torch

    torch.manual_seed(SEED)
    q = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    k = (torch.randn(B, T, H, K) / math.sqrt(K)).to(dtype)
    v = torch.randn(B, T, H, V).to(dtype)
    g = (-torch.rand(B, T, H, K) * 0.1).to(dtype)
    b = torch.rand(B, T, H, K).to(dtype)
    w = torch.rand(B, T, H, V).to(dtype)
    h0 = torch.randn(B, H, K, V) * 0.01
    tensors = [x.to(device) for x in (q, k, v, g, b, w, h0)]
    cpu = [x.cpu() for x in (q, k, v, g, b, w, h0)]
    return tensors, cpu


def _metrics(expected, actual):
    import torch

    e = expected.double().flatten().cpu()
    a = actual.double().flatten().cpu()
    diff = (e - a).abs()
    rms = diff.square().mean().sqrt().item()
    ref_rms = e.square().mean().sqrt().item()
    return {
        "max_abs": diff.max().item(),
        "rms_rel": rms / (ref_rms + 1e-12),
        "ref_rms": ref_rms,
        "ref_max": e.abs().max().item(),
    }


def _gates(metrics):
    strict = (
        metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_rms"]
        and metrics["rms_rel"] <= GATE_RTOL
    )
    magnitude = (
        metrics["max_abs"] <= GATE_ATOL + GATE_RTOL * metrics["ref_max"]
        and metrics["rms_rel"] <= GATE_RTOL
    )
    return {"gate_a": bool(strict), "gate_b": bool(magnitude)}


def _install_kernel_tracer():
    """Record every Triton launch (kernel name, grid, BK/BV/tile constexprs)."""
    from triton.runtime.jit import JITFunction

    records = []
    original = JITFunction.run

    def traced(self, *args, **kwargs):
        grid = kwargs.get("grid")
        if callable(grid):
            grid_repr = "callable"
        elif isinstance(grid, (tuple, list)):
            grid_repr = [int(x) for x in grid]
        else:
            grid_repr = repr(grid)
        cfg = {}
        for key in ("BK", "BV", "BT", "GRID_MODE", "STATE_V_FIRST", "num_warps", "num_stages"):
            if key in kwargs:
                value = kwargs[key]
                cfg[key] = value if isinstance(value, (int, float, bool, str)) else repr(value)
        records.append({
            "kernel": getattr(self.fn, "__name__", str(self.fn)),
            "grid": grid_repr,
            "cfg": cfg,
        })
        return original(self, *args, **kwargs)

    JITFunction.run = traced
    return records


def _time_call(call, samples=STEADY_SAMPLES):
    import torch

    for _ in range(WARMUP_ITERS):
        call()
    torch.gcu.synchronize()
    begin = torch.gcu.Event(enable_timing=True)
    end = torch.gcu.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        begin.record()
        call()
        end.record()
        torch.gcu.synchronize()
        values.append(begin.elapsed_time(end))
    ordered = sorted(values)
    return {
        "samples_ms": [round(x, 6) for x in values],
        "median_ms": round(statistics.median(values), 6),
        "min_ms": round(ordered[0], 6),
        "p25_ms": round(ordered[len(ordered) // 4], 6),
        "max_ms": round(ordered[-1], 6),
    }


def command_run(args) -> int:
    import torch

    tree = Path(args.tree).resolve()
    sys.path.insert(0, str(tree / "src"))
    sys.path.insert(0, str(tree / "tests" / "flag_attn"))

    from gdn2_backward_ref import gdn2_forward  # noqa: E402
    from flag_attn import chunk_gdn2  # noqa: E402

    out = Path(args.out)
    dtype_name = args.dtype
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    shapes = BF16_SHAPES if dtype_name == "bfloat16" else FP16_SHAPES
    if args.shape:
        shapes = [tuple(int(x) for x in item.split(",")) for item in args.shape]

    pci = _efsmi_pci(0)
    meta = {
        "schema": 1,
        "task": "GDN2-003",
        "kind": "meta",
        "variant": args.variant,
        "round": args.round,
        "tree": str(tree),
        "commit": args.commit,
        "dtype": dtype_name,
        "hostname": platform.node(),
        "torch": torch.__version__,
        "device_count_visible": torch.gcu.device_count(),
        "current_device": torch.gcu.current_device(),
        "pci0": pci,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
        "tile_adapt_default": getattr(
            __import__("flag_attn.runtime.backend._enflame.FLA.gdn2.chunk_gdn2",
                       fromlist=["_S60_TILES_ADAPT"]), "_S60_TILES_ADAPT", None),
        "loadavg": _loadavg(),
        "efsmi_used_mib": _efsmi_used_mib(0),
        "started_at": time.time(),
    }
    _jsonl_write(out, meta)
    if pci is None or pci.strip().lower() != EXPECTED_PCI:
        _jsonl_write(out, {"kind": "error", "variant": args.variant,
                           "round": args.round, "error": f"unexpected GCU0 PCI {pci}"})
        return 2
    if torch.gcu.device_count() != 1:
        _jsonl_write(out, {"kind": "error", "variant": args.variant, "round": args.round,
                           "error": f"TOPS_VISIBLE_DEVICES did not pin one device: "
                                    f"{torch.gcu.device_count()}"})
        return 2

    tracer = _install_kernel_tracer()
    exit_code = 0
    for shape in shapes:
        B, T, H, K, V = shape
        scale = K ** -0.5
        try:
            inputs, cpu_inputs = _build_inputs(B, T, H, K, V, dtype, DEVICE)
            ref_o, ref_ht = gdn2_forward(*cpu_inputs[:6], scale, cpu_inputs[6])

            def call():
                with torch.inference_mode():
                    return chunk_gdn2(
                        *inputs[:6], scale=scale, initial_state=inputs[6],
                        output_final_state=True, chunk_size=CHUNK_SIZE,
                    )

            tracer.clear()
            torch.gcu.synchronize()
            wall_start = time.perf_counter()
            o, ht = call()
            torch.gcu.synchronize()
            cold_s = time.perf_counter() - wall_start
            cold_launches = [dict(item) for item in tracer]

            torch.gcu.synchronize()
            if hasattr(torch.gcu, "reset_peak_memory_stats"):
                torch.gcu.reset_peak_memory_stats()
            base_alloc = torch.gcu.memory_allocated()
            o, ht = call()
            torch.gcu.synchronize()
            peak_mib = (torch.gcu.max_memory_allocated() - base_alloc) / 2**20

            tracer.clear()
            steady = _time_call(call)
            steady_launches = [dict(item) for item in tracer]

            metrics = {}
            for label, ref, act in (("o", ref_o, o), ("ht", ref_ht, ht)):
                item = _metrics(ref, act)
                item.update(_gates(item))
                item["finite"] = bool(torch.isfinite(act).all().item())
                metrics[label] = item

            _jsonl_write(out, {
                "schema": 1,
                "kind": "shape",
                "task": "GDN2-003",
                "variant": args.variant,
                "round": args.round,
                "commit": args.commit,
                "dtype": dtype_name,
                "shape": [B, T, H, K, V],
                "cold_compile_s": round(cold_s, 6),
                "cold_launches": cold_launches,
                "steady_launches": steady_launches,
                "steady": steady,
                "peak_alloc_mib": round(peak_mib, 6),
                "efsmi_used_mib": _efsmi_used_mib(0),
                "metrics": metrics,
                "loadavg": _loadavg(),
                "wall_clock_s": round(time.perf_counter() - wall_start, 6),
            })
            del inputs, cpu_inputs, o, ht
        except Exception as error:  # noqa: BLE001 - one bad shape must be recorded
            import traceback
            traceback.print_exc()
            exit_code = 1
            _jsonl_write(out, {
                "schema": 1, "kind": "error", "task": "GDN2-003",
                "variant": args.variant, "round": args.round,
                "commit": args.commit, "dtype": dtype_name, "shape": list(shape),
                "error": f"{type(error).__name__}: {error}",
            })
    _jsonl_write(out, {"schema": 1, "kind": "meta_end", "task": "GDN2-003",
                       "variant": args.variant, "round": args.round,
                       "exit_code": exit_code, "ended_at": time.time(),
                       "loadavg": _loadavg(), "efsmi_used_mib": _efsmi_used_mib(0)})
    return exit_code


def _extract(workspace: Path, ref: str, dest: Path) -> str:
    sha = subprocess.run(["git", "-C", str(workspace), "rev-parse", ref],
                         capture_output=True, text=True, check=True).stdout.strip()
    archive = subprocess.run(["git", "-C", str(workspace), "archive", sha],
                             capture_output=True, check=True).stdout
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return sha


def command_drive(args) -> int:
    workspace = Path(args.workspace).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        out.unlink()
    root = Path(tempfile.mkdtemp(prefix="gdn2-003-ab-"))
    code = 0
    try:
        trees = {
            "legacy": _extract(workspace, args.old_ref, root / "legacy"),
            "adaptive": _extract(workspace, args.new_ref, root / "adaptive"),
        }
        _jsonl_write(out, {
            "schema": 1, "kind": "driver", "task": "GDN2-003",
            "workspace": str(workspace), "old_ref": args.old_ref,
            "new_ref": args.new_ref, "commits": trees,
            "rounds": args.rounds, "dtypes": args.dtypes,
            "cpu_list": CPU_LIST, "device": DEVICE, "pci_expected": "23:00.0",
            "hostname": platform.node(), "started_at": time.time(),
            "loadavg": _loadavg(),
        })
        for round_index in range(1, args.rounds + 1):
            order = ("legacy", "adaptive") if round_index % 2 else ("adaptive", "legacy")
            for variant in order:
                for dtype_name in args.dtypes:
                    cache = root / f"triton-{round_index}-{variant}-{dtype_name}"
                    cache.mkdir()
                    env = os.environ.copy()
                    env["PYTHONPATH"] = os.pathsep.join(
                        [str(trees[variant] / "src"), str(trees[variant]),
                         env.get("PYTHONPATH", "")]
                    ).rstrip(os.pathsep)
                    env["TOPS_VISIBLE_DEVICES"] = "0"
                    env["TRITON_CACHE_DIR"] = str(cache)
                    env.pop("CUDA_VISIBLE_DEVICES", None)
                    if variant == "legacy":
                        env["FLAG_ATTN_GDN2_FIXED_TILES"] = "1"
                    else:
                        env.pop("FLAG_ATTN_GDN2_FIXED_TILES", None)
                    per_run = root / f"records-{round_index}-{variant}-{dtype_name}.jsonl"
                    argv = [
                        "taskset", "-c", CPU_LIST, sys.executable,
                        str(Path(__file__).resolve()), "run",
                        "--tree", str(trees[variant]), "--variant", variant,
                        "--commit", trees[variant], "--round", str(round_index),
                        "--dtype", dtype_name, "--out", str(per_run),
                    ]
                    started = time.time()
                    proc = subprocess.run(argv, env=env, capture_output=True, text=True)
                    log = root / f"log-{round_index}-{variant}-{dtype_name}.txt"
                    log.write_text(
                        f"$ {' '.join(argv)}\n\n[stdout]\n{proc.stdout}\n[stderr]\n{proc.stderr}",
                        encoding="utf-8",
                    )
                    if proc.returncode != 0:
                        code = 1
                    if per_run.exists():
                        with out.open("a", encoding="utf-8") as sink, per_run.open(encoding="utf-8") as src:
                            for line in src:
                                record = json.loads(line)
                                record["round"] = round_index
                                record["variant"] = variant
                                record["commit"] = trees[variant]
                                record["child_exit_code"] = proc.returncode
                                record["child_log"] = str(log)
                                sink.write(json.dumps(record, sort_keys=True) + "\n")
                    _jsonl_write(out, {
                        "schema": 1, "kind": "invocation", "task": "GDN2-003",
                        "round": round_index, "variant": variant, "dtype": dtype_name,
                        "child_exit_code": proc.returncode, "argv": argv,
                        "env": {k: env.get(k) for k in
                                ("PYTHONPATH", "TOPS_VISIBLE_DEVICES", "TRITON_CACHE_DIR",
                                 "FLAG_ATTN_GDN2_FIXED_TILES")},
                        "duration_s": round(time.time() - started, 3),
                        "loadavg": _loadavg(),
                    })
        _jsonl_write(out, {"schema": 1, "kind": "driver_end", "task": "GDN2-003",
                           "exit_code": code, "ended_at": time.time(),
                           "loadavg": _loadavg()})
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="measure one variant (single process)")
    run.add_argument("--tree", required=True)
    run.add_argument("--variant", choices=["legacy", "adaptive"], required=True)
    run.add_argument("--commit", default="unknown")
    run.add_argument("--round", type=int, default=0)
    run.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    run.add_argument("--shape", action="append", help="B,T,H,K,V; repeatable")
    run.add_argument("--out", required=True)
    run.set_defaults(func=command_run)

    drive = sub.add_parser("drive", help="extract both commits and run alternating rounds")
    drive.add_argument("--workspace", default=".")
    drive.add_argument("--old-ref", required=True)
    drive.add_argument("--new-ref", default="HEAD")
    drive.add_argument("--rounds", type=int, default=3)
    drive.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"],
                       choices=["bfloat16", "float16"])
    drive.add_argument("--out", required=True)
    drive.set_defaults(func=command_drive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
