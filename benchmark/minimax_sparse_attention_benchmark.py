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

"""CUDA benchmark for the MiniMax M3 paged MSA kernels.

The benchmark compares the FlagAttention and vLLM implementations on the same
inputs.  ``fp8`` means FP8 index queries/index keys and FP8 main KV cache,
with scalar K/V dequantization scales passed to both implementations.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import triton
import triton.knobs
import triton.testing as triton_testing

from flag_attn.minimax_sparse_attention import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)

try:
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_decode as vllm_index_decode,
    )
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_score as vllm_index_score,
    )
    from vllm.models.minimax_m3.common.ops.index_topk import (
        minimax_m3_index_topk as vllm_index_topk,
    )
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn as vllm_sparse_attn,
    )
    from vllm.models.minimax_m3.common.ops.sparse_attn import (
        minimax_m3_sparse_attn_decode as vllm_sparse_attn_decode,
    )

    VLLM_AVAILABLE = True
    VLLM_IMPORT_ERROR = ""
except Exception as exc:  # vLLM is an optional benchmark baseline.
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

warnings.filterwarnings("ignore", message="tl.make_block_ptr is deprecated")
triton.knobs.autotuning.adjust_block_size = False


class _CachedPlatform:
    """Return one cached PDL decision during benchmark iterations."""

    def __init__(self, platform):
        self._platform = platform
        self._supports_pdl = platform.is_arch_support_pdl()

    def is_arch_support_pdl(self) -> bool:
        return self._supports_pdl

    def __getattr__(self, name: str):
        return getattr(self._platform, name)


_flag_attn_index_module = sys.modules[minimax_m3_index_decode.__module__]
_flag_attn_sparse_module = sys.modules[minimax_m3_sparse_attn_decode.__module__]
_flag_attn_platform = _CachedPlatform(
    _flag_attn_index_module.current_platform
)
_flag_attn_index_module.current_platform = _flag_attn_platform
_flag_attn_sparse_module.current_platform = _flag_attn_platform

if VLLM_AVAILABLE:
    _vllm_index_module = sys.modules[vllm_index_decode.__module__]
    _vllm_sparse_module = sys.modules[vllm_sparse_attn_decode.__module__]
    _vllm_platform = _CachedPlatform(
        _vllm_index_module.current_platform
    )
    _vllm_index_module.current_platform = _vllm_platform
    _vllm_sparse_module.current_platform = _vllm_platform


BLOCK = SPARSE_BLOCK_SIZE
HEAD_DIM = 128
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
DEFAULT_WARMUP = 200
DEFAULT_REP = 300
KV_SCALE = 0.5
SEED = 0
TOPK = 16
INIT_BLOCKS = 1
LOCAL_BLOCKS = 2
DECODE_QLEN = 1

PREFILL_SHAPES = [
    (1, 8192, 16, 96),
    (2, 16384, 8, 96),
    (1, 32768, 16, 96),
    (2, 8192, 8, 96),
    (4, 4096, 16, 384),
    (4, 4096, 16, 256),
]

DECODE_SHAPES = [
    (1, 4096, 16, 96),
    (1, 16384, 16, 96),
    (1, 65536, 16, 96),
    (4, 4096, 8, 96),
    (4, 16384, 8, 96),
    (16, 4096, 8, 96),
    (32, 2048, 4, 48),
    (64, 1024, 4, 48),
]


@dataclass
class MSAData:
    q: torch.Tensor
    idx_q: torch.Tensor
    kv_cache: torch.Tensor
    index_kv_cache: torch.Tensor
    block_table: torch.Tensor
    cu_q: torch.Tensor
    seq_lens: torch.Tensor
    prefix_lens: torch.Tensor
    sm_scale: float
    k_scale: torch.Tensor | None
    v_scale: torch.Tensor | None


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")


def _supports_fp8() -> bool:
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    # NVIDIA FP8 Tensor Core support starts with Ada (8.9) and Hopper (9.0).
    return torch.cuda.get_device_capability() >= (8, 9)


def _encode_fp8(value: torch.Tensor, scale: float) -> torch.Tensor:
    if FP8_DTYPE is None:
        raise RuntimeError("This PyTorch build does not provide float8_e4m3fn.")
    return (value / scale).to(FP8_DTYPE)


def _random_storage(
    shape: tuple[int, ...],
    device: torch.device,
    fp8: bool,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    # Generate in BF16 because torch.randn support for FP8 is version-dependent.
    value = (
        torch.randn(
            shape,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5
    )
    return _encode_fp8(value, scale) if fp8 else value


def make_data(
    batch: int,
    seq_len: int,
    num_kv_heads: int,
    num_heads: int,
    device: torch.device,
    dtype_name: str,
    *,
    decode: bool,
    decode_qlen: int,
    randomize_pages: bool = True,
    generator: torch.Generator | None = None,
) -> MSAData:
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if dtype_name not in {"bf16", "fp8"}:
        raise ValueError(f"unsupported dtype: {dtype_name}")
    if dtype_name == "fp8" and FP8_DTYPE is None:
        raise RuntimeError("FP8 was requested but float8_e4m3fn is unavailable.")

    storage_dtype = torch.bfloat16 if dtype_name == "bf16" else FP8_DTYPE
    blocks_per_request = (seq_len + BLOCK - 1) // BLOCK
    total_blocks = batch * blocks_per_request
    total_q = batch * decode_qlen if decode else batch * seq_len

    q = (
        torch.randn(
            (total_q, num_heads, HEAD_DIM),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.5
    )
    idx_q = _random_storage(
        (total_q, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        generator=generator,
    )
    k_cont = _random_storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        KV_SCALE,
        generator,
    )
    v_cont = _random_storage(
        (total_blocks * BLOCK, num_kv_heads, HEAD_DIM),
        device,
        dtype_name == "fp8",
        KV_SCALE,
        generator,
    )
    index_k_cont = _random_storage(
        (total_blocks * BLOCK, HEAD_DIM),
        device,
        dtype_name == "fp8",
        generator=generator,
    )

    kv_cache = torch.empty(
        (total_blocks, num_kv_heads, BLOCK, 2 * HEAD_DIM),
        device=device,
        dtype=storage_dtype,
    )
    k_paged = k_cont.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    v_paged = v_cont.reshape(total_blocks, BLOCK, num_kv_heads, HEAD_DIM).permute(
        0, 2, 1, 3
    )
    kv_cache[..., :HEAD_DIM] = k_paged
    kv_cache[..., HEAD_DIM:] = v_paged
    index_kv_cache = index_k_cont.reshape(total_blocks, BLOCK, HEAD_DIM)

    physical_pages = torch.randperm(total_blocks, device=device, generator=generator)
    if not randomize_pages:
        physical_pages = torch.arange(total_blocks, device=device)
    # Force identity page ordering for FP8 inputs.
    if dtype_name == "fp8":
        physical_pages = torch.arange(total_blocks, device=device)
        randomize_pages = False  # Skip the page-remapping step below.
    block_table = physical_pages.reshape(batch, blocks_per_request).to(torch.int32)
    if randomize_pages:
        kv_cache = kv_cache[physical_pages.argsort()].contiguous()
        index_kv_cache = index_kv_cache[physical_pages.argsort()].contiguous()

    q_stride = decode_qlen if decode else seq_len
    cu_q = torch.arange(
        0,
        (batch + 1) * q_stride,
        q_stride,
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    prefix_lens = torch.zeros_like(seq_lens)
    if dtype_name == "fp8":
        k_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32)
        v_scale = torch.tensor([KV_SCALE], device=device, dtype=torch.float32)
    else:
        k_scale = v_scale = None
    return MSAData(
        q,
        idx_q,
        kv_cache,
        index_kv_cache,
        block_table,
        cu_q,
        seq_lens,
        prefix_lens,
        HEAD_DIM**-0.5,
        k_scale,
        v_scale,
    )


def _call_sparse(
    fn: Callable,
    data: MSAData,
    topk_idx: torch.Tensor,
    output: torch.Tensor,
    *,
    decode: bool,
    max_query_len: int,
    num_kv_heads: int,
    decode_qlen: int,
) -> None:
    common = dict(
        q=data.q,
        kv_cache=data.kv_cache,
        topk_idx=topk_idx,
        block_table=data.block_table,
        sm_scale=data.sm_scale,
        output=output,
    )
    if decode:
        common.update(
            seq_lens=data.seq_lens,
            num_kv_heads=num_kv_heads,
            decode_query_len=decode_qlen,
        )
    else:
        common.update(
            cu_seqlens_q=data.cu_q,
            seq_lens=data.seq_lens,
            prefix_lens=data.prefix_lens,
            max_query_len=max_query_len,
            num_kv_heads=num_kv_heads,
        )
    if data.k_scale is not None:
        common.update(k_scale=data.k_scale, v_scale=data.v_scale)
    fn(**common)


def run_prefill(
    index_score: Callable,
    index_topk: Callable,
    sparse_attn: Callable,
    data: MSAData,
    seq_len: int,
    num_kv_heads: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    output: torch.Tensor,
) -> None:
    scores = index_score(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.cu_q,
        data.seq_lens,
        data.prefix_lens,
        seq_len,
        seq_len,
        num_kv_heads,
    )
    topk_idx = index_topk(
        scores,
        data.cu_q,
        data.prefix_lens,
        seq_len,
        topk,
        init_blocks,
        local_blocks,
    )
    _call_sparse(
        sparse_attn,
        data,
        topk_idx,
        output,
        decode=False,
        max_query_len=seq_len,
        num_kv_heads=num_kv_heads,
        decode_qlen=1,
    )


def run_decode(
    index_decode: Callable,
    sparse_attn_decode: Callable,
    data: MSAData,
    seq_len: int,
    num_kv_heads: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    decode_qlen: int,
    output: torch.Tensor,
) -> None:
    topk_idx = index_decode(
        data.idx_q,
        data.index_kv_cache,
        data.block_table,
        data.seq_lens,
        seq_len,
        topk,
        init_blocks,
        local_blocks,
        num_kv_heads,
        decode_qlen,
        decode_qlen,
    )
    _call_sparse(
        sparse_attn_decode,
        data,
        topk_idx,
        output,
        decode=True,
        max_query_len=1,
        num_kv_heads=num_kv_heads,
        decode_qlen=decode_qlen,
    )


def bench_fn(fn: Callable, warmup: int, rep: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return float(
        triton_testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")
    )


def _supports_fp8_scales() -> bool:
    if not VLLM_AVAILABLE:
        return False
    try:
        prefill_params = inspect.signature(vllm_sparse_attn).parameters
        decode_params = inspect.signature(vllm_sparse_attn_decode).parameters
    except (TypeError, ValueError):
        return False
    return (
        "k_scale" in prefill_params
        and "v_scale" in prefill_params
        and ("k_scale" in decode_params and "v_scale" in decode_params)
    )


def _select_impl(
    provider: str,
) -> tuple[Callable, Callable, Callable, Callable, Callable]:
    if provider == "flag_attn":
        return (
            minimax_m3_index_decode,
            minimax_m3_index_score,
            minimax_m3_index_topk,
            minimax_m3_sparse_attn,
            minimax_m3_sparse_attn_decode,
        )
    if provider == "vllm":
        return (
            vllm_index_decode,
            vllm_index_score,
            vllm_index_topk,
            vllm_sparse_attn,
            vllm_sparse_attn_decode,
        )
    raise ValueError(f"unknown provider: {provider}")


def _provider_vals(dtype_name: str) -> list[str]:
    vals = ["flag_attn"]
    if VLLM_AVAILABLE:
        # vLLM cannot run FP8 without k_scale/v_scale support.
        if dtype_name == "fp8" and not _supports_fp8_scales():
            pass
        else:
            vals.append("vllm")
    return vals


_DTYPES = ["bf16"] + (["fp8"] if _supports_fp8() else [])

configs = [
    triton.testing.Benchmark(
        x_names=["batch", "seq_len", "num_kv_heads", "num_heads"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=_provider_vals(dtype_name),
        line_names=_provider_vals(dtype_name),
        styles=[("red", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name=f"minimax_m3_sparse_attention-{mode}-{dtype_name}",
        args={"mode": mode, "dtype": dtype_name},
    )
    for mode, SHAPES in [("prefill", PREFILL_SHAPES), ("decode", DECODE_SHAPES)]
    for dtype_name in _DTYPES
]


@triton.testing.perf_report(configs)
def bench_minimax_sparse_attention(
    batch,
    seq_len,
    num_kv_heads,
    num_heads,
    mode,
    provider,
    dtype="bf16",
    device="cuda",
):
    _require_cuda()
    shape = (batch, seq_len, num_kv_heads, num_heads)
    decode = mode == "decode"

    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"Invalid shape {shape}: num_heads must be divisible by num_kv_heads"
        )
    if decode and DECODE_QLEN > seq_len:
        raise ValueError(
            f"Invalid shape {shape}: decode_qlen={DECODE_QLEN} cannot exceed seq_len"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(SEED + (100_000 if decode else 0))
    data = make_data(
        batch,
        seq_len,
        num_kv_heads,
        num_heads,
        torch.device(device),
        dtype,
        decode=decode,
        decode_qlen=DECODE_QLEN,
        randomize_pages=dtype != "fp8",
        generator=generator,
    )
    output = torch.empty_like(data.q)

    index_decode, index_score, index_topk, sparse_attn, sparse_attn_decode = (
        _select_impl(provider)
    )

    if decode:

        def run() -> None:
            run_decode(
                index_decode,
                sparse_attn_decode,
                data,
                seq_len,
                num_kv_heads,
                TOPK,
                INIT_BLOCKS,
                LOCAL_BLOCKS,
                DECODE_QLEN,
                output,
            )

    else:

        def run() -> None:
            run_prefill(
                index_score,
                index_topk,
                sparse_attn,
                data,
                seq_len,
                num_kv_heads,
                TOPK,
                INIT_BLOCKS,
                LOCAL_BLOCKS,
                output,
            )

    return bench_fn(run, DEFAULT_WARMUP, DEFAULT_REP)


# only works on post-Ampere GPUs right now
bench_minimax_sparse_attention.run(print_data=True)
