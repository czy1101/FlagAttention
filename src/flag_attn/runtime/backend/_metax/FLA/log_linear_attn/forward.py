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

import math
from functools import lru_cache

import torch
import triton

from .kernels import (
    _fused_output_kernel,
    _inter_fwd_kernel,
    _local_fwd_kernel,
    _merge_states_kernel,
    _paired_states_kernel,
    _prepare_query_factors_kernel,
    _quad_states_kernel,
    _reduce_output_flat_kernel,
    _scores_kernel,
    _shared_score_output_kernel,
)

CHUNK_SIZE = 64
BLOCK_KEY = 64
BLOCK_VALUE = 64
REDUCTION_BLOCK = 4096


@lru_cache(maxsize=None)
def _level_lut(device_index: int) -> torch.Tensor:
    device = torch.device("cuda", device_index)
    lut = torch.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=torch.int32, device=device)
    rows = torch.arange(CHUNK_SIZE, device=device)[:, None]
    cols = torch.arange(CHUNK_SIZE, device=device)[None, :]
    for level in range(1, int(math.log2(CHUNK_SIZE)) + 1):
        half = 1 << (level - 1)
        anchor = rows - rows % half
        selected = (rows % (2 * half) >= half) & (cols + half >= anchor) & (cols < anchor)
        lut = torch.where(selected, level, lut)
    return lut


def _chunk_local_cumsum(g: torch.Tensor) -> torch.Tensor:
    batch, sequence, heads = g.shape
    return g.view(batch, sequence // CHUNK_SIZE, CHUNK_SIZE, heads).cumsum(dim=2).view_as(g)


def _validate_inputs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, level_scales: torch.Tensor
) -> tuple[int, int, int, int, int, int]:
    tensors = (q, k, v, g, level_scales)
    if not all((tensor.is_cuda for tensor in tensors)):
        raise ValueError("chunk_log_linear_attn requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")
    if any((tensor.requires_grad for tensor in tensors)):
        raise RuntimeError("chunk_log_linear_attn is forward-only and does not support autograd")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("chunk_log_linear_attn supports BF16 q, k, and v")
    if g.dtype != torch.float32:
        raise ValueError("g must use float32")
    if level_scales.dtype != q.dtype:
        raise ValueError("level_scales must have the same dtype as q")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 tensors with matching q/k shapes")
    if q.shape[2] != 1 or k.shape[2] != 1:
        raise ValueError("chunk_log_linear_attn requires one shared Q/K head")
    batch, sequence, _, key_dim = q.shape
    heads, value_dim = (v.shape[2], v.shape[-1])
    if v.shape[:2] != (batch, sequence):
        raise ValueError("v must have shape [B, T, H, V]")
    if g.shape != (batch, sequence, heads):
        raise ValueError("g must have shape [B, T, H]")
    if batch < 1 or sequence < 1 or heads < 1:
        raise ValueError("batch, sequence, and heads must be positive")
    if sequence % CHUNK_SIZE != 0 or sequence & (sequence - 1):
        raise ValueError("sequence must be a power of two and divisible by 64")
    if key_dim not in (64, 128, 256) or value_dim not in (64, 128, 256):
        raise ValueError("chunk_log_linear_attn supports K/V in {64, 128, 256}")
    if level_scales.ndim != 4:
        raise ValueError("level_scales must be rank 4")
    levels = level_scales.shape[-1]
    if level_scales.shape[:3] != (batch, sequence, heads):
        raise ValueError("level_scales must have shape [B, T, H, L]")
    if levels != math.ceil(math.log2(sequence)) + 1:
        raise ValueError("invalid level_scales level count")
    return (batch, sequence, heads, key_dim, value_dim, levels)


def chunk_log_linear_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, level_scales: torch.Tensor
) -> torch.Tensor:
    batch, sequence, _, key_dim, _, _ = _validate_inputs(q=q, k=k, v=v, g=g, level_scales=level_scales)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    level_scales = level_scales.contiguous()
    if batch > 1 and sequence > 2 * CHUNK_SIZE and key_dim // BLOCK_KEY > 2:
        group_size = _batch_group_size(v=v, g=g, key_dim=key_dim)
        if group_size < batch:
            output = torch.empty_like(v)
            for start in range(0, batch, group_size):
                group = slice(start, start + group_size)
                _chunk_forward(
                    q=q[group],
                    k=k[group],
                    v=v[group],
                    g=g[group],
                    level_scales=level_scales[group],
                    output=output[group],
                )
            return output
    return _chunk_forward(q=q, k=k, v=v, g=g, level_scales=level_scales)


def _batch_group_size(v: torch.Tensor, g: torch.Tensor, key_dim: int) -> int:
    batch, sequence, heads, value_dim = v.shape
    chunks = sequence // CHUNK_SIZE
    key_tiles = key_dim // BLOCK_KEY
    output_bytes = v.nbytes // batch
    partial_bytes = output_bytes * key_tiles
    state_bytes = (chunks - 1) * heads * key_dim * value_dim * v.element_size()
    gate_bytes = (sequence + chunks) * heads * g.element_size()
    score_bytes = chunks * key_tiles * CHUNK_SIZE**2 * g.element_size()
    factor_bytes = (chunks.bit_length() - 1) * (g.nbytes // batch) // 2
    workspace_per_batch = partial_bytes + max(state_bytes, output_bytes) + gate_bytes + score_bytes + factor_bytes
    free_bytes, _ = torch.cuda.mem_get_info(v.device)
    required_bytes = batch * workspace_per_batch
    if required_bytes <= free_bytes:
        return batch
    stats = torch.cuda.memory_stats(v.device)
    available_bytes = free_bytes + stats["reserved_bytes.all.current"] - stats["allocated_bytes.all.current"]
    if required_bytes <= available_bytes:
        return batch
    compute_units = torch.cuda.get_device_properties(v.device).multi_processor_count
    batches_per_wave = triton.cdiv(compute_units, chunks * key_tiles)
    return max(1, min(batches_per_wave, (free_bytes - v.nbytes) // workspace_per_batch))


def _chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    level_scales: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, sequence, heads, value_dim = v.shape
    key_dim = q.shape[-1]
    levels = level_scales.shape[-1]
    num_chunks = sequence // CHUNK_SIZE
    cache_states = num_chunks > 1
    num_levels = max(1, num_chunks.bit_length() - 1)
    first_level = min(2, num_levels - 1)
    fuse_forward = first_level > 0
    state_levels = num_levels - int(cache_states)
    num_key_tiles = key_dim // BLOCK_KEY
    num_value_tiles = value_dim // BLOCK_VALUE
    num_tiles = num_key_tiles * num_value_tiles
    local_g = _chunk_local_cumsum(g=g)
    level_lut = _level_lut(device_index=q.device.index or 0)
    states = cached_states = chunk_gates = None
    if cache_states:
        if fuse_forward:
            first_nodes = num_chunks >> first_level
            state_nodes = first_nodes + (first_nodes // 2 if state_levels > first_level + 1 else 0)
            if state_levels <= first_level:
                state_nodes = 0
            states = torch.empty(
                state_nodes * batch * heads * key_dim * value_dim, device=v.device, dtype=torch.float32
            )
        cached_states = torch.empty(
            (num_chunks - 1) * batch * heads * key_dim * value_dim, device=v.device, dtype=torch.bfloat16
        )
        chunk_gates = torch.empty(batch, num_chunks, heads, device=v.device, dtype=torch.float32)
    output_grid = (num_chunks * batch * heads * num_tiles,)
    if not fuse_forward:
        output_partial = torch.empty((batch, sequence, heads, num_key_tiles, value_dim), device=v.device, dtype=v.dtype)
        _local_fwd_kernel[output_grid](
            q=q,
            k=k,
            v=v,
            g=local_g,
            level_scales=level_scales,
            level_lut=level_lut,
            local_output=output_partial,
            chunk_gates=chunk_gates,
            cached_states=cached_states,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NK=num_key_tiles,
            NV=num_value_tiles,
            num_warps=4,
            num_stages=2,
        )
    else:
        state_kernel = _quad_states_kernel if first_level == 2 else _paired_states_kernel
        state_grid = ((num_chunks >> first_level) * batch * heads * num_tiles,)
        state_kernel[state_grid](
            k=k,
            v=v,
            g=local_g,
            states=states,
            cached_states=cached_states,
            chunk_gates=chunk_gates,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            NT=num_chunks,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NV=num_value_tiles,
            STORE_STATES=first_level < state_levels,
            num_warps=4,
            num_stages=2,
        )
    for level in range(first_level + 1, num_levels):
        block = 1 << level
        half = block >> 1
        store_states = level < state_levels
        merge_nodes = num_chunks // block if store_states else num_chunks // (2 * block)
        merge_grid = (merge_nodes * batch * heads * num_tiles,)
        _merge_states_kernel[merge_grid](
            states=states,
            cached_states=cached_states,
            chunk_gates=chunk_gates,
            B=batch,
            H=heads,
            K=key_dim,
            V=value_dim,
            NT=num_chunks,
            LEVEL=level,
            HALF=half,
            BLOCK=block,
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NV=num_value_tiles,
            CACHE_STATES=cache_states,
            FIRST_LEVEL=first_level,
            STORE_STATES=store_states,
            num_warps=4,
            num_stages=2,
        )
    shared_grid_size = batch * num_chunks * heads * num_key_tiles
    share_value_tiles = False
    if fuse_forward and num_value_tiles > 1:
        compute_units = torch.cuda.get_device_properties(q.device).multi_processor_count
        if num_key_tiles == 1:
            share_value_tiles = shared_grid_size >= compute_units
        else:
            score_tiles = batch * num_chunks * num_key_tiles
            score_bytes = score_tiles * CHUNK_SIZE**2 * g.element_size()
            query_factor_bytes = g.nbytes * num_levels // 2
            share_value_tiles = (
                heads > 1 and score_tiles >= compute_units and score_bytes + query_factor_bytes <= states.nbytes
            )
    if fuse_forward:
        del states
    if share_value_tiles and num_key_tiles > 1:
        scores = torch.empty(
            batch, num_chunks, num_key_tiles, CHUNK_SIZE, CHUNK_SIZE, device=q.device, dtype=torch.float32
        )
        _scores_kernel[num_chunks, batch, num_key_tiles](
            q=q, k=k, scores=scores, T=sequence, K=key_dim, NT=num_chunks, NK=num_key_tiles, BK=BLOCK_KEY, num_warps=4
        )
        query_factors = torch.empty(
            batch, num_levels, num_chunks // 2, heads, CHUNK_SIZE, device=q.device, dtype=torch.float32
        )
        _prepare_query_factors_kernel[num_chunks, batch * heads](
            g=local_g,
            level_scales=level_scales,
            chunk_gates=chunk_gates,
            query_factors=query_factors,
            T=sequence,
            H=heads,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            num_warps=4,
            num_stages=2,
        )
    else:
        scores = query_factors = None
    key_tiles_per_program = 2 if share_value_tiles and num_key_tiles == 2 else 1
    output_key_tiles = num_key_tiles // key_tiles_per_program
    if fuse_forward:
        output_partial = torch.empty(
            (batch, sequence, heads, output_key_tiles, value_dim), device=v.device, dtype=v.dtype
        )
        if share_value_tiles:
            _shared_score_output_kernel[shared_grid_size // key_tiles_per_program,](
                q=q,
                k=k,
                v=v,
                g=local_g,
                level_scales=level_scales,
                level_lut=level_lut,
                scores=scores,
                states=cached_states,
                chunk_gates=chunk_gates,
                output=output_partial,
                query_factors=query_factors,
                B=batch,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                L=levels,
                NT=num_chunks,
                NUM_LEVELS=num_levels,
                MAX_BLOCK=1 << (num_levels - 1),
                BK=BLOCK_KEY,
                BV=BLOCK_VALUE,
                NK=num_key_tiles,
                NV=num_value_tiles,
                KEY_TILES_PER_PROGRAM=key_tiles_per_program,
                num_warps=4,
                num_stages=2,
            )
        else:
            _fused_output_kernel[output_grid](
                q=q,
                k=k,
                v=v,
                g=local_g,
                level_scales=level_scales,
                level_lut=level_lut,
                states=cached_states,
                chunk_gates=chunk_gates,
                output=output_partial,
                B=batch,
                T=sequence,
                H=heads,
                K=key_dim,
                V=value_dim,
                L=levels,
                NT=num_chunks,
                NUM_LEVELS=num_levels,
                MAX_BLOCK=1 << (num_levels - 1),
                BK=BLOCK_KEY,
                BV=BLOCK_VALUE,
                NK=num_key_tiles,
                NV=num_value_tiles,
                num_warps=4,
                num_stages=2,
            )
    elif cache_states:
        _inter_fwd_kernel[output_grid](
            q=q,
            g=local_g,
            level_scales=level_scales,
            local_output=output_partial,
            states=cached_states,
            chunk_gates=chunk_gates,
            output=output_partial,
            B=batch,
            T=sequence,
            H=heads,
            K=key_dim,
            V=value_dim,
            L=levels,
            NT=num_chunks,
            NUM_LEVELS=num_levels,
            MAX_BLOCK=1 << (num_levels - 1),
            BK=BLOCK_KEY,
            BV=BLOCK_VALUE,
            NK=num_key_tiles,
            NV=num_value_tiles,
            num_warps=4,
            num_stages=2,
        )
    del cached_states, scores, query_factors
    if output_key_tiles == 1:
        return output_partial.squeeze(3)
    if output is None:
        output = torch.empty_like(v)
    _reduce_output_flat_kernel[triton.cdiv(batch * sequence * heads * value_dim, REDUCTION_BLOCK),](
        partial=output_partial,
        output=output,
        N=batch * sequence * heads * value_dim,
        V=value_dim,
        NK=num_key_tiles,
        BLOCK=REDUCTION_BLOCK,
        num_warps=4,
    )
    return output
