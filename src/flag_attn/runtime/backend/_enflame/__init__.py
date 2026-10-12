# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 attention backend."""

from __future__ import annotations

import importlib

_OPERATOR_EXPORTS = {
    "parallel_parallax": (".FLA.parallax", "parallel_parallax"),
    "parallax_decode": (".FLA.parallax", "parallax_decode"),
    "parallax_attn_with_kvcache": (".FLA.parallax", "parallax_attn_with_kvcache"),
    "hy3_attention": (".hpc_ops_attention.prefill", "attention_with_kvcache_blocksparse_prefill_fp8"),
    "hy3_attention_decode_bf16_static_prepare_static_bf16_workspace": (
        ".hpc_ops_attention.decode.static.bf16_static",
        "prepare_static_bf16_workspace",
    ),
    "hy3_attention_decode_bf16_static": (".hpc_ops_attention.decode.static.bf16_static", "attention_decode_bf16_tle"),
    "hy3_attention_decode_bf16_static_select_static_bf16_policy": (
        ".hpc_ops_attention.decode.static.bf16_static",
        "select_static_bf16_policy",
    ),
    "hy3_attention_decode_bf16_dynamic_prepare_dynamic_bf16_workspace": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic",
        "prepare_dynamic_bf16_workspace",
    ),
    "hy3_attention_decode_bf16_dynamic": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic",
        "attention_decode_bf16_dynamic",
    ),
    "hy3_attention_decode_bf16_dynamic_refresh_dynamic_bf16_task_map": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic",
        "refresh_dynamic_bf16_task_map",
    ),
    "hy3_attention_decode_bf16_dynamic_bf16_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic",
        "bf16_dynamic_workspace_is_reset",
    ),
    "hy3_attention_decode_fp8_qk_static_prepare_decode_workspace": (
        ".hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "prepare_decode_workspace",
    ),
    "hy3_attention_decode_fp8_qk_static": (
        ".hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "attention_decode_fp8",
    ),
    "hy3_attention_decode_fp8_qk_static_workspace_is_reset": (
        ".hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "workspace_is_reset",
    ),
    "hy3_attention_decode_fp8_qk_static_select_decode_policy": (
        ".hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "select_decode_policy",
    ),
    "hy3_attention_decode_fp8_qk_dynamic_prepare_decode_workspace": (
        ".hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic",
        "prepare_decode_workspace",
    ),
    "hy3_attention_decode_fp8_qk_dynamic": (
        ".hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic",
        "attention_decode_fp8",
    ),
    "hy3_attention_decode_fp8_qk_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic",
        "workspace_is_reset",
    ),
    "hy3_attention_decode_fp8_qk_dynamic_select_decode_policy": (
        ".hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic",
        "select_decode_policy",
    ),
    "hy3_attention_decode_fp8_kv_static_prepare_decode_workspace": (
        ".hpc_ops_attention.decode.static.fp8_qpertoken_perhead_kvpertensor_static",
        "prepare_decode_workspace",
    ),
    "hy3_attention_decode_fp8_kv_static": (
        ".hpc_ops_attention.decode.static.fp8_qpertoken_perhead_kvpertensor_static",
        "attention_decode_fp8",
    ),
    "hy3_attention_decode_fp8_kv_static_workspace_is_reset": (
        ".hpc_ops_attention.decode.static.fp8_qpertoken_perhead_kvpertensor_static",
        "workspace_is_reset",
    ),
    "hy3_attention_decode_fp8_kv_static_select_decode_policy": (
        ".hpc_ops_attention.decode.static.fp8_qpertoken_perhead_kvpertensor_static",
        "select_decode_policy",
    ),
    "hy3_attention_decode_fp8_kv_dynamic_prepare_decode_workspace": (
        ".hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic",
        "prepare_decode_workspace",
    ),
    "hy3_attention_decode_fp8_kv_dynamic": (
        ".hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic",
        "attention_decode_fp8",
    ),
    "hy3_attention_decode_fp8_kv_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic",
        "workspace_is_reset",
    ),
    "hy3_attention_decode_fp8_kv_dynamic_select_decode_policy": (
        ".hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic",
        "select_decode_policy",
    ),
    "SPARSE_BLOCK_SIZE": (".msa.index_topk", "SPARSE_BLOCK_SIZE"),
    "chunk_gdn2": (".FLA.gdn2", "chunk_gdn2"),
    "chunk_gdn2_native": (".FLA.gdn2.native.chunk_fwd", "chunk_gdn2_fwd"),
    "chunk_gla": (".FLA.gla", "chunk_gla"),
    "chunk_kda": (".FLA.kda", "chunk_kda"),
    "chunk_kda_fwd_infer": (".FLA.kda.chunk_kda", "chunk_kda_fwd_infer"),
    "install_msa_prefill": (".msa", "install_msa_prefill"),
    "minimax_m3_index_decode": (".msa", "minimax_m3_index_decode"),
    "minimax_m3_index_decode_score": (".msa", "minimax_m3_index_decode_score"),
    "minimax_m3_index_score": (".msa", "minimax_m3_index_score"),
    "minimax_m3_index_score_topk": (".msa", "minimax_m3_index_score_topk"),
    "minimax_m3_index_topk": (".msa", "minimax_m3_index_topk"),
    "minimax_m3_sparse_attn": (".msa", "minimax_m3_sparse_attn"),
    "minimax_m3_sparse_attn_decode": (".msa", "minimax_m3_sparse_attn_decode"),
    "parallel_nsa": (".FLA.nsa.parallel_nsa", "parallel_nsa"),
    "parallel_nsa_compression": (".FLA.nsa.parallel_nsa_compression", "parallel_nsa_compression"),
    "sage_attention": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".sage_attention.attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (".sage_attention.quant_per_block", "per_block_int8"),
}


def __getattr__(name: str):
    try:
        module_name, symbol = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = sorted(_OPERATOR_EXPORTS)
