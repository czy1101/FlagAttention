# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
import importlib

_OPERATOR_EXPORTS = {
    "chunk_gdn2": (".gdn2", "chunk_gdn2"),
    "forward": (".attn_qk_int8_per_block", "forward"),
    "hy3_attention": (
        ".hpc_ops_attention.prefill.attention_blocksparse_prefill_fp8",
        "attention_with_kvcache_blocksparse_prefill_fp8",
    ),
    "hy3_attention_decode_bf16_static": (
        ".hpc_ops_attention.decode.static.bf16_static", "attention_decode_bf16_tle"),
    "hy3_attention_decode_bf16_static_prepare": (
        ".hpc_ops_attention.decode.static.bf16_static", "prepare_static_bf16_workspace"),
    "hy3_attention_decode_bf16_dynamic": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic", "attention_decode_bf16_dynamic"),
    "hy3_attention_decode_bf16_dynamic_prepare": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic", "prepare_dynamic_bf16_workspace"),
    "hy3_attention_decode_bf16_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic", "bf16_dynamic_workspace_is_reset"),
    "hy3_attention_decode_bf16_dynamic_refresh_task_map": (
        ".hpc_ops_attention.decode.dynamic.bf16_dynamic", "refresh_dynamic_bf16_task_map"),
    "hy3_attention_decode_fp8_qk_static": (
        ".hpc_ops_attention.decode.api", "attention_fp8_qk_static"),
    "hy3_attention_decode_fp8_qk_static_prepare": (
        ".hpc_ops_attention.decode.api", "prepare_fp8_qk_static"),
    "hy3_attention_decode_fp8_qk_static_workspace_is_reset": (
        ".hpc_ops_attention.decode.static.fp8_qkpertoken_perhead_vperhead_static",
        "workspace_is_reset"),
    "hy3_attention_decode_fp8_qk_dynamic": (
        ".hpc_ops_attention.decode.api", "attention_fp8_qk_dynamic"),
    "hy3_attention_decode_fp8_qk_dynamic_prepare": (
        ".hpc_ops_attention.decode.api", "prepare_fp8_qk_dynamic"),
    "hy3_attention_decode_fp8_qk_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.fp8_qkpertoken_perhead_vperhead_dynamic",
        "workspace_is_reset"),
    "hy3_attention_decode_fp8_kv_static": (
        ".hpc_ops_attention.decode.api", "attention_fp8_kv_static"),
    "hy3_attention_decode_fp8_kv_static_prepare": (
        ".hpc_ops_attention.decode.api", "prepare_fp8_kv_static"),
    "hy3_attention_decode_fp8_kv_static_workspace_is_reset": (
        ".hpc_ops_attention.decode.static.fp8_qpertoken_perhead_kvpertensor_static",
        "workspace_is_reset"),
    "hy3_attention_decode_fp8_kv_dynamic": (
        ".hpc_ops_attention.decode.api", "attention_fp8_kv_dynamic"),
    "hy3_attention_decode_fp8_kv_dynamic_prepare": (
        ".hpc_ops_attention.decode.api", "prepare_fp8_kv_dynamic"),
    "hy3_attention_decode_fp8_kv_dynamic_workspace_is_reset": (
        ".hpc_ops_attention.decode.dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic",
        "workspace_is_reset"),
    "per_block_int8": (".ops", "quant_per_block_int8"),
    "quant_per_block_int8": (".ops", "quant_per_block_int8"),
    "sage_attention": (".attn_qk_int8_per_block", "forward"),
    "sage_attention_forward": (".attn_qk_int8_per_block", "forward"),
    "sage_attention_per_block_int8": (".ops", "quant_per_block_int8"),
}

__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
