# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Adapters from public HY3 FP8 inputs to the Ascend decode entry points."""

from .fp8 import FP8DecodeInputs
from .static.fp8_qkpertoken_perhead_vperhead_static import (
    attention_decode_fp8 as _attention_qk_static,
    prepare_decode_workspace as _prepare_qk_static,
)
from .dynamic.fp8_qkpertoken_perhead_vperhead_dynamic import (
    attention_decode_fp8 as _attention_qk_dynamic,
    prepare_decode_workspace as _prepare_qk_dynamic,
)
from .static.fp8_qpertoken_perhead_kvpertensor_static import (
    attention_decode_fp8 as _attention_kv_static,
    prepare_decode_workspace as _prepare_kv_static,
)
from .dynamic.fp8_qpertoken_perhead_kvpertensor_dynamic import (
    attention_decode_fp8 as _attention_kv_dynamic,
    prepare_decode_workspace as _prepare_kv_dynamic,
)


def _ascend_inputs(inputs):
    if isinstance(inputs, FP8DecodeInputs):
        return inputs
    return FP8DecodeInputs(
        q=inputs.q,
        k_cache=inputs.k_cache,
        v_cache=inputs.v_cache,
        block_ids=inputs.block_ids,
        kv_lens=inputs.kv_lens,
        q_scale=inputs.q_scale,
        k_scale=inputs.k_scale,
        v_scale=inputs.v_scale,
    )


def prepare_fp8_qk_static(inputs):
    return _prepare_qk_static(_ascend_inputs(inputs))


def attention_fp8_qk_static(inputs, workspace):
    return _attention_qk_static(_ascend_inputs(inputs), workspace)


def prepare_fp8_qk_dynamic(inputs):
    return _prepare_qk_dynamic(_ascend_inputs(inputs))


def attention_fp8_qk_dynamic(inputs, workspace):
    return _attention_qk_dynamic(_ascend_inputs(inputs), workspace)


def prepare_fp8_kv_static(inputs):
    return _prepare_kv_static(_ascend_inputs(inputs))


def attention_fp8_kv_static(inputs, workspace):
    return _attention_kv_static(_ascend_inputs(inputs), workspace)


def prepare_fp8_kv_dynamic(inputs):
    return _prepare_kv_dynamic(_ascend_inputs(inputs))


def attention_fp8_kv_dynamic(inputs, workspace):
    return _attention_kv_dynamic(_ascend_inputs(inputs), workspace)


__all__ = [
    "attention_fp8_kv_dynamic",
    "attention_fp8_kv_static",
    "attention_fp8_qk_dynamic",
    "attention_fp8_qk_static",
    "prepare_fp8_kv_dynamic",
    "prepare_fp8_kv_static",
    "prepare_fp8_qk_dynamic",
    "prepare_fp8_qk_static",
]
