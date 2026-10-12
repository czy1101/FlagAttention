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


from ._fp8 import (
    BLOCK_SIZE, HEAD_DIM, OFFICIAL_CASES, SUPPORTED_MTP,
    FP8DecodeInputs, FP8DecodePolicy, FP8DecodeWorkspace,
    attention_decode_fp8 as _attention_decode_fp8,
    prepare_decode_workspace as _prepare_decode_workspace,
    select_decode_policy,
    workspace_is_reset as _workspace_is_reset,
)

QUANT_TYPE = "qpertoken_perhead_kvpertensor"
QUANT_TYPE_ID = 1
QUANT_TYPES = (QUANT_TYPE,)


def prepare_decode_workspace(inputs):
    return _prepare_decode_workspace(inputs, QUANT_TYPE)


def attention_decode_fp8(inputs, workspace):
    return _attention_decode_fp8(inputs, workspace, QUANT_TYPE)


def workspace_is_reset(workspace):
    return _workspace_is_reset(workspace, QUANT_TYPE)


__all__ = [
    "BLOCK_SIZE", "HEAD_DIM", "OFFICIAL_CASES", "SUPPORTED_MTP",
    "QUANT_TYPE", "QUANT_TYPE_ID", "QUANT_TYPES",
    "FP8DecodeInputs", "FP8DecodePolicy", "FP8DecodeWorkspace",
    "attention_decode_fp8", "prepare_decode_workspace", "select_decode_policy",
    "workspace_is_reset",
]
