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

# Concrete primitive exports keep the generic Hy3 facade and its official
# tests unchanged. All variants accept upstream-prepared workspace objects.
_OPERATOR_EXPORTS = {
    "hy3_attention": (".hy3", "attention_with_kvcache_blocksparse_prefill_fp8"),
    "hy3_attention_decode_bf16_static": (".hy3", "attention_decode_bf16_static"),
    "hy3_attention_decode_bf16_dynamic": (".hy3", "attention_decode_bf16_dynamic"),
    "hy3_attention_decode_fp8_qk_static": (".hy3", "attention_decode_fp8_qk"),
    "hy3_attention_decode_fp8_qk_dynamic": (".hy3", "attention_decode_fp8_qk"),
    "hy3_attention_decode_fp8_kv_static": (".hy3", "attention_decode_fp8_kv"),
    "hy3_attention_decode_fp8_kv_dynamic": (".hy3", "attention_decode_fp8_kv"),
}
