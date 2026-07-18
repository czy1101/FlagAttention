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

import torch
import flag_attn


B, H, M, N, D = 2, 4, 10, 20, 128
causal = False
q = torch.randn((B, H, M, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
v = torch.randn((B, H, N, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)

o, logz, tot_attn = flag_attn.flash_attention(
	q, k, v, causal=causal, return_log_normalizer=True, return_total_attention=True)
o_ref, logz_ref, tot_attn_ref = flag_attn.testing.flash_attention(
	q, k, v, causal=causal, 
	return_log_normalizer=True, return_total_attention=True, pcast=True)

print("log normalizer")
print(logz[0, 0]) 
print(logz_ref[0, 0])

print("total attention")
print(tot_attn[0, 0]) 
print(tot_attn_ref[0, 0])
