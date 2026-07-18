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

B, H, M, N, D = 2, 16, 4000, 4000, 128
causal = True

q = torch.randn(B, H, M, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)
v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device="cuda:0", requires_grad=True)


o_ref = flag_attn.testing.flash_attention(q, k, v, causal=causal, upcast=True)
o = flag_attn.flash_attention(q, k, v, causal=causal)
o_torch = flag_attn.testing.flash_attention(q, k, v, causal=causal)

go = torch.randn_like(o)
gq_ref, gk_ref, gv_ref = torch.autograd.grad(o_ref, (q, k, v), go)
gq, gk, gv = torch.autograd.grad(o, (q, k, v), go)
gq_torch, gk_torch, gv_torch = torch.autograd.grad(o_torch, (q, k, v), go)
