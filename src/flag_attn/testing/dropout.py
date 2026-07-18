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
import triton
import triton.language as tl

@triton.jit
def recompute_mask_kernel(mask, B, H, M, N, dropout_p, seed, offset):
    row, b, h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_base = b * H * M * N + h * M * N + row * N
    BLOCK: tl.constexpr = 1024
    offs_base += tl.arange(0, BLOCK)
    for start_n in range(0, N, BLOCK):
        offs = start_n + offs_base
        rng_offs = offset + offs
        pmask = tl.rand(seed, rng_offs, n_rounds=6) > dropout_p
        row_mask = start_n + tl.arange(0, BLOCK) < N
        tl.store(mask + offs, pmask, mask=row_mask)

def recompute_mask(B, H, M, N, dropout_p, seed, offset, device):
    mask = torch.full((B, H, M, N), True, dtype=torch.bool, device=device)
    if dropout_p == 0:
        return mask
    grid = (M, B, H)
    with torch.cuda.device(device):
        recompute_mask_kernel[grid](mask, B, H, M, N, dropout_p, seed, offset)
    return mask
