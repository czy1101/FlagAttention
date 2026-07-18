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
import pytest
from flag_attn.testing import recompute_mask


@pytest.mark.parametrize('B, H, M, N', [
    (2, 4, 512, 612),
    (2, 4, 1024, 1034),
    (2, 4, 2048, 2048),
    (2, 4, 4096, 4096),
    (2, 4, 4001, 4001),
    (2, 4, 4001, 4096),
    (2, 4, 4096, 4000),
    (1, 2, 8192, 8202),
    (1, 2, 8192, 8192),
])
@pytest.mark.parametrize('p', [0.5, 0.8])
def test_recompute_mask(B, H, M, N, p):
    import math
    seed = 123456789
    offset = 123456789123456789
    device = torch.cuda.current_device()
    mask = recompute_mask(B, H, M, N, p, seed, offset, device)
    # zeros indicate to drop
    # k follows Binomial distributio B(k; n, p)
    n = mask.numel()
    k = torch.sum(mask == 0)
    p_cap = k / n
    tol = 0.01
    assert math.fabs(p_cap - p) < tol * p
