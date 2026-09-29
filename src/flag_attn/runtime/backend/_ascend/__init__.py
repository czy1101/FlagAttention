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

"""Ascend backend of the HPC-Ops attention operators.

The operator packages are bound on first access instead of at import time, so
importing this backend pulls in neither torch-npu nor a Triton kernel.  The two
phases are also reachable directly, with the names the operator package uses:

    from flag_attn.runtime.backend._ascend import decode, prefill
"""

import importlib

# name -> (module relative to this package, attribute; None binds the module)
_OPERATOR_EXPORTS = {
    "decode": (".hpc_ops_attention.decode", None),
    "hpc_ops_attention": (".hpc_ops_attention", None),
    "prefill": (".hpc_ops_attention.prefill", None),
}

__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = importlib.import_module(module_name, __name__)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value
