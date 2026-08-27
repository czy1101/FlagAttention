# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0.

"""Enflame S60 attention backend."""

from .msa import install_msa_prefill
from .nsa import parallel_nsa

__all__ = [
    "install_msa_prefill",
    "parallel_nsa",
]
