# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

from typing import Literal

from .operator import InfLLMV2Config, infllmv2_decode
from .operator import infllmv2_attention as _infllmv2_prefill


def infllmv2_attention(*args, stage: Literal["prefill", "decode"] = "prefill", **kwargs):
    """Run InfLLMv2 attention for the explicitly selected stage.

    Arguments and return values follow operator.infllmv2_attention for
    prefill (the default) and operator.infllmv2_decode for decode.
    The original config, cache layout and autograd behavior are preserved.
    """
    if stage == "prefill":
        return _infllmv2_prefill(*args, **kwargs)
    if stage == "decode":
        return infllmv2_decode(*args, **kwargs)
    raise ValueError(f"Unsupported InfLLMv2 stage: {stage!r}; expected 'prefill' or 'decode'")


__all__ = [
    "InfLLMV2Config",
    "infllmv2_attention",
    "infllmv2_decode",
]
