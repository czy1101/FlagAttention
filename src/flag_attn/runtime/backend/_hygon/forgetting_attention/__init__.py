"""Public Forgetting Attention exports."""

from .parallel import (
    _forgetting_attention_impl as _naive_forgetting_attention,
    parallel_forgetting_attn,
)


def __getattr__(name):
    if name != "naive_forgetting_attention":
        raise AttributeError(name)
    globals()[name] = _naive_forgetting_attention
    return _naive_forgetting_attention


__all__ = [
    "parallel_forgetting_attn", "naive_forgetting_attention",
]
