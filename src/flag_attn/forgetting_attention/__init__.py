"""Public Forgetting Attention exports."""
import importlib

from flag_attn.runtime.backend import resolve_operator

parallel_forgetting_attn = resolve_operator(
    "parallel_forgetting_attn", "flag_attn.forgetting_attention.parallel"
)
forgetting_attention = parallel_forgetting_attn


def __getattr__(name):
    if name != "naive_forgetting_attention":
        raise AttributeError(name)
    value = importlib.import_module(".naive", __name__).forgetting_attention
    globals()[name] = value
    return value


__all__ = [
    "parallel_forgetting_attn", "forgetting_attention", "naive_forgetting_attention",
]
