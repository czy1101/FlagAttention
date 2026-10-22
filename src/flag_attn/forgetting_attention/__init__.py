"""Forgetting Attention / ACP: optional H100 TLE and plain Triton paths."""
import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def has_tle():
    """Probe required Python APIs, not compiler correctness or GPU support.

    Only missing imports/APIs select the fallback. Kernel/compilation failures
    and other unexpected import errors are deliberately not swallowed.
    """
    try:
        tle = importlib.import_module("triton.experimental.tle.language")
        types = importlib.import_module("triton.experimental.tle.language.gpu.types")
        descriptors = importlib.import_module("triton.tools.tensor_descriptor")
        from triton import knobs
        from triton.runtime import jit
    except ImportError:
        return False
    gpu = getattr(tle, "gpu", None)
    required = (
        "alloc", "alloc_barrier", "alloc_barriers", "barrier_wait", "copy",
        "local_ptr", "set_layout", "wgmma", "wgmma_wait",
    )
    return (
        all(callable(getattr(gpu, name, None)) for name in required)
        and callable(getattr(types, "BlockEncoding", None))
        and callable(getattr(descriptors, "TensorDescriptor", None))
        and callable(getattr(jit, "JITFunction", None))
        and hasattr(getattr(knobs, "runtime", None), "debug")
        and hasattr(getattr(knobs, "compilation", None), "instrumentation_mode")
    )


def __getattr__(name):
    if name == "forgetting_attention":
        module, attribute = ((".parallel", "forgetting_attention") if has_tle()
                             else (".naive", "triton_forgetting_attention"))
    elif name == "naive_forgetting_attention":
        module, attribute = ".naive", "forgetting_attention"
    elif name == "triton_forgetting_attention":
        module, attribute = ".naive", "triton_forgetting_attention"
    else:
        raise AttributeError(name)
    value = getattr(importlib.import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = ["forgetting_attention", "naive_forgetting_attention",
           "triton_forgetting_attention", "has_tle"]
