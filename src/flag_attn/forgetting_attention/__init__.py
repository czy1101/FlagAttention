"""V7.6 Forgetting Attention / Adaptive Computation Pruning, SM90 inference.

Importing this package does not require a TLE compiler. Calling the operator does.
The public entry is cached after first resolution, preserving the V7.6 hot path.
"""
import importlib
import importlib.util


def has_tle():
    try:
        return importlib.util.find_spec("triton.experimental.tle") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def __getattr__(name):
    if name != "forgetting_attention":
        raise AttributeError(name)
    if not has_tle():
        raise RuntimeError("Forgetting Attention V7.6 requires Triton 3.6 with compatible FlagTree/TLE")
    value = importlib.import_module(".tle", __name__).forgetting_attention
    globals()[name] = value
    return value


__all__ = ["forgetting_attention", "has_tle"]
