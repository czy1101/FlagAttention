"""Backend dispatcher for Inkling FA4 relative attention.

CuTe is optional. Importing :mod:`inkling_fa4` therefore never requires the
out-of-tree ``cute_sm90_backend`` module.  The default order is TLE -> Triton;
CuTe is selected only when explicitly requested.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Callable


_ALIASES = {
    "auto": "auto",
    "tle": "tle",
    "triton_tle": "tle",
    "triton-tle": "tle",
    "triton": "triton",
    "base": "triton",
    "cute": "cute",
    "cute_sm90": "cute",
    "cute-sm90": "cute",
}


def _load_tle() -> Callable[..., Any]:
    module = import_module("inkling_fa4.triton_tle_kernel")
    return module.inkling_fa4_rel_attention_tle


def _load_triton() -> Callable[..., Any]:
    module = import_module("inkling_fa4.triton_kernel")
    return getattr(
        module,
        "inkling_fa4_rel_attention_triton",
        module.inkling_fa4_rel_attention,
    )


def _load_cute() -> Callable[..., Any]:
    try:
        module = import_module("inkling_fa4.cute_sm90_backend")
    except ModuleNotFoundError as exc:
        if exc.name != "inkling_fa4.cute_sm90_backend":
            raise
        raise RuntimeError(
            "请求了 CuTe 后端，但 src/inkling_fa4/cute_sm90_backend.py 不存在。"
            "请安装/恢复 CuTe 适配模块，或使用 backend='tle' / backend='triton'。"
        ) from exc

    for name in (
        "inkling_fa4_rel_attention_cute",
        "inkling_fa4_rel_attention",
    ):
        function = getattr(module, name, None)
        if function is not None:
            return function
    raise RuntimeError("cute_sm90_backend.py 未导出 Inkling FA4 算子函数")


def get_backend(name: str | None = None) -> Callable[..., Any]:
    """Return an implementation without launching the operator."""
    requested = (name or os.getenv("INKLING_FA4_BACKEND", "auto")).lower()
    try:
        selected = _ALIASES[requested]
    except KeyError as exc:
        choices = ", ".join(sorted(_ALIASES))
        raise ValueError(f"未知 Inkling FA4 backend={requested!r}；可选：{choices}") from exc

    if selected == "tle":
        return _load_tle()
    if selected == "triton":
        return _load_triton()
    if selected == "cute":
        return _load_cute()

    # auto：优先使用目标优化版本。仅当 TLE 本身无法导入时回退；算子编译或
    # 运行错误不能静默回退，否则会掩盖 TLE kernel 的真实问题。
    try:
        return _load_tle()
    except (ImportError, AttributeError):
        return _load_triton()


def inkling_fa4_rel_attention(
    *args: Any,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Dispatch to TLE, Triton, or optional CuTe implementation.

    Select with ``backend=...`` or ``INKLING_FA4_BACKEND``.  The keyword is
    consumed here and is not forwarded to the kernel wrapper.
    """
    implementation = get_backend(backend)
    return implementation(*args, **kwargs)


__all__ = ["get_backend", "inkling_fa4_rel_attention"]
