from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """
    Minimal LangChain-like tool wrapper.

    WorkBench action traces call tools as: `<tool_name>.func(...)`.
    This wrapper provides `.name` and `.func` and remains callable.
    """

    name: str
    func: Callable[..., Any]
    return_direct: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(name: str, return_direct: bool = False) -> Callable[[Callable[..., Any]], Tool]:
    def _decorator(fn: Callable[..., Any]) -> Tool:
        wrapped = Tool(name=str(name), func=fn, return_direct=bool(return_direct))
        # Best-effort metadata for debugging/introspection.
        wrapped.__doc__ = getattr(fn, "__doc__", None)
        wrapped.__module__ = getattr(fn, "__module__", None)  # type: ignore[attr-defined]
        return wrapped

    return _decorator

