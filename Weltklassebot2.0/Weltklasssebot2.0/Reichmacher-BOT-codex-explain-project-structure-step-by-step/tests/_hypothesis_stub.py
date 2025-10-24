"""Minimal Hypothesis stub to keep tests runnable in offline environments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class _Strategy:
    factory: Callable[[], Any]

    def example(self) -> Any:
        return self.factory()


class _Strategies:
    """Tiny subset of Hypothesis strategies used in the test-suite."""

    @staticmethod
    def floats(
        min_value: float | None = None,
        max_value: float | None = None,
        allow_nan: bool = False,
        allow_infinity: bool = False,
    ) -> _Strategy:
        def _factory() -> float:
            lower = min_value if min_value is not None else 0.0
            upper = max_value if max_value is not None else lower + 1.0
            value = (lower + upper) / 2
            if not allow_nan and value != value:
                value = lower
            if not allow_infinity:
                return float(value)
            return float(value)

        return _Strategy(factory=_factory)

    @staticmethod
    def integers(
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> _Strategy:
        lower = 0 if min_value is None else int(min_value)
        upper = lower if max_value is None else int(max_value)
        if upper < lower:
            lower, upper = upper, lower

        def _factory() -> int:
            return (lower + upper) // 2

        return _Strategy(factory=_factory)

    @staticmethod
    def lists(
        strategy: _Strategy,
        min_size: int = 0,
        max_size: int | None = None,
    ) -> _Strategy:
        size = max(min_size, 1)
        if max_size is not None:
            size = min(size, max_size)

        def _factory() -> list[Any]:
            return [strategy.example() for _ in range(size)]

        return _Strategy(factory=_factory)


strategies = _Strategies()


def given(
    *strategies_args: _Strategy, **strategies_kwargs: _Strategy
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Simplistic replacement for :func:`hypothesis.given` used in tests."""

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            positional = [strategy.example() for strategy in strategies_args]
            named = {name: strategy.example() for name, strategy in strategies_kwargs.items()}
            return func(*args, *positional, **named)

        return _wrapper

    return _decorator
