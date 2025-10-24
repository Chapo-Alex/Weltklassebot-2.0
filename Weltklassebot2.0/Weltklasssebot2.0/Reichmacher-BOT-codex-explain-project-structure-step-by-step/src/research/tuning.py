"""Simple configuration search utilities and evaluation objectives."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfigSpace:
    """Discrete configuration space used for grid or random search."""

    parameters: Mapping[str, Sequence[float | int | str]]

    def __post_init__(self) -> None:
        if not self.parameters:
            msg = "ConfigSpace requires at least one parameter"
            raise ValueError(msg)
        for key, values in self.parameters.items():
            if not values:
                msg = f"parameter {key!r} must provide at least one option"
                raise ValueError(msg)


def grid(params: ConfigSpace) -> list[dict[str, Any]]:
    """Return the cartesian product of all parameter values."""

    items = [(key, list(values)) for key, values in params.parameters.items()]
    if not items:
        return [{}]

    def _product(level: int, prefix: dict[str, Any]) -> list[dict[str, Any]]:
        if level == len(items):
            return [dict(prefix)]
        key, values = items[level]
        combinations: list[dict[str, Any]] = []
        for value in values:
            prefix[key] = value
            combinations.extend(_product(level + 1, prefix))
        prefix.pop(key, None)
        return combinations

    return _product(0, {})


def random(params: ConfigSpace, n: int, rng: Any) -> list[dict[str, Any]]:
    """Sample ``n`` configurations using ``rng``."""

    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)
    keys = list(params.parameters.keys())
    if not keys:
        return [{}]

    samples: list[dict[str, Any]] = []
    for _ in range(n):
        choice: dict[str, Any] = {}
        for key in keys:
            options = list(params.parameters[key])
            if not options:
                msg = f"parameter {key!r} must provide at least one option"
                raise ValueError(msg)
            index = int(rng.integers(0, len(options)))
            choice[key] = options[index]
        samples.append(choice)
    return samples


def objective_sharpe_penalized(
    equity: Sequence[float],
    turnover: float,
    max_dd: float,
    w_turnover: float = 0.1,
    w_dd: float = 0.2,
) -> float:
    """Return a Sharpe-like score penalised by turnover and drawdown.

    Sharpe is computed on the first difference of ``equity``.  When the equity
    series is constant or too short, the Sharpe contribution becomes zero.
    Penalties are subtracted in proportion to ``turnover`` and ``max_dd``.
    """

    if len(equity) < 2:
        sharpe = 0.0
    else:
        returns = [equity[i] - equity[i - 1] for i in range(1, len(equity))]
        avg = mean(returns)
        variance = mean((value - avg) ** 2 for value in returns)
        if variance <= 0.0:
            sharpe = 0.0
        else:
            sharpe = avg / sqrt(variance)

    penalty = (w_turnover * turnover) + (w_dd * max_dd)
    return sharpe - penalty


__all__ = ["ConfigSpace", "grid", "objective_sharpe_penalized", "random"]

