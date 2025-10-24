"""Research utilities for walk-forward evaluation and tuning."""

from __future__ import annotations

from .diagnostics import permutation_test, whites_reality_check
from .splits import time_kfold
from .tuning import ConfigSpace, grid, objective_sharpe_penalized, random

__all__ = [
    "ConfigSpace",
    "permutation_test",
    "grid",
    "objective_sharpe_penalized",
    "random",
    "time_kfold",
    "whites_reality_check",
]

