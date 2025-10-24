from __future__ import annotations

from collections.abc import Sequence
from random import Random
from statistics import mean
from typing import Any

import pytest

from research.diagnostics import permutation_test, whites_reality_check


def _split_mean_diff(values: Sequence[float]) -> float:
    midpoint = len(values) // 2
    first = mean(values[:midpoint])
    second = mean(values[midpoint:])
    return float(first - second)


def test_permutation_detects_signal() -> None:
    strong = [0.015] * 20 + [-0.005] * 20
    observed, p_value = permutation_test(
        strong,
        _split_mean_diff,
        n=300,
        rng=Random(1337),
    )
    assert observed > 0.0
    assert p_value < 0.05


def test_permutation_accepts_noise() -> None:
    generator = Random(1337)
    noise = [generator.gauss(0.0, 0.01) for _ in range(40)]
    observed, p_value = permutation_test(
        noise,
        _split_mean_diff,
        n=300,
        rng=Random(2024),
    )
    assert abs(observed) < 0.01
    assert p_value > 0.2


def _call_whites(matrix: Any, boot: int = 300) -> float:
    np = pytest.importorskip("numpy")
    try:
        return whites_reality_check(matrix, boot=boot, rng=np.random.default_rng(1337))
    except RuntimeError as exc:  # pragma: no cover - exercised when numpy missing at runtime
        if str(exc) == "numpy-unavailable":
            pytest.skip("numpy required for White's reality check")
        raise


@pytest.mark.needs_numpy
def test_white_reality_check_strong_strategy() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1337)
    samples = 120
    strat_a = 0.002 + rng.normal(0.0, 0.0005, size=samples)
    strat_b = rng.normal(0.0, 0.0005, size=samples)
    matrix = np.vstack([strat_a, strat_b])

    p_value = _call_whites(matrix, boot=300)
    assert p_value < 0.05


@pytest.mark.needs_numpy
def test_white_reality_check_noise_controls() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1337)
    samples = 120
    strat_a = rng.normal(0.0, 0.0005, size=samples)
    strat_b = rng.normal(0.0, 0.0005, size=samples)
    matrix = np.vstack([strat_a, strat_b])

    p_value = _call_whites(matrix, boot=300)
    assert p_value > 0.2
