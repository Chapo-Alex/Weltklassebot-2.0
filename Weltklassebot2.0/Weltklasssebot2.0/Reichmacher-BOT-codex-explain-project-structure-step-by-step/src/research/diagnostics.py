"""Statistical diagnostics to guard against overfitting."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import sqrt
from random import Random
from typing import TYPE_CHECKING, Any

try:  # NumPy is optional for the permutation tests
    import numpy as _np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - import guard
    _np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.random import Generator as NpGenerator
    from numpy.typing import NDArray
else:  # pragma: no cover - runtime fallback when numpy missing
    NDArray = Any
    NpGenerator = Any


def _ensure_random(rng: Random | None) -> Random:
    if rng is not None:
        return rng
    return Random(1337)


def _ensure_numpy_generator(rng: NpGenerator | None) -> NpGenerator:
    if _np is None:
        raise RuntimeError("numpy-unavailable")
    if rng is not None:
        return rng
    return _np.random.default_rng(1337)


def permutation_test(
    returns: Sequence[float],
    stat_fn: Callable[[Sequence[float]], float],
    n: int = 200,
    rng: Random | None = None,
) -> tuple[float, float]:
    """Return (observed_statistic, p_value) from a permutation test.

    The test shuffles ``returns`` ``n`` times, evaluates ``stat_fn`` on each
    permutation, and computes a two-sided p-value based on the proportion of
    permuted statistics whose absolute value exceeds the observed statistic.
    """

    if n <= 0:
        msg = "n must be positive"
        raise ValueError(msg)

    data = [float(x) for x in returns]
    if len(data) < 2:
        msg = "returns must contain at least two observations"
        raise ValueError(msg)

    generator = _ensure_random(rng)
    observed = float(stat_fn(tuple(data)))
    threshold = abs(observed)

    permuted = data.copy()
    exceed = 0
    for _ in range(n):
        generator.shuffle(permuted)
        stat = float(stat_fn(tuple(permuted)))
        if abs(stat) >= threshold:
            exceed += 1

    p_value = (exceed + 1) / (n + 1)
    return observed, p_value


def whites_reality_check(
    stats_matrix: NDArray[Any],
    boot: int = 500,
    rng: NpGenerator | None = None,
) -> float:
    """Return the White's Reality Check p-value for a strategy ensemble.

    Given a matrix of strategy performance statistics with shape
    ``(num_strategies, samples)``, this function estimates the probability that
    the maximum observed mean performance could arise under the null hypothesis
    of no skill.  Bootstrapped samples are drawn with replacement across the
    time dimension and compared against the observed maximum.  The p-value is
    computed as::

        p = (1 + sum(max_b >= max_obs)) / (boot + 1)

    where ``max_b`` are the bootstrap maxima of the centred statistics.
    """

    if boot <= 0:
        msg = "boot must be positive"
        raise ValueError(msg)

    generator = _ensure_numpy_generator(rng)
    data = _np.asarray(stats_matrix, dtype=float)
    if data.ndim != 2:
        msg = "stats_matrix must be two-dimensional"
        raise ValueError(msg)

    strategies, samples = data.shape
    if strategies == 0 or samples == 0:
        msg = "stats_matrix must contain at least one strategy and one sample"
        raise ValueError(msg)

    means = data.mean(axis=1)
    observed = sqrt(float(samples)) * float(_np.max(means))

    exceed = 0
    for _ in range(boot):
        indices = generator.integers(0, samples, size=samples)
        sample = data[:, indices]
        centred = sample.mean(axis=1) - means
        stat = sqrt(float(samples)) * float(_np.max(centred))
        if stat >= observed:
            exceed += 1

    p_value = (exceed + 1) / (boot + 1)
    return p_value


__all__ = ["permutation_test", "whites_reality_check"]

