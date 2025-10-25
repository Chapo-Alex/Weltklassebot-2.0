"""Slippage strategy implementations with deterministic RNG support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Protocol

try:  # pragma: no cover - numpy is an optional dependency
    from numpy.random import Generator, PCG64
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    import random as _random

    class Generator:  # type: ignore[override]
        """Minimal RNG stub exposing the subset of the numpy API we rely on."""

        __slots__ = ("_rng",)

        def __init__(self, seed: int | None = None) -> None:
            self._rng = _random.Random(seed)

        def random(self) -> float:
            return self._rng.random()

    class PCG64:  # type: ignore[override]
        def __init__(self, seed: int | None = None) -> None:
            self.seed = seed

    def _build_rng(seed: int | None = None) -> Generator:
        return Generator(seed)
else:  # pragma: no cover - numpy path

    def _build_rng(seed: int | None = None) -> Generator:
        return Generator(PCG64(seed))


class SlippageModel(Protocol):
    """Interface describing slippage adjustments for taker and maker flow."""

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        """Return the effective execution price for a taker order."""

    def maker_fill_prob(self, queue_eta: float) -> float:
        """Return the probability that a maker order fills over the next interval."""

    def bind_rng(self, rng: Generator) -> None:
        """Inject a deterministic RNG instance."""


@dataclass(slots=True)
class _BaseSlippage:
    seed: int | None = None
    _rng: Generator | None = field(default=None, init=False, repr=False)
    _local_rng: Generator | None = field(default=None, init=False, repr=False)

    def bind_rng(self, rng: Generator) -> None:
        self._rng = rng

    def _rng_instance(self) -> Generator | None:
        if self._rng is not None:
            return self._rng
        if self.seed is None:
            return None
        if self._local_rng is None:
            self._local_rng = _build_rng(self.seed)
        return self._local_rng


@dataclass(slots=True)
class LinearSlippage(_BaseSlippage):
    """Linear impact model proportional to order notional."""

    bps_per_notional: float = 0.0
    maker_queue_eta: float = 1.0

    def __post_init__(self) -> None:
        if self.bps_per_notional < 0:
            msg = "bps_per_notional must be non-negative"
            raise ValueError(msg)
        if self.maker_queue_eta <= 0:
            msg = "maker_queue_eta must be positive"
            raise ValueError(msg)

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        _validate_inputs(notional, mid)
        impact = self.bps_per_notional * notional
        adjustment = 1.0 + impact if side == "buy" else 1.0 - impact
        if adjustment <= 0:
            return 0.0
        return mid * adjustment

    def maker_fill_prob(self, queue_eta: float) -> float:
        scale = max(self.maker_queue_eta, 1e-9)
        impact_scale = 1.0 + self.bps_per_notional
        base = math.exp(-max(queue_eta, 0.0) * impact_scale / scale)
        rng = self._rng_instance()
        if rng is not None:
            # deterministic smoothing to avoid perfect 0/1 probabilities
            epsilon = 1e-3 * rng.random()
            base = max(0.0, min(1.0, base + epsilon))
        return _bounded_probability(base)


@dataclass(slots=True)
class SquareRootSlippage(_BaseSlippage):
    """Square-root impact model based on the order notional."""

    k: float = 0.0
    maker_queue_eta: float = 1.0

    def __post_init__(self) -> None:
        if self.k < 0:
            msg = "k must be non-negative"
            raise ValueError(msg)
        if self.maker_queue_eta <= 0:
            msg = "maker_queue_eta must be positive"
            raise ValueError(msg)

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        _validate_inputs(notional, mid)
        impact = self.k * math.sqrt(notional)
        adjustment = 1.0 + impact if side == "buy" else 1.0 - impact
        if adjustment <= 0:
            return 0.0
        return mid * adjustment

    def maker_fill_prob(self, queue_eta: float) -> float:
        scale = max(self.maker_queue_eta, 1e-9)
        base = math.exp(-max(queue_eta, 0.0) / scale)
        rng = self._rng_instance()
        if rng is not None:
            epsilon = 1e-3 * rng.random()
            base = max(0.0, min(1.0, base + epsilon))
        return _bounded_probability(base)


def _validate_inputs(notional: float, mid: float) -> None:
    if notional < 0:
        msg = "notional must be non-negative"
        raise ValueError(msg)
    if mid <= 0:
        msg = "mid price must be positive"
        raise ValueError(msg)


def _bounded_probability(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


__all__ = ["SlippageModel", "LinearSlippage", "SquareRootSlippage"]
