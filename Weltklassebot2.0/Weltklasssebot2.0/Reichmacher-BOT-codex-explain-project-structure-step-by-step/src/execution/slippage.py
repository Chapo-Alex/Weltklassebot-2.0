"""Deterministic slippage and queue position models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol


class SlippageModel(Protocol):
    """Interface describing slippage adjustments for taker and maker flow."""

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        """Return the effective execution price for a taker order."""

    def maker_fill_prob(self, queue_eta: float) -> float:
        """Return the probability that a maker order fills over the next interval."""


@dataclass(slots=True)
class ImpactLinear:
    """Linear impact model proportional to order notional."""

    k: float = 0.0

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        _validate_inputs(notional, mid)
        impact = self.k * notional
        adjustment = 1.0 + impact if side == "buy" else 1.0 - impact
        return max(mid * adjustment, 0.0)

    def maker_fill_prob(self, queue_eta: float) -> float:
        return _bounded_probability(math.exp(-max(queue_eta, 0.0) * (1.0 + self.k)))


@dataclass(slots=True)
class ImpactSqrt:
    """Square-root impact model based on the order notional."""

    k: float = 0.0

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        _validate_inputs(notional, mid)
        impact = self.k * math.sqrt(notional)
        adjustment = 1.0 + impact if side == "buy" else 1.0 - impact
        return max(mid * adjustment, 0.0)

    def maker_fill_prob(self, queue_eta: float) -> float:
        scale = 1.0 + self.k
        return _bounded_probability(math.exp(-max(queue_eta, 0.0) * scale))


@dataclass(slots=True)
class QueuePosition:
    """Ex-ante maker fill probability derived from queue position."""

    eta: float = 1.0

    def taker_price(self, side: Literal["buy", "sell"], notional: float, mid: float) -> float:
        _validate_inputs(notional, mid)
        return mid

    def maker_fill_prob(self, queue_eta: float) -> float:
        scale = max(self.eta, 1e-12)
        exponent = -max(queue_eta, 0.0) / scale
        return _bounded_probability(math.exp(exponent))


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


__all__ = [
    "SlippageModel",
    "ImpactLinear",
    "ImpactSqrt",
    "QueuePosition",
]
