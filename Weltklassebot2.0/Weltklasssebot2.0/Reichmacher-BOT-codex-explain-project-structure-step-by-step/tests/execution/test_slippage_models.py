"""Slippage model behavioural checks."""

from __future__ import annotations

import pytest

from execution.slippage import ImpactLinear, ImpactSqrt, QueuePosition


def test_linear_taker_prices_are_monotonic() -> None:
    model = ImpactLinear(k=0.01)
    mid = 100.0
    small = model.taker_price("buy", 1.0, mid)
    large = model.taker_price("buy", 5.0, mid)
    assert small > mid
    assert large > small
    assert model.taker_price("sell", 1.0, mid) < mid
    assert model.taker_price("buy", 0.0, mid) == pytest.approx(mid)


def test_sqrt_model_impacts_less_aggressively() -> None:
    model = ImpactSqrt(k=0.02)
    mid = 200.0
    linear = ImpactLinear(k=0.02)
    notional = 9.0
    sqrt_price = model.taker_price("buy", notional, mid)
    linear_price = linear.taker_price("buy", notional, mid)
    assert sqrt_price < linear_price
    assert model.taker_price("sell", notional, mid) < mid


def test_queue_position_probability_bounds() -> None:
    model = QueuePosition(eta=2.0)
    assert model.maker_fill_prob(-1.0) == pytest.approx(1.0)
    assert 0.0 <= model.maker_fill_prob(0.5) <= 1.0
    high_eta = model.maker_fill_prob(10.0)
    assert high_eta < model.maker_fill_prob(1.0)
    zero_impact = model.taker_price("buy", 5.0, 50.0)
    assert zero_impact == pytest.approx(50.0)
