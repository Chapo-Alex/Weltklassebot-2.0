"""Slippage strategy behavioural checks."""

from __future__ import annotations

import pytest

from execution.slippage import LinearSlippage, SquareRootSlippage


def test_linear_slippage_taker_prices_increase_with_size() -> None:
    model = LinearSlippage(bps_per_notional=0.01)
    mid = 100.0
    small = model.taker_price("buy", 1.0, mid)
    large = model.taker_price("buy", 5.0, mid)
    assert small > mid
    assert large > small
    assert model.taker_price("sell", 1.0, mid) < mid


def test_square_root_slippage_is_less_aggressive_than_linear() -> None:
    linear = LinearSlippage(bps_per_notional=0.02)
    sqrt = SquareRootSlippage(k=0.02)
    mid = 150.0
    notional = 9.0
    sqrt_price = sqrt.taker_price("buy", notional, mid)
    linear_price = linear.taker_price("buy", notional, mid)
    assert sqrt_price < linear_price
    assert sqrt.taker_price("sell", notional, mid) < mid


@pytest.mark.parametrize("queue_eta", [0.5, 1.5, 3.0])
def test_slippage_models_respect_seed_determinism(queue_eta: float) -> None:
    linear_a = LinearSlippage(bps_per_notional=0.01, seed=42)
    linear_b = LinearSlippage(bps_per_notional=0.01, seed=42)
    sqrt_a = SquareRootSlippage(k=0.015, seed=99)
    sqrt_b = SquareRootSlippage(k=0.015, seed=99)

    assert linear_a.maker_fill_prob(queue_eta) == pytest.approx(
        linear_b.maker_fill_prob(queue_eta)
    )
    assert sqrt_a.maker_fill_prob(queue_eta) == pytest.approx(
        sqrt_b.maker_fill_prob(queue_eta)
    )
