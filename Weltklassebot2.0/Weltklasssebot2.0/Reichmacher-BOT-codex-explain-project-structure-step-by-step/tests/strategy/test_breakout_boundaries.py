"""Boundary-focused regression tests for the breakout bias strategy."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time, timedelta

import pytest

from core.events import CandleEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


def _clone(candle: CandleEvent, **kwargs) -> CandleEvent:
    return replace(candle, **kwargs)


def test_breakout_equality_emits_single_entry(tiny_candles) -> None:
    candles = tiny_candles(n=40)
    lookback = 8
    pivot = lookback
    history_window = candles[pivot - lookback : pivot]
    previous_high = max(candle.high for candle in history_window)
    modified = list(candles)
    base = candles[pivot]
    modified[pivot] = _clone(
        base,
        close=previous_high,
        high=max(base.high, previous_high),
        low=min(base.low, previous_high - 0.25),
    )

    config = StrategyConfig(
        order_size=0.4,
        lookback=lookback,
        bias_lookback=lookback,
        atr_lookback=lookback,
        breakout_threshold=0.0,
        threshold_mode="fixed",
        pyramid_steps=(1.0,),
        max_pyramids=1,
        news_blackout=(),
        session=(),
        bias_vol_ratio=5.0,
        bias_min_slope=0.0,
        atr_trailing_multiplier=6.0,
    )
    strategy = BreakoutBiasStrategy(config)

    orders = strategy.generate_orders(modified)
    buys = [order for order in orders if order.side is OrderSide.BUY]
    sells = [order for order in orders if order.side is OrderSide.SELL]

    assert len(buys) == 1
    assert buys[0].ts == modified[pivot].end
    assert all(order.reduce_only for order in sells)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["fixed", "geometric"])
def test_pyramiding_boundary_respects_limits(tiny_candles, mode: str) -> None:
    candles = tiny_candles(n=60)
    kwargs: dict[str, object] = {
        "order_size": 0.5,
        "lookback": 10,
        "bias_lookback": 10,
        "atr_lookback": 10,
        "breakout_threshold": 0.004,
        "threshold_mode": "fixed",
        "pyramid_mode": mode,
        "max_pyramids": 2,
        "news_blackout": (),
        "session": (),
        "bias_vol_ratio": 5.0,
        "bias_min_slope": 0.0,
        "atr_trailing_multiplier": 8.0,
        "vol_shock_multiple": 50.0,
        "flatten_on_session_close": False,
    }
    if mode == "fixed":
        kwargs["pyramid_steps"] = (1.0, 1.2)
    else:
        kwargs["pyramid_steps"] = (1.0,)
        kwargs["pyramid_scale"] = 1.35

    config = StrategyConfig(**kwargs)
    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)

    layers = 0
    open_qty = 0.0
    for order in orders:
        if order.side is OrderSide.BUY:
            layers += 1
            open_qty += order.qty
            assert layers <= config.max_pyramids
            assert order.reduce_only is False
        else:
            assert order.reduce_only is True
            open_qty = max(open_qty - order.qty, 0.0)
            layers = 0
        assert open_qty >= -1e-9

    total_buys = sum(1 for order in orders if order.side is OrderSide.BUY)
    assert total_buys <= config.max_pyramids


def test_session_and_vol_shock_resets() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)

    def _make_candle(
        minute: int,
        close: float,
        *,
        high: float | None = None,
        low: float | None = None,
    ) -> CandleEvent:
        start = base + timedelta(minutes=minute)
        end = start + timedelta(minutes=1)
        upper = high if high is not None else close + 0.2
        lower = low if low is not None else max(0.1, close - 0.2)
        return CandleEvent(
            symbol="BTCUSDT",
            open=close - 0.05,
            high=upper,
            low=lower,
            close=close,
            volume=25.0,
            start=start,
            end=end,
        )

    candles = [
        _make_candle(0, 100.0),
        _make_candle(1, 100.2),
        _make_candle(2, 100.4),
        _make_candle(3, 100.6),
        _make_candle(4, 100.8),
        _make_candle(5, 101.05, high=101.1),
        _make_candle(15, 101.2, high=101.25),
        _make_candle(21, 101.6, high=101.7),
        _make_candle(22, 101.4, high=108.0, low=94.0),
    ]

    config = StrategyConfig(
        order_size=0.6,
        lookback=5,
        bias_lookback=5,
        atr_lookback=5,
        breakout_threshold=0.0,
        threshold_mode="fixed",
        pyramid_steps=(1.0,),
        max_pyramids=1,
        news_blackout=(),
        session=(
            (time(0, 0), time(0, 10), False),
            (time(0, 20), time(0, 40), False),
        ),
        bias_vol_ratio=5.0,
        bias_min_slope=0.0,
        atr_trailing_multiplier=5.0,
        vol_shock_multiple=0.5,
        flatten_on_session_close=True,
    )
    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)

    assert any(order.side is OrderSide.BUY for order in orders)
    exits = [order for order in orders if order.side is OrderSide.SELL]
    assert exits, "Expected forced exits for session and volatility resets"
    reasons = {order.client_tag for order in exits}
    assert "session" in reasons
    assert "vol_shock" in reasons
    for exit_order in exits:
        assert exit_order.reduce_only is True
