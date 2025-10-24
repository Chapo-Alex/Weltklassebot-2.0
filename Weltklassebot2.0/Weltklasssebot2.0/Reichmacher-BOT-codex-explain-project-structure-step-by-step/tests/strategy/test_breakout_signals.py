"""Signal validation for the breakout-bias strategy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.events import CandleEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


def _first_breakout_index(candles, lookback: int, threshold: float) -> int:
    for index in range(lookback, len(candles)):
        window = candles[index - lookback : index]
        previous_high = max(candle.high for candle in window)
        if candles[index].close >= previous_high * (1 + threshold):
            return index
    raise AssertionError("breakout not detected in fixture data")


def test_breakout_emits_single_entry(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        pyramid_steps=(1.0,),
        max_pyramids=1,
        atr_trailing_multiplier=2.0,
        bias_vol_ratio=0.5,
        bias_min_slope=0.0,
        news_blackout=(),
        session=(),
    )
    strategy = BreakoutBiasStrategy(config)

    orders = strategy.generate_orders(candles)
    buy_orders = [order for order in orders if order.side is OrderSide.BUY]
    sell_orders = [order for order in orders if order.side is OrderSide.SELL]

    assert len(buy_orders) == 1
    entry = buy_orders[0]
    assert entry.reduce_only is False
    assert entry.qty > 0

    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    assert entry.ts == candles[breakout_idx].end
    assert sell_orders == []


def test_weekend_entries_blocked(tiny_candles) -> None:
    candles = tiny_candles(n=60, start="2024-01-06T00:00:00Z")  # Saturday start
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        allow_weekends=False,
        news_blackout=(),
        session=(),
    )
    strategy = BreakoutBiasStrategy(config)

    orders = strategy.generate_orders(candles)
    assert all(order.side is not OrderSide.BUY for order in orders)


def test_percentile_threshold_uses_span(tiny_candles) -> None:
    candles = tiny_candles(n=12)
    config = StrategyConfig(threshold_mode="percentile", threshold_lookback=4)
    strategy = BreakoutBiasStrategy(config)
    history = candles[:4]
    reference_high = max(candle.high for candle in history)
    threshold = strategy._compute_breakout_threshold(history, reference_high)
    assert threshold > 0


def test_atr_k_threshold_calculates_ratio(tiny_candles) -> None:
    candles = tiny_candles(n=15)
    config = StrategyConfig(
        threshold_mode="atr_k",
        atr_lookback=5,
        atr_k_high=2.0,
        atr_k_low=0.5,
    )
    strategy = BreakoutBiasStrategy(config)
    history = candles[:6]
    reference_high = max(candle.high for candle in history)
    threshold = strategy._compute_breakout_threshold(history, reference_high)
    assert threshold > 0


def test_bias_blocks_on_high_volatility() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        CandleEvent(
            symbol="BTCUSDT",
            open=100 + i,
            high=120 + i,
            low=80 + i,
            close=110 + i,
            volume=1.0,
            start=start + timedelta(minutes=i),
            end=start + timedelta(minutes=i + 1),
        )
        for i in range(5)
    ]
    config = StrategyConfig(
        lookback=5,
        bias_lookback=5,
        breakout_threshold=0.0,
        bias_vol_ratio=0.01,
        threshold_mode="fixed",
    )
    strategy = BreakoutBiasStrategy(config)
    assert strategy._bias_allows_entry(candles) is False
