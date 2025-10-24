"""Exit logic coverage for the breakout-bias strategy."""

from __future__ import annotations

from dataclasses import replace
from datetime import time

from core.events import OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


def _first_breakout_index(candles, lookback: int, threshold: float) -> int:
    for index in range(lookback, len(candles)):
        window = candles[index - lookback : index]
        previous_high = max(candle.high for candle in window)
        if candles[index].close >= previous_high * (1 + threshold):
            return index
    raise AssertionError("breakout not detected in fixture data")


def _net_position(orders) -> float:  # type: ignore[no-untyped-def]
    position = 0.0
    for order in orders:
        if order.side is OrderSide.BUY:
            position += order.qty
        else:
            position -= order.qty
    return position


def _assert_exit_reason(orders, reason: str) -> None:
    exit_orders = [order for order in orders if order.side is OrderSide.SELL]
    assert exit_orders, "expected at least one exit order"
    exit_order = exit_orders[0]
    assert exit_order.reduce_only is True
    assert exit_order.client_tag == reason
    assert abs(_net_position(orders)) < 1e-6


def test_atr_trailing_exit_is_reduce_only(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        exit_mode="atr_trail",
        atr_trailing_multiplier=1.0,
        pyramid_steps=(1.0, 0.75),
        max_pyramids=2,
        flatten_on_session_close=False,
        vol_shock_multiple=50.0,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    drop_idx = min(len(candles) - 1, breakout_idx + 5)
    base_close = candles[breakout_idx].close
    shock_close = max(base_close * 0.75, 0.05)
    candles[drop_idx] = replace(
        candles[drop_idx],
        open=shock_close * 1.02,
        high=max(shock_close * 1.05, candles[drop_idx].high),
        low=shock_close * 0.95,
        close=shock_close,
    )

    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)
    _assert_exit_reason(orders, "atr_trail")


def test_chandelier_exit_closes_position(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        exit_mode="chandelier",
        chandelier_lookback=10,
        chandelier_atr_mult=1.2,
        flatten_on_session_close=False,
        vol_shock_multiple=50.0,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    drop_idx = min(len(candles) - 1, breakout_idx + 6)
    shock_close = max(candles[breakout_idx].close * 0.7, 0.05)
    candles[drop_idx] = replace(
        candles[drop_idx],
        open=shock_close * 1.03,
        high=max(shock_close * 1.08, candles[drop_idx].high),
        low=shock_close * 0.9,
        close=shock_close,
    )

    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)
    _assert_exit_reason(orders, "chandelier")


def test_session_reset_flatten(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        exit_mode="atr_trail",
        atr_trailing_multiplier=3.0,
        flatten_on_session_close=True,
        session=((time(hour=0, minute=0), time(hour=0, minute=45), False),),
        news_blackout=(),
        vol_shock_multiple=50.0,
    )

    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)
    _assert_exit_reason(orders, "session")


def test_vol_shock_reset_triggers_exit(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        exit_mode="atr_trail",
        atr_trailing_multiplier=10.0,
        flatten_on_session_close=False,
        vol_shock_multiple=1.2,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    shock_idx = min(len(candles) - 1, breakout_idx + 3)
    high = candles[shock_idx].close + 4.0
    low = max(candles[shock_idx].close - 3.5, 0.05)
    candles[shock_idx] = replace(
        candles[shock_idx],
        high=high,
        low=low,
        close=low,
    )

    strategy = BreakoutBiasStrategy(config)
    orders = strategy.generate_orders(candles)
    _assert_exit_reason(orders, "vol_shock")


def test_risk_notification_forces_exit(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.01,
        threshold_mode="fixed",
        exit_mode="atr_trail",
        atr_trailing_multiplier=4.0,
        flatten_on_session_close=False,
        vol_shock_multiple=50.0,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    strategy = BreakoutBiasStrategy(config)

    initial_orders = strategy.generate_orders(candles[: breakout_idx + 1])
    assert any(order.side is OrderSide.BUY for order in initial_orders)

    strategy.notify_risk_drawdown("BTCUSDT")
    remaining_orders = strategy.generate_orders(candles[breakout_idx + 1 :])
    exits = [order for order in remaining_orders if order.side is OrderSide.SELL]
    assert exits
    exit_order = exits[0]
    assert exit_order.client_tag == "risk"
    assert exit_order.reduce_only is True


def test_take_profit_exit_triggers(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.0,
        threshold_mode="fixed",
        take_profit_pct=0.02,
        stop_loss_pct=None,
        exit_mode="atr_trail",
        atr_trailing_multiplier=10.0,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    tp_idx = min(len(candles) - 1, breakout_idx + 2)
    base_price = candles[breakout_idx].close
    candles[tp_idx] = replace(
        candles[tp_idx],
        high=base_price * 1.05,
        close=base_price * 1.04,
    )

    strategy = BreakoutBiasStrategy(config)
    partial = strategy.generate_orders(candles[: tp_idx + 1])
    _assert_exit_reason(partial, "take_profit")


def test_stop_loss_exit_triggers(tiny_candles) -> None:
    candles = tiny_candles(n=60)
    config = StrategyConfig(
        order_size=1.0,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.0,
        threshold_mode="fixed",
        take_profit_pct=None,
        stop_loss_pct=0.02,
        exit_mode="atr_trail",
        atr_trailing_multiplier=10.0,
        news_blackout=(),
        session=(),
    )
    breakout_idx = _first_breakout_index(candles, config.lookback, config.breakout_threshold)
    sl_idx = min(len(candles) - 1, breakout_idx + 3)
    base_price = candles[breakout_idx].close
    candles[sl_idx] = replace(
        candles[sl_idx],
        low=base_price * 0.95,
        close=base_price * 0.96,
    )

    strategy = BreakoutBiasStrategy(config)
    partial = strategy.generate_orders(candles[: sl_idx + 1])
    _assert_exit_reason(partial, "stop_loss")
