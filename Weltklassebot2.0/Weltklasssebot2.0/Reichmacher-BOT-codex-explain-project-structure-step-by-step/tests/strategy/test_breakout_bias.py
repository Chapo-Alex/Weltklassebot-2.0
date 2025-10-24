from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import UTC, datetime, timedelta

import pytest

try:  # pragma: no cover - optional dependency for CI
    from hypothesis import given, strategies as st
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    stub_name = "_hypothesis_stub"
    if stub_name in sys.modules:
        module = sys.modules[stub_name]
    else:
        stub_path = pathlib.Path(__file__).resolve().parents[1] / "_hypothesis_stub.py"
        spec = importlib.util.spec_from_file_location(stub_name, stub_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[stub_name] = module
        spec.loader.exec_module(module)
    given = module.given
    st = module.strategies

from core.events import CandleEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


def _build_candle(
    symbol: str,
    start: datetime,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
) -> CandleEvent:
    return CandleEvent(
        symbol=symbol,
        open=open_price,
        high=high_price if high_price is not None else max(open_price, close_price),
        low=low_price if low_price is not None else min(open_price, close_price),
        close=close_price,
        volume=1.0,
        start=start,
        end=start + timedelta(minutes=1),
    )


def test_breakout_requires_bias_alignment() -> None:
    config = StrategyConfig(
        order_size=0.5,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        threshold_mode="fixed",
    )
    strategy = BreakoutBiasStrategy(config)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        _build_candle("BTCUSDT", start + timedelta(minutes=i), 100.0 + i, 100.5 + i)
        for i in range(3)
    ]
    candles.append(
        _build_candle(
            "BTCUSDT",
            start + timedelta(minutes=3),
            103.0,
            106.0,
            high_price=106.5,
            low_price=102.5,
        )
    )

    orders = strategy.generate_orders(candles)
    buys = [order for order in orders if order.side is OrderSide.BUY]

    assert len(buys) == 1
    assert buys[0].qty == pytest.approx(0.5)


def test_bias_filter_blocks_when_trend_negative() -> None:
    config = StrategyConfig(
        order_size=0.4,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        bias_min_slope=0.02,
    )
    strategy = BreakoutBiasStrategy(config)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        _build_candle("ETHUSDT", start, 100.0, 99.5, high_price=100.5, low_price=99.0),
        _build_candle(
            "ETHUSDT",
            start + timedelta(minutes=1),
            99.5,
            99.0,
            high_price=100.0,
            low_price=98.5,
        ),
        _build_candle(
            "ETHUSDT",
            start + timedelta(minutes=2),
            99.0,
            98.8,
            high_price=99.4,
            low_price=98.2,
        ),
        _build_candle(
            "ETHUSDT",
            start + timedelta(minutes=3),
            98.8,
            100.6,
            high_price=101.2,
            low_price=98.6,
        ),
    ]

    orders = strategy.generate_orders(candles)

    assert all(order.side is not OrderSide.BUY for order in orders)


def test_exit_orders_are_reduce_only() -> None:
    config = StrategyConfig(
        order_size=1.0,
        breakout_threshold=0.0,
        lookback=2,
        bias_lookback=2,
        take_profit_pct=0.01,
        stop_loss_pct=None,
    )
    strategy = BreakoutBiasStrategy(config)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        _build_candle("SOLUSDT", start, 50.0, 50.5, high_price=50.6, low_price=49.8),
        _build_candle(
            "SOLUSDT",
            start + timedelta(minutes=1),
            50.5,
            51.5,
            high_price=51.6,
            low_price=50.2,
        ),
        _build_candle(
            "SOLUSDT",
            start + timedelta(minutes=2),
            51.5,
            52.0,
            high_price=52.2,
            low_price=51.0,
        ),
        _build_candle(
            "SOLUSDT",
            start + timedelta(minutes=3),
            52.0,
            52.6,
            high_price=53.5,
            low_price=51.8,
        ),
    ]

    orders = strategy.generate_orders(candles)
    sells = [order for order in orders if order.side is OrderSide.SELL]

    assert sells, "expected a reduce-only exit"
    assert all(order.reduce_only for order in sells)


def test_news_blackout_blocks_entries() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    config = StrategyConfig(
        order_size=0.3,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        news_blackout=((start + timedelta(minutes=1), start + timedelta(minutes=4)),),
    )
    strategy = BreakoutBiasStrategy(config)
    candles = [
        _build_candle("ADAUSDT", start + timedelta(minutes=i), 10.0 + i, 10.4 + i)
        for i in range(5)
    ]

    orders = strategy.generate_orders(candles)
    buys = [order for order in orders if order.side is OrderSide.BUY]

    assert len(buys) == 1
    assert buys[0].ts == candles[-1].end


@given(
    moves=st.lists(
        st.floats(min_value=-0.05, max_value=0.15, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    )
)
def test_generated_orders_never_negative_and_pyramids_limited(moves: list[float]) -> None:
    config = StrategyConfig(
        order_size=0.25,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        pyramid_mode="geometric",
        pyramid_scale=0.5,
        max_pyramids=3,
    )
    strategy = BreakoutBiasStrategy(config)
    price = 100.0
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles: list[CandleEvent] = []
    for index, move in enumerate(moves):
        close = max(price * (1 + move), 1e-6)
        high = max(price, close) + 0.2
        low = min(price, close) - 0.2
        candles.append(
            CandleEvent(
                symbol="XRPUSDT",
                open=price,
                high=high,
                low=max(low, 1e-6),
                close=close,
                volume=1.0,
                start=start + timedelta(minutes=index),
                end=start + timedelta(minutes=index + 1),
            )
        )
        price = close

    orders = strategy.generate_orders(candles)
    assert all(order.qty >= 0 for order in orders)
    for order in orders:
        if order.side is OrderSide.BUY:
            assert "layer" in order.id
    position = strategy._positions.get("XRPUSDT")  # noqa: SLF001 - test hook
    if position:
        assert position.layers <= config.max_pyramids
