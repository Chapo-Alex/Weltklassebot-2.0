from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import UTC, datetime, timedelta

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
    index: int,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
) -> CandleEvent:
    start = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=index)
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


def test_percentile_threshold_monotonicity() -> None:
    candles = [
        _build_candle(
            "ETHUSDT",
            idx,
            100.0 + idx,
            101.5 + idx,
            high_price=102 + idx,
            low_price=99 + idx,
        )
        for idx in range(8)
    ]
    base = dict(
        order_size=0.2,
        breakout_threshold=0.0,
        lookback=4,
        bias_lookback=4,
        threshold_mode="percentile",
        threshold_lookback=4,
    )
    lower = BreakoutBiasStrategy(StrategyConfig(threshold_percentile=0.25, **base))
    higher = BreakoutBiasStrategy(StrategyConfig(threshold_percentile=0.75, **base))

    low_orders = [o for o in lower.generate_orders(candles) if o.side is OrderSide.BUY]
    high_orders = [o for o in higher.generate_orders(candles) if o.side is OrderSide.BUY]

    assert len(high_orders) <= len(low_orders)


def test_atr_regime_threshold_respects_volatility() -> None:
    candles = []
    price = 100.0
    for idx in range(10):
        close = price * (1.01 if idx % 2 == 0 else 1.02)
        high = max(price, close) + (5.0 if idx >= 4 else 1.0)
        low = min(price, close) - (5.0 if idx >= 4 else 1.0)
        candles.append(_build_candle("BTCUSDT", idx, price, close, high, low))
        price = close

    base = dict(
        order_size=0.4,
        breakout_threshold=0.0,
        lookback=5,
        bias_lookback=5,
        threshold_mode="atr_k",
        atr_lookback=5,
        atr_percentile_split=0.4,
    )

    permissive = BreakoutBiasStrategy(StrategyConfig(atr_k_high=1.0, atr_k_low=0.5, **base))
    strict = BreakoutBiasStrategy(StrategyConfig(atr_k_high=2.0, atr_k_low=0.5, **base))

    perm_orders = [o for o in permissive.generate_orders(candles) if o.side is OrderSide.BUY]
    strict_orders = [o for o in strict.generate_orders(candles) if o.side is OrderSide.BUY]

    assert len(strict_orders) <= len(perm_orders)


def test_exit_modes_trigger_reductions() -> None:
    atr_config = StrategyConfig(
        order_size=0.3,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        exit_mode="atr_trail",
        atr_trailing_multiplier=0.6,
    )
    atr_strategy = BreakoutBiasStrategy(atr_config)
    atr_candles = [
        _build_candle("SOLUSDT", 0, 50.0, 52.0, high_price=53.0, low_price=49.8),
        _build_candle("SOLUSDT", 1, 52.0, 54.5, high_price=55.0, low_price=51.5),
        _build_candle("SOLUSDT", 2, 54.5, 55.0, high_price=55.5, low_price=54.0),
        _build_candle("SOLUSDT", 3, 55.0, 49.0, high_price=55.2, low_price=48.5),
    ]
    atr_orders = atr_strategy.generate_orders(atr_candles)
    assert any(order.side is OrderSide.SELL for order in atr_orders)

    chandelier_config = StrategyConfig(
        order_size=0.3,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        exit_mode="chandelier",
        chandelier_lookback=3,
        chandelier_atr_mult=1.0,
    )
    chandelier = BreakoutBiasStrategy(chandelier_config)
    chandelier_candles = [
        _build_candle("ADAUSDT", 0, 10.0, 10.8, high_price=11.0, low_price=9.9),
        _build_candle("ADAUSDT", 1, 10.8, 11.2, high_price=11.4, low_price=10.5),
        _build_candle("ADAUSDT", 2, 11.2, 11.6, high_price=11.9, low_price=11.0),
        _build_candle("ADAUSDT", 3, 11.6, 11.8, high_price=12.0, low_price=11.4),
        _build_candle("ADAUSDT", 4, 11.8, 10.6, high_price=11.9, low_price=10.5),
    ]
    orders: list = []
    stop_levels: list[float] = []
    for candle in chandelier_candles:
        orders.extend(chandelier.generate_orders([candle]))
        position = chandelier._positions.get("ADAUSDT")  # noqa: SLF001 - test helper
        if position and position.chandelier_stop is not None:
            stop_levels.append(position.chandelier_stop)

    assert stop_levels == sorted(stop_levels)
    assert any(order.side is OrderSide.SELL for order in orders)


@given(
    moves=st.lists(
        st.floats(min_value=-0.03, max_value=0.08, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    )
)
def test_deterministic_orders_for_repeated_runs(moves: list[float]) -> None:
    price = 100.0
    candles = []
    for idx, move in enumerate(moves):
        next_price = max(price * (1 + move), 1e-6)
        high = max(price, next_price) + 0.5
        low = min(price, next_price) - 0.5
        candles.append(_build_candle("XRPUSDT", idx, price, next_price, high, low))
        price = next_price

    config = StrategyConfig(
        order_size=0.2,
        breakout_threshold=0.0,
        lookback=3,
        bias_lookback=3,
        threshold_mode="percentile",
        threshold_lookback=3,
    )
    strat_a = BreakoutBiasStrategy(config)
    strat_b = BreakoutBiasStrategy(config)

    assert strat_a.generate_orders(candles) == strat_b.generate_orders(candles)
