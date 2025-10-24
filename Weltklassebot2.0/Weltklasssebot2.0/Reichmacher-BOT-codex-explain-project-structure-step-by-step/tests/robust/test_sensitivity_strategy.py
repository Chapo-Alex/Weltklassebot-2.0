import itertools
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from core.events import CandleEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig

THRESHOLD_PERCENTILES = (0.6, 0.7, 0.8, 0.9)
ATR_K_HIGH_VALUES = (1.1, 1.3, 1.6)
TAKE_PROFITS = (0.01, 0.02, 0.03)


def _sensitivity_candles(length: int = 30) -> list[CandleEvent]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0
    candles: list[CandleEvent] = []
    for index in range(length):
        drift = 0.006 if index < length // 2 else -0.005
        oscillation = 0.0015 * ((index % 4) - 1.5)
        close = max(price * (1.0 + drift + oscillation), 1e-6)
        high = max(price, close) * (1.0 + 0.004 + 0.001 * (index % 3))
        low = min(price, close) * (1.0 - 0.004 - 0.001 * ((index + 1) % 3))
        start = base + timedelta(minutes=index)
        candles.append(
            CandleEvent(
                symbol="BTCUSDT",
                open=price,
                high=high,
                low=max(low, 0.01),
                close=close,
                volume=1.0,
                start=start,
                end=start + timedelta(minutes=1),
            )
        )
        price = close
    return candles


def _count_orders(
    strategy: BreakoutBiasStrategy, candles: Sequence[CandleEvent]
) -> tuple[int, int]:
    orders = strategy.generate_orders(candles)
    buys = sum(1 for order in orders if order.side is OrderSide.BUY)
    sells = sum(1 for order in orders if order.side is OrderSide.SELL)
    return buys, sells


def _exit_reasons(
    strategy: BreakoutBiasStrategy, candles: Sequence[CandleEvent]
) -> list[str]:
    return [
        order.client_tag or "unknown"
        for order in strategy.generate_orders(candles)
        if order.side is OrderSide.SELL
    ]


def test_parameter_grid_is_deterministic() -> None:
    candles = _sensitivity_candles()
    results: dict[tuple[float, float, float, str], tuple[int, int]] = {}
    for threshold_pct, atr_high, tp in itertools.product(
        THRESHOLD_PERCENTILES, ATR_K_HIGH_VALUES, TAKE_PROFITS
    ):
        configs = (
            StrategyConfig(
                breakout_threshold=0.0,
                order_size=0.2,
                lookback=6,
                bias_lookback=6,
                pyramid_mode="fixed",
                pyramid_steps=(1.0, 0.5),
                threshold_mode="percentile",
                threshold_percentile=threshold_pct,
                threshold_lookback=5,
                take_profit_pct=tp,
            ),
            StrategyConfig(
                breakout_threshold=0.0,
                order_size=0.2,
                lookback=6,
                bias_lookback=6,
                pyramid_mode="fixed",
                pyramid_steps=(1.0, 0.5),
                threshold_mode="atr_k",
                atr_k_low=0.8,
                atr_k_high=atr_high,
                atr_lookback=5,
                atr_percentile_split=0.5,
                take_profit_pct=tp,
            ),
        )
        for variant, config in zip(("percentile", "atr"), configs, strict=True):
            first = _count_orders(BreakoutBiasStrategy(config), candles)
            second = _count_orders(BreakoutBiasStrategy(config), candles)
            assert first == second
            results[(threshold_pct, atr_high, tp, variant)] = first
    assert results


def test_threshold_percentile_monotonic_entries() -> None:
    candles = _sensitivity_candles()
    previous = None
    for value in THRESHOLD_PERCENTILES:
        strategy = BreakoutBiasStrategy(
            StrategyConfig(
                breakout_threshold=0.0,
                order_size=0.2,
                lookback=6,
                bias_lookback=6,
                pyramid_mode="fixed",
                pyramid_steps=(1.0, 0.5, 0.33),
                threshold_mode="percentile",
                threshold_percentile=value,
                threshold_lookback=6,
                take_profit_pct=0.02,
            )
        )
        buys, _ = _count_orders(strategy, candles)
        if previous is not None:
            assert buys <= previous
        previous = buys


def test_atr_high_monotonic_entries() -> None:
    candles = _sensitivity_candles()
    previous = None
    for value in ATR_K_HIGH_VALUES:
        strategy = BreakoutBiasStrategy(
            StrategyConfig(
                breakout_threshold=0.0,
                order_size=0.2,
                lookback=6,
                bias_lookback=6,
                pyramid_mode="geometric",
                pyramid_scale=0.6,
                max_pyramids=4,
                threshold_mode="atr_k",
                atr_k_low=0.7,
                atr_k_high=value,
                atr_lookback=6,
                atr_percentile_split=0.4,
                take_profit_pct=0.02,
            )
        )
        buys, _ = _count_orders(strategy, candles)
        if previous is not None:
            assert buys <= previous
        previous = buys


@pytest.mark.parametrize("exit_mode", ("atr_trail", "chandelier"))
def test_take_profit_increases_positive_exit_share(exit_mode: str) -> None:
    candles = _sensitivity_candles()
    ratios: list[float] = []
    for tp in TAKE_PROFITS:
        config = StrategyConfig(
            breakout_threshold=0.0,
            order_size=0.15,
            lookback=6,
            bias_lookback=6,
            pyramid_mode="fixed",
            pyramid_steps=(1.0, 0.5),
            threshold_mode="percentile",
            threshold_percentile=0.65,
            threshold_lookback=5,
            exit_mode=exit_mode,
            take_profit_pct=tp,
        )
        strategy = BreakoutBiasStrategy(config)
        reasons = _exit_reasons(strategy, candles)
        if reasons:
            ratios.append(sum(1 for reason in reasons if reason == "take_profit") / len(reasons))
        else:
            ratios.append(0.0)
    assert ratios == sorted(ratios)
