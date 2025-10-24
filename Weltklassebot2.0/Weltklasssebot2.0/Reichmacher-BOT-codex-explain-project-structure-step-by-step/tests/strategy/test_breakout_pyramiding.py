"""Pyramiding behaviour for the breakout-bias strategy."""

from __future__ import annotations

from collections import Counter

import pytest

from core.events import OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["fixed", "geometric"])
@pytest.mark.parametrize("max_pyramids", [1, 2, 3])
def test_pyramiding_respects_limits(tiny_candles, mode: str, max_pyramids: int) -> None:
    candles = tiny_candles(n=60)
    kwargs: dict[str, object] = {
        "order_size": 0.6,
        "lookback": 12,
        "bias_lookback": 12,
        "atr_lookback": 12,
        "breakout_threshold": 0.008,
        "threshold_mode": "fixed",
        "pyramid_mode": mode,
        "max_pyramids": max_pyramids,
        "atr_trailing_multiplier": 5.0,
        "vol_shock_multiple": 50.0,
        "flatten_on_session_close": False,
        "news_blackout": (),
        "session": (),
        "bias_vol_ratio": 0.5,
        "bias_min_slope": 0.0,
    }
    if mode == "fixed":
        kwargs["pyramid_steps"] = tuple(max(1.0, 1.0 + 0.1 * i) for i in range(max_pyramids)) or (1.0,)
    else:
        kwargs["pyramid_scale"] = 1.1

    config = StrategyConfig(**kwargs)
    strategy = BreakoutBiasStrategy(config)

    orders = strategy.generate_orders(candles)
    timestamp_index = {candle.end: idx for idx, candle in enumerate(candles)}
    active_layers = 0
    open_qty = 0.0
    buy_layers: Counter[int] = Counter()

    for order in orders:
        if order.side is OrderSide.BUY:
            active_layers += 1
            open_qty += order.qty
            assert order.reduce_only is False
            assert active_layers <= max_pyramids
            buy_layers[active_layers] += 1
        else:
            assert order.side is OrderSide.SELL
            assert order.reduce_only is True
            open_qty = max(open_qty - order.qty, 0.0)
            active_layers = 0
        assert open_qty >= -1e-9

    buy_indices = [timestamp_index[order.ts] for order in orders if order.side is OrderSide.BUY]
    assert buy_indices == sorted(buy_indices)
    assert len(buy_indices) <= max_pyramids
    assert buy_layers  # at least one pyramid add occurred
