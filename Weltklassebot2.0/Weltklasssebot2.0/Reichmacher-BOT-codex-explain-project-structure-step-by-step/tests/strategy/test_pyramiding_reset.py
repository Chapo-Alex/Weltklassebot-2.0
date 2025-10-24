from datetime import UTC, datetime, timedelta

from core.events import CandleEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


def _make_candle(
    symbol: str,
    index: int,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    start_dt: datetime | None = None,
) -> CandleEvent:
    start = start_dt or (datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=index))
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


def test_fixed_and_geometric_pyramids_respect_caps() -> None:
    candles = [
        _make_candle("BTCUSDT", 0, 100.0, 105.0, high_price=106.0, low_price=99.5),
        _make_candle("BTCUSDT", 1, 105.0, 110.0, high_price=111.0, low_price=104.5),
        _make_candle("BTCUSDT", 2, 110.0, 116.0, high_price=117.0, low_price=109.0),
        _make_candle("BTCUSDT", 3, 116.0, 120.0, high_price=121.0, low_price=115.0),
    ]

    fixed = BreakoutBiasStrategy(
        StrategyConfig(
            order_size=0.2,
            breakout_threshold=0.0,
            lookback=3,
            bias_lookback=2,
            pyramid_mode="fixed",
            pyramid_steps=(1.0, 0.5, 0.33),
            max_pyramids=3,
            pyramid_dd_reset_pct=1.0,
        )
    )
    fixed_orders = [o for o in fixed.generate_orders(candles) if o.side is OrderSide.BUY]
    assert len(fixed_orders) == 3

    geometric = BreakoutBiasStrategy(
        StrategyConfig(
            order_size=0.2,
            breakout_threshold=0.0,
            lookback=3,
            bias_lookback=2,
            pyramid_mode="geometric",
            pyramid_scale=0.5,
            max_pyramids=3,
            pyramid_dd_reset_pct=1.0,
        )
    )
    geo_orders = [o for o in geometric.generate_orders(candles) if o.side is OrderSide.BUY]
    assert len(geo_orders) == 3
    assert geo_orders[-1].qty < geo_orders[0].qty


def test_vol_shock_reset_triggers_exit() -> None:
    candles = [
        _make_candle("ETHUSDT", 0, 50.0, 52.5, high_price=53.0, low_price=49.5),
        _make_candle("ETHUSDT", 1, 52.5, 55.0, high_price=55.5, low_price=53.5),
        _make_candle("ETHUSDT", 2, 55.0, 51.0, high_price=60.0, low_price=45.0),
    ]
    strategy = BreakoutBiasStrategy(
        StrategyConfig(
            order_size=0.3,
            breakout_threshold=0.0,
            lookback=2,
            bias_lookback=2,
            vol_shock_multiple=1.5,
        )
    )

    orders = strategy.generate_orders(candles)
    sells = [o for o in orders if o.side is OrderSide.SELL]

    assert sells, "expected vol-shock triggered exit"
    assert sells[0].client_tag == "vol_shock"


def test_session_close_flatten_position() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    candles = [
        _make_candle(
            "LINKUSDT",
            0,
            20.0,
            21.0,
            high_price=21.2,
            low_price=19.8,
            start_dt=base.replace(hour=8, minute=55),
        ),
        _make_candle(
            "LINKUSDT",
            1,
            21.0,
            22.5,
            high_price=22.8,
            low_price=20.9,
            start_dt=base.replace(hour=9, minute=5),
        ),
        _make_candle(
            "LINKUSDT",
            2,
            22.5,
            23.5,
            high_price=23.8,
            low_price=22.4,
            start_dt=base.replace(hour=23, minute=10),
        ),
    ]

    strategy = BreakoutBiasStrategy(
        StrategyConfig(
            order_size=0.1,
            breakout_threshold=0.0,
            lookback=2,
            bias_lookback=2,
            session=(
                (
                    datetime(2024, 1, 1, 7, 0, tzinfo=UTC).time(),
                    datetime(2024, 1, 1, 22, 0, tzinfo=UTC).time(),
                ),
            ),
            flatten_on_session_close=True,
        )
    )

    orders = strategy.generate_orders(candles)
    sells = [o for o in orders if o.side is OrderSide.SELL]

    assert sells, "expected session close exit"
    assert sells[0].client_tag == "session"


def test_risk_drawdown_hook_forces_exit() -> None:
    candles = [
        _make_candle("BNBUSDT", 0, 300.0, 309.0, high_price=310.0, low_price=299.0),
        _make_candle("BNBUSDT", 1, 309.0, 315.0, high_price=316.0, low_price=305.0),
        _make_candle("BNBUSDT", 2, 315.0, 314.0, high_price=316.5, low_price=313.0),
    ]
    strategy = BreakoutBiasStrategy(
        StrategyConfig(
            order_size=0.2,
            breakout_threshold=0.0,
            lookback=2,
            bias_lookback=2,
            vol_shock_multiple=10.0,
        )
    )

    first_two = strategy.generate_orders(candles[:2])
    assert any(o.side is OrderSide.BUY for o in first_two)

    strategy.notify_risk_drawdown("BNBUSDT")
    orders = strategy.generate_orders([candles[2]])
    sells = [o for o in orders if o.side is OrderSide.SELL]

    assert sells and sells[0].client_tag == "risk"
