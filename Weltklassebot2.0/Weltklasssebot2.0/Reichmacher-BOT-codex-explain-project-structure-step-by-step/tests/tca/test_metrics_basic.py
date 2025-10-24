from datetime import UTC, datetime

import pytest

from core.events import FillEvent, LiquidityFlag, OrderSide
from tca.metrics import adverse_selection, implementation_shortfall, vwap


def _fill(
    order_id: str,
    qty: float,
    price: float,
    side: OrderSide,
    liquidity: LiquidityFlag,
) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        ts=datetime(2024, 1, 1, tzinfo=UTC),
        qty=qty,
        price=price,
        fee=0.0,
        liquidity_flag=liquidity,
        symbol="BTCUSDT",
        side=side,
    )


def test_vwap_and_shortfall_buy() -> None:
    fills = [
        _fill("o1", 1.0, 100.0, OrderSide.BUY, LiquidityFlag.TAKER),
        _fill("o2", 2.0, 101.0, OrderSide.BUY, LiquidityFlag.MAKER),
    ]
    mid = 100.5

    computed_vwap = vwap(fills)
    assert computed_vwap == pytest.approx(302.0 / 3.0, abs=1e-9)

    is_bps = implementation_shortfall(mid, fills, OrderSide.BUY)
    assert is_bps == pytest.approx(((computed_vwap / mid) - 1.0) * 10_000.0, abs=1e-9)
    assert is_bps > 0.0

    adverse = adverse_selection(fills, next_mid=99.5, side=OrderSide.BUY)
    expected = ((computed_vwap / 99.5) - 1.0) * 10_000.0
    assert adverse == pytest.approx(expected, abs=1e-9)
    assert adverse > 0.0


def test_vwap_and_shortfall_sell() -> None:
    fills = [
        _fill("s1", 1.5, 100.0, OrderSide.SELL, LiquidityFlag.TAKER),
        _fill("s2", 0.5, 99.0, OrderSide.SELL, LiquidityFlag.MAKER),
    ]
    mid = 100.4
    next_mid = 101.2

    computed_vwap = vwap(fills)
    assert computed_vwap == pytest.approx((1.5 * 100.0 + 0.5 * 99.0) / 2.0, abs=1e-9)

    is_bps = implementation_shortfall(mid, fills, OrderSide.SELL)
    assert is_bps == pytest.approx(((computed_vwap / mid) - 1.0) * -10_000.0, abs=1e-9)
    assert is_bps > 0.0

    adverse = adverse_selection(fills, next_mid, OrderSide.SELL)
    expected = ((computed_vwap / next_mid) - 1.0) * -10_000.0
    assert adverse == pytest.approx(expected, abs=1e-9)
    assert adverse > 0.0
