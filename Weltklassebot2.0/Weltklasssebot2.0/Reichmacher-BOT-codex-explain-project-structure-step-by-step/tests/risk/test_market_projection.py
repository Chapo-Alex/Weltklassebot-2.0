from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.events import OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters


def _market_order(*, qty: float = 1.0) -> OrderEvent:
    return OrderEvent(
        id="order-market-1",
        ts=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=qty,
        type=OrderType.MARKET,
        price=None,
        stop=None,
        tif="GTC",
        reduce_only=False,
        post_only=False,
        client_tag=None,
    )


def _context(
    *,
    notional: float,
    trades: int,
    orderbook_mid: float | None,
    last_close: float | None,
) -> RiskContext:
    return RiskContext(
        equity=100_000.0,
        drawdown=0.0,
        notional=notional,
        trades_today=trades,
        now=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
        session="primary",
        orderbook_mid=orderbook_mid,
        last_close=last_close,
    )


def test_market_order_projects_with_orderbook_mid() -> None:
    manager = RiskManagerV2(RiskParameters(max_notional=1_000_000.0))
    ctx = _context(notional=50_000.0, trades=0, orderbook_mid=101.25, last_close=99.5)
    order = _market_order(qty=2.0)

    decision = manager.allow(order, ctx)

    assert decision is True
    assert ctx.orderbook_mid is not None
    expected = ctx.notional + abs(order.qty) * ctx.orderbook_mid
    snapshot = manager.metrics_snapshot()
    assert snapshot["projected_exposure"] == pytest.approx(expected, rel=1e-3)


def test_market_order_projects_with_last_close_when_no_orderbook() -> None:
    manager = RiskManagerV2(RiskParameters(max_notional=1_000_000.0))
    ctx = _context(notional=10_000.0, trades=0, orderbook_mid=None, last_close=120.0)
    order = _market_order(qty=3.5)

    decision = manager.allow(order, ctx)

    assert decision is True
    assert ctx.last_close is not None
    expected = ctx.notional + abs(order.qty) * ctx.last_close
    snapshot = manager.metrics_snapshot()
    assert snapshot["projected_exposure"] == pytest.approx(expected, rel=1e-3)


def test_market_order_without_reference_price_errors() -> None:
    manager = RiskManagerV2(RiskParameters(max_notional=1_000_000.0))
    ctx = _context(notional=5_000.0, trades=0, orderbook_mid=None, last_close=None)
    order = _market_order(qty=1.0)

    with pytest.raises(ValueError, match="missing price data"):
        manager.allow(order, ctx)
