"""Venue constraint enforcement unit tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.events import OrderEvent, OrderSide, OrderType
from execution.normalizer import round_price, round_qty
from execution.oms_simulator import ExecConfig, OmsSimulator
from execution.orderbook import OrderBook
from execution.slippage import LinearSlippage


def _order(
    *,
    order_id: str,
    side: OrderSide,
    qty: float,
    order_type: OrderType,
    price: float | None,
) -> OrderEvent:
    ts = datetime.now(timezone.utc)
    return OrderEvent(
        id=order_id,
        ts=ts,
        symbol="XYZ",
        side=side,
        qty=qty,
        type=order_type,
        price=price,
        stop=None,
        tif="GTC",
    )


def test_round_price_clamps_to_tick() -> None:
    assert round_price(5e-9, 1e-8) == pytest.approx(1e-8)
    assert round_price(0.0126, 0.005) == pytest.approx(0.015)
    assert round_price(None, 0.01) is None


def test_round_qty_respects_minimum() -> None:
    assert round_qty(0.26, 0.1) == pytest.approx(0.3)
    assert round_qty(5e-7, 1e-6) == pytest.approx(1e-6)


def test_limit_order_resting_uses_normalised_values() -> None:
    book = OrderBook()
    book.add("bid", 99.0, 5.0)
    book.add("ask", 101.0, 5.0)
    cfg = ExecConfig(
        slippage=LinearSlippage(),
        tick_size=0.05,
        min_qty=0.1,
    )
    sim = OmsSimulator(book, cfg, seed=7)
    order = _order(
        order_id="limit-1",
        side=OrderSide.BUY,
        qty=0.1234,
        order_type=OrderType.LIMIT,
        price=100.061,
    )
    result = sim.send(order, now=order.ts)
    assert result == []
    resting = sim._resting[0]
    assert resting.price == pytest.approx(100.05)
    assert resting.order.qty == pytest.approx(0.1)


def test_market_order_clamps_to_min_qty() -> None:
    book = OrderBook()
    book.add("ask", 10.0, 5.0)
    cfg = ExecConfig(
        slippage=LinearSlippage(),
        tick_size=0.01,
        min_qty=0.5,
    )
    sim = OmsSimulator(book, cfg, seed=11)
    order = _order(
        order_id="market-1",
        side=OrderSide.BUY,
        qty=0.1,
        order_type=OrderType.MARKET,
        price=None,
    )
    fills = sim.send(order, now=order.ts)
    assert isinstance(fills, list)
    assert fills
    assert fills[0].qty == pytest.approx(0.5)
