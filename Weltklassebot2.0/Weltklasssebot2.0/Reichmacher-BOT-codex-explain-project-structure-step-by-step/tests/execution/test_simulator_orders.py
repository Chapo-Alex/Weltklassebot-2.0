"""OMS simulator regression tests covering order types and flags."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

import pytest

from core.events import LiquidityFlag, OrderEvent, OrderSide, OrderType
from execution.orderbook import OrderBook
from execution.simulator import ExecConfig, OmsSimulator, Rejection
from execution.slippage import ImpactLinear, QueuePosition, SlippageModel

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"


def _make_order(
    order_id: str,
    *,
    side: OrderSide,
    qty: float,
    order_type: OrderType,
    price: float | None = None,
    stop: float | None = None,
    tif: str = "GTC",
    reduce_only: bool = False,
    post_only: bool = False,
) -> OrderEvent:
    return OrderEvent(
        id=order_id,
        ts=BASE_TS,
        symbol=SYMBOL,
        side=side,
        qty=qty,
        type=order_type,
        price=price,
        stop=stop,
        tif=tif,
        reduce_only=reduce_only,
        post_only=post_only,
    )


def _default_config(slippage: SlippageModel | None = None) -> ExecConfig:
    model = slippage or ImpactLinear(k=0.0)
    return ExecConfig(
        slippage=model,
        taker_fee=0.001,
        maker_fee=0.0005,
        latency_ms=0,
        jitter_ms=0,
        seed=1337,
    )


def _assert_fill_flags(fills: Iterable) -> None:
    for fill in fills:
        assert fill.ts >= BASE_TS
        assert fill.symbol == SYMBOL


def test_market_buy_matches_vwap_and_fee() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    book.add("ask", 102.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order("mkt-buy", side=OrderSide.BUY, qty=1.5, order_type=OrderType.MARKET)
    fills = sim.send(order, BASE_TS)
    assert isinstance(fills, list)
    assert len(fills) == 1
    fill = fills[0]
    expected_price = ((1.0 * 101.0) + (0.5 * 102.0)) / 1.5
    assert fill.price == pytest.approx(expected_price)
    assert fill.qty == pytest.approx(1.5)
    assert fill.fee == pytest.approx(fill.price * fill.qty * 0.001)
    assert fill.liquidity_flag is LiquidityFlag.TAKER
    _assert_fill_flags(fills)


def test_market_sell_matches_vwap_and_fee() -> None:
    book = OrderBook()
    book.add("bid", 99.0, 1.0)
    book.add("bid", 98.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order("mkt-sell", side=OrderSide.SELL, qty=1.5, order_type=OrderType.MARKET)
    fills = sim.send(order, BASE_TS)
    assert isinstance(fills, list)
    fill = fills[0]
    expected_price = ((1.0 * 99.0) + (0.5 * 98.0)) / 1.5
    assert fill.price == pytest.approx(expected_price)
    assert fill.liquidity_flag is LiquidityFlag.TAKER
    _assert_fill_flags(fills)


def test_post_only_limit_rests_in_book() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "limit-maker",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.LIMIT,
        price=100.0,
        post_only=True,
    )
    fills = sim.send(order, BASE_TS)
    assert fills == []
    bids = book.depth("bid")
    assert bids[0] == (100.0, 1.0)
    assert len(sim._resting) == 1


def test_post_only_crossing_order_rejected() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "post-cross",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.LIMIT,
        price=105.0,
        post_only=True,
    )
    result = sim.send(order, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "post_only_would_cross"


def test_aggressive_limit_partial_fill_and_rest() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    book.add("ask", 102.0, 0.5)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "limit-partial",
        side=OrderSide.BUY,
        qty=2.0,
        order_type=OrderType.LIMIT,
        price=102.0,
    )
    fills = sim.send(order, BASE_TS)
    assert isinstance(fills, list)
    assert len(fills) == 1
    fill = fills[0]
    expected_price = ((1.0 * 101.0) + (0.5 * 102.0)) / 1.5
    assert fill.qty == pytest.approx(1.5)
    assert fill.price == pytest.approx(expected_price)
    bids = book.depth("bid")
    bid_price, bid_size = bids[0]
    assert bid_price == pytest.approx(102.0)
    assert bid_size == pytest.approx(0.5)
    assert len(sim._resting) == 1


def test_ioc_limit_does_not_leave_residual() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    book.add("ask", 102.0, 0.5)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "limit-ioc",
        side=OrderSide.BUY,
        qty=2.0,
        order_type=OrderType.LIMIT,
        price=102.0,
        tif="IOC",
    )
    fills = sim.send(order, BASE_TS)
    assert isinstance(fills, list)
    assert len(fills) == 1
    assert book.depth("bid") == []
    assert len(sim._resting) == 0


def test_fok_limit_rejects_when_not_filled() -> None:
    book = OrderBook()
    book.add("ask", 101.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "limit-fok",
        side=OrderSide.BUY,
        qty=2.0,
        order_type=OrderType.LIMIT,
        price=101.0,
        tif="FOK",
    )
    result = sim.send(order, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "fok_unfilled"


def test_reduce_only_orders_enforce_position_limits() -> None:
    book = OrderBook()
    book.add("bid", 99.0, 2.0)
    book.add("ask", 101.0, 2.0)
    sim = OmsSimulator(book, _default_config())
    sim._positions[SYMBOL] = 1.0
    sell_order = _make_order(
        "reduce-sell",
        side=OrderSide.SELL,
        qty=0.5,
        order_type=OrderType.MARKET,
        reduce_only=True,
    )
    fills = sim.send(sell_order, BASE_TS)
    assert isinstance(fills, list)
    assert fills[0].side is OrderSide.SELL
    buy_order = _make_order(
        "reduce-buy",
        side=OrderSide.BUY,
        qty=0.5,
        order_type=OrderType.MARKET,
        reduce_only=True,
    )
    rejection = sim.send(buy_order, BASE_TS)
    assert isinstance(rejection, Rejection)
    assert rejection.reason == "reduce_only_violation"


def test_stop_orders_trigger_on_touch() -> None:
    book = OrderBook()
    book.add("ask", 105.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    triggered = _make_order(
        "stop-buy",
        side=OrderSide.BUY,
        qty=0.5,
        order_type=OrderType.STOP,
        stop=104.0,
    )
    fills = sim.send(triggered, BASE_TS)
    assert isinstance(fills, list)
    assert fills[0].qty == pytest.approx(0.5)
    untriggered = _make_order(
        "stop-miss",
        side=OrderSide.BUY,
        qty=0.5,
        order_type=OrderType.STOP,
        stop=106.0,
    )
    result = sim.send(untriggered, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "stop_not_triggered"


def test_stop_limit_resting_behaviour() -> None:
    book = OrderBook()
    book.add("ask", 105.0, 1.0)
    sim = OmsSimulator(book, _default_config())
    order = _make_order(
        "stop-limit",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.STOP_LIMIT,
        price=104.0,
        stop=103.0,
    )
    fills = sim.send(order, BASE_TS)
    assert fills == []
    bids = book.depth("bid")
    assert bids[0] == (104.0, 1.0)


def test_maker_fill_occurs_on_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    book = OrderBook()
    cfg = _default_config(slippage=QueuePosition(eta=5.0))
    sim = OmsSimulator(book, cfg)
    order = _make_order(
        "maker-fill",
        side=OrderSide.BUY,
        qty=1.0,
        order_type=OrderType.LIMIT,
        price=99.0,
    )
    fills = sim.send(order, BASE_TS)
    assert fills == []
    bids = book.depth("bid")
    assert bids[0] == (99.0, 1.0)
    class _DeterministicRng:
        def random(self) -> float:
            return 0.0

        def integers(self, low: int, high: int | None = None, size: None = None) -> int:
            if high is None:
                high = low
                low = 0
            if high <= low:
                return low
            return low

    monkeypatch.setattr(sim, "_rng", _DeterministicRng())
    maker_fills = sim.tick(BASE_TS)
    assert len(maker_fills) == 1
    fill = maker_fills[0]
    assert fill.liquidity_flag is LiquidityFlag.MAKER
    assert fill.price == pytest.approx(99.0)
    assert fill.fee == pytest.approx(99.0 * 1.0 * cfg.maker_fee)
    assert book.depth("bid") == []
    assert sim.tick(BASE_TS) == []
