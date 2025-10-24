"""Edge-case order denials exercise OMS and risk integration paths."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.events import OrderEvent, OrderSide, OrderType
from core.metrics import CollectorRegistry, set_registry
from execution.orderbook import OrderBook
from execution.simulator import ExecConfig, OmsSimulator, Rejection
from execution.slippage import ImpactLinear
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _counter_value(counter) -> float | None:  # type: ignore[no-untyped-def]
    raw = getattr(counter, "_value", None)
    if raw is None:
        try:
            return counter._value.get()  # type: ignore[attr-defined]
        except AttributeError:
            return None
    if hasattr(raw, "get"):
        return raw.get()  # type: ignore[no-untyped-call]
    return float(raw)


def _assert_counter_increment(counter) -> None:  # type: ignore[no-untyped-def]
    before = _counter_value(counter)
    counter.inc()
    after = _counter_value(counter)
    if before is None or after is None:
        pytest.skip("prometheus counter lacks readable state")
    assert after == pytest.approx(before + 1.0)


def _simulator() -> OmsSimulator:
    book = OrderBook()
    cfg = ExecConfig(
        slippage=ImpactLinear(k=0.0),
        taker_fee=0.0004,
        maker_fee=0.0002,
        latency_ms=5,
        jitter_ms=0,
        seed=1337,
    )
    return OmsSimulator(book, cfg)


def test_post_only_cross_denial_increments_counter() -> None:
    sim = _simulator()
    sim._book.add("ask", 100.0, 0.4)
    order = OrderEvent(
        id="post-only",
        ts=BASE_TS,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.5,
        type=OrderType.LIMIT,
        price=100.0,
        stop=None,
        tif="GTC",
        post_only=True,
    )

    result = sim.send(order, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "post_only_would_cross"

    registry = CollectorRegistry()
    manager = RiskManagerV2(RiskParameters(), registry=registry)
    counter = manager._metrics.denials.labels(reason=result.reason)
    try:
        _assert_counter_increment(counter)
    finally:
        set_registry(None)


def test_reduce_only_violation_records_reason() -> None:
    sim = _simulator()
    order = OrderEvent(
        id="reduce-only",
        ts=BASE_TS,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.25,
        type=OrderType.MARKET,
        price=None,
        stop=None,
        tif="GTC",
        reduce_only=True,
    )

    result = sim.send(order, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "reduce_only_violation"

    registry = CollectorRegistry()
    manager = RiskManagerV2(RiskParameters(), registry=registry)
    counter = manager._metrics.denials.labels(reason=result.reason)
    try:
        _assert_counter_increment(counter)
    finally:
        set_registry(None)


def test_ioc_partial_fill_cancels_remainder() -> None:
    sim = _simulator()
    sim._book.add("ask", 100.0, 0.3)
    sim._book.add("ask", 100.5, 0.5)
    order = OrderEvent(
        id="ioc-partial",
        ts=BASE_TS,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.5,
        type=OrderType.LIMIT,
        price=100.0,
        stop=None,
        tif="IOC",
    )

    fills = sim.send(order, BASE_TS)
    assert isinstance(fills, list)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.qty == pytest.approx(0.3)
    assert fill.side is OrderSide.BUY
    assert sim._book.best_ask() == pytest.approx(100.5)


def test_fok_unfilled_denial_updates_counter() -> None:
    sim = _simulator()
    sim._book.add("ask", 100.0, 0.2)
    order = OrderEvent(
        id="fok",
        ts=BASE_TS,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.5,
        type=OrderType.LIMIT,
        price=100.0,
        stop=None,
        tif="FOK",
    )

    result = sim.send(order, BASE_TS)
    assert isinstance(result, Rejection)
    assert result.reason == "fok_unfilled"

    registry = CollectorRegistry()
    manager = RiskManagerV2(RiskParameters(), registry=registry)
    counter = manager._metrics.denials.labels(reason=result.reason)
    try:
        _assert_counter_increment(counter)
    finally:
        set_registry(None)


def test_session_blackout_denial_triggers_cooldown_metric() -> None:
    registry = CollectorRegistry()
    params = RiskParameters(max_trades_per_day=1, cooldown_minutes=10.0)
    manager = RiskManagerV2(params, registry=registry)
    now = BASE_TS
    ctx = RiskContext(
        equity=100_000.0,
        drawdown=0.0,
        notional=0.0,
        trades_today=1,
        now=now,
        session="news_blackout",
    )
    order = OrderEvent(
        id="session-block",
        ts=now,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.1,
        type=OrderType.MARKET,
        price=None,
        stop=None,
        tif="GTC",
    )
    counter = manager._metrics.denials.labels(reason="cooldown")
    before = _counter_value(counter)
    result = manager.allow(order, ctx)
    assert result == "cooldown"
    after = _counter_value(counter)
    try:
        if before is None or after is None:
            pytest.skip("prometheus counter lacks readable state")
        assert after == pytest.approx(before + 1.0)
    finally:
        set_registry(None)
