"""Paper connector regression tests for retries and idempotency."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from connectors.paper import PaperConnector
from core.events import OrderEvent, OrderSide, OrderType
from execution.orderbook import OrderBook
from execution.oms_simulator import ExecConfig, OmsSimulator
from execution.slippage import LinearSlippage

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"


def _make_order(order_id: str, *, price: float | None = None) -> OrderEvent:
    return OrderEvent(
        id=order_id,
        ts=BASE_TS,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        qty=1.0,
        type=OrderType.MARKET if price is None else OrderType.LIMIT,
        price=price,
        stop=None,
        tif="GTC",
    )


def _build_simulator() -> OmsSimulator:
    book = OrderBook()
    book.add("ask", 101.0, 0.6)
    book.add("ask", 102.0, 0.6)
    cfg = ExecConfig(
        slippage=LinearSlippage(bps_per_notional=0.0),
        taker_fee=0.0,
        maker_fee=0.0,
        latency_ms=0,
        jitter_ms=0,
    )
    return OmsSimulator(book, cfg, seed=1337)


def test_idempotent_submission_reuses_order_id() -> None:
    sim = _build_simulator()
    connector = PaperConnector(sim=sim, seed=1337, max_retries=2, base_backoff_ms=10)
    order = _make_order("ord-1")

    first_id = connector.send_order(order, idempotency_key="idem-1")
    fills = connector.fetch_fills()
    assert first_id == order.id
    assert len(fills) == 1

    second_id = connector.send_order(order, idempotency_key="idem-1")
    assert second_id == order.id
    assert connector.fetch_fills() == fills

    cutoff = fills[0].ts + timedelta(microseconds=1)
    assert connector.fetch_fills(since=cutoff) == ()


def test_retry_on_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    sim = _build_simulator()
    connector = PaperConnector(sim=sim, seed=1337, max_retries=1, base_backoff_ms=20)
    expected_sleep = connector._jitter_ms(0) / 1000.0

    original_send = sim.send
    call_count = {"value": 0}

    def flaky_send(order: OrderEvent, now: datetime):  # type: ignore[override]
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("transient")
        return original_send(order, now)

    monkeypatch.setattr(sim, "send", flaky_send)

    sleep_calls: list[float] = []

    def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr("connectors.paper.time.sleep", fake_sleep)

    order = _make_order("ord-2")
    order_id = connector.send_order(order, idempotency_key="idem-2")

    assert order_id == order.id
    assert call_count["value"] == 2
    assert sleep_calls == [pytest.approx(expected_sleep)]


def test_fetch_fills_since_filters() -> None:
    sim = _build_simulator()
    connector = PaperConnector(sim=sim, seed=42)
    order = _make_order("ord-3")
    connector.send_order(order, idempotency_key="idem-3")
    fills = list(connector.fetch_fills())
    assert len(fills) == 1

    later = fills[0].ts + timedelta(seconds=1)
    assert connector.fetch_fills(since=later) == ()

    earlier = fills[0].ts - timedelta(seconds=1)
    assert connector.fetch_fills(since=earlier) == tuple(fills)
