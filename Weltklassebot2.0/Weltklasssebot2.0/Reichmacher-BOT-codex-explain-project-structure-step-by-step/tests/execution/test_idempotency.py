"""Execution-layer idempotency regression tests."""

from __future__ import annotations

from datetime import UTC, datetime

from connectors.paper import PaperConnector
from core.events import OrderEvent, OrderSide, OrderType
from execution.orderbook import OrderBook
from execution.oms_simulator import ExecConfig, OmsSimulator
from execution.slippage import LinearSlippage

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"


def _make_order(order_id: str) -> OrderEvent:
    return OrderEvent(
        id=order_id,
        ts=BASE_TS,
        symbol=SYMBOL,
        side=OrderSide.BUY,
        qty=1.0,
        type=OrderType.MARKET,
        price=None,
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


def test_duplicate_coids_do_not_duplicate_fills() -> None:
    sim = _build_simulator()
    connector = PaperConnector(
        sim=sim,
        seed=1337,
        max_retries=0,
        base_backoff_ms=0,
        run_id="backtest-run",
    )
    coid = "backtest-run-1"
    order = _make_order("ord-duplicate")

    first_id = connector.send_order(order, idempotency_key=coid)
    baseline_fills = tuple(connector.fetch_fills())
    assert len(baseline_fills) == 1

    for _ in range(1000):
        duplicate_id = connector.send_order(order, idempotency_key=coid)
        assert duplicate_id == first_id

    final_fills = tuple(connector.fetch_fills())
    assert final_fills == baseline_fills
    assert len(final_fills) == 1
