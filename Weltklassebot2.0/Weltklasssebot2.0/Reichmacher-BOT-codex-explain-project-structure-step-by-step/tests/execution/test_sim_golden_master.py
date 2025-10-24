"""Golden-master regression for the deterministic OMS simulator."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from core.events import FillEvent, OrderEvent, OrderSide, OrderType
from execution.orderbook import OrderBook
from execution.simulator import ExecConfig, OmsSimulator, Rejection
from execution.slippage import ImpactLinear
from tests.execution.utils import stable_hash, vwap

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)
SYMBOL = "BTCUSDT"
EXPECTED_HASH = "6c69ef5979f68099924c26401184b0dbb07302ce10b14d64bbe56a654d516cd0"


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


def _record_rows(
    rows: list[tuple[str, str, str, float, float, float, str, str]],
    fills: Iterable[FillEvent],
    mid_reference: float | None,
    *,
    buy_prices: list[tuple[float, float]],
    sell_prices: list[tuple[float, float]],
    buy_mids: list[tuple[float, float]],
    sell_mids: list[tuple[float, float]],
) -> None:
    for fill in fills:
        rows.append(
            (
                fill.ts.isoformat(),
                fill.symbol,
                fill.side.value,
                fill.qty,
                fill.price,
                fill.fee,
                fill.liquidity_flag.value,
                fill.order_id,
            )
        )
        pair = (fill.qty, fill.price)
        if fill.side is OrderSide.BUY:
            buy_prices.append(pair)
            if mid_reference is not None:
                buy_mids.append((fill.qty, mid_reference))
        else:
            sell_prices.append(pair)
            if mid_reference is not None:
                sell_mids.append((fill.qty, mid_reference))


def test_simulator_golden_master() -> None:
    book = OrderBook()
    for price, size in [
        (99.5, 3.0),
        (99.0, 2.0),
        (98.5, 1.0),
    ]:
        book.add("bid", price, size)
    for price, size in [
        (100.5, 3.0),
        (101.0, 2.0),
        (101.5, 1.0),
    ]:
        book.add("ask", price, size)

    cfg = ExecConfig(
        slippage=ImpactLinear(k=1e-4),
        taker_fee=0.0004,
        maker_fee=0.0002,
        latency_ms=30,
        jitter_ms=5,
        seed=1337,
    )
    sim = OmsSimulator(book, cfg)

    rows: list[tuple[str, str, str, float, float, float, str, str]] = []
    buy_prices: list[tuple[float, float]] = []
    sell_prices: list[tuple[float, float]] = []
    buy_mids: list[tuple[float, float]] = []
    sell_mids: list[tuple[float, float]] = []

    # MARKET buy
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "mkt-buy",
            side=OrderSide.BUY,
            qty=2.5,
            order_type=OrderType.MARKET,
        ),
        BASE_TS,
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # MARKET sell
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "mkt-sell",
            side=OrderSide.SELL,
            qty=1.8,
            order_type=OrderType.MARKET,
        ),
        BASE_TS + timedelta(minutes=1),
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # LIMIT aggressive buy (partial taker, remainder resting)
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "limit-cross",
            side=OrderSide.BUY,
            qty=3.0,
            order_type=OrderType.LIMIT,
            price=101.0,
        ),
        BASE_TS + timedelta(minutes=2),
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # LIMIT passive post-only sell
    fills = sim.send(
        _make_order(
            "post-only",
            side=OrderSide.SELL,
            qty=0.7,
            order_type=OrderType.LIMIT,
            price=102.0,
            post_only=True,
        ),
        BASE_TS + timedelta(minutes=3),
    )
    assert fills == []

    # LIMIT sell IOC consuming resting bid
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "ioc-sell",
            side=OrderSide.SELL,
            qty=1.0,
            order_type=OrderType.LIMIT,
            price=101.0,
            tif="IOC",
        ),
        BASE_TS + timedelta(minutes=4),
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # LIMIT buy FOK that cannot complete
    result = sim.send(
        _make_order(
            "fok-buy",
            side=OrderSide.BUY,
            qty=3.0,
            order_type=OrderType.LIMIT,
            price=99.5,
            tif="FOK",
        ),
        BASE_TS + timedelta(minutes=5),
    )
    assert isinstance(result, Rejection)
    assert result.reason == "fok_unfilled"

    # STOP sell (market on trigger)
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "stop-sell",
            side=OrderSide.SELL,
            qty=1.0,
            order_type=OrderType.STOP,
            stop=100.0,
        ),
        BASE_TS + timedelta(minutes=6),
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # STOP LIMIT buy with partial taker fill
    mid_before = book.mid()
    assert mid_before is not None
    fills = sim.send(
        _make_order(
            "stop-limit-buy",
            side=OrderSide.BUY,
            qty=1.5,
            order_type=OrderType.STOP_LIMIT,
            price=101.6,
            stop=101.0,
        ),
        BASE_TS + timedelta(minutes=7),
    )
    assert not isinstance(fills, Rejection)
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    # Process maker fills deterministically
    mid_before = book.mid()
    fills = sim.tick(BASE_TS + timedelta(minutes=8))
    _record_rows(
        rows,
        fills,
        mid_before,
        buy_prices=buy_prices,
        sell_prices=sell_prices,
        buy_mids=buy_mids,
        sell_mids=sell_mids,
    )

    assert rows, "expected at least one fill event"

    digest = stable_hash(rows)
    assert digest == EXPECTED_HASH

    # VWAP validation for buy and sell flows
    buy_total = sum(q for q, _ in buy_prices)
    sell_total = sum(q for q, _ in sell_prices)
    assert buy_total > 0 and sell_total > 0

    buy_vwap = vwap(buy_prices)
    buy_manual = sum(q * price for q, price in buy_prices) / buy_total
    assert abs(buy_vwap - buy_manual) < 1e-9
    buy_mid = vwap(buy_mids)
    assert buy_vwap > buy_mid

    sell_vwap = vwap(sell_prices)
    sell_manual = sum(q * price for q, price in sell_prices) / sell_total
    assert abs(sell_vwap - sell_manual) < 1e-9
    sell_mid = vwap(sell_mids)
    assert sell_vwap < sell_mid
