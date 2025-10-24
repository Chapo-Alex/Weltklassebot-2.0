"""Golden master regression for portfolio accounting flows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from core.events import FillEvent, LiquidityFlag, OrderSide
from portfolio.accounting import Portfolio
from tests._utils.hash import stable_hash, to_csv


def _make_fill(
    order_id: str,
    ts: datetime,
    qty: float,
    price: float,
    fee: float,
    liquidity: LiquidityFlag,
    side: OrderSide,
    symbol: str = "BTCUSDT",
) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        ts=ts,
        qty=qty,
        price=price,
        fee=fee,
        liquidity_flag=liquidity,
        symbol=symbol,
        side=side,
    )


def test_accounting_golden_master() -> None:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    fills = [
        _make_fill(
            "buy-taker",
            base,
            qty=1.0,
            price=100.0,
            fee=0.04,
            liquidity=LiquidityFlag.TAKER,
            side=OrderSide.BUY,
        ),
        _make_fill(
            "buy-maker",
            base + timedelta(minutes=1),
            qty=0.5,
            price=100.5,
            fee=0.01005,
            liquidity=LiquidityFlag.MAKER,
            side=OrderSide.BUY,
        ),
        _make_fill(
            "sell-taker",
            base + timedelta(minutes=2),
            qty=0.8,
            price=102.0,
            fee=0.03264,
            liquidity=LiquidityFlag.TAKER,
            side=OrderSide.SELL,
        ),
        _make_fill(
            "sell-maker",
            base + timedelta(minutes=3),
            qty=0.7,
            price=101.5,
            fee=0.01421,
            liquidity=LiquidityFlag.MAKER,
            side=OrderSide.SELL,
        ),
    ]

    portfolio = Portfolio(cash=100_000.0)
    marks: dict[str, float] = {}
    peak_equity = portfolio.cash
    rows: list[tuple[str, float, float, float, float, float]] = []

    for fill in fills:
        marks[fill.symbol] = fill.price
        portfolio.apply_fill(fill)
        equity = portfolio.equity(marks)
        peak_equity = max(peak_equity, equity)
        drawdown_pct = 0.0 if peak_equity <= 0 else max(0.0, (peak_equity - equity) / peak_equity)
        position = portfolio.positions.get(fill.symbol)
        qty = position.qty if position is not None else 0.0
        realized = sum(pos.realized_pnl + pos.fees_accum for pos in portfolio.positions.values())
        unrealized = sum(
            pos.qty * (marks[pos.symbol] - pos.avg_price)
            for pos in portfolio.positions.values()
            if pos.qty != 0.0
        )
        rows.append((fill.ts.isoformat(), qty, equity, realized, unrealized, drawdown_pct))

    headers = ["timestamp", "qty", "equity", "realized", "unrealized", "drawdown_pct"]
    csv_payload = to_csv(rows, headers)
    digest = stable_hash(rows)

    assert csv_payload
    assert digest == "6b1543260ad065c290257c7f848ce8e711c182b36fef24ebe72af47fb952117e"
