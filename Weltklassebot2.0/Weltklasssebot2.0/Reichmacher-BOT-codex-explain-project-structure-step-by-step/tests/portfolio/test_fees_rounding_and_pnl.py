"""Deterministic regression around fee rounding and PnL accounting."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import isfinite

import pytest

from core.events import FillEvent, LiquidityFlag, OrderSide
from portfolio.accounting import Portfolio

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)


class _RoundingFeeModel:
    def fee(self, qty: float, price: float, taker: bool) -> float:  # noqa: D401
        rate = 0.0004 if taker else 0.0001
        return round(qty * price * rate, 4)


def _make_fill(
    order_id: str,
    *,
    side: OrderSide,
    qty: float,
    price: float,
    liquidity: LiquidityFlag,
    offset: int,
) -> FillEvent:
    return FillEvent(
        order_id=order_id,
        ts=BASE_TS + timedelta(minutes=offset),
        qty=qty,
        price=price,
        fee=0.0,
        liquidity_flag=liquidity,
        symbol="BTCUSDT",
        side=side,
    )


def test_fee_rounding_and_pnl_consistency() -> None:
    portfolio = Portfolio(cash=1_000.0, fee_model=_RoundingFeeModel())
    fills = [
        _make_fill(
            "fill-1",
            side=OrderSide.BUY,
            qty=0.5,
            price=101.2345,
            liquidity=LiquidityFlag.TAKER,
            offset=0,
        ),
        _make_fill(
            "fill-2",
            side=OrderSide.SELL,
            qty=0.2,
            price=102.3456,
            liquidity=LiquidityFlag.MAKER,
            offset=1,
        ),
        _make_fill(
            "fill-3",
            side=OrderSide.BUY,
            qty=0.3,
            price=99.8765,
            liquidity=LiquidityFlag.TAKER,
            offset=2,
        ),
        _make_fill(
            "fill-4",
            side=OrderSide.SELL,
            qty=0.6,
            price=100.4321,
            liquidity=LiquidityFlag.TAKER,
            offset=3,
        ),
    ]
    marks = [100.9, 100.1, 99.2, 98.8]

    equities: list[float] = []
    drawdowns: list[float] = []
    peak = float("-inf")
    for fill, mark in zip(fills, marks, strict=True):
        portfolio.apply_fill(fill)
        equity = portfolio.equity({"BTCUSDT": mark})
        assert isfinite(equity)
        equities.append(equity)
        peak = max(peak, equity)
        drawdown = 0.0 if peak <= 0 else max(0.0, (peak - equity) / peak)
        drawdowns.append(drawdown)

    for prev, curr, eq_prev, eq_curr in zip(
        drawdowns,
        drawdowns[1:],
        equities,
        equities[1:],
        strict=True,
    ):
        if eq_curr <= eq_prev + 1e-9:
            assert curr >= prev - 1e-9

    position = portfolio.positions["BTCUSDT"]
    assert position.qty == pytest.approx(0.0, abs=1e-9)
    assert position.realized_pnl == pytest.approx(0.14818, abs=1e-6)
    assert position.fees_accum == pytest.approx(-0.0583, abs=1e-6)

    expected_equity = portfolio.cash + position.realized_pnl + position.fees_accum
    assert equities[-1] == pytest.approx(expected_equity)
