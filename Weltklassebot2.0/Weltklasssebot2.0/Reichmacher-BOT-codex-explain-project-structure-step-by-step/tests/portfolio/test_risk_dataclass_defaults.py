from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.events import OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters, State


def _order(price: float = 100.0, qty: float = 1.0) -> OrderEvent:
    return OrderEvent(
        id="order-1",
        ts=datetime(2024, 1, 1, tzinfo=UTC),
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=qty,
        type=OrderType.LIMIT,
        price=price,
        stop=None,
        tif="GTC",
        reduce_only=False,
        post_only=False,
        client_tag=None,
    )


def _context(
    *,
    equity: float,
    drawdown: float,
    notional: float,
    trades: int,
    now: datetime,
    session: str = "primary",
) -> RiskContext:
    return RiskContext(
        equity=equity,
        drawdown=drawdown,
        notional=notional,
        trades_today=trades,
        now=now,
        session=session,
    )


def test_allow_permits_within_limits() -> None:
    manager = RiskManagerV2(
        RiskParameters(
            max_drawdown=10.0,
            max_notional=1_000.0,
            max_trades_per_day=5,
            cooldown_minutes=30.0,
        )
    )
    ctx = _context(
        equity=100_000.0,
        drawdown=2.0,
        notional=200.0,
        trades=0,
        now=datetime(2024, 1, 1, 9, 0, tzinfo=UTC),
    )

    decision = manager.allow(_order(price=100.0, qty=2.0), ctx)

    assert decision is True
    snapshot = manager.metrics_snapshot()
    assert snapshot["state"] == "RUNNING"
    assert snapshot["projected_exposure"] == pytest.approx(400.0)


def test_trade_limit_triggers_cooldown_state() -> None:
    params = RiskParameters(
        max_drawdown=10.0,
        max_notional=10_000.0,
        max_trades_per_day=1,
        cooldown_minutes=5.0,
    )
    manager = RiskManagerV2(params)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    ctx = _context(
        equity=50_000.0,
        drawdown=0.0,
        notional=500.0,
        trades=1,
        now=now,
    )

    denial = manager.allow(_order(), ctx)

    assert denial == "cooldown"
    assert manager.state is State.COOLDOWN

    resumed = manager.transition(
        _context(
            equity=50_000.0,
            drawdown=0.0,
            notional=0.0,
            trades=0,
            now=now + timedelta(minutes=6),
        )
    )

    assert resumed is State.RUNNING
    assert manager.state is State.RUNNING


def test_trade_limit_denial_without_cooldown_window() -> None:
    params = RiskParameters(
        max_drawdown=10.0,
        max_notional=10_000.0,
        max_trades_per_day=1,
        cooldown_minutes=0.0,
    )
    manager = RiskManagerV2(params)
    ctx = _context(
        equity=25_000.0,
        drawdown=0.0,
        notional=0.0,
        trades=1,
        now=datetime(2024, 1, 1, 13, 0, tzinfo=UTC),
    )

    decision = manager.allow(_order(), ctx)

    assert decision == "trade_limit"
    assert manager.state is State.RUNNING


def test_halted_when_drawdown_exceeded() -> None:
    params = RiskParameters(
        max_drawdown=5.0,
        max_notional=10_000.0,
        max_trades_per_day=10,
        cooldown_minutes=1.0,
    )
    manager = RiskManagerV2(params)
    ctx = _context(
        equity=80_000.0,
        drawdown=5.0,
        notional=0.0,
        trades=0,
        now=datetime(2024, 1, 1, 10, 0, tzinfo=UTC),
    )

    reason = manager.allow(_order(), ctx)

    assert reason == "halted"
    assert manager.state is State.HALTED
    halted_snapshot = manager.metrics_snapshot()
    assert halted_snapshot["reason"] == "drawdown"


def test_notional_limit_denial_sets_metric() -> None:
    params = RiskParameters(
        max_drawdown=10.0,
        max_notional=1_000.0,
        max_trades_per_day=5,
        cooldown_minutes=10.0,
    )
    manager = RiskManagerV2(params)
    ctx = _context(
        equity=110_000.0,
        drawdown=1.0,
        notional=950.0,
        trades=0,
        now=datetime(2024, 1, 1, 14, 0, tzinfo=UTC),
    )

    reason = manager.allow(_order(price=200.0, qty=1.0), ctx)

    assert reason == "max_notional"
    assert manager.metrics_snapshot()["projected_exposure"] == pytest.approx(1_150.0)


def test_new_session_resets_cooldown_timer() -> None:
    params = RiskParameters(
        max_drawdown=10.0,
        max_notional=10_000.0,
        max_trades_per_day=1,
        cooldown_minutes=15.0,
    )
    manager = RiskManagerV2(params)
    first_ctx = _context(
        equity=90_000.0,
        drawdown=0.0,
        notional=1_000.0,
        trades=1,
        now=datetime(2024, 1, 1, 15, 0, tzinfo=UTC),
        session="session-a",
    )
    manager.allow(_order(), first_ctx)

    assert manager.state is State.COOLDOWN

    resumed_state = manager.transition(
        _context(
            equity=90_000.0,
            drawdown=0.0,
            notional=0.0,
            trades=0,
            now=datetime(2024, 1, 1, 15, 5, tzinfo=UTC),
            session="session-b",
        )
    )

    assert resumed_state is State.RUNNING
    assert manager.state is State.RUNNING
