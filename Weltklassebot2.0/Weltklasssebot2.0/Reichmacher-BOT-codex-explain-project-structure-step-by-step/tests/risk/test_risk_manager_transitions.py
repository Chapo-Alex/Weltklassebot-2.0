"""State machine transition tests for :mod:`portfolio.risk`."""

from __future__ import annotations

from datetime import timedelta

from core.events import OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters, State


def _make_order(ts, *, qty: float = 1.0, price: float = 100.0) -> OrderEvent:
    return OrderEvent(
        id="order-1",
        ts=ts,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=qty,
        type=OrderType.MARKET,
        price=price,
        stop=None,
        tif="GTC",
    )


def _ctx(
    *,
    now,
    equity: float = 100_000.0,
    drawdown: float = 0.0,
    notional: float = 0.0,
    trades_today: int = 0,
    session: str = "regular",
) -> RiskContext:
    return RiskContext(
        equity=equity,
        drawdown=drawdown,
        notional=notional,
        trades_today=trades_today,
        now=now,
        session=session,
    )


def test_running_to_cooldown_and_back(fixed_clock) -> None:
    params = RiskParameters(max_trades_per_day=1, cooldown_minutes=5)
    manager = RiskManagerV2(params)

    first = fixed_clock.now()
    state = manager.transition(_ctx(now=first))
    assert state is State.RUNNING

    limited_now = fixed_clock.advance(minutes=1)
    limit_ctx = _ctx(now=limited_now, trades_today=1)
    assert manager.transition(limit_ctx) is State.COOLDOWN
    cooldown_snapshot = manager.metrics_snapshot()
    assert cooldown_snapshot["state"] == State.COOLDOWN.name
    assert cooldown_snapshot["cooldown_until"] is not None
    assert manager.allow(_make_order(limited_now), limit_ctx) == "cooldown"

    resume_now = fixed_clock.advance(minutes=10)
    resume_ctx = _ctx(now=resume_now, trades_today=0)
    assert manager.transition(resume_ctx) is State.RUNNING
    assert manager.allow(_make_order(resume_now), resume_ctx) is True

    snapshot = manager.metrics_snapshot()
    assert snapshot["state"] == State.RUNNING.name
    assert snapshot["cooldown_until"] is None


def test_running_to_halted_on_drawdown(fixed_clock) -> None:
    params = RiskParameters(max_drawdown=50.0)
    manager = RiskManagerV2(params)

    start = fixed_clock.now()
    ctx = _ctx(now=start, drawdown=60.0)
    assert manager.transition(ctx) is State.HALTED

    result = manager.allow(_make_order(start), ctx)
    assert result == "halted"

    later = fixed_clock.advance(minutes=1)
    snapshot = manager.metrics_snapshot()
    assert snapshot["state"] == State.HALTED.name
    assert snapshot["cooldown_until"] is None
    assert snapshot["reason"] == "drawdown"

    # Even after time advances, the manager remains halted until external reset.
    future_ctx = _ctx(now=later + timedelta(minutes=30), drawdown=60.0)
    assert manager.transition(future_ctx) is State.HALTED
    assert manager.allow(_make_order(future_ctx.now), future_ctx) == "halted"
