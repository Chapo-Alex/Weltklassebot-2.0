"""Denial reason coverage for :mod:`portfolio.risk`."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.events import OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters


def _make_order(ts: datetime, *, qty: float = 1.0, price: float | None = 100.0) -> OrderEvent:
    return OrderEvent(
        id=f"order-{ts.isoformat()}",
        ts=ts,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=qty,
        type=OrderType.MARKET,
        price=price,
        stop=None,
        tif="GTC",
    )


def _make_ctx(
    *,
    now: datetime,
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


def _assert_increment(counter, before) -> None:  # type: ignore[no-untyped-def]
    after = _counter_value(counter)
    if before is None or after is None:
        pytest.skip("prometheus counter lacks readable state")
    assert after == pytest.approx(before + 1.0)


@pytest.mark.parametrize(
    ("params", "ctx_kwargs", "order_kwargs", "expected_reason"),
    [
        (
            RiskParameters(max_drawdown=0.05),
            {"drawdown": 0.06},
            {},
            "halted",
        ),
        (
            RiskParameters(max_notional=1_000.0),
            {"notional": 950.0},
            {"qty": 1.0, "price": 100.0},
            "max_notional",
        ),
        (
            RiskParameters(max_trades_per_day=1, cooldown_minutes=0.0),
            {"trades_today": 1},
            {},
            "trade_limit",
        ),
        (
            RiskParameters(max_trades_per_day=1, cooldown_minutes=5.0),
            {"trades_today": 1},
            {},
            "cooldown",
        ),
    ],
)
def test_denial_reasons_increment_counters(
    params: RiskParameters,
    ctx_kwargs: dict[str, object],
    order_kwargs: dict[str, object],
    expected_reason: str,
) -> None:
    manager = RiskManagerV2(params)
    now = datetime(2024, 1, 1, tzinfo=UTC)
    ctx = _make_ctx(now=now, **ctx_kwargs)
    counter = manager._metrics.denials.labels(reason=expected_reason)
    before = _counter_value(counter)

    result = manager.allow(_make_order(now, **order_kwargs), ctx)
    assert result == expected_reason
    _assert_increment(counter, before)


def test_cooldown_resets_on_session_change() -> None:
    params = RiskParameters(max_trades_per_day=1, cooldown_minutes=5.0)
    manager = RiskManagerV2(params)
    now = datetime(2024, 1, 1, tzinfo=UTC)

    cooldown_ctx = _make_ctx(now=now, trades_today=1, session="regular")
    assert manager.allow(_make_order(now), cooldown_ctx) == "cooldown"

    later = now + timedelta(minutes=1)
    reset_ctx = _make_ctx(now=later, trades_today=0, session="news_blackout")
    assert manager.allow(_make_order(later), reset_ctx) is True


def test_allow_updates_metrics_with_registry(fixed_clock) -> None:
    from portfolio.risk import CollectorRegistry, State

    registry = CollectorRegistry()
    params = RiskParameters(max_notional=5_000.0)
    manager = RiskManagerV2(params, registry=registry)
    now = fixed_clock.now()
    ctx = _make_ctx(now=now, notional=1_200.0)
    order = _make_order(now, qty=0.0, price=None)

    assert manager.allow(order, ctx) is True
    snapshot = manager.metrics_snapshot()
    assert snapshot["state"] == State.RUNNING.name
    assert snapshot["projected_exposure"] == pytest.approx(ctx.notional)
