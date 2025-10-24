"""Property-based invariants for risk manager order allowances."""

from __future__ import annotations

from datetime import UTC, datetime

from core.events import OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters

try:  # pragma: no cover - optional dependency path
    from hypothesis import given, settings  # type: ignore[no-redef]
    from hypothesis import strategies as st  # type: ignore[no-redef]
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    from tests._hypothesis_stub import given, strategies as st

    def settings(*args, **kwargs):  # type: ignore[no-untyped-def]
        def _decorator(func):
            return func

        return _decorator


MAX_NOTIONAL = 10_000.0
MAX_DRAWDOWN = 0.05
MAX_TRADES = 3


@settings(max_examples=15, deadline=None)
@given(
    notional=st.floats(min_value=0.0, max_value=MAX_NOTIONAL * 0.8, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=10.0, max_value=250.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=0.0, max_value=3.0, allow_nan=False, allow_infinity=False),
    drawdown=st.floats(min_value=0.0, max_value=MAX_DRAWDOWN * 0.8, allow_nan=False, allow_infinity=False),
    trades=st.integers(min_value=0, max_value=MAX_TRADES - 1),
)
def test_orders_within_limits_are_allowed(
    notional: float,
    price: float,
    qty: float,
    drawdown: float,
    trades: int,
) -> None:
    params = RiskParameters(
        max_drawdown=MAX_DRAWDOWN,
        max_notional=MAX_NOTIONAL,
        max_trades_per_day=MAX_TRADES,
        cooldown_minutes=1.0,
    )
    manager = RiskManagerV2(params)
    now = datetime(2024, 1, 1, tzinfo=UTC)

    projected = notional + qty * price
    if projected >= params.max_notional:
        room = max(params.max_notional - notional, 0.0)
        qty = room / max(price, 1e-6)
        projected = notional + qty * price
    qty = max(qty, 0.0)

    ctx = RiskContext(
        equity=100_000.0,
        drawdown=min(drawdown, params.max_drawdown * 0.99),
        notional=min(notional, params.max_notional * 0.99),
        trades_today=min(trades, params.max_trades_per_day - 1),
        now=now,
        session="regular",
    )
    order = OrderEvent(
        id="order-prop",
        ts=now,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=qty,
        type=OrderType.MARKET,
        price=price,
        stop=None,
        tif="GTC",
    )

    assert manager.allow(order, ctx) is True
