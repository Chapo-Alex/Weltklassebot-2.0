"""Additional property-based validation tests for fill events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.events import FillEvent, LiquidityFlag, OrderSide

try:  # pragma: no cover - optional dependency path
    from hypothesis import given, settings, strategies as st
except ModuleNotFoundError:  # pragma: no cover - fallback stub
    from tests._hypothesis_stub import given, strategies as st  # type: ignore[assignment]

    def settings(*args, **kwargs):  # type: ignore[no-untyped-def]
        def _decorator(func):
            return func

        return _decorator

REFERENCE_TS = datetime(2024, 1, 1, tzinfo=UTC)


@settings(max_examples=12, deadline=None, seed=2025)
@given(
    qty=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=0.1, max_value=10_000.0, allow_nan=False, allow_infinity=False),
)
def test_fill_event_rejects_non_positive_values(qty: float, price: float) -> None:
    with pytest.raises(ValueError):
        FillEvent(
            order_id="invalid-qty",
            ts=REFERENCE_TS,
            qty=-abs(qty),
            price=price,
            fee=0.0,
            liquidity_flag=LiquidityFlag.TAKER,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
        )

    with pytest.raises(ValueError):
        FillEvent(
            order_id="invalid-price",
            ts=REFERENCE_TS,
            qty=qty,
            price=-abs(price),
            fee=0.0,
            liquidity_flag=LiquidityFlag.MAKER,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
        )


@settings(max_examples=12, deadline=None, seed=313)
@given(spaces=st.integers(min_value=0, max_value=5))
def test_fill_event_rejects_blank_identifier(spaces: int) -> None:
    order_id = " " * spaces
    with pytest.raises(ValueError):
        FillEvent(
            order_id=order_id,
            ts=REFERENCE_TS,
            qty=1.0,
            price=1.0,
            fee=0.0,
            liquidity_flag=LiquidityFlag.TAKER,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
        )


@settings(max_examples=12, deadline=None, seed=177)
@given(code=st.integers(min_value=0, max_value=5))
def test_fill_event_rejects_invalid_side(code: int) -> None:
    invalid_side = f"SIDE_{code}"
    with pytest.raises(ValueError):
        FillEvent(
            order_id="invalid-side",
            ts=REFERENCE_TS,
            qty=1.0,
            price=1.0,
            fee=0.0,
            liquidity_flag=LiquidityFlag.MAKER,
            symbol="BTCUSDT",
            side=invalid_side,
        )


@settings(max_examples=12, deadline=None, seed=733)
@given(
    qty=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=0.1, max_value=10_000.0, allow_nan=False, allow_infinity=False),
)
def test_fill_event_normalises_string_side(qty: float, price: float) -> None:
    fill = FillEvent(
        order_id="string-side",
        ts=REFERENCE_TS,
        qty=qty,
        price=price,
        fee=0.0,
        liquidity_flag=LiquidityFlag.MAKER,
        symbol="BTCUSDT",
        side="buy",
    )
    assert fill.side is OrderSide.BUY
