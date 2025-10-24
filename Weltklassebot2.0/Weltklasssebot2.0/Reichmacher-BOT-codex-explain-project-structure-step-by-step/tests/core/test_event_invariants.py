"""Invariant checks for core event dataclasses."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from core.events import FillEvent, LiquidityFlag, OrderSide

try:  # pragma: no cover - optional dependency path
    from hypothesis import (  # type: ignore[no-redef]
        given,
        settings,
        strategies as st,  # type: ignore[no-redef]
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    from tests._hypothesis_stub import given, strategies as st

    def settings(*args, **kwargs):  # type: ignore[no-untyped-def]
        def _decorator(func):
            return func

        return _decorator


REFERENCE_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _side_from_flag(flag: int) -> OrderSide:
    return OrderSide.BUY if flag % 2 == 0 else OrderSide.SELL


@settings(max_examples=15, deadline=None, seed=1337)
@given(
    qty=st.floats(min_value=1e-9, max_value=10.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=1e-9, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    order_idx=st.integers(min_value=1, max_value=10_000),
    side_flag=st.integers(min_value=0, max_value=3),
    liquidity_flag=st.integers(min_value=0, max_value=1),
)
def test_fill_event_accepts_valid_inputs(
    qty: float,
    price: float,
    order_idx: int,
    side_flag: int,
    liquidity_flag: int,
) -> None:
    side_enum = _side_from_flag(side_flag)
    side_input: OrderSide | str
    if side_flag % 2 == 0:
        side_input = side_enum
    else:
        side_input = side_enum.value  # exercise string parsing branch

    liquidity = LiquidityFlag.MAKER if liquidity_flag == 0 else LiquidityFlag.TAKER
    fill = FillEvent(
        order_id=f"order-{order_idx}",
        ts=REFERENCE_TS,
        qty=qty,
        price=price,
        fee=0.0001,
        liquidity_flag=liquidity,
        symbol="BTCUSDT",
        side=side_input,
    )

    assert isinstance(fill.side, OrderSide)
    assert fill.qty == pytest.approx(qty)
    assert fill.price == pytest.approx(price)


@settings(max_examples=15, deadline=None, seed=2024)
@given(
    qty=st.floats(max_value=0.0, min_value=-10.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=1e-6, max_value=100_000.0, allow_nan=False, allow_infinity=False),
)
def test_fill_event_rejects_non_positive_qty(qty: float, price: float) -> None:
    with pytest.raises(ValueError):
        FillEvent(
            order_id="order-invalid",
            ts=REFERENCE_TS,
            qty=qty,
            price=price,
            fee=0.0,
            liquidity_flag=LiquidityFlag.TAKER,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
        )


@settings(max_examples=15, deadline=None, seed=7)
@given(
    price=st.floats(max_value=0.0, min_value=-10.0, allow_nan=False, allow_infinity=False),
    qty=st.floats(min_value=1e-6, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_fill_event_rejects_non_positive_price(price: float, qty: float) -> None:
    with pytest.raises(ValueError):
        FillEvent(
            order_id="order-invalid",
            ts=REFERENCE_TS,
            qty=qty,
            price=price,
            fee=0.0,
            liquidity_flag=LiquidityFlag.MAKER,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
        )


@settings(max_examples=15, deadline=None, seed=99)
@given(spaces=st.integers(min_value=0, max_value=4))
def test_fill_event_rejects_blank_order_id(spaces: int) -> None:
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


@settings(max_examples=15, deadline=None, seed=313)
@given(code=st.integers(min_value=0, max_value=5))
def test_fill_event_rejects_invalid_side(code: int) -> None:
    invalid_side = f"INVALID_{code}"
    with pytest.raises(ValueError):
        FillEvent(
            order_id="order-invalid",
            ts=REFERENCE_TS,
            qty=1.0,
            price=1.0,
            fee=0.0,
            liquidity_flag=LiquidityFlag.MAKER,
            symbol="BTCUSDT",
            side=invalid_side,
        )
