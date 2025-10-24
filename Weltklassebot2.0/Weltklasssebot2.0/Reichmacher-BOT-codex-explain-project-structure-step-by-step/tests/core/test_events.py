"""Tests for the event primitives."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import pytest

from src.core.events import (
    CandleEvent,
    FillEvent,
    LiquidityFlag,
    MarketEvent,
    OrderEvent,
    OrderSide,
    OrderType,
)


def _round_trip(instance: Any) -> None:
    data = asdict(instance)
    clone = type(instance)(**data)
    assert clone == instance
    assert clone is not instance


def test_market_event_round_trip() -> None:
    event = MarketEvent(
        symbol="BTC-USD",
        price=42_000.0,
        quantity=0.5,
        ts=datetime(2024, 5, 1, 12, 0),
    )
    _round_trip(event)


def test_candle_event_round_trip() -> None:
    event = CandleEvent(
        symbol="ETH-USD",
        open=2_500.0,
        high=2_550.0,
        low=2_480.0,
        close=2_520.0,
        volume=1_234.5,
        start=datetime(2024, 5, 1, 12, 0),
        end=datetime(2024, 5, 1, 12, 5),
    )
    _round_trip(event)


def test_order_event_round_trip() -> None:
    event = OrderEvent(
        id="order-123",
        ts=datetime(2024, 5, 1, 12, 0),
        symbol="BTC-USD",
        side=OrderSide.BUY,
        qty=1.25,
        type=OrderType.LIMIT,
        price=41_950.0,
        stop=None,
        tif="GTC",
        reduce_only=True,
        post_only=True,
        client_tag="strategy-alpha",
    )
    _round_trip(event)


def test_fill_event_round_trip() -> None:
    event = FillEvent(
        order_id="order-123",
        ts=datetime(2024, 5, 1, 12, 0) + timedelta(minutes=1),
        qty=1.25,
        price=41_950.0,
        fee=4.5,
        liquidity_flag=LiquidityFlag.MAKER,
        symbol="BTC-USD",
        side=OrderSide.BUY,
    )
    _round_trip(event)


EnumT = TypeVar("EnumT", bound=Enum)


@pytest.mark.parametrize(
    ("enum", "value", "expected"),
    [
        (OrderSide, "buy", OrderSide.BUY),
        (OrderSide, "SELL", OrderSide.SELL),
        (OrderType, "market", OrderType.MARKET),
        (OrderType, "Stop_Limit", OrderType.STOP_LIMIT),
        (LiquidityFlag, "maker", LiquidityFlag.MAKER),
        (LiquidityFlag, "TAKER", LiquidityFlag.TAKER),
    ],
)
def test_enum_parsing(enum: type[EnumT], value: str, expected: EnumT) -> None:
    assert enum(value) is expected


@pytest.mark.parametrize(
    "enum",
    [OrderSide, OrderType, LiquidityFlag],
)
def test_enum_parsing_invalid(enum: type[EnumT]) -> None:
    with pytest.raises(ValueError):
        enum("invalid")

