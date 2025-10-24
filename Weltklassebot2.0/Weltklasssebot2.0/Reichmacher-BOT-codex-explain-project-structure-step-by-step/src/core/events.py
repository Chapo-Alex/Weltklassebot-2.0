"""Event primitives shared across the trading system."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal, Self


class _CaseInsensitiveStrEnum(StrEnum):
    """`StrEnum` variant that falls back to case-insensitive matching."""

    @classmethod
    def _missing_(cls, value: object) -> Self | None:
        if isinstance(value, str):
            value_upper = value.upper()
            for member in cls:  # pragma: no branch - deterministic iteration
                if member.value == value_upper:
                    return member
        return None


class OrderSide(_CaseInsensitiveStrEnum):
    """Buy or sell direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(_CaseInsensitiveStrEnum):
    """Supported order execution instructions."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class LiquidityFlag(_CaseInsensitiveStrEnum):
    """Whether a fill removed or added liquidity."""

    MAKER = "MAKER"
    TAKER = "TAKER"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """Represents a trade tick received from an exchange."""

    symbol: str
    price: float
    quantity: float
    ts: datetime


@dataclass(frozen=True, slots=True)
class CandleEvent:
    """Aggregated candle built from trade data."""

    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """State transition for an order lifecycle."""

    id: str
    ts: datetime
    symbol: str
    side: OrderSide
    qty: float
    type: OrderType
    price: float | None
    stop: float | None
    tif: Literal["GTC", "IOC", "FOK"]
    reduce_only: bool = False
    post_only: bool = False
    client_tag: str | None = None


@dataclass(frozen=True, slots=True)
class FillEvent:
    """Fill received after submitting an order."""

    order_id: str
    ts: datetime
    qty: float
    price: float
    fee: float
    liquidity_flag: LiquidityFlag
    symbol: str
    side: OrderSide

    def __post_init__(self) -> None:  # noqa: D401 - simple validation hook
        """Validate fill invariants after dataclass initialisation."""

        if not isinstance(self.order_id, str) or not self.order_id.strip():
            msg = "order_id must be a non-empty string"
            raise ValueError(msg)

        if isinstance(self.qty, bool) or not isinstance(self.qty, int | float):
            msg = "qty must be a positive number"
            raise ValueError(msg)
        if float(self.qty) <= 0.0:
            msg = "qty must be positive"
            raise ValueError(msg)

        if isinstance(self.price, bool) or not isinstance(self.price, int | float):
            msg = "price must be a positive number"
            raise ValueError(msg)
        if float(self.price) <= 0.0:
            msg = "price must be positive"
            raise ValueError(msg)

        if isinstance(self.side, OrderSide):
            return

        if isinstance(self.side, str):
            try:
                side_value = OrderSide(self.side)
            except ValueError as exc:  # pragma: no cover - error path asserted in tests
                msg = "side must be BUY or SELL"
                raise ValueError(msg) from exc
            object.__setattr__(self, "side", side_value)
            return

        msg = "side must be an OrderSide"
        raise ValueError(msg)


__all__ = [
    "OrderSide",
    "OrderType",
    "LiquidityFlag",
    "MarketEvent",
    "CandleEvent",
    "OrderEvent",
    "FillEvent",
]

