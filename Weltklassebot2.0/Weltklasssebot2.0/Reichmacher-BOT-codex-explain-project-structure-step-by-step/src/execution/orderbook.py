"""Order book primitives for execution simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(slots=True)
class PriceLevel:
    """Represents a single price level in the order book."""

    price: float
    size: float

    def __post_init__(self) -> None:
        if self.size <= 0:
            msg = "price level size must be positive"
            raise ValueError(msg)
        if self.price <= 0:
            msg = "price level price must be positive"
            raise ValueError(msg)


@dataclass(slots=True)
class OrderBook:
    """Simple level-2 order book with FIFO consumption semantics."""

    bids: list[PriceLevel] = field(default_factory=list)
    asks: list[PriceLevel] = field(default_factory=list)
    ts: datetime | None = None

    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    def mid(self) -> float | None:
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    def depth(self, side: Literal["bid", "ask"]) -> list[tuple[float, float]]:
        levels = self._levels(side)
        return [(level.price, level.size) for level in levels]

    def add(self, side: Literal["bid", "ask"], price: float, size: float) -> None:
        if size <= 0:
            msg = "order size must be positive"
            raise ValueError(msg)
        if price <= 0:
            msg = "price must be positive"
            raise ValueError(msg)
        levels = self._levels(side)
        for level in levels:
            if level.price == price:
                level.size += size
                return
        new_level = PriceLevel(price=price, size=size)
        levels.append(new_level)
        reverse = side == "bid"
        levels.sort(key=lambda lvl: lvl.price, reverse=reverse)

    def remove(self, side: Literal["bid", "ask"], price: float, size: float) -> float:
        if size <= 0:
            msg = "removal size must be positive"
            raise ValueError(msg)
        levels = self._levels(side)
        for index, level in enumerate(levels):
            if level.price == price:
                filled = min(size, level.size)
                level.size -= filled
                if level.size <= 0:
                    levels.pop(index)
                return filled
        return 0.0

    def sweep(self, side: Literal["ask", "bid"], qty: float) -> tuple[float, float]:
        if qty <= 0:
            msg = "sweep quantity must be positive"
            raise ValueError(msg)
        levels = self._levels(side)
        remaining = qty
        notional = 0.0
        filled = 0.0
        idx = 0
        while remaining > 0 and idx < len(levels):
            level = levels[idx]
            take = min(remaining, level.size)
            notional += take * level.price
            filled += take
            remaining -= take
            level.size -= take
            if level.size <= 0:
                levels.pop(idx)
                continue
            idx += 1
        if filled == 0.0:
            return 0.0, 0.0
        return notional / filled, filled

    def _levels(self, side: Literal["bid", "ask"]) -> list[PriceLevel]:
        if side == "bid":
            return self.bids
        if side == "ask":
            return self.asks
        msg = f"unknown side: {side}"
        raise ValueError(msg)


__all__ = ["PriceLevel", "OrderBook"]
