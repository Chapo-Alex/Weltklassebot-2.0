"""Portfolio accounting primitives with deterministic FIFO matching."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import copysign, isclose
from typing import Protocol

from core.events import FillEvent, LiquidityFlag, OrderSide
from core.metrics import DRAWDOWN_PCT, FILL_Q, PNL_REALIZED

_CLOSE_EPS = 1e-12


class FeeModel(Protocol):
    """Interface for fee calculation."""

    def fee(self, qty: float, price: float, taker: bool) -> float:
        """Return the absolute fee for a trade.

        Implementations must return a non-negative value. ``qty`` represents the
        filled quantity (always positive) and ``price`` the trade price.
        ``taker`` indicates whether the fill removed liquidity.
        """


@dataclass(slots=True)
class Position:
    """Tracks a symbol level position under FIFO realisation."""

    symbol: str
    avg_price: float = 0.0
    qty: float = 0.0
    realized_pnl: float = 0.0
    fees_accum: float = 0.0
    last_ts: datetime | None = None
    _lots: deque[tuple[float, float]] = field(default_factory=deque, init=False, repr=False)

    def apply_fill(self, fill: FillEvent, *, fee_override: float | None = None) -> None:
        """Apply a fill to the position using FIFO matching."""

        if fill.symbol != self.symbol:
            msg = f"Fill symbol {fill.symbol!r} does not match position {self.symbol!r}"
            raise ValueError(msg)
        if fill.qty <= 0:
            msg = "Fill quantity must be strictly positive"
            raise ValueError(msg)

        fee_value = fill.fee if fee_override is None else fee_override
        if fee_value < 0:
            msg = "Fee must be non-negative"
            raise ValueError(msg)

        signed_qty = fill.qty if fill.side is OrderSide.BUY else -fill.qty
        remaining = signed_qty

        while (
            not isclose(remaining, 0.0, abs_tol=_CLOSE_EPS)
            and self._lots
            and remaining * self._lots[0][0] < 0
        ):
            lot_qty, lot_price = self._lots[0]
            matched = min(abs(lot_qty), abs(remaining))
            if lot_qty > 0:
                self.realized_pnl += (fill.price - lot_price) * matched
            else:
                self.realized_pnl += (lot_price - fill.price) * matched

            if isclose(matched, abs(lot_qty), abs_tol=_CLOSE_EPS):
                self._lots.popleft()
            else:
                updated_qty = lot_qty - copysign(matched, lot_qty)
                if abs(updated_qty) <= _CLOSE_EPS:
                    self._lots.popleft()
                else:
                    self._lots[0] = (updated_qty, lot_price)

            remaining -= copysign(matched, remaining)

        if not isclose(remaining, 0.0, abs_tol=_CLOSE_EPS):
            self._lots.append((remaining, fill.price))

        self.qty = sum(qty for qty, _ in self._lots)
        if abs(self.qty) <= _CLOSE_EPS:
            self.qty = 0.0
            self.avg_price = 0.0
        else:
            total_qty = sum(abs(qty) for qty, _ in self._lots)
            total_cost = sum(abs(qty) * price for qty, price in self._lots)
            self.avg_price = total_cost / total_qty

        self.fees_accum -= fee_value
        self.last_ts = fill.ts

    def mark_to_market(self, price: float) -> float:
        """Return total PnL (realised + unrealised + fees) at ``price``."""

        unrealized = self.qty * (price - self.avg_price)
        return self.realized_pnl + self.fees_accum + unrealized


class Portfolio:
    """Account for cash and symbol level positions."""

    __slots__ = ("cash", "positions", "_fee_model", "_equity_peak", "_last_equity")

    def __init__(self, cash: float = 0.0, *, fee_model: FeeModel | None = None) -> None:
        self.cash = float(cash)
        self.positions: dict[str, Position] = {}
        self._fee_model = fee_model
        self._equity_peak = float(cash)
        self._last_equity = float(cash)
        PNL_REALIZED.set(0.0)
        DRAWDOWN_PCT.set(0.0)

    def apply_fill(self, fill: FillEvent) -> None:
        """Apply a fill updating cash and symbol position."""

        position = self.positions.get(fill.symbol)
        if position is None:
            position = self.positions[fill.symbol] = Position(fill.symbol)

        cash_flow = fill.price * fill.qty
        if fill.side is OrderSide.BUY:
            self.cash -= cash_flow
        elif fill.side is OrderSide.SELL:
            self.cash += cash_flow
        else:  # pragma: no cover - exhaustive guard
            msg = f"Unsupported order side {fill.side!r}"
            raise ValueError(msg)

        fee_value = fill.fee
        if self._fee_model is not None:
            taker = fill.liquidity_flag is LiquidityFlag.TAKER
            fee_value = self._fee_model.fee(fill.qty, fill.price, taker)

        position.apply_fill(fill, fee_override=fee_value)
        FILL_Q.labels(liquidity=fill.liquidity_flag.value).inc(fill.qty)
        self._update_realized_metric()

        if position.qty == 0.0 and not position._lots:
            # retain realised information but drop empty inventory containers
            position._lots.clear()

    def equity(self, price_map: Mapping[str, float]) -> float:
        """Compute marked-to-market equity given a price map."""

        total = self.cash
        for symbol, position in self.positions.items():
            price = price_map.get(symbol)
            if price is None:
                if position.qty == 0.0:
                    price = 0.0
                else:
                    msg = f"Missing mark price for symbol {symbol}"
                    raise KeyError(msg)
            total += position.qty * price + position.realized_pnl + position.fees_accum
        self._update_drawdown_metric(total)
        return total

    def exposure(self) -> dict[str, float]:
        """Return quantity exposure per symbol."""

        return {
            symbol: position.qty
            for symbol, position in self.positions.items()
            if position.qty != 0.0
        }

    def _update_realized_metric(self) -> None:
        realized_total = sum(
            position.realized_pnl + position.fees_accum
            for position in self.positions.values()
        )
        PNL_REALIZED.set(realized_total)

    def _update_drawdown_metric(self, equity: float) -> None:
        self._equity_peak = max(self._equity_peak, equity)
        self._last_equity = equity
        if self._equity_peak <= 0:
            drawdown_pct = 0.0
        else:
            drawdown_pct = max(0.0, (self._equity_peak - equity) / self._equity_peak)
        DRAWDOWN_PCT.set(drawdown_pct)


def fifo_pnl(fills: Iterable[FillEvent]) -> float:
    """Compute realised PnL via FIFO by replaying fills."""

    positions: dict[str, Position] = {}
    realized = 0.0
    for fill in fills:
        position = positions.get(fill.symbol)
        if position is None:
            position = positions[fill.symbol] = Position(fill.symbol)
        before = position.realized_pnl
        position.apply_fill(fill)
        realized += position.realized_pnl - before
    return realized


__all__ = ["FeeModel", "Position", "Portfolio", "fifo_pnl"]
