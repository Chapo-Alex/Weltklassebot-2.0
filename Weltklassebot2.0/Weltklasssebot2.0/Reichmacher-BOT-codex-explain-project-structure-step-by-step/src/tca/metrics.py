"""Transaction cost analysis helpers for fills produced by the engine."""

from __future__ import annotations

from collections.abc import Sequence

from core.events import FillEvent, OrderSide

_EPS = 1e-12
_BPS = 10_000.0


def vwap(fills: Sequence[FillEvent]) -> float:
    """Return the volume-weighted average price for ``fills``.

    The VWAP is defined as ``sum(price * qty) / sum(qty)``. When ``fills`` is
    empty or the cumulative quantity is negligible, ``0.0`` is returned.
    """

    total_qty = sum(fill.qty for fill in fills)
    if total_qty <= _EPS:
        return 0.0
    total_notional = sum(fill.qty * fill.price for fill in fills)
    return total_notional / total_qty


def implementation_shortfall(
    entry_mid: float, fills: Sequence[FillEvent], side: OrderSide
) -> float:
    """Return implementation shortfall in basis points relative to ``entry_mid``.

    The metric is expressed as ``bps = 10_000 * (VWAP / entry_mid - 1) * sign``
    with ``sign = +1`` for ``BUY`` trades and ``sign = -1`` for ``SELL`` trades.
    Positive values indicate worse execution relative to the mid-price
    benchmark (i.e. higher VWAP for buys, lower VWAP for sells).
    """

    if entry_mid <= _EPS:
        return 0.0
    trade_vwap = vwap(fills)
    if trade_vwap <= _EPS:
        return 0.0
    ratio = trade_vwap / entry_mid
    sign = 1.0 if side is OrderSide.BUY else -1.0
    return _BPS * (ratio - 1.0) * sign


def adverse_selection(
    fills: Sequence[FillEvent], next_mid: float, side: OrderSide
) -> float:
    """Return adverse selection in basis points versus the subsequent mid-price.

    The computation follows ``bps = 10_000 * (VWAP / next_mid - 1) * sign`` with
    ``sign`` identical to :func:`implementation_shortfall`. Positive values
    represent price moves against the trade direction after execution.
    """

    if next_mid <= _EPS:
        return 0.0
    trade_vwap = vwap(fills)
    if trade_vwap <= _EPS:
        return 0.0
    ratio = trade_vwap / next_mid
    sign = 1.0 if side is OrderSide.BUY else -1.0
    return _BPS * (ratio - 1.0) * sign


__all__ = ["vwap", "implementation_shortfall", "adverse_selection"]

