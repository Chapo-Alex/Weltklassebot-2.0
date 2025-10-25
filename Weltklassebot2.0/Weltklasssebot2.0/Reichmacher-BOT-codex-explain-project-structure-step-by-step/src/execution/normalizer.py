"""Utilities for normalising orders to venue-specific constraints."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import math


def _validate_step(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid_{field}")


def _decimal(value: float) -> Decimal:
    try:
        return Decimal(str(value))
    except InvalidOperation as exc:  # pragma: no cover - defensive guard
        raise ValueError("invalid_numeric") from exc


def _quantize(value: float, step: float, minimum: float | None) -> float:
    decimal_value = _decimal(value)
    decimal_step = _decimal(step)
    try:
        steps = (decimal_value / decimal_step).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    except InvalidOperation as exc:  # pragma: no cover - defensive guard
        raise ValueError("invalid_numeric") from exc
    quantized = steps * decimal_step
    if minimum is not None:
        min_decimal = _decimal(minimum)
        if quantized < min_decimal:
            quantized = min_decimal
    return float(quantized)


def round_price(price: float | None, tick_size: float) -> float | None:
    """Round ``price`` to the nearest valid tick for the venue."""

    _validate_step(tick_size, field="tick_size")
    if price is None:
        return None
    if not math.isfinite(price) or price <= 0.0:
        raise ValueError("invalid_price")
    return _quantize(price, tick_size, tick_size)


def round_qty(qty: float, min_qty: float) -> float:
    """Round ``qty`` to the nearest valid lot size for the venue."""

    _validate_step(min_qty, field="min_qty")
    if not math.isfinite(qty) or qty <= 0.0:
        raise ValueError("invalid_quantity")
    return _quantize(qty, min_qty, min_qty)


__all__ = ["round_price", "round_qty"]
