"""Execution test utilities for hashing and VWAP calculations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

NumberOrStr = str | float
Row = Sequence[NumberOrStr]


def stable_hash(rows: Iterable[Row]) -> str:
    """Return a deterministic SHA-256 hash for the given rows."""

    formatted_lines: list[str] = []
    for row in rows:
        formatted_cells: list[str] = []
        for cell in row:
            if isinstance(cell, float):
                formatted_cells.append(f"{cell:.10f}")
            else:
                formatted_cells.append(str(cell))
        formatted_lines.append(",".join(formatted_cells))
    payload = "\n".join(formatted_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def vwap(qp: Sequence[tuple[float, float]]) -> float:
    """Compute the volume-weighted average price for (qty, price) pairs."""

    total_qty = sum(q for q, _ in qp)
    if total_qty <= 0.0:
        raise ValueError("total quantity must be positive for VWAP computation")
    total_notional = sum(q * price for q, price in qp)
    return total_notional / total_qty


__all__ = ["stable_hash", "vwap"]
