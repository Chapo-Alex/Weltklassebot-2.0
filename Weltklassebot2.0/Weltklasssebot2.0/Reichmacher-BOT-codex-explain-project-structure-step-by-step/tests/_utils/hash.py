"""Helpers for deterministic hashing and CSV serialisation in tests."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

Cell = Any
Row = Sequence[Cell]


def _format_cell(value: Cell, float_fmt: str) -> str:
    if isinstance(value, float):
        spec = float_fmt[1:] if float_fmt.startswith(":") else float_fmt
        return format(value, spec)
    if isinstance(value, int):
        return str(int(value))
    if value is None:
        return ""
    return str(value)


def to_csv(
    rows: Iterable[Row],
    headers: Sequence[str] | None = None,
    *,
    float_fmt: str = ":.10f",
) -> str:
    """Render ``rows`` into a CSV string using deterministic formatting."""

    lines: list[str] = []
    if headers:
        lines.append(",".join(headers))
    for row in rows:
        formatted = [_format_cell(cell, float_fmt) for cell in row]
        lines.append(",".join(formatted))
    return "\n".join(lines)


def stable_hash(rows: Iterable[Row], *, float_fmt: str = ":.10f") -> str:
    """Return a SHA-256 hash for ``rows`` with stable float formatting."""

    payload = to_csv(rows, None, float_fmt=float_fmt)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["stable_hash", "to_csv"]
