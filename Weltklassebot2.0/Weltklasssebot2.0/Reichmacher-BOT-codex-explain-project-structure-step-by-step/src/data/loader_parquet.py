"""Utilities for loading UTC-normalised candle data from Parquet."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

try:  # pragma: no cover - optional dependency
    import pandas as _pd
except ModuleNotFoundError:  # pragma: no cover - optional dependency missing
    _pd = None

pd: Any | None = _pd

UTC = timezone.utc

if TYPE_CHECKING:
    from core.events import CandleEvent


class ParquetDataLoader:
    """Load candle events from Parquet files with JSON/CSV fallbacks."""

    _EXPECTED_COLUMNS = {
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "start",
        "end",
    }

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        start_utc = _ensure_utc_datetime(start)
        end_utc = _ensure_utc_datetime(end)
        if end_utc < start_utc:
            msg = "End timestamp must be greater than or equal to start timestamp"
            raise ValueError(msg)
        suffix = self._path.suffix.lower()
        if suffix == ".csv":
            records = self._load_from_csv()
        elif pd is not None and suffix in {".parquet", ".pq", ""}:
            records = self._load_with_pandas()
        else:
            records = self._load_from_json()
        CandleEvent = _resolve_candle_event()
        candles: list[CandleEvent] = []
        for record in records:
            if record.get("symbol") != symbol:
                continue
            start_ts = _ensure_utc_datetime(record.get("start"))
            end_ts = _ensure_utc_datetime(record.get("end"))
            if end_ts < start_utc or start_ts > end_utc:
                continue
            candles.append(
                CandleEvent(
                    symbol=symbol,
                    open=float(record["open"]),
                    high=float(record["high"]),
                    low=float(record["low"]),
                    close=float(record["close"]),
                    volume=float(record["volume"]),
                    start=start_ts,
                    end=end_ts,
                )
            )
        candles.sort(key=lambda candle: candle.start)
        return candles

    def _load_with_pandas(self) -> Iterable[dict[str, Any]]:
        if pd is None:  # pragma: no cover - defensive guard
            raise RuntimeError("pandas is required for Parquet loading")
        frame = pd.read_parquet(self._path)
        missing = self._EXPECTED_COLUMNS.difference(frame.columns)
        if missing:
            missing_fmt = ", ".join(sorted(missing))
            msg = f"Parquet dataset missing required columns: {missing_fmt}"
            raise ValueError(msg)
        if "symbol" not in frame.columns:
            return []
        records = frame.to_dict("records")
        return records

    def _load_from_json(self) -> Iterable[dict[str, Any]]:
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            records = payload["records"]
        elif isinstance(payload, list):
            records = payload
        else:  # pragma: no cover - invalid structure guard
            msg = "JSON dataset must be a list of records or contain a 'records' field"
            raise ValueError(msg)
        return records

    def _load_from_csv(self) -> Iterable[dict[str, Any]]:
        buffer = self._path.read_text(encoding="utf-8")
        reader = csv.DictReader(buffer.splitlines())
        records: list[dict[str, Any]] = []
        for row in reader:
            record: dict[str, Any] = {
                "symbol": row["symbol"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "start": row["start"],
                "end": row["end"],
            }
            records.append(record)
        return records


def _ensure_utc_datetime(candidate: Any) -> datetime:
    """Return ``candidate`` as a timezone-aware UTC datetime."""

    if isinstance(candidate, datetime):
        dt = candidate
    elif pd is not None:
        try:
            dt = pd.Timestamp(candidate).to_pydatetime()
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError(f"Unsupported datetime value: {candidate!r}") from exc
    elif isinstance(candidate, str):
        dt = datetime.fromisoformat(candidate)
    else:
        raise TypeError(f"Unsupported datetime value: {candidate!r}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)

    offset = dt.utcoffset()
    if offset is None:
        raise ValueError("Timestamp must have a valid UTC offset")
    if offset != timedelta(0):
        dt = dt.astimezone(UTC)
        offset = dt.utcoffset()
        if offset != timedelta(0):
            raise ValueError("Timestamp offset must normalise to UTC")
    return dt


@lru_cache(maxsize=1)
def _resolve_candle_event() -> type["CandleEvent"]:
    module = import_module("core.events")
    return getattr(module, "CandleEvent")


__all__ = ["ParquetDataLoader"]
