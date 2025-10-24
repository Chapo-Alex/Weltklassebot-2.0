"""Reusable building blocks for deterministic backtests."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from core.events import CandleEvent, FillEvent

pd: Any | None = None
try:  # pragma: no cover - optional dependency for richer IO
    import pandas as _pd
except ModuleNotFoundError:  # pragma: no cover - fallback path
    _pd = None
pd = cast(Any, _pd)

if TYPE_CHECKING:
    from core.engine import BacktestConfig


@dataclass(slots=True)
class EquityPoint:
    """Single sample of the equity curve."""

    ts: datetime
    equity: float
    cash: float
    notional: float
    drawdown: float


@dataclass(slots=True)
class BacktestResult:
    """Container for the artefacts produced by a backtest run."""

    fills: list[FillEvent]
    equity_curve: list[EquityPoint]
    candles_processed: int
    config: BacktestConfig
    tca: list[dict[str, float]] = field(default_factory=list)

    def trades_records(self) -> list[dict[str, Any]]:
        """Return executed trades encoded as serialisable dictionaries."""

        return [
            {
                "order_id": fill.order_id,
                "timestamp": fill.ts.astimezone(UTC).isoformat(),
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": fill.qty,
                "price": fill.price,
                "fee": fill.fee,
                "liquidity": fill.liquidity_flag.value,
            }
            for fill in self.fills
        ]

    def equity_records(self) -> list[dict[str, Any]]:
        """Return the equity curve as serialisable dictionaries."""

        return [
            {
                "timestamp": point.ts.astimezone(UTC).isoformat(),
                "equity": point.equity,
                "cash": point.cash,
                "notional": point.notional,
                "drawdown": point.drawdown,
            }
            for point in self.equity_curve
        ]

    def trades_csv(self) -> str:
        """Render trade records to CSV."""

        header = ["order_id", "timestamp", "symbol", "side", "qty", "price", "fee", "liquidity"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=header)
        writer.writeheader()
        for record in self.trades_records():
            writer.writerow(record)
        return buffer.getvalue()

    def equity_payload(self) -> bytes:
        """Encode the equity curve to a deterministic JSON payload."""

        payload = {
            "schema": "equity_timeseries",
            "records": self.equity_records(),
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    def metrics(self) -> dict[str, Any]:
        """Compute summary metrics for reporting."""

        max_drawdown = max((point.drawdown for point in self.equity_curve), default=0.0)
        final_equity = (
            self.equity_curve[-1].equity if self.equity_curve else self.config.initial_cash
        )
        realised_trades = len(self.fills)
        return {
            "fills": realised_trades,
            "candles": self.candles_processed,
            "final_equity": final_equity,
            "max_drawdown": max_drawdown,
        }


class ParquetDataLoader:  # pragma: no cover - IO heavy and exercised via integration tests
    """Load candle events from a Parquet file or JSON fallback."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        if pd is not None:
            return self._load_with_pandas(symbol, start, end)
        return self._load_from_json(symbol, start, end)

    def _load_with_pandas(  # pragma: no cover - optional dependency path
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CandleEvent]:
        if pd is None:  # pragma: no cover - defensive fallback
            raise RuntimeError("pandas is required for parquet loading")
        assert pd is not None
        frame = pd.read_parquet(self._path)
        expected = {"symbol", "open", "high", "low", "close", "volume", "start", "end"}
        missing = expected.difference(frame.columns)
        if missing:
            missing_fmt = ", ".join(sorted(missing))
            msg = f"Parquet dataset missing required columns: {missing_fmt}"
            raise ValueError(msg)

        filtered = frame[frame["symbol"] == symbol].copy()
        if filtered.empty:
            return []

        filtered["start"] = pd.to_datetime(filtered["start"], utc=True)
        filtered["end"] = pd.to_datetime(filtered["end"], utc=True)
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        mask = (filtered["end"] >= start_ts) & (filtered["start"] <= end_ts)
        scoped = filtered.loc[mask]
        scoped = scoped.sort_values("start")

        candles: list[CandleEvent] = []
        for row in scoped.itertuples(index=False):
            candles.append(
                CandleEvent(
                    symbol=row.symbol,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    start=row.start.to_pydatetime(),
                    end=row.end.to_pydatetime(),
                )
            )
        return candles

    def _load_from_json(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "records" in raw:
            records = raw["records"]
        elif isinstance(raw, list):
            records = raw
        else:
            msg = "JSON dataset must be a list of records or contain a 'records' field"
            raise ValueError(msg)  # pragma: no cover - defensive guard

        candles: list[CandleEvent] = []
        for record in records:
            if record.get("symbol") != symbol:
                continue
            start_ts = self._coerce_datetime(record.get("start"))
            end_ts = self._coerce_datetime(record.get("end"))
            if end_ts < start or start_ts > end:
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

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        msg = f"Unsupported datetime value: {value!r}"
        raise TypeError(msg)  # pragma: no cover - defensive guard


@dataclass(slots=True)
class MakerTakerFeeModel:
    """Simple maker/taker percentage fee model."""

    maker_rate: float
    taker_rate: float

    def fee(self, qty: float, price: float, taker: bool) -> float:
        rate = self.taker_rate if taker else self.maker_rate
        return abs(qty) * price * rate


__all__ = [
    "BacktestResult",
    "EquityPoint",
    "MakerTakerFeeModel",
    "ParquetDataLoader",
]
