"""Reusable building blocks for deterministic backtests."""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from core.events import CandleEvent, FillEvent, OrderEvent
from data.loader_parquet import ParquetDataLoader

if TYPE_CHECKING:
    from core.config import BacktestConfig


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
    orders: list[OrderEvent] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

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

    def order_records(self) -> list[dict[str, Any]]:
        """Return submitted orders encoded as serialisable dictionaries."""

        return [
            {
                "order_id": order.id,
                "timestamp": order.ts.astimezone(UTC).isoformat(),
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "type": order.type.value,
                "price": order.price if order.price is not None else "",
                "stop": order.stop if order.stop is not None else "",
                "tif": order.tif,
                "reduce_only": order.reduce_only,
                "post_only": order.post_only,
                "client_tag": order.client_tag or "",
            }
            for order in self.orders
        ]

    def event_records(self) -> list[dict[str, Any]]:
        """Return replayable events in deterministic order."""

        return list(self.events)

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
