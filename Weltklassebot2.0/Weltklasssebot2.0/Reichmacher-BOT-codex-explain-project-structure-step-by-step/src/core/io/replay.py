"""Helpers for loading deterministic replay logs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest.primitives import EquityPoint
from core.events import (
    FillEvent,
    LiquidityFlag,
    OrderEvent,
    OrderSide,
    OrderType,
)


@dataclass(slots=True)
class ReplayLogs:
    """Container aggregating replay artefacts."""

    orders: list[OrderEvent]
    fills: list[FillEvent]
    equity: list[EquityPoint]
    events: list[dict[str, Any]]


def load_orders_csv(path: Path) -> list[OrderEvent]:
    """Load order submissions from ``orders.csv``."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        orders: list[OrderEvent] = []
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"]).astimezone(UTC)
            price = _parse_optional_float(row.get("price"))
            stop = _parse_optional_float(row.get("stop"))
            reduce_only = _parse_bool(row.get("reduce_only"))
            post_only = _parse_bool(row.get("post_only"))
            client_tag = row.get("client_tag") or None
            order = OrderEvent(
                id=row["order_id"],
                ts=ts,
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
                qty=float(row["qty"]),
                type=OrderType(row["type"]),
                price=price,
                stop=stop,
                tif=row["tif"],
                reduce_only=reduce_only,
                post_only=post_only,
                client_tag=client_tag,
            )
            orders.append(order)
    return orders


def load_fills_csv(path: Path) -> list[FillEvent]:
    """Load fills from ``fills.csv``."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fills: list[FillEvent] = []
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"]).astimezone(UTC)
            fill = FillEvent(
                order_id=row["order_id"],
                ts=ts,
                qty=float(row["qty"]),
                price=float(row["price"]),
                fee=float(row["fee"]),
                liquidity_flag=LiquidityFlag(row["liquidity"]),
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
            )
            fills.append(fill)
    return fills


def load_equity_csv(path: Path) -> list[EquityPoint]:
    """Load equity curve samples from ``equity.csv``."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        points: list[EquityPoint] = []
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"]).astimezone(UTC)
            points.append(
                EquityPoint(
                    ts=ts,
                    equity=float(row["equity"]),
                    cash=float(row["cash"]),
                    notional=float(row["notional"]),
                    drawdown=float(row["drawdown"]),
                )
            )
    return points


def load_events_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL structured events."""

    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def load_replay_logs(directory: Path) -> ReplayLogs:
    """Load all replay artefacts from ``directory``."""

    directory = Path(directory)
    orders = load_orders_csv(directory / "orders.csv")
    fills = load_fills_csv(directory / "fills.csv")
    equity = load_equity_csv(directory / "equity.csv")
    events = load_events_jsonl(directory / "events.jsonl")
    return ReplayLogs(orders=orders, fills=fills, equity=equity, events=events)


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes"}


__all__ = [
    "ReplayLogs",
    "load_replay_logs",
    "load_orders_csv",
    "load_fills_csv",
    "load_equity_csv",
    "load_events_jsonl",
]
