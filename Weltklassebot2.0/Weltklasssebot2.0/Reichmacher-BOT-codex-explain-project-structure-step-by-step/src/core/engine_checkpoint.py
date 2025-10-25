"""Engine checkpoint persistence helpers for crash recovery."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from portfolio.accounting import Portfolio, Position
from state.jsonl_store import atomic_write


@dataclass(slots=True)
class PositionSnapshot:
    """Serializable representation of a ``Position``."""

    symbol: str
    avg_price: float
    qty: float
    realized_pnl: float
    fees_accum: float
    lots: list[tuple[float, float]]
    last_ts: str | None

    @classmethod
    def from_position(cls, position: Position) -> PositionSnapshot:
        return cls(
            symbol=position.symbol,
            avg_price=position.avg_price,
            qty=position.qty,
            realized_pnl=position.realized_pnl,
            fees_accum=position.fees_accum,
            lots=[(qty, price) for qty, price in position._lots],
            last_ts=position.last_ts.isoformat() if position.last_ts else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "avg_price": self.avg_price,
            "qty": self.qty,
            "realized_pnl": self.realized_pnl,
            "fees_accum": self.fees_accum,
            "lots": self.lots,
            "last_ts": self.last_ts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PositionSnapshot:
        return cls(
            symbol=str(payload["symbol"]),
            avg_price=float(payload.get("avg_price", 0.0)),
            qty=float(payload.get("qty", 0.0)),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            fees_accum=float(payload.get("fees_accum", 0.0)),
            lots=[tuple(map(float, entry)) for entry in payload.get("lots", [])],
            last_ts=payload.get("last_ts"),
        )

    def restore(self) -> Position:
        position = Position(self.symbol)
        position.avg_price = self.avg_price
        position.qty = self.qty
        position.realized_pnl = self.realized_pnl
        position.fees_accum = self.fees_accum
        position._lots = deque(self.lots)
        if self.last_ts is not None:
            position.last_ts = datetime.fromisoformat(self.last_ts)
        else:
            position.last_ts = None
        return position


@dataclass(slots=True)
class PortfolioSnapshot:
    """Serializable snapshot of ``Portfolio`` state."""

    cash: float
    equity_peak: float
    last_equity: float
    positions: list[PositionSnapshot]

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio) -> PortfolioSnapshot:
        return cls(
            cash=float(portfolio.cash),
            equity_peak=float(portfolio._equity_peak),
            last_equity=float(portfolio._last_equity),
            positions=[
                PositionSnapshot.from_position(position)
                for position in portfolio.positions.values()
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "equity_peak": self.equity_peak,
            "last_equity": self.last_equity,
            "positions": [snapshot.to_dict() for snapshot in self.positions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PortfolioSnapshot:
        positions_payload = payload.get("positions", [])
        return cls(
            cash=float(payload.get("cash", 0.0)),
            equity_peak=float(payload.get("equity_peak", 0.0)),
            last_equity=float(payload.get("last_equity", 0.0)),
            positions=[
                PositionSnapshot.from_dict(entry)
                for entry in positions_payload
            ],
        )

    def restore(self, portfolio: Portfolio) -> None:
        portfolio.cash = self.cash
        portfolio.positions = {
            snapshot.symbol: snapshot.restore() for snapshot in self.positions
        }
        portfolio._equity_peak = self.equity_peak
        portfolio._last_equity = self.last_equity
        portfolio._update_realized_metric()


@dataclass(slots=True)
class EngineCheckpointState:
    """Structured checkpoint payload persisted between runs."""

    run_id: str
    coid_sequence: int
    portfolio: PortfolioSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "coid_sequence": self.coid_sequence,
            "portfolio": self.portfolio.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EngineCheckpointState:
        return cls(
            run_id=str(payload.get("run_id", "")),
            coid_sequence=int(payload.get("coid_sequence", 0)),
            portfolio=PortfolioSnapshot.from_dict(payload.get("portfolio", {})),
        )

    def apply(self, *, connector: Any | None = None, portfolio: Portfolio | None = None) -> None:
        """Restore connector and portfolio state from the checkpoint."""

        if connector is not None:
            if hasattr(connector, "run_id"):
                connector.run_id = self.run_id
            if hasattr(connector, "_sequence"):
                setattr(connector, "_sequence", int(self.coid_sequence))
        if portfolio is not None:
            self.portfolio.restore(portfolio)


@dataclass(slots=True)
class EngineCheckpoint:
    """Persist and restore engine state for crash recovery."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def load(self) -> EngineCheckpointState | None:
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return EngineCheckpointState.from_dict(data)

    def save(self, state: EngineCheckpointState) -> None:
        payload = json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        atomic_write(self.path, payload, tmp_suffix=".chkpt.tmp")

    def persist(
        self,
        *,
        run_id: str,
        coid_sequence: int,
        portfolio: Portfolio,
    ) -> EngineCheckpointState:
        state = EngineCheckpointState(
            run_id=run_id,
            coid_sequence=int(coid_sequence),
            portfolio=PortfolioSnapshot.from_portfolio(portfolio),
        )
        self.save(state)
        return state

    def restore(
        self,
        *,
        connector: Any | None = None,
        portfolio: Portfolio | None = None,
    ) -> EngineCheckpointState | None:
        state = self.load()
        if state is None:
            return None
        state.apply(connector=connector, portfolio=portfolio)
        return state


__all__ = [
    "EngineCheckpoint",
    "EngineCheckpointState",
    "PortfolioSnapshot",
    "PositionSnapshot",
]
