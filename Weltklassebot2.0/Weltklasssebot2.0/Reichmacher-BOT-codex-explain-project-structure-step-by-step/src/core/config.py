"""Backtest configuration container for deterministic execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from portfolio.risk import RiskParameters
from strategy.breakout_bias import StrategyConfig


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for orchestrating a single backtest run."""

    data_path: Path
    symbol: str
    start: datetime
    end: datetime
    venue: str = "default"
    seed: int = 42
    maker_fee: float = 0.0
    taker_fee: float = 0.0
    latency_ms: float = 0.0
    impact_coefficient: float = 0.0
    initial_cash: float = 1_000_000.0
    session_name: str = "backtest"
    execution: Literal["sim", "paper"] = "sim"
    exec_params: dict[str, Any] = field(default_factory=dict)
    risk_store_dir: Path | None = None
    risk_store_rotate_lines: int | None = 100_000
    risk_store_rotate_mb: int | None = 64
    risk_store_fsync: bool = False
    risk: RiskParameters = field(
        default_factory=lambda: RiskParameters(
            max_drawdown=1e12,
            max_notional=1e12,
            max_trades_per_day=1_000_000,
            cooldown_minutes=0.0,
        )
    )
    strategy_config: StrategyConfig | None = None

    def __post_init__(self) -> None:
        self.data_path = self.data_path.expanduser().resolve()
        self.start = self._ensure_utc(self.start)
        self.end = self._ensure_utc(self.end)
        if self.end < self.start:
            msg = "End timestamp must not be before start timestamp"
            raise ValueError(msg)
        if self.risk_store_dir is not None:
            self.risk_store_dir = Path(self.risk_store_dir).expanduser().resolve()
        if self.risk_store_rotate_lines is not None and self.risk_store_rotate_lines <= 0:
            self.risk_store_rotate_lines = None
        if self.risk_store_rotate_mb is not None and self.risk_store_rotate_mb <= 0:
            self.risk_store_rotate_mb = None

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


__all__ = ["BacktestConfig"]
