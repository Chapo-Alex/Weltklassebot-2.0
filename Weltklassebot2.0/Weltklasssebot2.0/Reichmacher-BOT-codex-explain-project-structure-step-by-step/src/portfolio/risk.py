"""Risk management state machine with Prometheus instrumentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    from pydantic import BaseModel
    try:  # pragma: no cover - pydantic v1 fallback
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - compatibility for v1
        ConfigDict = None
except ModuleNotFoundError:  # pragma: no cover - unit test fallback
    class BaseModel:  # type: ignore[no-redef]
        """Lightweight stub mimicking the pydantic ``BaseModel`` interface."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    ConfigDict = None

from core.events import OrderEvent
from core.metrics import (
    CFG_MAX_DRAWDOWN,
    RISK_DENIALS,
    RISK_DRAWDOWN,
    RISK_EQUITY,
    RISK_PROJECTED_EXPOSURE,
    RISK_STATE,
    CollectorRegistry,
    Counter,
    Gauge,
    configure_metrics,
)


class State(Enum):
    """Trading gate state maintained by :class:`RiskManagerV2`."""

    RUNNING = 0
    COOLDOWN = 1
    HALTED = 2


class RiskParameters(BaseModel):  # type: ignore[misc]
    """Configuration bundle that describes the guardrail thresholds."""

    max_drawdown: float = 0.0
    max_notional: float = 0.0
    max_trades_per_day: int = 0
    cooldown_minutes: float = 0.0

    if "ConfigDict" in globals() and ConfigDict is not None:  # pragma: no cover - v2 path
        model_config = ConfigDict(frozen=True)

    class Config:  # pragma: no cover - v1 path
        allow_mutation = False
        frozen = True


@dataclass(frozen=True, slots=True)
class RiskContext:
    """Snapshot of portfolio state required to evaluate risk gates."""

    equity: float
    drawdown: float
    notional: float
    trades_today: int
    now: datetime
    session: str


@dataclass(slots=True)
class _Metrics:
    """Container for Prometheus metric handles."""

    state: Gauge
    denials: Counter
    projected_exposure: Gauge
    drawdown: Gauge
    equity: Gauge

    @classmethod
    def create(cls, registry: CollectorRegistry | None) -> _Metrics:
        configure_metrics(registry)
        return cls(
            state=RISK_STATE,
            denials=RISK_DENIALS,
            projected_exposure=RISK_PROJECTED_EXPOSURE,
            drawdown=RISK_DRAWDOWN,
            equity=RISK_EQUITY,
        )


@dataclass(slots=True)
class RiskManagerV2:
    """Finite state machine that enforces drawdown, notional and trade gates."""

    params: RiskParameters
    registry: CollectorRegistry | None = None
    _state: State = field(init=False, default=State.RUNNING)
    _cooldown_until: datetime | None = field(init=False, default=None)
    _session: str | None = field(init=False, default=None)
    _metrics: _Metrics = field(init=False)
    _last_projected: float = field(init=False, default=0.0)
    _last_drawdown: float = field(init=False, default=0.0)
    _last_equity: float = field(init=False, default=0.0)
    _last_state_reason: str = field(init=False, default="init")

    def __post_init__(self) -> None:
        self._metrics = _Metrics.create(self.registry)
        CFG_MAX_DRAWDOWN.set(self.params.max_drawdown)
        self._set_state(State.RUNNING, reason="init", ctx=None)

    @property
    def state(self) -> State:
        """Current finite state."""

        return self._state

    def allow(self, order: OrderEvent, ctx: RiskContext) -> bool | str:
        """Determine whether an order is permitted under the current limits."""

        state = self.transition(ctx)
        projected = ctx.notional
        if order.price is not None:
            projected += abs(order.qty) * order.price
        self._last_projected = projected
        self._metrics.projected_exposure.set(projected)

        if state is State.HALTED:
            return self._deny("halted")
        if state is State.COOLDOWN:
            return self._deny("cooldown")

        if self.params.max_notional > 0.0 and projected > self.params.max_notional:
            return self._deny("max_notional")
        if (
            self.params.max_trades_per_day > 0
            and ctx.trades_today >= self.params.max_trades_per_day
        ):
            return self._deny("trade_limit")
        return True

    def transition(self, ctx: RiskContext) -> State:
        """Evaluate the state machine against the provided context."""

        if self._session != ctx.session:
            self._session = ctx.session
            self._cooldown_until = None

        self._last_equity = ctx.equity
        self._metrics.equity.set(ctx.equity)
        self._last_drawdown = ctx.drawdown
        self._metrics.drawdown.set(ctx.drawdown)

        if self.params.max_drawdown > 0.0 and ctx.drawdown >= self.params.max_drawdown:
            self._cooldown_until = None
            self._set_state(State.HALTED, reason="drawdown", ctx=ctx)
            return self._state

        if self._cooldown_until is not None and ctx.now >= self._cooldown_until:
            self._cooldown_until = None

        if (
            self.params.max_trades_per_day > 0
            and ctx.trades_today >= self.params.max_trades_per_day
        ) and self.params.cooldown_minutes > 0.0:
            cooldown = timedelta(minutes=self.params.cooldown_minutes)
            expires = ctx.now + cooldown
            if self._cooldown_until is None or expires > self._cooldown_until:
                self._cooldown_until = expires

        if self._cooldown_until is not None and ctx.now < self._cooldown_until:
            self._set_state(State.COOLDOWN, reason="cooldown", ctx=ctx)
        else:
            self._set_state(State.RUNNING, reason="running", ctx=ctx)
        return self._state

    def metrics_snapshot(self) -> dict[str, float | str | None]:
        """Return the latest metric values tracked by the manager."""

        cooldown_until = self._cooldown_until.isoformat() if self._cooldown_until else None
        return {
            "state": self._state.name,
            "state_value": float(self._state.value),
            "equity": self._last_equity,
            "drawdown": self._last_drawdown,
            "projected_exposure": self._last_projected,
            "cooldown_until": cooldown_until,
            "reason": self._last_state_reason,
        }

    def _set_state(
        self, new_state: State, *, reason: str, ctx: RiskContext | None
    ) -> None:
        previous = self._state
        if new_state is not previous:
            self._log_transition(new_state, reason, ctx)
        self._state = new_state
        self._last_state_reason = reason
        self._metrics.state.set(float(new_state.value))

    def _deny(self, reason: str) -> str:
        self._metrics.denials.labels(reason=reason).inc()
        return reason

    def _log_transition(
        self, new_state: State, reason: str, ctx: RiskContext | None
    ) -> None:  # pragma: no cover - logging side effect
        import logging

        logger = logging.getLogger("portfolio.risk")
        level = logging.INFO if new_state is State.RUNNING else logging.WARNING
        details: dict[str, Any] = {"state": new_state.name, "reason": reason}
        if ctx is not None:
            details["session"] = ctx.session
            details["timestamp"] = ctx.now.isoformat()
        logger.log(level, "risk state transition", extra=details)


RiskManager = RiskManagerV2

__all__ = [
    "CollectorRegistry",
    "RiskContext",
    "RiskManager",
    "RiskManagerV2",
    "RiskParameters",
    "State",
]
