"""Centralised Prometheus metric handles with offline fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

try:  # pragma: no cover - optional dependency
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
except ModuleNotFoundError:  # pragma: no cover - offline environments
    from core._prometheus_stub import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

@dataclass(slots=True)
class _MetricHandles:
    """Bundle of lazily cached metric handles per registry."""

    latency: Histogram
    risk_denials: Counter
    risk_state: Gauge
    pnl_realized: Gauge
    drawdown_pct: Gauge
    fill_quantity: Counter
    projected_exposure: Gauge
    risk_drawdown: Gauge
    risk_equity: Gauge
    cfg_max_drawdown: Gauge

_REGISTRY: CollectorRegistry | None = None
_CACHE: dict[int, _MetricHandles] = {}

LAT: Histogram = cast(Histogram, None)
RISK_DENIALS: Counter = cast(Counter, None)
RISK_STATE: Gauge = cast(Gauge, None)
PNL_REALIZED: Gauge = cast(Gauge, None)
DRAWDOWN_PCT: Gauge = cast(Gauge, None)
FILL_Q: Counter = cast(Counter, None)
RISK_PROJECTED_EXPOSURE: Gauge = cast(Gauge, None)
RISK_DRAWDOWN: Gauge = cast(Gauge, None)
RISK_EQUITY: Gauge = cast(Gauge, None)
CFG_MAX_DRAWDOWN: Gauge = cast(Gauge, None)


def _metric_kwargs() -> dict[str, CollectorRegistry]:
    if _REGISTRY is None:
        return {}
    return {"registry": _REGISTRY}


def _create_handles() -> _MetricHandles:
    kwargs = _metric_kwargs()
    latency = Histogram(
        "stage_latency_seconds",
        "Latency per execution stage in seconds.",
        labelnames=("stage",),
        **kwargs,
    )
    risk_denials = Counter(
        "risk_denials_total",
        "Total number of risk manager order denials.",
        labelnames=("reason",),
        **kwargs,
    )
    risk_state = Gauge(
        "risk_state",
        "Current finite state of the risk manager (RUNNING=0, COOLDOWN=1, HALTED=2).",
        **kwargs,
    )
    pnl_realized = Gauge(
        "pnl_realized",
        "Realised profit and loss across all positions.",
        **kwargs,
    )
    drawdown_pct = Gauge(
        "drawdown_pct",
        "Portfolio drawdown expressed as a fraction of the high-water mark.",
        **kwargs,
    )
    fill_quantity = Counter(
        "fills_total",
        "Quantity of fills observed by liquidity flag.",
        labelnames=("liquidity",),
        **kwargs,
    )
    projected_exposure = Gauge(
        "risk_projected_exposure",
        "Projected notional exposure after evaluating an order.",
        **kwargs,
    )
    risk_drawdown = Gauge(
        "risk_drawdown",
        "Drawdown forwarded to the risk manager.",
        **kwargs,
    )
    risk_equity = Gauge(
        "risk_equity",
        "Equity forwarded to the risk manager.",
        **kwargs,
    )
    cfg_max_drawdown = Gauge(
        "cfg_max_dd",
        "Configured maximum drawdown threshold.",
        **kwargs,
    )
    return _MetricHandles(
        latency=latency,
        risk_denials=risk_denials,
        risk_state=risk_state,
        pnl_realized=pnl_realized,
        drawdown_pct=drawdown_pct,
        fill_quantity=fill_quantity,
        projected_exposure=projected_exposure,
        risk_drawdown=risk_drawdown,
        risk_equity=risk_equity,
        cfg_max_drawdown=cfg_max_drawdown,
    )


def configure_metrics(registry: CollectorRegistry | None = None) -> None:
    """Bind all metric handles to ``registry`` or a default registry."""

    global _REGISTRY, LAT, RISK_DENIALS, RISK_STATE, PNL_REALIZED, DRAWDOWN_PCT, FILL_Q
    global RISK_PROJECTED_EXPOSURE, RISK_DRAWDOWN, RISK_EQUITY, CFG_MAX_DRAWDOWN

    if registry is None and _REGISTRY is not None:
        new_registry = _REGISTRY
    else:
        new_registry = registry or CollectorRegistry()
    if _REGISTRY is new_registry:
        return
    _REGISTRY = new_registry
    key = id(new_registry)
    handles = _CACHE.get(key)
    if handles is None:
        handles = _create_handles()
        _CACHE[key] = handles
    LAT = handles.latency
    RISK_DENIALS = handles.risk_denials
    RISK_STATE = handles.risk_state
    PNL_REALIZED = handles.pnl_realized
    DRAWDOWN_PCT = handles.drawdown_pct
    FILL_Q = handles.fill_quantity
    RISK_PROJECTED_EXPOSURE = handles.projected_exposure
    RISK_DRAWDOWN = handles.risk_drawdown
    RISK_EQUITY = handles.risk_equity
    CFG_MAX_DRAWDOWN = handles.cfg_max_drawdown


configure_metrics()


def get_registry() -> CollectorRegistry | None:
    """Return the currently bound registry instance."""

    return _REGISTRY


def set_registry(registry: CollectorRegistry | None) -> None:
    """Bind metrics to ``registry`` or reset to the default registry."""

    global _REGISTRY
    if registry is None:
        _REGISTRY = None
        configure_metrics()
        return
    configure_metrics(registry)

__all__ = [
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "LAT",
    "RISK_DENIALS",
    "RISK_STATE",
    "PNL_REALIZED",
    "DRAWDOWN_PCT",
    "FILL_Q",
    "RISK_PROJECTED_EXPOSURE",
    "RISK_DRAWDOWN",
    "RISK_EQUITY",
    "CFG_MAX_DRAWDOWN",
    "configure_metrics",
    "get_registry",
    "set_registry",
    "start_http_server",
]
