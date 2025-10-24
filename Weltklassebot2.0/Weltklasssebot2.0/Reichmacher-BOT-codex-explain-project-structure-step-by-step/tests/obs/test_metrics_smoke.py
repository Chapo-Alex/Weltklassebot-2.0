"""Smoke-test metrics instrumentation without requiring an HTTP server."""

from __future__ import annotations

import pytest

from core import metrics
from tests.obs._prom_helpers import ensure_prometheus

PROM_MODULE = ensure_prometheus(None)
CollectorRegistry = PROM_MODULE.CollectorRegistry
generate_latest = PROM_MODULE.generate_latest

pytestmark = pytest.mark.obs


def _prepare_registry(monkeypatch: pytest.MonkeyPatch):
    prom = ensure_prometheus(monkeypatch)
    monkeypatch.setattr(metrics, "CollectorRegistry", prom.CollectorRegistry, raising=False)
    monkeypatch.setattr(metrics, "Counter", prom.Counter, raising=False)
    monkeypatch.setattr(metrics, "Gauge", prom.Gauge, raising=False)
    monkeypatch.setattr(metrics, "Histogram", prom.Histogram, raising=False)

    registry = CollectorRegistry()
    previous = metrics.get_registry()
    metrics.configure_metrics(registry)
    return registry, previous, metrics, generate_latest


def test_metrics_emit_into_custom_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, previous, metrics_module, exporter = _prepare_registry(monkeypatch)
    monkeypatch.setattr(metrics_module, "get_registry", lambda: registry, raising=False)

    metrics_module.LAT.labels(stage="strategy").observe(0.05)
    metrics_module.RISK_DENIALS.labels(reason="max_drawdown").inc()
    metrics_module.RISK_STATE.set(1)
    metrics_module.PNL_REALIZED.set(123.45)
    metrics_module.DRAWDOWN_PCT.set(0.25)
    metrics_module.FILL_Q.labels(liquidity="taker").inc(2.0)
    metrics_module.RISK_PROJECTED_EXPOSURE.set(1000.0)
    metrics_module.RISK_DRAWDOWN.set(0.1)
    metrics_module.RISK_EQUITY.set(101_000.0)
    metrics_module.CFG_MAX_DRAWDOWN.set(0.5)

    payload = exporter(registry).decode("utf-8")
    expected_metrics = [
        "stage_latency_seconds_count",
        "risk_denials_total",
        "risk_state",
        "pnl_realized",
        "drawdown_pct",
        "fills_total",
        "risk_projected_exposure",
        "risk_drawdown",
        "risk_equity",
        "cfg_max_dd",
    ]
    for metric_name in expected_metrics:
        assert metric_name in payload

    assert registry.get_sample_value(
        "stage_latency_seconds_count",
        {"stage": "strategy"},
    ) == pytest.approx(1.0)
    assert registry.get_sample_value(
        "risk_denials_total",
        {"reason": "max_drawdown"},
    ) == pytest.approx(1.0)
    assert registry.get_sample_value("risk_state", {}) == pytest.approx(1.0)
    assert registry.get_sample_value("pnl_realized", {}) == pytest.approx(123.45)
    assert registry.get_sample_value("drawdown_pct", {}) == pytest.approx(0.25)
    assert registry.get_sample_value(
        "fills_total",
        {"liquidity": "taker"},
    ) == pytest.approx(2.0)
    assert registry.get_sample_value("risk_projected_exposure", {}) == pytest.approx(1000.0)
    assert registry.get_sample_value("risk_drawdown", {}) == pytest.approx(0.1)
    assert registry.get_sample_value("risk_equity", {}) == pytest.approx(101_000.0)
    assert registry.get_sample_value("cfg_max_dd", {}) == pytest.approx(0.5)

    if previous is None:
        metrics.configure_metrics(None)
    else:
        metrics.configure_metrics(previous)
