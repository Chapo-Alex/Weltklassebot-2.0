"""Exercise engine instrumentation hooks without network access."""

from __future__ import annotations

import pytest

from core import engine as engine_module, metrics
from tests.obs._prom_helpers import ensure_prometheus

PROM_MODULE = ensure_prometheus(None)
CollectorRegistry = PROM_MODULE.CollectorRegistry

pytestmark = pytest.mark.obs


def _configure_engine_metrics(monkeypatch: pytest.MonkeyPatch):
    prom = ensure_prometheus(monkeypatch)
    monkeypatch.setattr(metrics, "CollectorRegistry", prom.CollectorRegistry, raising=False)
    monkeypatch.setattr(metrics, "Counter", prom.Counter, raising=False)
    monkeypatch.setattr(metrics, "Gauge", prom.Gauge, raising=False)
    monkeypatch.setattr(metrics, "Histogram", prom.Histogram, raising=False)

    registry = CollectorRegistry()
    previous = metrics.get_registry()
    metrics.configure_metrics(registry)
    monkeypatch.setattr(engine_module, "LAT", metrics.LAT, raising=False)
    return registry, previous


def test_time_stage_records_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    registry, previous = _configure_engine_metrics(monkeypatch)

    dummy_engine = engine_module.BacktestEngine.__new__(engine_module.BacktestEngine)  # type: ignore[misc]

    stages = ("loader", "strategy", "accounting")
    for stage in stages:
        result = engine_module.BacktestEngine._time_stage(dummy_engine, stage, lambda s=stage: s)
        assert result == stage

    count = registry.get_sample_value(
        "stage_latency_seconds_count",
        {"stage": "accounting"},
    )
    assert count is not None and count >= 1.0

    bucket = registry.get_sample_value(
        "stage_latency_seconds_bucket",
        {"stage": "accounting", "le": "+Inf"},
    )
    assert bucket is not None and bucket >= 1.0

    if previous is None:
        metrics.configure_metrics(None)
    else:
        metrics.configure_metrics(previous)
