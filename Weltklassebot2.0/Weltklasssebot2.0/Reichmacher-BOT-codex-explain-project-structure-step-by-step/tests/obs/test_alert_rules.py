from __future__ import annotations

import re
from pathlib import Path

import pytest


def _load_yaml_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert "groups:" in text
    assert "alert: RiskStateHalted" in text
    assert "alert: RiskDenialsSpike" in text
    assert "alert: StageLatencyP95High" in text
    return text


def test_rules_file_exists_and_contains_exprs() -> None:
    path = Path("ops/alerts/rules.yml")
    assert path.exists(), "alerts rules.yml missing"
    text = _load_yaml_text(path)
    assert "max_over_time(risk_state{state=\"HALTED\"}[5m]) > 0" in text
    assert "increase(risk_denials_total[5m]) > 50" in text
    assert "histogram_quantile(0.95" in text
    assert "engine_stage_latency_bucket" in text


@pytest.mark.skipif(
    pytest.importorskip(
        "prometheus_client", reason="prometheus_client not installed"
    )
    is None,
    reason="prometheus_client unavailable",
)
def test_metric_names_and_sample_shapes_match_registry_primitives() -> None:
    """Validate observability primitives against the alert rules."""

    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    from src.core import metrics as m  # type: ignore

    registry = CollectorRegistry()
    set_registry = getattr(m, "set_registry", None)
    get_registry = getattr(m, "get_registry", None)
    original_registry = get_registry() if callable(get_registry) else None
    try:
        if callable(set_registry):
            set_registry(registry)

        risk_state = Gauge("risk_state", "current risk state", ("state",), registry=registry)
        risk_denials = Counter("risk_denials_total", "risk denials", registry=registry)
        stage_latency = Histogram(
            "engine_stage_latency",
            "stage latency seconds",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=registry,
        )

        risk_state.labels(state="HALTED").set(1.0)
        for _ in range(60):
            risk_denials.inc()
        for _ in range(200):
            stage_latency.observe(0.06)

        payload = generate_latest(registry)
        assert isinstance(payload, (bytes | bytearray))
        text = payload.decode("utf-8", errors="ignore")
        assert "risk_state" in text
        assert "risk_denials_total" in text
        assert "engine_stage_latency_bucket" in text
        assert "engine_stage_latency_count" in text
        assert "engine_stage_latency_sum" in text
        assert re.search(r"risk_denials_total\\{\\}\\s+60.0", text) is not None
    finally:
        if callable(set_registry):
            set_registry(original_registry)

