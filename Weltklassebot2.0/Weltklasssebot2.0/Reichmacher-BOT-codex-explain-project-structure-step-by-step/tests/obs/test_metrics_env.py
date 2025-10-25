"""Unit tests for environment-driven metrics exporter behaviour."""

from __future__ import annotations

from typing import Any

from core import metrics


def test_maybe_start_server_uses_env_port(monkeypatch):
    calls: dict[str, Any] = {}

    def fake_start_http_server(port: int, registry: Any) -> None:
        calls["port"] = port
        calls["registry"] = registry

    monkeypatch.setenv("WELTKLASSE_METRICS_PORT", "12345")
    monkeypatch.setattr(metrics, "_PROM_CLIENT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(metrics, "start_http_server", fake_start_http_server, raising=False)

    previous = metrics.get_registry()
    try:
        metrics.configure_metrics()
        metrics.maybe_start_server()
    finally:
        metrics.set_registry(previous)

    assert calls["port"] == 12345
    assert calls["registry"] is metrics.get_registry()


def test_maybe_start_server_skips_without_prometheus_client(monkeypatch):
    called = False

    def fake_start_http_server(port: int, registry: Any) -> None:  # pragma: no cover - should not execute
        nonlocal called
        called = True

    monkeypatch.delenv("WELTKLASSE_METRICS_PORT", raising=False)
    monkeypatch.setattr(metrics, "_PROM_CLIENT_AVAILABLE", False, raising=False)
    monkeypatch.setattr(metrics, "start_http_server", fake_start_http_server, raising=False)

    previous = metrics.get_registry()
    try:
        metrics.maybe_start_server()
    finally:
        metrics.set_registry(previous)

    assert called is False
