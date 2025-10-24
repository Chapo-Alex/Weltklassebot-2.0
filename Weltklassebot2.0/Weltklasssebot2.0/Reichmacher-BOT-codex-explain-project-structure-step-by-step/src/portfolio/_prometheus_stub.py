"""Minimal Prometheus client stub for offline test environments."""

from __future__ import annotations

from core._prometheus_stub import CollectorRegistry, Counter, Gauge

__all__ = ["CollectorRegistry", "Counter", "Gauge"]
