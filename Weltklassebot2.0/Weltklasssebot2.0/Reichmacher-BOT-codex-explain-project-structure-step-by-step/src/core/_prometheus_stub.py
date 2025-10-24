"""Prometheus metric stubs for offline or test environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CollectorRegistry:
    """Placeholder registry mirroring the prometheus_client interface."""

    def register(self, metric: Any) -> None:  # pragma: no cover - compatibility hook
        del metric


class _MetricChild:
    def __init__(self) -> None:
        self._value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def set(self, value: float) -> None:
        self._value = value

    def observe(self, value: float) -> None:
        self._value += value


class _Metric:
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub path
        del args, kwargs
        self._value = 0.0
        self._children: dict[tuple[tuple[str, str], ...], _MetricChild] = {}

    def labels(self, **labels: str) -> _MetricChild:
        key = tuple(sorted(labels.items()))
        if key not in self._children:
            self._children[key] = _MetricChild()
        return self._children[key]

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def set(self, value: float) -> None:
        self._value = value

    def observe(self, value: float) -> None:
        self._value += value


class Counter(_Metric):
    """Stubbed counter metric."""


class Gauge(_Metric):
    """Stubbed gauge metric."""


class Histogram(_Metric):
    """Stubbed histogram metric."""


def start_http_server(port: int, registry: CollectorRegistry | None = None) -> None:
    del port, registry
    print("prometheus_client not installed; metrics exporter disabled", flush=True)


__all__ = [
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "start_http_server",
]
