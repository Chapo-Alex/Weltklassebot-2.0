"""Utilities to guarantee a Prometheus-like interface for observability tests."""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from types import ModuleType
from typing import Any


class _MetricBase:
    metric_type: str

    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: CollectorRegistry | None = None,
    ) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self._samples: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}
        if registry is not None:
            registry.register(self)

    def _key(self, labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(labels.items()))

    def _increment(self, sample: str, labels: Mapping[str, str], amount: float) -> None:
        bucket = self._samples.setdefault(sample, {})
        key = self._key(labels)
        bucket[key] = bucket.get(key, 0.0) + amount

    def _set(self, sample: str, labels: Mapping[str, str], value: float) -> None:
        bucket = self._samples.setdefault(sample, {})
        key = self._key(labels)
        bucket[key] = value

    def get_sample_value(
        self, sample: str, labels: Mapping[str, str] | None = None
    ) -> float | None:
        bucket = self._samples.get(sample)
        if bucket is None:
            return None
        key = self._key(labels or {})
        return bucket.get(key)

    def _child(self, labels: Mapping[str, str]) -> _MetricChild:
        return _MetricChild(self, labels)

    def all_samples(self) -> list[tuple[str, Mapping[str, str], float]]:
        emitted: list[tuple[str, Mapping[str, str], float]] = []
        for sample, bucket in self._samples.items():
            for key, value in bucket.items():
                emitted.append((sample, dict(key), value))
        return emitted


class _MetricChild:
    def __init__(self, parent: _MetricBase, labels: Mapping[str, str]) -> None:
        self._parent = parent
        self._labels = dict(labels)

    def inc(self, amount: float = 1.0) -> None:
        self._parent.inc(amount, labels=self._labels)

    def set(self, value: float) -> None:
        self._parent.set(value, labels=self._labels)

    def observe(self, value: float) -> None:
        self._parent.observe(value, labels=self._labels)


class CollectorRegistry:
    def __init__(self) -> None:
        self._metrics: list[_MetricBase] = []

    def register(self, metric: _MetricBase) -> None:
        if metric not in self._metrics:
            self._metrics.append(metric)

    def get_sample_value(
        self, name: str, labels: Mapping[str, str] | None = None
    ) -> float | None:
        for metric in self._metrics:
            value = metric.get_sample_value(name, labels)
            if value is not None:
                return value
        return None

    def collect(self) -> list[_MetricBase]:
        return list(self._metrics)


class Counter(_MetricBase):
    metric_type = "counter"

    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: CollectorRegistry | None = None,
    ) -> None:
        super().__init__(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
        )
        self._primary = name

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        self._increment(self._primary, labels or {}, amount)

    def labels(self, **labels: str) -> _MetricChild:
        return self._child(labels)

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self._set(self._primary, labels or {}, value)

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self.inc(value, labels=labels)


class Gauge(_MetricBase):
    metric_type = "gauge"

    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: CollectorRegistry | None = None,
    ) -> None:
        super().__init__(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
        )
        self._primary = name

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        current = self.get_sample_value(self._primary, labels or {}) or 0.0
        self._set(self._primary, labels or {}, current + amount)

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self._set(self._primary, labels or {}, value)

    def labels(self, **labels: str) -> _MetricChild:
        return self._child(labels)

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self.set(value, labels=labels)


class Histogram(_MetricBase):
    metric_type = "histogram"

    def __init__(
        self,
        name: str,
        documentation: str,
        *,
        labelnames: tuple[str, ...] = (),
        registry: CollectorRegistry | None = None,
    ) -> None:
        super().__init__(
            name,
            documentation,
            labelnames=labelnames,
            registry=registry,
        )
        self._count = f"{name}_count"
        self._sum = f"{name}_sum"
        self._bucket = f"{name}_bucket"

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        base_labels = dict(labels or {})
        self._increment(self._count, base_labels, 1.0)
        self._increment(self._sum, base_labels, float(value))
        bucket_labels = dict(base_labels)
        bucket_labels["le"] = "+Inf"
        self._increment(self._bucket, bucket_labels, 1.0)

    def labels(self, **labels: str) -> _MetricChild:
        return self._child(labels)

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        self.observe(amount, labels=labels)

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        self.observe(value, labels=labels)


def generate_latest(registry: CollectorRegistry) -> bytes:
    lines: list[str] = []
    for metric in registry.collect():
        lines.append(f"# HELP {metric.name} {metric.documentation}")
        lines.append(f"# TYPE {metric.name} {metric.metric_type}")
        for sample_name, labels, value in metric.all_samples():
            if labels:
                label_payload = ",".join(f'{key}="{val}"' for key, val in sorted(labels.items()))
                lines.append(f"{sample_name}{{{label_payload}}} {value}")
            else:
                lines.append(f"{sample_name} {value}")
    lines.append(f"# EOF {int(time.time())}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def start_http_server(
    port: int, registry: CollectorRegistry | None = None
) -> None:  # pragma: no cover
    del port, registry


def _install_stub() -> ModuleType:
    module = ModuleType("prometheus_client")
    module.CollectorRegistry = CollectorRegistry
    module.Counter = Counter
    module.Gauge = Gauge
    module.Histogram = Histogram
    module.generate_latest = generate_latest
    module.start_http_server = start_http_server
    module.__all__ = [
        "CollectorRegistry",
        "Counter",
        "Gauge",
        "Histogram",
        "generate_latest",
        "start_http_server",
    ]
    return module


def ensure_prometheus(monkeypatch: Any | None = None) -> ModuleType:
    try:
        import prometheus_client as prom  # type: ignore import-not-found
    except ModuleNotFoundError:
        prom = _install_stub()
        if monkeypatch is not None:
            monkeypatch.setitem(sys.modules, "prometheus_client", prom)
        else:
            sys.modules.setdefault("prometheus_client", prom)
    return prom


__all__ = [
    "ensure_prometheus",
    "CollectorRegistry",
    "Counter",
    "Gauge",
    "Histogram",
    "generate_latest",
]
