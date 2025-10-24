"""Exercise the Prometheus exporter without opening real sockets."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from types import FrameType

import pytest

from core import metrics
from scripts import exporter
from tests.obs._prom_helpers import ensure_prometheus

pytestmark = pytest.mark.obs


@pytest.fixture
def bound_registry(monkeypatch: pytest.MonkeyPatch):
    prom = ensure_prometheus(monkeypatch)
    monkeypatch.setattr(metrics, "CollectorRegistry", prom.CollectorRegistry, raising=False)
    monkeypatch.setattr(metrics, "Counter", prom.Counter, raising=False)
    monkeypatch.setattr(metrics, "Gauge", prom.Gauge, raising=False)
    monkeypatch.setattr(metrics, "Histogram", prom.Histogram, raising=False)

    registry = prom.CollectorRegistry()
    previous = metrics.get_registry()
    metrics.set_registry(registry)
    try:
        yield registry
    finally:
        metrics.set_registry(previous)


def test_exporter_starts_and_binds(
    bound_registry: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_calls: list[tuple[int, str, object | None]] = []

    def fake_start(port: int, *, addr: str | None = None, registry: object | None = None) -> None:
        start_calls.append((port, addr or "", registry))

    monkeypatch.setattr(exporter, "start_http_server", fake_start)
    stopper = threading.Event()
    stopper.set()
    monkeypatch.setattr(exporter, "Event", lambda: stopper)

    handlers: dict[int, Callable[[int, FrameType | None], None]] = {}

    def fake_signal(
        sig: int, handler: Callable[[int, FrameType | None], None]
    ) -> Callable[[int, FrameType | None], None]:
        handlers[sig] = handler
        return handler

    monkeypatch.setattr(exporter.signal, "signal", fake_signal)
    monkeypatch.delenv("WELTKLASSE_EXPORTER_TICK", raising=False)

    exit_code = exporter.run(["--addr", "127.0.0.1", "--port", "1234", "--grace-ms", "0"])
    assert exit_code == 0
    assert handlers
    assert start_calls == [(1234, "127.0.0.1", bound_registry)]


def test_exporter_handles_port_conflict(
    bound_registry: object, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    del bound_registry

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("address already in use")

    monkeypatch.setattr(exporter, "start_http_server", boom)
    monkeypatch.delenv("WELTKLASSE_EXPORTER_TICK", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        exporter.main(["--addr", "127.0.0.1", "--port", "9999", "--grace-ms", "0"])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    err_lines = [line for line in captured.err.splitlines() if line]
    assert len(err_lines) == 1
    assert "failed to start exporter" in err_lines[0]
    assert "addr=127.0.0.1" in err_lines[0]
    assert "port=9999" in err_lines[0]


def test_signal_handler_triggers_shutdown(
    bound_registry: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    start_calls: list[tuple[int, str, object | None]] = []

    def fake_start(port: int, *, addr: str | None = None, registry: object | None = None) -> None:
        start_calls.append((port, addr or "", registry))

    monkeypatch.setattr(exporter, "start_http_server", fake_start)

    stop_event = threading.Event()
    monkeypatch.setattr(exporter, "Event", lambda: stop_event)

    handlers: dict[int, Callable[[int, FrameType | None], None]] = {}
    ready = threading.Event()

    def fake_signal(
        sig: int, handler: Callable[[int, FrameType | None], None]
    ) -> Callable[[int, FrameType | None], None]:
        handlers[sig] = handler
        if sig == signal.SIGTERM:
            ready.set()
        return handler

    monkeypatch.setattr(exporter.signal, "signal", fake_signal)
    monkeypatch.delenv("WELTKLASSE_EXPORTER_TICK", raising=False)

    thread = threading.Thread(target=exporter.run, args=(["--grace-ms", "0"],), daemon=True)
    thread.start()
    assert ready.wait(1.0)
    assert start_calls == [(9000, "0.0.0.0", bound_registry)]

    handlers[signal.SIGINT](signal.SIGINT, None)
    assert stop_event.wait(1.0)
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_heartbeat_emits_metric(
    bound_registry: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    del bound_registry
    heartbeat_fired = threading.Event()

    class DummyLat:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.last_value: float | None = None

        def labels(self, *, stage: str) -> DummyLat:
            self.calls.append(stage)
            return self

        def observe(self, value: float) -> None:
            self.last_value = value
            heartbeat_fired.set()

    dummy_lat = DummyLat()
    monkeypatch.setattr(exporter, "LAT", dummy_lat)

    class FakeEvent:
        def __init__(self) -> None:
            self._set = False
            self.wait_calls = 0

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_calls += 1
            if self._set:
                return True
            assert timeout == exporter._HEARTBEAT_INTERVAL
            return False

        def set(self) -> None:
            self._set = True

    fake_event = FakeEvent()
    monkeypatch.setattr(exporter, "Event", lambda: fake_event)

    handlers: dict[int, Callable[[int, FrameType | None], None]] = {}
    ready = threading.Event()

    def fake_signal(
        sig: int, handler: Callable[[int, FrameType | None], None]
    ) -> Callable[[int, FrameType | None], None]:
        handlers[sig] = handler
        if sig == signal.SIGTERM:
            ready.set()
        return handler

    monkeypatch.setattr(exporter.signal, "signal", fake_signal)

    def fake_start(port: int, *, addr: str | None = None, registry: object | None = None) -> None:
        del port, addr, registry

    monkeypatch.setattr(exporter, "start_http_server", fake_start)
    monkeypatch.setenv("WELTKLASSE_EXPORTER_TICK", "1")

    thread = threading.Thread(target=exporter.run, args=(["--grace-ms", "0"],), daemon=True)
    thread.start()
    assert ready.wait(1.0)
    assert heartbeat_fired.wait(1.0)
    assert dummy_lat.calls, "heartbeat should emit at least one observation"
    assert set(dummy_lat.calls) == {"exporter_heartbeat"}
    assert dummy_lat.last_value == pytest.approx(0.0)

    handlers[signal.SIGINT](signal.SIGINT, None)
    fake_event.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert fake_event.wait_calls >= 2
