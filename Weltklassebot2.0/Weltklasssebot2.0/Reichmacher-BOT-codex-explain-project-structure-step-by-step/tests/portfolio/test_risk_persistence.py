from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from core.engine import BacktestConfig, BacktestEngine
from portfolio.risk import State
from portfolio.risk_admin import force_cooldown, halt, resume
from portfolio.risk_state_store import JsonlStateStore


def _patch_time(monkeypatch: pytest.MonkeyPatch, values: Iterable[float]) -> None:
    iterator = iter(values)
    from portfolio import risk_state_store as module

    monkeypatch.setattr(module.time, "time", lambda: next(iterator))


def test_state_store_default_and_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_time = 1_700_000_000.0
    from portfolio import risk_state_store as module

    monkeypatch.setattr(module.time, "time", lambda: fixed_time)
    store = JsonlStateStore(tmp_path)

    state = store.load_state()
    assert state == {"state": "RUNNING", "since": 0.0, "meta": {}}

    result = force_cooldown(store, 5, actor="ops", reason="maintenance")
    assert result["state"] == "COOLDOWN"
    assert result["meta"]["cooldown_minutes"] == 5
    assert result["since"] == fixed_time

    written = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert written["state"] == "COOLDOWN"
    assert written["meta"]["cooldown_minutes"] == 5
    assert written["since"] == fixed_time

    audit_lines = store.audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 1
    event = json.loads(audit_lines[0])
    assert event["actor"] == "ops"
    assert event["action"] == "cooldown"
    assert event["reason"] == "maintenance"


def test_halt_and_resume_append_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_time(monkeypatch, [1_700_000_100.0, 1_700_000_200.0, 1_700_000_300.0, 1_700_000_400.0])
    store = JsonlStateStore(tmp_path)

    halt(store, actor="ops", reason="issue detected")
    resume(store, actor="ops", reason="resolved")

    saved = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert saved["state"] == "RUNNING"
    assert saved["since"] == pytest.approx(1_700_000_400.0)

    audit_lines = store.audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(audit_lines) == 2
    first, second = map(json.loads, audit_lines)
    assert first["action"] == "halted"
    assert second["action"] == "running"


def test_engine_bootstraps_and_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store_dir = tmp_path / "store"
    store = JsonlStateStore(store_dir)
    store.save_state({"state": "COOLDOWN", "since": 42.0, "meta": {"reason": "ops"}})

    _patch_time(monkeypatch, [1_700_000_500.0, 1_700_000_600.0])

    data_path = tmp_path / "data.parquet"
    data_path.write_text("", encoding="utf-8")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1)

    config = BacktestConfig(
        data_path=data_path,
        symbol="BTCUSDT",
        start=start,
        end=end,
        risk_store_dir=store_dir,
    )

    class _DummyLoader:
        def load(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

    class _DummyStrategy:
        def generate_orders(self, candles: list[Any]) -> list[Any]:
            return []

    class _DummyExecution:
        def send(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        def tick(self, *args: Any, **kwargs: Any) -> None:
            return None

    engine = BacktestEngine(
        config,
        data_loader=_DummyLoader(),
        strategy=_DummyStrategy(),
        execution_client=_DummyExecution(),
    )

    assert engine._risk.state is State.COOLDOWN
    engine._update_risk_context(timestamp=end)

    persisted = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "RUNNING"
    assert persisted["meta"]["reason"] == "running"

    audit_lines = store.audit_path.read_text(encoding="utf-8").strip().splitlines()
    assert audit_lines, "audit trail should capture engine transition"
    event = json.loads(audit_lines[-1])
    assert event["state"] == "RUNNING"
    assert event["previous_state"] == "COOLDOWN"
    assert event["source"] == "transition"


def test_save_state_atomicity(tmp_path: Path) -> None:
    store = JsonlStateStore(tmp_path)
    payload = {"state": "HALTED", "since": 1_700_000_777.0, "meta": {}}
    store.save_state(payload)

    loaded = json.loads(store.state_path.read_text(encoding="utf-8"))
    assert loaded["state"] == "HALTED"
    assert loaded["since"] >= 0.0

    leftovers = [path for path in tmp_path.iterdir() if path.name.startswith(".state.")]
    assert leftovers == []
