from __future__ import annotations

from pathlib import Path

import pytest

from state import JsonlStore


def test_fsync_invoked_when_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _track(fd: int) -> None:
        calls.append(fd)

    monkeypatch.setattr("state.jsonl_store.os.fsync", _track)

    store = JsonlStore(tmp_path / "audit.jsonl", rotate_lines=None, rotate_mb=None, fsync=True)
    store.append({"event": 1})
    store.append({"event": 2})

    assert len(calls) == 2


def test_fsync_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    called = False

    def _track(fd: int) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("state.jsonl_store.os.fsync", _track)

    store = JsonlStore(tmp_path / "audit.jsonl", rotate_lines=None, rotate_mb=None, fsync=False)
    store.append({"event": 1})

    assert called is False
