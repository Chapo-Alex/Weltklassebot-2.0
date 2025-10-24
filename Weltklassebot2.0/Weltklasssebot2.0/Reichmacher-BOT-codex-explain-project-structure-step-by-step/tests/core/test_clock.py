from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core import clock as clock_module
from core.clock import DeterministicClock, SystemClock


def test_deterministic_clock_advances_in_steps() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    clock = DeterministicClock(start=start, step=timedelta(seconds=5))

    first = clock.now()
    second = clock.now()

    assert first == start
    assert second == start + timedelta(seconds=5)


def test_deterministic_clock_reset_rejects_past() -> None:
    clock = DeterministicClock(start=datetime(2024, 1, 1, tzinfo=UTC))
    clock.now()

    with pytest.raises(ValueError):
        clock.reset(datetime(2023, 12, 31, tzinfo=UTC))


def test_deterministic_clock_advance_updates_state() -> None:
    clock = DeterministicClock(start=datetime(2024, 1, 1, tzinfo=UTC))
    clock.advance(10)

    assert clock.peek() == datetime(2024, 1, 1, 0, 0, 10, tzinfo=UTC)


def test_deterministic_clock_validates_inputs() -> None:
    with pytest.raises(ValueError):
        DeterministicClock(step=timedelta(seconds=0))

    clock = DeterministicClock(start=datetime(2024, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        clock.advance(-1)


def test_system_clock_respects_configured_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDatetime(datetime):
        calls: list[tuple[datetime, object | None]] = []

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            cls.calls.append((datetime(2024, 1, 1, tzinfo=UTC), tz))
            return datetime(2024, 1, 1, tzinfo=UTC)

    monkeypatch.setattr(clock_module, "datetime", _FakeDatetime)
    clock = SystemClock(tz=UTC)

    result = clock.now()

    assert result == datetime(2024, 1, 1, tzinfo=UTC)
    assert _FakeDatetime.calls[-1][1] is UTC
