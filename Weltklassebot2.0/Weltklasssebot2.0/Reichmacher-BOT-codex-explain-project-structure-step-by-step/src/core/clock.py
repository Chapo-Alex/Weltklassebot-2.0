"""Deterministic clock utilities for reproducible backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Protocol


class SupportsNow(Protocol):
    """Protocol describing objects that expose a :meth:`now` method."""

    def now(self) -> datetime:
        """Return the current timestamp."""


@dataclass(slots=True)
class SystemClock:
    """Thin wrapper around :func:`datetime.datetime.now` with a fixed timezone."""

    tz: tzinfo = UTC

    def now(self) -> datetime:
        """Return the current wall-clock time using the configured timezone."""

        return datetime.now(self.tz)


@dataclass(slots=True)
class DeterministicClock:
    """Monotonic clock that advances in deterministic steps for simulations."""

    start: datetime | None = None
    step: timedelta = timedelta(seconds=1)
    _current: datetime = field(init=False)

    def __post_init__(self) -> None:
        if self.step.total_seconds() <= 0:
            msg = "step must be a positive duration"
            raise ValueError(msg)
        base = self.start or datetime.now(UTC)
        if base.tzinfo is None:
            base = base.replace(tzinfo=UTC)
        self.start = base
        self._current = base

    def peek(self) -> datetime:
        """Return the timestamp that will be emitted by the next :meth:`now` call."""

        return self._current

    def now(self) -> datetime:
        """Return the current timestamp and advance by one configured step."""

        current = self._current
        self._current = current + self.step
        return current

    def advance(self, seconds: float) -> datetime:
        """Manually advance the internal clock by an arbitrary duration."""

        if seconds < 0:
            msg = "Cannot move deterministic clock backwards"
            raise ValueError(msg)
        delta = timedelta(seconds=seconds)
        self._current = self._current + delta
        return self._current

    def reset(self, ts: datetime) -> None:
        """Reset the clock to a later timestamp, ensuring monotonic progression."""

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if ts < self._current:
            msg = "DeterministicClock cannot be reset backwards"
            raise ValueError(msg)
        self._current = ts


__all__ = ["DeterministicClock", "SupportsNow", "SystemClock"]
