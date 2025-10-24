"""Paper-trading connector backed by the deterministic OMS simulator."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, cast

from core.events import FillEvent, OrderEvent
from execution.simulator import OmsSimulator, Rejection

from .base import Connector


class _SupportsIntegers(Protocol):
    def integers(self, low: int, high: int | None = None) -> int: ...

try:  # optional dependency guard
    from numpy.random import default_rng as _numpy_default_rng
except ModuleNotFoundError:  # pragma: no cover - numpy is optional for the connector
    import random as _random

    class _StubGenerator:
        __slots__ = ("_rng",)

        def __init__(self, seed: int | None = None) -> None:
            self._rng = _random.Random(seed)

        def integers(self, low: int, high: int | None = None) -> int:
            if high is None:
                high = low
                low = 0
            if high <= low:
                return low
            return self._rng.randrange(low, high)

    def _default_rng(seed: int | None = None) -> _SupportsIntegers:
        return _StubGenerator(seed)
else:

    def _default_rng(seed: int | None = None) -> _SupportsIntegers:  # pragma: no cover - numpy path
        return cast(_SupportsIntegers, _numpy_default_rng(seed))


@dataclass(slots=True)
class PaperConnector(Connector):
    """Deterministic paper connector with idempotent submission semantics."""

    sim: OmsSimulator
    seed: int = 1337
    max_retries: int = 3
    base_backoff_ms: int = 50
    _idem: dict[str, str] = field(default_factory=dict)
    _fills: list[FillEvent] = field(default_factory=list)

    def _jitter_ms(self, attempt: int) -> int:
        base = int(self.base_backoff_ms * (2**attempt))
        rng = _default_rng(self.seed + attempt)
        jitter = int(rng.integers(0, 7))
        return base + jitter

    def send_order(self, order: OrderEvent, *, idempotency_key: str) -> str:
        if idempotency_key in self._idem:
            return self._idem[idempotency_key]
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.sim.send(order, now=order.ts)
            except Exception as exc:  # pragma: no cover - transient simulator failure
                last_error = exc
            else:
                if isinstance(result, Rejection):
                    reason = f"order_rejected:{result.reason}"
                    raise RuntimeError(reason)
                if result:
                    self._fills.extend(result)
                self._idem[idempotency_key] = order.id
                return order.id
            if attempt < self.max_retries:
                time.sleep(self._jitter_ms(attempt) / 1000.0)
        reason = f"send_order_failed:{last_error!s}" if last_error else "send_order_failed:unknown"
        raise RuntimeError(reason)

    def amend_order(self, order_id: str, **kwargs: object) -> None:
        """Paper connector does not mutate resting orders."""

    def cancel_order(self, order_id: str) -> None:
        """Paper connector does not maintain cancel state."""

    def fetch_fills(self, since: datetime | None = None) -> Sequence[FillEvent]:
        if since is None:
            return tuple(self._fills)
        return tuple(fill for fill in self._fills if fill.ts >= since)


__all__ = ["PaperConnector"]
