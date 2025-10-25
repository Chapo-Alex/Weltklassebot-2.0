"""Paper-trading connector backed by the deterministic OMS simulator."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from core.events import FillEvent, OrderEvent
from execution.oms_simulator import OmsSimulator, Rejection

from .base import Connector


class _SupportsIntegers(Protocol):
    def integers(self, low: int, high: int | None = None) -> int: ...


try:  # optional dependency guard
    from numpy.random import Generator as _NPGenerator, PCG64 as _NPPCG64
except ModuleNotFoundError:  # pragma: no cover - numpy is optional for the connector
    import random as _random

    class _NPGenerator:  # type: ignore[override]
        __slots__ = ("_rng",)

        def __init__(self, bit_generator: "_NPPCG64 | None") -> None:
            seed = None if bit_generator is None else bit_generator.seed
            self._rng = _random.Random(seed)

        def integers(self, low: int, high: int | None = None) -> int:
            if high is None:
                high = low
                low = 0
            if high <= low:
                return low
            return self._rng.randrange(low, high)

    class _NPPCG64:  # type: ignore[override]
        __slots__ = ("seed",)

        def __init__(self, seed: int | None = None) -> None:
            self.seed = seed

    def _build_rng(seed: int | None = None) -> _SupportsIntegers:
        return _NPGenerator(_NPPCG64(seed))
else:  # pragma: no cover - numpy path

    def _build_rng(seed: int | None = None) -> _SupportsIntegers:
        return _NPGenerator(_NPPCG64(seed))


@dataclass(frozen=True, slots=True)
class _OrderState:
    order_id: str
    fills: tuple[FillEvent, ...]


def _default_run_id() -> str:
    return uuid4().hex


@dataclass(slots=True)
class PaperConnector(Connector):
    """Deterministic paper connector with idempotent submission semantics."""

    sim: OmsSimulator
    seed: int | None = None
    max_retries: int = 3
    base_backoff_ms: int = 50
    run_id: str = field(default_factory=_default_run_id)
    _fills: list[FillEvent] = field(default_factory=list)
    _sequence: int = 0
    _orders: dict[str, _OrderState] = field(default_factory=dict)

    def _next_coid(self) -> str:
        self._sequence += 1
        return f"{self.run_id}-{self._sequence}"

    def _jitter_ms(self, attempt: int) -> int:
        base = int(self.base_backoff_ms * (2**attempt))
        jitter_seed = None if self.seed is None else self.seed + attempt
        rng = _build_rng(jitter_seed)
        jitter = int(rng.integers(0, 7))
        return base + jitter

    def send_order(self, order: OrderEvent, *, idempotency_key: str | None) -> str:
        coid = idempotency_key or self._next_coid()
        cached = self._orders.get(coid)
        if cached is not None:
            return cached.order_id
        try:
            order = self.sim.normalise_order(order)
        except ValueError as exc:
            raise RuntimeError(f"order_normalisation_failed:{exc}") from exc
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
                fills: tuple[FillEvent, ...] = tuple(result) if result else ()
                if fills:
                    self._fills.extend(fills)
                state = _OrderState(order_id=order.id, fills=fills)
                self._orders[coid] = state
                return state.order_id
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
