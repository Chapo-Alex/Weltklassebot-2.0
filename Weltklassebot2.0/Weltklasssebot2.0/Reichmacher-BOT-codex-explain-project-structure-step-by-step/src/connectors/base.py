"""Connector protocol definitions for execution adapters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from core.events import FillEvent, OrderEvent


class Connector(Protocol):
    """Abstraction for order routing backends."""

    def send_order(self, order: OrderEvent, *, idempotency_key: str | None = None) -> str:
        """Submit an order and return its broker-assigned identifier."""

    def amend_order(self, order_id: str, **kwargs: object) -> None:
        """Modify an existing working order."""

    def cancel_order(self, order_id: str) -> None:
        """Cancel an existing working order."""

    def fetch_fills(self, since: datetime | None = None) -> Sequence[FillEvent]:
        """Retrieve fills observed since the provided timestamp."""


__all__ = ["Connector"]
