"""Connector abstractions for execution backends."""

from __future__ import annotations

from .base import Connector
from .paper import PaperConnector

__all__ = ["Connector", "PaperConnector"]
