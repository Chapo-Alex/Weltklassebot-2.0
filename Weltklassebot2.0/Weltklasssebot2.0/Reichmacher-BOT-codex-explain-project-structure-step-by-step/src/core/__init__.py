"""Core package exports for convenience imports."""

from .config import BacktestConfig
from .engine import BacktestEngine, ExecutionClient

__all__ = ["BacktestConfig", "BacktestEngine", "ExecutionClient"]
