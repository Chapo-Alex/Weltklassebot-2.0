from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pytest

pytest.importorskip("pytest_benchmark", reason="pytest-benchmark plugin not installed")
pytest.importorskip("prometheus_client")

from core.config import BacktestConfig
from core.engine import BacktestEngine
from core.events import CandleEvent, OrderEvent, OrderSide, OrderType
from core.metrics import LAT, CollectorRegistry, get_registry, set_registry


@dataclass(slots=True)
class _SyntheticLoader:
    candles: Sequence[CandleEvent]

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        return [
            candle
            for candle in self.candles
            if candle.symbol == symbol and not (candle.end < start or candle.start > end)
        ]


class _DeterministicStrategy:
    def __init__(self) -> None:
        self._counter = 0

    def generate_orders(self, candles: Sequence[CandleEvent]) -> list[OrderEvent]:
        candle = candles[0]
        self._counter += 1
        side = OrderSide.BUY if self._counter % 2 else OrderSide.SELL
        order_id = f"perf-{self._counter}"
        return [
            OrderEvent(
                id=order_id,
                ts=candle.end,
                symbol=candle.symbol,
                side=side,
                qty=0.5,
                type=OrderType.MARKET,
                price=None,
                stop=None,
                tif="GTC",
            )
        ]


def _generate_candles(count: int, symbol: str) -> list[CandleEvent]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles: list[CandleEvent] = []
    price = 25_000.0
    for idx in range(count):
        shock = (idx % 5 - 2) * 12.0
        volume = 1_000.0 + idx * 10.0
        open_price = price
        close_price = max(1.0, price + shock)
        high_price = max(open_price, close_price) + abs(shock) * 0.25
        low_price = max(0.5, min(open_price, close_price) - abs(shock) * 0.25)
        start_ts = start + timedelta(minutes=idx)
        end_ts = start_ts + timedelta(minutes=1)
        candles.append(
            CandleEvent(
                symbol=symbol,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                start=start_ts,
                end=end_ts,
            )
        )
        price = close_price
    return candles


def _histogram_quantile(stage: str, quantile: float) -> float | None:
    collect = getattr(LAT, "collect", None)
    if collect is None:  # pragma: no cover - stubbed metrics path
        return None

    total = 0.0
    buckets: dict[float, float] = {}
    for metric in collect():
        for sample in metric.samples:
            name = getattr(sample, "name", sample[0])
            labels = getattr(sample, "labels", sample[1])
            value = float(getattr(sample, "value", sample[2]))
            if labels.get("stage") != stage:
                continue
            if name.endswith("_count"):
                total = value
            elif name.endswith("_bucket"):
                bound_raw = labels.get("le")
                if bound_raw is None:
                    continue
                if bound_raw in {"+Inf", "+inf", "inf"}:
                    buckets[math.inf] = value
                    continue
                try:
                    bound = float(bound_raw)
                except ValueError:
                    continue
                buckets[bound] = value
    if total <= 0.0:
        return None

    threshold = total * quantile
    for bound in sorted(buckets):
        if buckets[bound] >= threshold:
            if math.isinf(bound):
                return None
            return bound
    return None


_BUDGET_ENV = "STAGE_P95_BUDGET_SECONDS"


def _read_budget(default: float = 0.25) -> float:
    raw = os.getenv(_BUDGET_ENV)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(0.0, value)


_P95_BUDGET_SECONDS = _read_budget()


@pytest.fixture
def _metrics_registry() -> CollectorRegistry:
    previous = get_registry()
    registry = CollectorRegistry()
    set_registry(registry)
    try:
        yield registry
    finally:
        set_registry(previous)


def _build_engine(candles: Sequence[CandleEvent], data_dir: Path) -> BacktestEngine:
    start = candles[0].start
    end = candles[-1].end
    config = BacktestConfig(
        data_path=data_dir,
        symbol=candles[0].symbol,
        start=start,
        end=end,
        seed=1337,
        maker_fee=0.0002,
        taker_fee=0.0004,
        latency_ms=25.0,
        impact_coefficient=2e-7,
        initial_cash=500_000.0,
    )
    loader = _SyntheticLoader(candles)
    strategy = _DeterministicStrategy()
    return BacktestEngine(config, data_loader=loader, strategy=strategy)


@pytest.mark.perf
def test_stage_latency_budget(benchmark, tmp_path: Path, _metrics_registry) -> None:
    candles = _generate_candles(128, "BTCUSDT")

    def run_once() -> None:
        engine = _build_engine(candles, tmp_path)
        engine.run()

    benchmark(run_once)

    stages = ("data_load", "strategy", "risk_allow", "execution", "execution_tick")
    for stage in stages:
        quantile = _histogram_quantile(stage, 0.95)
        assert quantile is not None, f"missing latency samples for stage {stage}"
        assert (
            quantile < _P95_BUDGET_SECONDS
        ), f"{stage} p95 latency {quantile * 1000:.2f} ms exceeds {_P95_BUDGET_SECONDS * 1000:.0f} ms budget"
