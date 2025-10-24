"""Benchmark the deterministic backtest engine without touching disk."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from types import ModuleType
from typing import TYPE_CHECKING, Any

from core.engine import BacktestConfig, BacktestEngine
from core.events import CandleEvent
from core.metrics import LAT, CollectorRegistry, configure_metrics, set_registry

if TYPE_CHECKING:
    from backtest.primitives import BacktestResult

try:  # pragma: no cover - numpy is optional
    import numpy as _np
except ModuleNotFoundError:  # pragma: no cover - fallback path
    _np = None

_resource: ModuleType | None
try:  # pragma: no cover - optional on non-unix platforms
    import resource as _resource_module
except ModuleNotFoundError:  # pragma: no cover - platform specific
    _resource = None
else:  # pragma: no cover - only executed when resource is available
    _resource = _resource_module


@dataclass(slots=True)
class _SyntheticLoader:
    """Static data loader that returns a prepared candle series."""

    candles: list[CandleEvent]

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        return [
            candle
            for candle in self.candles
            if candle.symbol == symbol and not (candle.end < start or candle.start > end)
        ]


def _generate_candles(count: int, seed: int, symbol: str) -> list[CandleEvent]:
    if count <= 0:
        return []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    candles: list[CandleEvent] = []
    base_price = 20_000.0
    price = base_price
    if _np is not None:
        rng = _np.random.default_rng(seed)
        shocks = rng.normal(0.0, 25.0, size=count)
        vols = rng.uniform(80.0, 240.0, size=count)
    else:
        from random import Random

        rng = Random(seed)
        shocks = [rng.uniform(-25.0, 25.0) for _ in range(count)]
        vols = [rng.uniform(80.0, 240.0) for _ in range(count)]
    for idx in range(count):
        shock = float(shocks[idx])
        volume = float(vols[idx])
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


def _build_config(candles: list[CandleEvent], seed: int, symbol: str) -> BacktestConfig:
    if not candles:
        now = datetime.now(tz=UTC)
        start = now
        end = now + timedelta(minutes=1)
    else:
        start = candles[0].start
        end = candles[-1].end
    return BacktestConfig(
        data_path=Path("."),
        symbol=symbol,
        start=start,
        end=end,
        seed=seed,
        maker_fee=0.0002,
        taker_fee=0.0004,
        latency_ms=30.0,
        impact_coefficient=1e-6,
        session_name="bench",
        execution="sim",
        exec_params={},
    )


def _run_once(candles: list[CandleEvent], seed: int, symbol: str) -> BacktestResult:
    config = _build_config(candles, seed, symbol)
    loader = _SyntheticLoader(list(candles))
    engine = BacktestEngine(config, data_loader=loader)
    return engine.run()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _candles_per_sec_summary(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"min": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "min": min(samples),
        "median": median(samples),
        "p95": _percentile(samples, 0.95),
    }


def _histogram_p95_ms() -> float | None:
    collect = getattr(LAT, "collect", None)
    if collect is None:  # pragma: no cover - stubbed histogram
        return None
    total = 0.0
    buckets: dict[float, float] = {}
    for metric in collect():
        for sample in metric.samples:
            name = getattr(sample, "name", sample[0])
            labels = getattr(sample, "labels", sample[1])
            value = float(getattr(sample, "value", sample[2]))
            if name.endswith("_count"):
                total += value
            elif name.endswith("_bucket"):
                bound_raw = labels.get("le")
                if bound_raw is None:
                    continue
                if bound_raw in {"+Inf", "+inf", "inf"}:
                    bound = math.inf
                else:
                    try:
                        bound = float(bound_raw)
                    except ValueError:
                        continue
                buckets[bound] = buckets.get(bound, 0.0) + value
    if total <= 0.0:
        return None
    threshold = total * 0.95
    running = 0.0
    for bound in sorted(buckets):
        running = buckets[bound]
        if running >= threshold:
            if math.isinf(bound):
                return None
            return bound * 1000.0
    return None


def _peak_rss_mb() -> float | None:
    if _resource is None:  # pragma: no cover - platform without resource module
        return None
    usage = _resource.getrusage(_resource.RUSAGE_SELF)
    rss = float(getattr(usage, "ru_maxrss", 0.0))
    if rss <= 0.0:
        return None
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def run_benchmark(
    *, candles: int, seed: int, repeats: int, symbol: str
) -> dict[str, Any]:
    series = _generate_candles(candles, seed, symbol)
    if not series:
        return {
            "candles": candles,
            "repeats": repeats,
            "seed": seed,
            "candles_per_sec": {"min": 0.0, "median": 0.0, "p95": 0.0},
            "p95_stage_latency_ms": None,
            "peak_rss_mb": _peak_rss_mb(),
        }
    warmup = min(len(series), max(64, len(series) // 20))
    _run_once(series[:warmup], seed, symbol)
    registry = CollectorRegistry()
    configure_metrics(registry)
    durations: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        _run_once(series, seed, symbol)
        duration = time.perf_counter() - start
        durations.append(duration)
    candles_per_sec = [candles / duration for duration in durations if duration > 0.0]
    summary = _candles_per_sec_summary(candles_per_sec)
    latency_ms = _histogram_p95_ms()
    rss_mb = _peak_rss_mb()
    set_registry(None)
    return {
        "candles": candles,
        "repeats": repeats,
        "seed": seed,
        "candles_per_sec": summary,
        "p95_stage_latency_ms": latency_ms,
        "peak_rss_mb": rss_mb,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candles", type=int, default=200_000, help="Number of candles to replay")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic RNG seed")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed benchmark runs")
    parser.add_argument("--symbol", default="BTCUSDT", help="Synthetic symbol to benchmark")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_benchmark(
        candles=max(1, args.candles),
        seed=args.seed,
        repeats=max(1, args.repeats),
        symbol=args.symbol,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
