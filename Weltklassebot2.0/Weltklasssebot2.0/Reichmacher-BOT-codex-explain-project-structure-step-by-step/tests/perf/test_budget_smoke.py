from __future__ import annotations

import os

import pytest

from scripts import bench_backtester


@pytest.mark.slow
def test_performance_budget() -> None:
    if os.environ.get("WELTKLASSE_SKIP_PERF") == "1":
        pytest.skip("performance budget check disabled")
    result = bench_backtester.run_benchmark(
        candles=1_000,
        seed=1337,
        repeats=1,
        symbol="BTCUSDT",
    )
    cps = result["candles_per_sec"]
    assert cps["min"] >= 833.0
    assert cps["median"] >= 833.0
    assert cps["p95"] >= 833.0
    latency = result["p95_stage_latency_ms"]
    if latency is not None:
        assert latency < 50.0
