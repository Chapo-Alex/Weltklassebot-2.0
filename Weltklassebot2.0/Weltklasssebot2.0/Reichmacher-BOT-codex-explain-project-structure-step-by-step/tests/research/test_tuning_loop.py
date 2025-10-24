from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backtest.primitives import BacktestResult, EquityPoint
from core.engine import BacktestConfig
from core.events import FillEvent, LiquidityFlag, OrderSide
from research.splits import time_kfold
from research.tuning import ConfigSpace, grid, objective_sharpe_penalized, random
from scripts.run_walkforward import run_walkforward_pipeline
from strategy.breakout_bias import StrategyConfig

np = pytest.importorskip("numpy")


def _timeline(count: int) -> list[datetime]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(count)]


def _make_config(start: datetime, end: datetime) -> BacktestConfig:
    return BacktestConfig(
        data_path=Path("/tmp/dataset.parquet"),
        symbol="BTCUSDT",
        start=start,
        end=end,
        seed=1337,
        strategy_config=StrategyConfig(),
    )


@dataclass(slots=True)
class _StubEngine:
    config: BacktestConfig

    def run(self) -> BacktestResult:
        params = self.config.strategy_config
        assert params is not None
        size = getattr(params, "order_size", 0.1)
        threshold = getattr(params, "breakout_threshold", 0.01)

        base = 100_000.0
        deltas = [size * 120 - threshold * 10, -size * 40, size * 90 - threshold * 5, size * 30]
        points: list[EquityPoint] = []
        equity = base
        max_equity = base
        for step, delta in enumerate(deltas, start=1):
            equity += delta
            max_equity = max(max_equity, equity)
            drawdown = max_equity - equity
            points.append(
                EquityPoint(
                    ts=self.config.start + timedelta(minutes=step),
                    equity=equity,
                    cash=equity,
                    notional=0.0,
                    drawdown=drawdown,
                )
            )

        fills = [
            FillEvent(
                order_id=f"fill-{step}",
                ts=self.config.start + timedelta(minutes=step),
                qty=size,
                price=100.0 + threshold,
                fee=0.0001 * size,
                liquidity_flag=LiquidityFlag.TAKER,
                symbol=self.config.symbol,
                side=OrderSide.BUY,
            )
            for step in range(1, 3)
        ]
        return BacktestResult(
            fills=fills,
            equity_curve=points,
            candles_processed=len(points),
            config=self.config,
        )


def test_grid_search_selects_lowest_penalty_params() -> None:
    index = _timeline(60)
    splits = time_kfold(index, folds=4, mode="expanding", embargo=timedelta(hours=2))
    base_config = _make_config(index[0], index[-1])

    space = ConfigSpace({
        "order_size": (0.4, 0.2),
        "breakout_threshold": (0.015, 0.01),
    })
    params = grid(space)

    result = run_walkforward_pipeline(
        index,
        splits,
        base_config,
        params,
        engine_factory=lambda cfg: _StubEngine(cfg),
        objective=objective_sharpe_penalized,
    )

    assert len(result["results"]) == len(params)
    assert result["best_params"] == {
        "order_size": 0.2,
        "breakout_threshold": 0.01,
    }


def test_random_search_deterministic_with_seed() -> None:
    index = _timeline(48)
    splits = time_kfold(index, folds=3, mode="rolling", embargo=timedelta(hours=1))
    base_config = _make_config(index[0], index[-1])

    space = ConfigSpace({"order_size": (0.1, 0.2, 0.3)})
    rng = np.random.default_rng(1337)
    samples = random(space, n=3, rng=rng)
    result_one = run_walkforward_pipeline(
        index,
        splits,
        base_config,
        samples,
        engine_factory=lambda cfg: _StubEngine(cfg),
        objective=objective_sharpe_penalized,
    )

    rng_repeat = np.random.default_rng(1337)
    repeat_samples = random(space, n=3, rng=rng_repeat)
    result_two = run_walkforward_pipeline(
        index,
        splits,
        base_config,
        repeat_samples,
        engine_factory=lambda cfg: _StubEngine(cfg),
        objective=objective_sharpe_penalized,
    )

    assert result_one["best_params"] == result_two["best_params"]
    assert len(result_one["results"]) == len(samples)
