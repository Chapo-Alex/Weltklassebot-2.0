"""Execute a walk-forward backtest grid or random search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from backtest.primitives import ParquetDataLoader
from core.engine import BacktestConfig, BacktestEngine
from research.splits import time_kfold
from research.tuning import (
    ConfigSpace,
    grid,
    objective_sharpe_penalized,
    random,
)
from strategy.breakout_bias import StrategyConfig

try:  # pragma: no cover - optional dependency for CLI convenience
    from numpy.random import default_rng
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    default_rng = None


def _parse_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"Invalid datetime value: {value}"
        raise argparse.ArgumentTypeError(msg) from exc
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_embargo(value: str) -> timedelta:
    token = value.strip()
    if not token:
        msg = "Embargo value must not be empty"
        raise argparse.ArgumentTypeError(msg)
    unit = token[-1].upper()
    magnitude_raw = token[:-1]
    try:
        magnitude = float(magnitude_raw)
    except ValueError as exc:
        msg = f"Invalid embargo magnitude: {magnitude_raw!r}"
        raise argparse.ArgumentTypeError(msg) from exc
    mapping = {
        "S": {"seconds": magnitude},
        "M": {"minutes": magnitude},
        "H": {"hours": magnitude},
        "D": {"days": magnitude},
    }
    if unit not in mapping:
        msg = f"Unsupported embargo unit: {unit!r}"
        raise argparse.ArgumentTypeError(msg)
    return timedelta(**mapping[unit])


def _load_index(
    loader: ParquetDataLoader,
    symbol: str,
    start: datetime,
    end: datetime,
) -> list[datetime]:
    candles = loader.load(symbol, start, end)
    return [candle.end for candle in candles]


def _strategy_to_kwargs(config: StrategyConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        dump = getattr(config, "model_dump")
        if callable(dump):
            payload = cast(Mapping[str, Any], dump())
            return dict(payload)
    if hasattr(config, "dict"):
        dump = getattr(config, "dict")
        if callable(dump):
            payload = cast(Mapping[str, Any], dump())
            return dict(payload)
    msg = "Unsupported strategy config type"
    raise TypeError(msg)


def run_walkforward_pipeline(
    index: Sequence[datetime],
    splits: Sequence[tuple[list[int], list[int]]],
    base_config: BacktestConfig,
    param_sets: Sequence[dict[str, Any]],
    engine_factory: Callable[[BacktestConfig], Any],
    objective: Callable[[Sequence[float], float, float], float] = objective_sharpe_penalized,
) -> dict[str, Any]:
    """Evaluate ``param_sets`` across ``splits`` and return scores."""

    if not param_sets:
        msg = "param_sets must contain at least one configuration"
        raise ValueError(msg)

    base_strategy_kwargs = _strategy_to_kwargs(base_config.strategy_config)

    results: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_params: dict[str, Any] | None = None

    for params in param_sets:
        strategy_kwargs = dict(base_strategy_kwargs)
        strategy_kwargs.update(params)
        fold_scores: list[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if not test_idx:
                continue
            start_pos = min(train_idx[0], test_idx[0]) if train_idx else test_idx[0]
            end_pos = test_idx[-1]
            start_ts = index[start_pos]
            end_ts = index[end_pos]
            strategy_config = StrategyConfig(**strategy_kwargs)
            config = replace(
                base_config,
                start=start_ts,
                end=end_ts,
                seed=base_config.seed + fold_idx,
                strategy_config=strategy_config,
            )
            engine = engine_factory(config)
            result = engine.run()
            equity = [point.equity for point in result.equity_curve]
            turnover = sum(abs(fill.qty * fill.price) for fill in result.fills)
            max_dd = max((point.drawdown for point in result.equity_curve), default=0.0)
            score = objective(equity, turnover, max_dd)
            fold_scores.append(score)
        aggregate = sum(fold_scores) / len(fold_scores) if fold_scores else float("-inf")
        results.append({"params": dict(params), "score": aggregate, "fold_scores": fold_scores})
        if aggregate > best_score:
            best_score = aggregate
            best_params = dict(params)

    payload = {"results": results, "best_params": best_params, "best_score": best_score}
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Parquet dataset containing candles",
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Symbol to evaluate",
    )
    parser.add_argument(
        "--from",
        dest="start",
        required=True,
        type=_parse_datetime,
        help="Inclusive start timestamp",
    )
    parser.add_argument(
        "--to",
        dest="end",
        required=True,
        type=_parse_datetime,
        help="Inclusive end timestamp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for random search",
    )
    parser.add_argument(
        "--split",
        choices=["expanding", "rolling"],
        default="expanding",
        help="Type of walk-forward split",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=6,
        help="Number of walk-forward folds",
    )
    parser.add_argument(
        "--embargo",
        type=_parse_embargo,
        default="0D",
        help="Embargo applied after each test window (e.g. 2D, 6H)",
    )
    parser.add_argument(
        "--search",
        choices=["grid", "random"],
        default="grid",
        help="Hyper-parameter search strategy",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="JSON encoded parameter grid",
    )
    parser.add_argument(
        "--objective",
        choices=["sharpe_penalized"],
        default="sharpe_penalized",
        help="Objective function used to score folds",
    )
    return parser


def _load_params(raw: str) -> ConfigSpace:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"Invalid params JSON: {exc}"
        raise SystemExit(msg) from exc
    if not isinstance(payload, dict):
        msg = "Params JSON must decode to a mapping"
        raise SystemExit(msg)
    normalised: dict[str, Sequence[Any]] = {}
    for key, value in payload.items():
        if isinstance(value, list | tuple):
            options = list(value)
        else:
            msg = f"Parameter {key!r} must map to a list of options"
            raise SystemExit(msg)
        if not options:
            msg = f"Parameter {key!r} requires at least one value"
            raise SystemExit(msg)
        normalised[key] = tuple(options)
    return ConfigSpace(normalised)


def _resolve_objective(name: str) -> Callable[[Sequence[float], float, float], float]:
    if name == "sharpe_penalized":
        return objective_sharpe_penalized
    msg = f"Unknown objective {name!r}"
    raise SystemExit(msg)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None and os.environ.get("WELTKLASSE_SUPPRESS_SCRIPT_WARNING") != "1":
        print(
            "[DEPRECATED] Use `python -m cli walkforward` instead of scripts/run_walkforward.py",
            file=sys.stderr,
        )
    parser = _build_parser()
    args = parser.parse_args(argv)

    params_space = _load_params(args.params)
    loader = ParquetDataLoader(args.data.expanduser().resolve())
    index = _load_index(loader, args.symbol, args.start, args.end)
    if len(index) < args.folds:
        msg = "Dataset shorter than requested number of folds"
        raise SystemExit(msg)

    splits = time_kfold(index, args.folds, args.split, args.embargo)

    grid_params = grid(params_space)
    if args.search == "grid":
        param_sets = grid_params
    else:
        if default_rng is None:
            msg = "numpy is required for random search"
            raise SystemExit(msg)
        sample_count = min(len(grid_params), max(1, args.folds))
        param_sets = random(params_space, sample_count, default_rng(args.seed))

    base_config = BacktestConfig(
        data_path=args.data,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        seed=args.seed,
    )

    def engine_factory(config: BacktestConfig) -> BacktestEngine:
        return BacktestEngine(config)
    objective = _resolve_objective(args.objective)
    payload = run_walkforward_pipeline(
        index,
        splits,
        base_config,
        param_sets,
        engine_factory,
        objective,
    )

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

