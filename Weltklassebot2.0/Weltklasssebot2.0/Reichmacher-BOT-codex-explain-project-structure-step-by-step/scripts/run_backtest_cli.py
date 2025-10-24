"""Deterministic in-memory backtest harness with SHA256 reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from core.engine import BacktestConfig, BacktestEngine  # type: ignore[import-untyped]
from core.events import CandleEvent  # type: ignore[import-untyped]
from strategy.breakout_bias import StrategyConfig  # type: ignore[import-untyped]


def _parse_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _synthetic_candles(
    symbol: str, start: datetime, count: int, seed: int
) -> list[CandleEvent]:
    """Generate a deterministic candle series without relying on NumPy."""

    if count <= 0:
        return []

    rng = seed & 0x7FFFFFFF

    def next_u() -> float:
        nonlocal rng
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        return rng / 0x7FFFFFFF

    candles: list[CandleEvent] = []
    price = 100.0
    current_start = start
    for _ in range(count):
        direction = 1.0 if next_u() > 0.5 else -1.0
        drift = 0.02 * direction
        noise = (next_u() - 0.5) * 0.2
        close = max(0.01, price * (1.0 + drift + noise))
        high = max(price, close) * 1.001
        low = min(price, close) * 0.999
        volume = 1_000.0 + int(next_u() * 500)
        end = current_start + timedelta(minutes=1)
        candles.append(
            CandleEvent(
                symbol=symbol,
                open=price,
                high=high,
                low=low,
                close=close,
                volume=volume,
                start=current_start,
                end=end,
            )
        )
        price = close
        current_start = end
    return candles


class _SyntheticLoader:
    """Simple loader that returns pre-generated candles."""

    __slots__ = ("_candles",)

    def __init__(self, candles: Sequence[CandleEvent]) -> None:
        self._candles = list(candles)

    def load(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CandleEvent]:  # pragma: no cover - exercised via integration
        scoped: list[CandleEvent] = []
        for candle in self._candles:
            if candle.symbol != symbol:
                continue
            if candle.end < start or candle.start > end:
                continue
            scoped.append(candle)
        return scoped


def _equity_at(ts: datetime, equity_curve: Sequence[Any], fallback: float) -> float:
    for point in equity_curve:
        point_ts = getattr(point, "ts", None)
        if point_ts is None:
            continue
        if point_ts >= ts:
            value = getattr(point, "equity", fallback)
            return float(value)
    if equity_curve:
        last = equity_curve[-1]
        return float(getattr(last, "equity", fallback))
    return fallback


def _csv_of_fills_and_equity(result: Any) -> str:
    rows = ["ts,side,qty,price,equity"]
    equity_curve = list(result.equity_curve)
    equity_curve.sort(key=lambda point: point.ts)
    for fill in sorted(result.fills, key=lambda item: item.ts):
        equity = _equity_at(fill.ts, equity_curve, result.config.initial_cash)
        rows.append(
            ",".join(
                [
                    fill.ts.astimezone(UTC).isoformat(),
                    fill.side.value,
                    f"{fill.qty:.8f}",
                    f"{fill.price:.4f}",
                    f"{equity:.4f}",
                ]
            )
        )
    return "\n".join(rows)


def _digest(data: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()


def _strategy_config(params: dict[str, object]) -> StrategyConfig:
    base = StrategyConfig(
        lookback=8,
        bias_lookback=8,
        threshold_lookback=8,
        atr_lookback=8,
        chandelier_lookback=8,
        order_size=0.3,
        breakout_threshold=0.0008,
        pyramid_steps=(1.0,),
        max_pyramids=1,
        atr_trailing_multiplier=1.1,
        bias_vol_ratio=0.2,
        bias_min_slope=0.0,
    )
    if not params:
        return base
    updates = base.model_dump()
    updates.update(params)
    return StrategyConfig(**updates)


def _build_engine(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    seed: int,
    candles: Sequence[CandleEvent],
    strategy_params: dict[str, object],
) -> BacktestEngine:
    config = BacktestConfig(
        data_path=Path("synthetic.parquet"),
        symbol=symbol,
        start=start,
        end=end,
        seed=seed,
        strategy_config=_strategy_config(strategy_params),
    )
    loader = _SyntheticLoader(candles)
    return BacktestEngine(config, data_loader=loader)


def _run_backtest(
    symbol: str,
    start: datetime,
    end: datetime,
    seed: int,
    candles: Sequence[CandleEvent],
    params: dict[str, object],
) -> Any:
    engine = _build_engine(
        symbol=symbol,
        start=start,
        end=end,
        seed=seed,
        candles=candles,
        strategy_params=params,
    )
    return engine.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an in-memory backtest and emit a deterministic digest.",
    )
    parser.add_argument("--symbol", default="SYNUSD")
    parser.add_argument("--from", dest="start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--to", dest="end", default="2024-01-01T01:00:00Z")
    parser.add_argument("--strategy", default="breakout")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--candles", type=int, default=60)
    parser.add_argument("--params", default="{}", help="JSON encoded strategy overrides")
    args = parser.parse_args(list(argv) if argv is not None else None)

    start = _parse_datetime(args.start)
    end = _parse_datetime(args.end)
    params = json.loads(args.params or "{}")
    if args.strategy != "breakout":
        msg = "Only the breakout strategy is supported"
        raise ValueError(msg)

    candles = _synthetic_candles(args.symbol, start, int(args.candles), int(args.seed))
    result = _run_backtest(
        symbol=args.symbol,
        start=start,
        end=end,
        seed=int(args.seed),
        candles=candles,
        params=params,
    )

    csv = _csv_of_fills_and_equity(result)
    payload = {"sha256": _digest(csv), "lines": csv.count("\n") + 1}
    print(json.dumps(payload, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
