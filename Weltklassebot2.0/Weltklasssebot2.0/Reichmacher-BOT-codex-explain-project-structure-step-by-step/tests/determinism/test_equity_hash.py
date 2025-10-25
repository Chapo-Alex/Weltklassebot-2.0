"""Ensure deterministic equity outputs for identical seeds and inputs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.config import BacktestConfig
from core.engine import BacktestEngine
from strategy.breakout_bias import StrategyConfig


def _write_equity_csv(path: Path, records: list[dict[str, object]]) -> None:
    header = ["timestamp", "equity", "cash", "notional", "drawdown"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _create_dataset(path: Path) -> tuple[Path, datetime, datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    price = 50_000.0
    last_end = start
    for idx in range(96):
        open_price = price + idx * 12.0
        close_price = open_price + 4.5 + (idx % 3)
        high_price = close_price + 6.0
        low_price = open_price - 6.0
        volume = 1_000.0 + idx * 5.0
        start_ts = start + timedelta(minutes=idx)
        end_ts = start_ts + timedelta(minutes=1)
        last_end = end_ts
        rows.append(
            {
                "symbol": "BTCUSDT",
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
                "start": start_ts.astimezone(UTC).isoformat(),
                "end": end_ts.astimezone(UTC).isoformat(),
            }
        )
    dataset = path / "candles.json"
    dataset.write_text(json.dumps({"records": rows}, sort_keys=True), encoding="utf-8")
    return dataset, start, last_end


def _build_config(data_path: Path, start: datetime, end: datetime, seed: int) -> BacktestConfig:
    strategy = StrategyConfig(
        lookback=8,
        bias_lookback=8,
        threshold_lookback=8,
        atr_lookback=8,
        chandelier_lookback=8,
        order_size=0.2,
        breakout_threshold=0.001,
        pyramid_steps=(1.0,),
        max_pyramids=1,
        atr_trailing_multiplier=1.05,
        bias_vol_ratio=0.25,
        bias_min_slope=0.0,
    )
    return BacktestConfig(
        data_path=data_path,
        symbol="BTCUSDT",
        start=start,
        end=end,
        seed=seed,
        maker_fee=0.0002,
        taker_fee=0.0004,
        latency_ms=25.0,
        impact_coefficient=5e-7,
        initial_cash=250_000.0,
        strategy_config=strategy,
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_equity_hash_deterministic(tmp_path: Path) -> None:
    dataset, start, end = _create_dataset(tmp_path)
    seed = 1234

    config_one = _build_config(dataset, start, end, seed)
    config_two = _build_config(dataset, start, end, seed)

    result_one = BacktestEngine(config_one).run()
    result_two = BacktestEngine(config_two).run()

    equity_one = tmp_path / "equity_run_one.csv"
    equity_two = tmp_path / "equity_run_two.csv"
    _write_equity_csv(equity_one, result_one.equity_records())
    _write_equity_csv(equity_two, result_two.equity_records())

    assert _hash_file(equity_one) == _hash_file(equity_two)
