from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from core.engine import BacktestConfig, BacktestEngine, BacktestResult, EquityPoint
from core.events import FillEvent, LiquidityFlag, OrderSide
from portfolio.risk import RiskParameters
from strategy.breakout_bias import StrategyConfig


def _create_dataset(tmp_path: Path) -> tuple[Path, datetime, datetime]:
    start = datetime(2023, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    base_price = 20_000.0
    for idx in range(100):
        open_price = base_price + idx * 35.0
        close_price = open_price + 22.0 + (idx % 5)
        high_price = close_price + 18.0
        low_price = open_price - 18.0
        volume = 5_000.0 + idx * 10.0
        start_ts = start + timedelta(hours=idx)
        end_ts = start_ts + timedelta(hours=1)
        rows.append(
            {
                "symbol": "BTCUSDT",
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": round(volume, 2),
                "start": start_ts,
                "end": end_ts,
            }
        )
    btc_end = rows[-1]["end"]
    rows.append(
        {
            "symbol": "ETHUSDT",
            "open": 1_500.0,
            "high": 1_520.0,
            "low": 1_480.0,
            "close": 1_510.0,
            "volume": 10_000.0,
            "start": start,
            "end": start + timedelta(hours=1),
        }
    )
    dataset_path = tmp_path / "candles.parquet"
    payload = {
        "schema": "candles",
        "records": [
            {
                **{key: value for key, value in row.items() if key not in {"start", "end"}},
                "start": row["start"].astimezone(UTC).isoformat(),
                "end": row["end"].astimezone(UTC).isoformat(),
            }
            for row in rows
        ],
    }
    dataset_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    assert isinstance(btc_end, datetime)
    return dataset_path, start, btc_end


def _engine_config(dataset: Path, start: datetime, end: datetime) -> BacktestConfig:
    strategy_config = StrategyConfig(
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
    return BacktestConfig(
        data_path=dataset,
        symbol="BTCUSDT",
        start=start,
        end=end,
        seed=1337,
        maker_fee=0.0002,
        taker_fee=0.0004,
        latency_ms=50.0,
        impact_coefficient=1e-6,
        initial_cash=100_000.0,
        session_name="test",
        strategy_config=strategy_config,
    )


EXPECTED_TRADES_HASH = "d8ac30952de2396f452ca0fc204407f42d5463b1573fb4646d29ac2289d9412f"
EXPECTED_EQUITY_HASH = "8df501bcfbc216e956343185d92d5b682858e30cd2c39a27987eca033165348d"


def test_backtest_config_normalises_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.parquet"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    start = datetime(2023, 1, 1, 12, 0, 0)
    end = datetime(2023, 1, 1, 13, 0, 0)
    config = BacktestConfig(
        data_path=dataset,
        symbol="BTCUSDT",
        start=start,
        end=end,
    )
    assert config.start.tzinfo is UTC
    assert config.end.tzinfo is UTC
    assert config.data_path.is_absolute()

    with pytest.raises(ValueError):
        BacktestConfig(
            data_path=dataset,
            symbol="BTCUSDT",
            start=end,
            end=start - timedelta(hours=1),
        )

    aware_start = datetime(2023, 1, 2, 12, 0, tzinfo=UTC)
    aware_end = aware_start + timedelta(hours=1)
    aware_config = BacktestConfig(
        data_path=dataset,
        symbol="ETHUSDT",
        start=aware_start,
        end=aware_end,
    )
    assert aware_config.start == aware_start
    assert aware_config.end == aware_end


def test_backtest_result_serialisation(tmp_path: Path) -> None:
    dataset = tmp_path / "dummy.parquet"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    config = BacktestConfig(
        data_path=dataset,
        symbol="BTCUSDT",
        start=datetime(2023, 1, 1, tzinfo=UTC),
        end=datetime(2023, 1, 2, tzinfo=UTC),
    )
    fill = FillEvent(
        order_id="fill-1",
        ts=datetime(2023, 1, 1, 10, tzinfo=UTC),
        qty=0.5,
        price=25_000.0,
        fee=5.0,
        liquidity_flag=LiquidityFlag.TAKER,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
    )
    equity_point = EquityPoint(
        ts=datetime(2023, 1, 1, 10, 5, tzinfo=UTC),
        equity=100_500.0,
        cash=49_750.0,
        notional=12_500.0,
        drawdown=50.0,
    )
    result = BacktestResult(
        fills=[fill],
        equity_curve=[equity_point],
        candles_processed=1,
        config=config,
    )

    trades_records = result.trades_records()
    assert trades_records[0]["timestamp"].endswith("+00:00")
    assert "fill-1" in result.trades_csv()

    equity_payload = json.loads(result.equity_payload().decode("utf-8"))
    assert equity_payload["schema"] == "equity_timeseries"
    assert len(equity_payload["records"]) == 1

    metrics = result.metrics()
    assert metrics["fills"] == 1
    assert metrics["candles"] == 1


def test_backtest_engine_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "empty.parquet"
    dataset.write_text(json.dumps({"records": []}), encoding="utf-8")
    start = datetime(2023, 1, 1, tzinfo=UTC)
    config = _engine_config(dataset, start, start)

    engine = BacktestEngine(config)
    result = engine.run()

    assert result.fills == []
    assert result.equity_curve == []
    metrics = result.metrics()
    assert metrics["final_equity"] == config.initial_cash
    assert metrics["candles"] == 0
    trades_csv = result.trades_csv()
    assert trades_csv.strip() == "order_id,timestamp,symbol,side,qty,price,fee,liquidity"
    payload = json.loads(result.equity_payload().decode("utf-8"))
    assert payload["records"] == []


def test_backtest_engine_risk_denial(tmp_path: Path) -> None:
    dataset, start, end = _create_dataset(tmp_path)
    base_config = _engine_config(dataset, start, end)
    denial_config = replace(
        base_config,
        risk=RiskParameters(
            max_drawdown=10.0,
            max_notional=1.0,
            max_trades_per_day=10,
            cooldown_minutes=0.0,
        ),
    )

    engine = BacktestEngine(denial_config)
    result = engine.run()

    assert result.fills == []
    metrics = result.metrics()
    assert metrics["fills"] == 0
    assert metrics["candles"] == len(result.equity_records())


def test_backtest_engine_golden_master(tmp_path: Path) -> None:
    dataset, start, end = _create_dataset(tmp_path)
    config = _engine_config(dataset, start, end)

    engine = BacktestEngine(config)
    result = engine.run()

    repeat_result = BacktestEngine(config).run()
    assert result.trades_records() == repeat_result.trades_records()
    assert result.equity_records() == repeat_result.equity_records()

    trades_data = result.trades_csv().encode("utf-8")
    equity_data = result.equity_payload()

    trades_hash = hashlib.sha256(trades_data).hexdigest()
    equity_hash = hashlib.sha256(equity_data).hexdigest()

    assert trades_hash == EXPECTED_TRADES_HASH
    assert equity_hash == EXPECTED_EQUITY_HASH


def test_run_backtest_cli_outputs(tmp_path: Path) -> None:
    dataset, start, end = _create_dataset(tmp_path)
    config = _engine_config(dataset, start, end)

    script = Path(__file__).resolve().parents[2] / "scripts" / "run_backtest.py"
    output_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        str(script),
        "--data",
        str(dataset),
        "--symbol",
        config.symbol,
        "--from",
        config.start.isoformat(),
        "--to",
        config.end.isoformat(),
        "--seed",
        str(config.seed),
        "--fees",
        f"taker={config.taker_fee}",
        f"maker={config.maker_fee}",
        "--latency-ms",
        str(config.latency_ms),
        "--impact-coef",
        str(config.impact_coefficient),
        "--initial-cash",
        str(config.initial_cash),
        "--session",
        config.session_name,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    trades_path = output_dir / "trades.csv"
    equity_path = output_dir / "equity.parquet"
    metrics_path = output_dir / "metrics.json"

    assert trades_path.exists()
    assert equity_path.exists()
    assert metrics_path.exists()

    trades_hash = hashlib.sha256(trades_path.read_bytes()).hexdigest()
    equity_hash = hashlib.sha256(equity_path.read_bytes()).hexdigest()

    assert trades_hash == EXPECTED_TRADES_HASH
    assert equity_hash == EXPECTED_EQUITY_HASH

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["seed"] == config.seed
    assert metrics["config"]["symbol"] == config.symbol
    assert metrics["config"]["start"] == config.start.isoformat()
    assert metrics["config"]["latency_ms"] == config.latency_ms
    assert metrics["candles"] == 100
    with trades_path.open(encoding="utf-8") as handle:
        fills_count = sum(1 for _ in csv.DictReader(handle))
    assert metrics["fills"] == fills_count
