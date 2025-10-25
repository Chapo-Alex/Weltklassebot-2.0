"""Ensure replay mode reproduces backtest artefacts bit-for-bit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from core.engine import BacktestEngine
from core.io.replay import load_replay_logs
from tests.determinism.test_equity_hash import (
    _build_config,
    _create_dataset,
    _hash_file,
    _write_equity_csv,
)


def _write_orders_csv(path: Path, records: list[dict[str, object]]) -> None:
    header = [
        "order_id",
        "timestamp",
        "symbol",
        "side",
        "qty",
        "type",
        "price",
        "stop",
        "tif",
        "reduce_only",
        "post_only",
        "client_tag",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_fills_csv(path: Path, records: list[dict[str, object]]) -> None:
    header = [
        "order_id",
        "timestamp",
        "symbol",
        "side",
        "qty",
        "price",
        "fee",
        "liquidity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _write_events_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _fills_digest(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = json.dumps(record, sort_keys=True).encode("utf-8")
        digest.update(payload)
    return digest.hexdigest()


def test_replay_parity(tmp_path: Path) -> None:
    dataset, start, end = _create_dataset(tmp_path)
    seed = 777
    config = _build_config(dataset, start, end, seed)

    backtest_result = BacktestEngine(config).run()

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    orders_path = logs_dir / "orders.csv"
    fills_path = logs_dir / "fills.csv"
    equity_path = logs_dir / "equity.csv"
    events_path = logs_dir / "events.jsonl"

    orders_records = backtest_result.order_records()
    fills_records = backtest_result.trades_records()
    events_records = backtest_result.event_records()

    _write_orders_csv(orders_path, orders_records)
    _write_fills_csv(fills_path, fills_records)
    _write_equity_csv(equity_path, backtest_result.equity_records())
    _write_events_jsonl(events_path, events_records)

    replay_logs = load_replay_logs(logs_dir)
    assert replay_logs.orders == backtest_result.orders
    assert replay_logs.fills == backtest_result.fills
    assert replay_logs.equity == backtest_result.equity_curve
    replay_engine = BacktestEngine(
        config,
        mode="replay",
        replay_logs=replay_logs,
    )
    replay_result = replay_engine.run()

    assert replay_result.trades_records() == fills_records
    assert _fills_digest(replay_result.trades_records()) == _fills_digest(fills_records)

    replay_equity_path = tmp_path / "replay_equity.csv"
    _write_equity_csv(replay_equity_path, replay_result.equity_records())

    assert _hash_file(equity_path) == _hash_file(replay_equity_path)
