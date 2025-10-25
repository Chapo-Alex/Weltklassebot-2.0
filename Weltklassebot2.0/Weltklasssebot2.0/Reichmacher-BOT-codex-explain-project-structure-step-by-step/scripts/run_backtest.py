"""Run a reproducible breakout-bias backtest and emit deterministic artefacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from core.engine import BacktestConfig, BacktestEngine
from strategy.breakout_bias import StrategyConfig


def _parse_datetime(value: str) -> datetime:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - argparse already validates
        msg = f"Invalid datetime value: {value}"  # pragma: no cover
        raise argparse.ArgumentTypeError(msg) from exc
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_fees(values: Sequence[str]) -> tuple[float, float]:
    maker = 0.0
    taker = 0.0
    for token in values:
        key, sep, raw = token.partition("=")
        if sep != "=":
            msg = f"Invalid fee token {token!r}; expected key=value"
            raise argparse.ArgumentTypeError(msg)
        try:
            value = float(raw)
        except ValueError as exc:  # pragma: no cover - argparse ensures float parsing
            msg = f"Invalid fee value for {key}: {raw!r}"
            raise argparse.ArgumentTypeError(msg) from exc
        key_lower = key.lower()
        if key_lower == "maker":
            maker = value
        elif key_lower == "taker":
            taker = value
        else:
            msg = f"Unknown fee key {key!r}; expected 'maker' or 'taker'"
            raise argparse.ArgumentTypeError(msg)
    return maker, taker


def _current_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):  # pragma: no cover - git optional
        return "unknown"
    return result.stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Parquet dataset with candles."
            " Provide a folder or single file."
        ),
    )
    parser.add_argument(
        "--symbol",
        default="BTCUSDT",
        help="Symbol to backtest",
    )
    parser.add_argument(
        "--from",
        dest="start",
        required=True,
        type=_parse_datetime,
        help="Start timestamp (UTC, e.g. 2024-01-01T00:00:00).",
    )
    parser.add_argument(
        "--to",
        dest="end",
        required=True,
        type=_parse_datetime,
        help="End timestamp (UTC).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed controlling jitter",
    )
    parser.add_argument(
        "--fees",
        nargs="*",
        default=("taker=0.0004", "maker=0.0002"),
        metavar="KEY=VALUE",
        help="Maker/taker fee specification (e.g. taker=0.0004 maker=0.0002)",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=50.0,
        help="Base execution latency in milliseconds",
    )
    parser.add_argument(
        "--impact-coef",
        type=float,
        default=1e-6,
        help="Linear price impact coefficient applied to notional size",
    )
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="Initial portfolio cash balance",
    )
    parser.add_argument(
        "--session",
        default="backtest",
        help="Risk session identifier forwarded to the state machine",
    )
    parser.add_argument(
        "--risk-store-dir",
        type=Path,
        help="Directory for persisting risk state snapshots and audit logs",
    )
    parser.add_argument(
        "--risk-store-rotate-lines",
        type=int,
        default=100_000,
        help="Rotate risk audit JSONL after this many lines (set <=0 to disable)",
    )
    parser.add_argument(
        "--risk-store-rotate-mb",
        type=int,
        default=64,
        help="Rotate risk audit JSONL after this many megabytes (set <=0 to disable)",
    )
    parser.add_argument(
        "--risk-store-fsync",
        action="store_true",
        help="Force fsync after each audit append for durability",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory to write trades.csv, equity.parquet and metrics.json",
    )
    return parser


def _default_strategy_config() -> StrategyConfig:
    return StrategyConfig(
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


def main() -> None:
    if os.environ.get("WELTKLASSE_SUPPRESS_SCRIPT_WARNING") != "1":
        print(
            "[DEPRECATED] Use `python -m cli backtest` instead of scripts/run_backtest.py",
            file=sys.stderr,
        )
    parser = _build_parser()
    args = parser.parse_args()
    maker_fee, taker_fee = _parse_fees(args.fees)

    rotate_lines = args.risk_store_rotate_lines
    if rotate_lines is not None and rotate_lines <= 0:
        rotate_lines = None
    rotate_mb = args.risk_store_rotate_mb
    if rotate_mb is not None and rotate_mb <= 0:
        rotate_mb = None

    config = BacktestConfig(
        data_path=args.data,
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        seed=args.seed,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        latency_ms=args.latency_ms,
        impact_coefficient=args.impact_coef,
        initial_cash=args.initial_cash,
        session_name=args.session,
        strategy_config=_default_strategy_config(),
        risk_store_dir=args.risk_store_dir,
        risk_store_rotate_lines=rotate_lines,
        risk_store_rotate_mb=rotate_mb,
        risk_store_fsync=args.risk_store_fsync,
    )

    engine = BacktestEngine(config)
    result = engine.run()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_path = output_dir / "trades.csv"
    equity_path = output_dir / "equity.parquet"
    metrics_path = output_dir / "metrics.json"

    trades_path.write_text(result.trades_csv(), encoding="utf-8")
    equity_path.write_bytes(result.equity_payload())

    project_root = Path(__file__).resolve().parents[1]
    commit_hash = _current_commit(project_root)
    metrics = result.metrics()
    metrics.update(
        {
            "seed": config.seed,
            "commit": commit_hash,
            "config": {
                "data": str(config.data_path),
                "symbol": config.symbol,
                "start": config.start.isoformat(),
                "end": config.end.isoformat(),
                "maker_fee": config.maker_fee,
                "taker_fee": config.taker_fee,
                "latency_ms": config.latency_ms,
                "impact_coefficient": config.impact_coefficient,
                "initial_cash": config.initial_cash,
                "session": config.session_name,
            },
        }
    )

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = (
        f"Processed {metrics['candles']} candles, "
        f"executed {metrics['fills']} fills; "
        f"final equity {metrics['final_equity']:.2f}"
    )
    print(summary)


if __name__ == "__main__":
    main()
