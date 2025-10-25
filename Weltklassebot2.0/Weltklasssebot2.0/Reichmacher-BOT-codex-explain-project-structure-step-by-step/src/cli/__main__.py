"""Unified Typer-based CLI for deterministic Weltklassebot workflows."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import typer

try:  # pragma: no cover - python >= 3.11 ships tomllib
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - runtime guard for <=3.10
    tomllib = None  # type: ignore[assignment]


@dataclass
class _GlobalState:
    seed: int
    data_snapshot: Path | None
    prom_port: int
    config_path: Path | None
    config_payload: dict[str, Any]


app = typer.Typer(help="Deterministic research CLI for Weltklassebot", add_completion=False)


def _load_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix in {".json", ".jsn"}:
        payload = json.loads(text)
    elif suffix in {".toml", ".tml"}:
        if tomllib is None:  # pragma: no cover - python < 3.11
            msg = "TOML configuration requires Python 3.11 or tomllib"
            raise typer.BadParameter(msg)
        payload = tomllib.loads(text)
    else:
        msg = f"Unsupported config format {path.suffix!r}; use JSON or TOML"
        raise typer.BadParameter(msg)
    if not isinstance(payload, dict):
        msg = "Configuration root must be a mapping"
        raise typer.BadParameter(msg)
    return payload


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    msg = f"Unsupported datetime payload: {value!r}"
    raise typer.BadParameter(msg)


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return Path(stripped)
    msg = f"Unsupported path payload: {value!r}"
    raise typer.BadParameter(msg)


def _command_defaults(payload: dict[str, Any], section: str) -> dict[str, Any]:
    block = payload.get(section, {})
    if not block and section in payload.get("commands", {}):
        block = payload["commands"][section]
    if block is None:
        return {}
    if not isinstance(block, dict):
        msg = f"Configuration section '{section}' must be a mapping"
        raise typer.BadParameter(msg)
    return dict(block)


def _ensure_args(mapping: dict[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if mapping.get(key) is None]
    if missing:
        joined = ", ".join(missing)
        msg = f"Missing required configuration values: {joined}"
        raise typer.BadParameter(msg)


def _format_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _apply_environment(seed: int, data_snapshot: Path | None, prom_port: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if data_snapshot is not None:
        os.environ["WELTKLASSE_DATA_SNAPSHOT"] = str(data_snapshot)
    else:
        os.environ.pop("WELTKLASSE_DATA_SNAPSHOT", None)
    os.environ["WELTKLASSE_METRICS_PORT"] = str(prom_port)

    try:
        import numpy.random as _np_random  # pragma: no cover - optional use

        _np_random.seed(seed)
    except Exception:  # pragma: no cover - defensive
        pass

    try:
        import random as _random

        _random.seed(seed)
    except Exception:  # pragma: no cover - defensive
        pass


def _run_backtest(argv: Sequence[str]) -> int:
    from scripts import run_backtest as legacy

    previous = os.environ.get("WELTKLASSE_SUPPRESS_SCRIPT_WARNING")
    os.environ["WELTKLASSE_SUPPRESS_SCRIPT_WARNING"] = "1"
    original_argv = sys.argv
    sys.argv = [original_argv[0], *map(str, argv)]
    try:
        legacy.main()
    finally:
        sys.argv = original_argv
        if previous is None:
            os.environ.pop("WELTKLASSE_SUPPRESS_SCRIPT_WARNING", None)
        else:
            os.environ["WELTKLASSE_SUPPRESS_SCRIPT_WARNING"] = previous
    return 0


def _run_synthetic(argv: Sequence[str]) -> int:
    from scripts import run_backtest_cli as synthetic

    return synthetic.main(list(argv))


def _run_walkforward(argv: Sequence[str]) -> int:
    from scripts import run_walkforward as legacy

    previous = os.environ.get("WELTKLASSE_SUPPRESS_SCRIPT_WARNING")
    os.environ["WELTKLASSE_SUPPRESS_SCRIPT_WARNING"] = "1"
    try:
        return legacy.main(list(argv))
    finally:
        if previous is None:
            os.environ.pop("WELTKLASSE_SUPPRESS_SCRIPT_WARNING", None)
        else:
            os.environ["WELTKLASSE_SUPPRESS_SCRIPT_WARNING"] = previous


def _extend_args(args: list[str], defaults: dict[str, Any], mapping: dict[str, str]) -> None:
    for key, flag in mapping.items():
        value = defaults.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        args.extend([flag, str(value)])


def _normalise_params(raw: Any) -> str:
    if raw is None:
        msg = "Walk-forward params must be provided via option or config"
        raise typer.BadParameter(msg)
    if isinstance(raw, (dict, list, tuple)):
        return json.dumps(raw)
    path = _coerce_path(raw)
    if path and path.exists():
        return path.read_text(encoding="utf-8")
    if isinstance(raw, str):
        cleaned = raw.strip()
        if not cleaned:
            msg = "Parameter specification must not be empty"
            raise typer.BadParameter(msg)
        return cleaned
    msg = f"Unsupported parameter specification: {raw!r}"
    raise typer.BadParameter(msg)


@app.callback()
def _main(  # type: ignore[override]
    ctx: typer.Context,
    seed: int = typer.Option(42, help="Global deterministic seed"),
    data_snapshot: Path | None = typer.Option(
        None,
        help="Path to a deterministic market data snapshot",
    ),
    prom_port: int = typer.Option(9000, help="Prometheus exporter port"),
    config: Path | None = typer.Option(
        None,
        help="Optional JSON/TOML defaults shared across sub-commands",
    ),
) -> None:
    payload = _load_payload(config) if config is not None else {}
    ctx.obj = _GlobalState(
        seed=seed,
        data_snapshot=data_snapshot,
        prom_port=prom_port,
        config_path=config,
        config_payload=payload,
    )


@app.command()
def backtest(
    ctx: typer.Context,
    *,
    seed: int | None = typer.Option(
        None, help="Override the deterministic seed for this run"
    ),
    data: Path | None = typer.Option(None, help="Path to Parquet candle data"),
    symbol: str | None = typer.Option(None, help="Trading symbol to replay"),
    start: datetime | None = typer.Option(None, help="Inclusive UTC start timestamp"),
    end: datetime | None = typer.Option(None, help="Inclusive UTC end timestamp"),
    output_dir: Path | None = typer.Option(
        None,
        help="Directory for artefacts (trades.csv, equity.parquet, metrics.json)",
    ),
    fees: list[str] | None = typer.Option(
        None,
        help="Maker/taker fee specification tokens (e.g. taker=0.0004)",
    ),
    latency_ms: float | None = typer.Option(None, help="Base execution latency"),
    impact_coef: float | None = typer.Option(
        None, help="Linear price impact coefficient"
    ),
    initial_cash: float | None = typer.Option(
        None, help="Initial cash balance for the portfolio"
    ),
    session: str | None = typer.Option(
        None, help="Risk session identifier propagated to the engine"
    ),
    risk_store_dir: Path | None = typer.Option(
        None, help="Directory for persisting risk state snapshots"
    ),
    risk_store_rotate_lines: int | None = typer.Option(
        None, help="Rotate risk JSONL after this many lines"
    ),
    risk_store_rotate_mb: int | None = typer.Option(
        None, help="Rotate risk JSONL after this many megabytes"
    ),
    risk_store_fsync: bool = typer.Option(
        False, help="Force fsync after each risk store append"
    ),
    config: Path | None = typer.Option(
        None, help="Alternate JSON/TOML defaults for this invocation"
    ),
    data_snapshot: Path | None = typer.Option(
        None, help="Override the global data snapshot for this run"
    ),
    prom_port: int | None = typer.Option(
        None, help="Override the exporter port for this run"
    ),
) -> None:
    state = ctx.ensure_object(_GlobalState)
    payload = state.config_payload
    if config is not None and config != state.config_path:
        payload = _load_payload(config)
    defaults = _command_defaults(payload, "backtest")

    effective_seed = seed if seed is not None else state.seed
    effective_snapshot = data_snapshot or state.data_snapshot
    effective_prom = prom_port if prom_port is not None else state.prom_port
    _apply_environment(effective_seed, effective_snapshot, effective_prom)

    dataset = data or _coerce_path(defaults.get("data")) or effective_snapshot
    start_dt = start or _coerce_datetime(defaults.get("start"))
    end_dt = end or _coerce_datetime(defaults.get("end"))
    symbol_value = symbol or defaults.get("symbol", "BTCUSDT")

    args: list[str] = []
    if dataset and start_dt and end_dt:
        args.extend(
            [
                "--data",
                str(Path(dataset)),
                "--symbol",
                str(symbol_value),
                "--from",
                _format_datetime(start_dt),
                "--to",
                _format_datetime(end_dt),
                "--seed",
                str(effective_seed),
            ]
        )
        output = output_dir or _coerce_path(defaults.get("output_dir"))
        if output is not None:
            args.extend(["--output-dir", str(output)])
        combined_fees = list(fees or [])
        default_fees = defaults.get("fees")
        if not combined_fees and isinstance(default_fees, (list, tuple)):
            combined_fees = [str(token) for token in default_fees]
        if combined_fees:
            args.append("--fees")
            args.extend(str(token) for token in combined_fees)

        option_mapping = {
            "latency_ms": latency_ms if latency_ms is not None else defaults.get("latency_ms"),
            "impact_coefficient": impact_coef if impact_coef is not None else defaults.get("impact_coefficient"),
            "initial_cash": initial_cash if initial_cash is not None else defaults.get("initial_cash"),
            "session": session if session is not None else defaults.get("session"),
            "risk_store_dir": risk_store_dir or _coerce_path(defaults.get("risk_store_dir")),
            "risk_store_rotate_lines": risk_store_rotate_lines if risk_store_rotate_lines is not None else defaults.get("risk_store_rotate_lines"),
            "risk_store_rotate_mb": risk_store_rotate_mb if risk_store_rotate_mb is not None else defaults.get("risk_store_rotate_mb"),
        }
        flag_map = {
            "latency_ms": "--latency-ms",
            "impact_coefficient": "--impact-coef",
            "initial_cash": "--initial-cash",
            "session": "--session",
            "risk_store_dir": "--risk-store-dir",
            "risk_store_rotate_lines": "--risk-store-rotate-lines",
            "risk_store_rotate_mb": "--risk-store-rotate-mb",
        }
        merged_defaults = {
            key: value if not isinstance(value, Path) else Path(value)
            for key, value in option_mapping.items()
            if value is not None
        }
        _extend_args(args, merged_defaults, flag_map)
        if risk_store_fsync or bool(defaults.get("risk_store_fsync")):
            args.append("--risk-store-fsync")
        code = _run_backtest(args)
    else:
        synthetic_args = ["--seed", str(effective_seed)]
        code = _run_synthetic(synthetic_args)

    raise typer.Exit(code)


def _walkforward_common(
    *,
    ctx: typer.Context,
    seed: int | None,
    data: Path | None,
    symbol: str | None,
    start: datetime | None,
    end: datetime | None,
    split: str | None,
    folds: int | None,
    embargo: str | None,
    params: Any,
    search: str,
    objective: str | None,
    config: Path | None,
    data_snapshot: Path | None,
    prom_port: int | None,
    section: str,
) -> None:
    state = ctx.ensure_object(_GlobalState)
    payload = state.config_payload
    if config is not None and config != state.config_path:
        payload = _load_payload(config)
    defaults = _command_defaults(payload, section)

    effective_seed = seed if seed is not None else state.seed
    effective_snapshot = data_snapshot or state.data_snapshot
    effective_prom = prom_port if prom_port is not None else state.prom_port
    _apply_environment(effective_seed, effective_snapshot, effective_prom)

    dataset = data or _coerce_path(defaults.get("data")) or effective_snapshot
    start_dt = start or _coerce_datetime(defaults.get("start"))
    end_dt = end or _coerce_datetime(defaults.get("end"))
    params_raw = params if params is not None else defaults.get("params")

    _ensure_args({"data": dataset, "start": start_dt, "end": end_dt, "params": params_raw}, ["data", "start", "end", "params"])

    args: list[str] = [
        "--data",
        str(Path(dataset)),
        "--symbol",
        str(symbol or defaults.get("symbol", "BTCUSDT")),
        "--from",
        _format_datetime(start_dt),
        "--to",
        _format_datetime(end_dt),
        "--seed",
        str(effective_seed),
        "--search",
        search,
        "--params",
        _normalise_params(params_raw),
    ]

    split_value = split or defaults.get("split")
    if split_value:
        args.extend(["--split", str(split_value)])
    folds_value = folds if folds is not None else defaults.get("folds")
    if folds_value is not None:
        args.extend(["--folds", str(folds_value)])
    embargo_value = embargo or defaults.get("embargo")
    if embargo_value:
        args.extend(["--embargo", str(embargo_value)])
    objective_value = objective or defaults.get("objective")
    if objective_value:
        args.extend(["--objective", str(objective_value)])

    code = _run_walkforward(args)
    raise typer.Exit(code)


@app.command()
def walkforward(
    ctx: typer.Context,
    *,
    seed: int | None = typer.Option(None, help="Override the deterministic seed"),
    data: Path | None = typer.Option(None, help="Path to Parquet candle data"),
    symbol: str | None = typer.Option(None, help="Trading symbol to evaluate"),
    start: datetime | None = typer.Option(None, help="Inclusive UTC start timestamp"),
    end: datetime | None = typer.Option(None, help="Inclusive UTC end timestamp"),
    split: str | None = typer.Option(None, help="expanding or rolling splits"),
    folds: int | None = typer.Option(None, help="Number of walk-forward folds"),
    embargo: str | None = typer.Option(None, help="Embargo duration (e.g. 2D, 6H)"),
    params: Any = typer.Option(None, help="JSON or file with parameter grid"),
    objective: str | None = typer.Option(None, help="Objective scoring function"),
    config: Path | None = typer.Option(
        None, help="Alternate JSON/TOML defaults for this invocation"
    ),
    data_snapshot: Path | None = typer.Option(
        None, help="Override the global data snapshot for this run"
    ),
    prom_port: int | None = typer.Option(
        None, help="Override the exporter port for this run"
    ),
) -> None:
    _walkforward_common(
        ctx=ctx,
        seed=seed,
        data=data,
        symbol=symbol,
        start=start,
        end=end,
        split=split,
        folds=folds,
        embargo=embargo,
        params=params,
        search="grid",
        objective=objective,
        config=config,
        data_snapshot=data_snapshot,
        prom_port=prom_port,
        section="walkforward",
    )


@app.command()
def tune(
    ctx: typer.Context,
    *,
    seed: int | None = typer.Option(None, help="Override the deterministic seed"),
    data: Path | None = typer.Option(None, help="Path to Parquet candle data"),
    symbol: str | None = typer.Option(None, help="Trading symbol to evaluate"),
    start: datetime | None = typer.Option(None, help="Inclusive UTC start timestamp"),
    end: datetime | None = typer.Option(None, help="Inclusive UTC end timestamp"),
    folds: int | None = typer.Option(None, help="Number of folds for sampling"),
    embargo: str | None = typer.Option(None, help="Embargo duration (e.g. 2D, 6H)"),
    params: Any = typer.Option(None, help="JSON or file with parameter grid"),
    config: Path | None = typer.Option(
        None, help="Alternate JSON/TOML defaults for this invocation"
    ),
    data_snapshot: Path | None = typer.Option(
        None, help="Override the global data snapshot for this run"
    ),
    prom_port: int | None = typer.Option(
        None, help="Override the exporter port for this run"
    ),
) -> None:
    _walkforward_common(
        ctx=ctx,
        seed=seed,
        data=data,
        symbol=symbol,
        start=start,
        end=end,
        split=None,
        folds=folds,
        embargo=embargo,
        params=params,
        search="random",
        objective=None,
        config=config,
        data_snapshot=data_snapshot,
        prom_port=prom_port,
        section="tune",
    )


def main() -> None:  # pragma: no cover - console script helper
    app()


if __name__ == "__main__":  # pragma: no cover - python -m cli
    main()
