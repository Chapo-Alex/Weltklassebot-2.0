"""Crash recovery integration covering checkpoint + JSONL durability."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.engine_checkpoint import EngineCheckpoint
from core.events import FillEvent, LiquidityFlag, OrderSide
from portfolio.accounting import Portfolio
from state import JsonlStore


SYMBOL = "BTCUSDT"
MARK_PRICE = 101.0
BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass
class _OrderSpec:
    side: OrderSide
    qty: float
    price: float


FILLS: list[_OrderSpec] = [
    _OrderSpec(OrderSide.BUY, 1.0, 100.0),
    _OrderSpec(OrderSide.SELL, 0.4, 101.0),
    _OrderSpec(OrderSide.SELL, 0.6, 102.0),
    _OrderSpec(OrderSide.BUY, 1.2, 99.5),
]


class _DeterministicConnector:
    def __init__(self, run_id: str, sequence: int = 0) -> None:
        self.run_id = run_id
        self._sequence = sequence

    def next_coid(self) -> str:
        self._sequence += 1
        return f"{self.run_id}-{self._sequence}"

    @property
    def sequence(self) -> int:
        return self._sequence


def _make_fill(idx: int, spec: _OrderSpec) -> FillEvent:
    ts = BASE_TS + timedelta(minutes=idx)
    return FillEvent(
        order_id=f"order-{idx}",
        ts=ts,
        qty=abs(spec.qty),
        price=spec.price,
        fee=0.0,
        liquidity_flag=LiquidityFlag.TAKER,
        symbol=SYMBOL,
        side=spec.side,
    )


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _process_orders(
    connector: _DeterministicConnector,
    portfolio: Portfolio,
    store: JsonlStore,
    checkpoint: EngineCheckpoint,
    *,
    start_index: int = 0,
    crash_after: int | None = None,
) -> bool:
    for idx in range(start_index, len(FILLS)):
        spec = FILLS[idx]
        fill = _make_fill(idx, spec)
        portfolio.apply_fill(fill)
        coid = connector.next_coid()
        store.append(
            {
                "coid": coid,
                "seq": idx + 1,
                "qty": spec.qty,
                "price": spec.price,
                "ts": fill.ts.isoformat(),
            }
        )
        checkpoint.persist(
            run_id=connector.run_id,
            coid_sequence=connector.sequence,
            portfolio=portfolio,
        )
        if crash_after is not None and (idx + 1) == crash_after:
            return False
    return True


def _final_equity(portfolio: Portfolio) -> float:
    return portfolio.equity({SYMBOL: MARK_PRICE})


def test_crash_recovery_resumes_without_duplication(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    baseline_store = JsonlStore(
        baseline_dir / "fills.jsonl",
        rotate_lines=None,
        rotate_mb=None,
        fsync=True,
        index_interval=1,
    )
    baseline_checkpoint = EngineCheckpoint(baseline_dir / "checkpoint.json")
    baseline_connector = _DeterministicConnector("recovery-run")
    baseline_portfolio = Portfolio(cash=1_000_000.0)

    completed = _process_orders(
        baseline_connector,
        baseline_portfolio,
        baseline_store,
        baseline_checkpoint,
    )
    assert completed is True
    baseline_records = _load_jsonl(baseline_store.path)
    baseline_equity = _final_equity(baseline_portfolio)

    crash_dir = tmp_path / "crash"
    crash_dir.mkdir()
    crash_store = JsonlStore(
        crash_dir / "fills.jsonl",
        rotate_lines=None,
        rotate_mb=None,
        fsync=True,
        index_interval=1,
    )
    crash_checkpoint = EngineCheckpoint(crash_dir / "checkpoint.json")
    crash_connector = _DeterministicConnector("recovery-run")
    crash_portfolio = Portfolio(cash=1_000_000.0)

    crashed = _process_orders(
        crash_connector,
        crash_portfolio,
        crash_store,
        crash_checkpoint,
        crash_after=2,
    )
    assert crashed is False

    intermediate_records = _load_jsonl(crash_store.path)
    assert len(intermediate_records) == 2

    state = crash_checkpoint.load()
    assert state is not None
    assert state.coid_sequence == 2
    restarted_connector = _DeterministicConnector("new-run")
    restarted_portfolio = Portfolio(cash=1_000_000.0)
    crash_checkpoint.restore(
        connector=restarted_connector,
        portfolio=restarted_portfolio,
    )
    assert restarted_connector.run_id == "recovery-run"
    assert restarted_connector.sequence == 2

    resumed = _process_orders(
        restarted_connector,
        restarted_portfolio,
        crash_store,
        crash_checkpoint,
        start_index=state.coid_sequence,
    )
    assert resumed is True

    recovered_records = _load_jsonl(crash_store.path)
    assert recovered_records == baseline_records

    index_meta_path = crash_store.path.with_name(f"{crash_store.path.name}.index")
    assert index_meta_path.exists()
    index_meta = json.loads(index_meta_path.read_text(encoding="utf-8"))
    assert index_meta["lines"] == len(FILLS)

    recovered_equity = _final_equity(restarted_portfolio)
    assert recovered_equity == baseline_equity
