"""Golden master regression for the breakout bias strategy pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.engine import BacktestConfig, BacktestEngine
from core.events import CandleEvent, FillEvent, OrderEvent, OrderSide
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig
from tests._utils.hash import stable_hash, to_csv


@dataclass(slots=True)
class _StaticLoader:
    candles: Sequence[CandleEvent]

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        del start, end
        return [candle for candle in self.candles if candle.symbol == symbol]


class _RecordingStrategy:
    def __init__(self, inner: BreakoutBiasStrategy) -> None:
        self._inner = inner
        self.records: list[tuple[datetime, list[OrderEvent]]] = []

    def generate_orders(self, candles: Sequence[CandleEvent]) -> list[OrderEvent]:
        orders = self._inner.generate_orders(candles)
        self.records.append((candles[-1].end, list(orders)))
        return orders


def _format_orders(orders: Sequence[OrderEvent]) -> str:
    if not orders:
        return "-"
    encoded: list[str] = []
    for order in orders:
        encoded.append(
            ":".join(
                [
                    order.id,
                    order.side.value,
                    f"{order.qty:.6f}",
                    order.type.value,
                    "1" if order.reduce_only else "0",
                ]
            )
        )
    return "|".join(encoded)


def _format_fills(fills: Sequence[FillEvent]) -> str:
    if not fills:
        return "-"
    encoded: list[str] = []
    for fill in fills:
        encoded.append(
            ":".join(
                [
                    fill.order_id,
                    fill.side.value,
                    f"{fill.qty:.6f}",
                    f"{fill.price:.6f}",
                    fill.liquidity_flag.value,
                ]
            )
        )
    return "|".join(encoded)


def test_breakout_golden_master(tiny_candles) -> None:
    candles = tiny_candles(n=60, symbol="BTCUSDT")
    loader = _StaticLoader(candles)
    strategy_cfg = StrategyConfig(
        order_size=0.75,
        lookback=12,
        bias_lookback=12,
        atr_lookback=12,
        breakout_threshold=0.008,
        threshold_mode="fixed",
        pyramid_steps=(1.0, 0.8),
        max_pyramids=2,
        atr_trailing_multiplier=2.5,
        flatten_on_session_close=False,
        news_blackout=(),
        session=(),
        bias_vol_ratio=0.4,
        bias_min_slope=0.0,
    )
    recorder = _RecordingStrategy(BreakoutBiasStrategy(strategy_cfg))

    config = BacktestConfig(
        data_path=Path("./dummy.parquet"),
        symbol="BTCUSDT",
        start=candles[0].start,
        end=candles[-1].end,
        seed=1337,
        execution="sim",
        exec_params={"latency_ms": 0, "jitter_ms": 0},
        impact_coefficient=0.0002,
        maker_fee=0.0002,
        taker_fee=0.0004,
        initial_cash=100_000.0,
    )

    engine = BacktestEngine(config, data_loader=loader, strategy=recorder)
    result = engine.run()

    fills_sorted = sorted(result.fills, key=lambda f: (f.ts, f.order_id))
    fill_index = 0
    position = 0.0
    rows: list[tuple[str, str, str, float]] = []

    for candle, (ts, orders) in zip(candles, recorder.records, strict=True):
        assert ts == candle.end
        order_repr = _format_orders(orders)
        step_fills = []
        while fill_index < len(fills_sorted) and fills_sorted[fill_index].ts <= candle.end:
            fill = fills_sorted[fill_index]
            step_fills.append(fill)
            if fill.side is OrderSide.BUY:
                position += fill.qty
            else:
                position -= fill.qty
            fill_index += 1
        fill_repr = _format_fills(step_fills)
        rows.append((candle.end.isoformat(), order_repr, fill_repr, position))

    assert fill_index == len(fills_sorted)

    headers = ["timestamp", "orders", "fills", "position"]
    csv_payload = to_csv(rows, headers)
    digest = stable_hash(rows)

    assert csv_payload  # sanity check
    assert digest == "c37e6009f6b8aba103ab9f4720728cf93de50c6f3f42a37b4320e0edc089f24a"
