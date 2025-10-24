import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.engine import BacktestConfig, BacktestEngine
from core.events import CandleEvent, OrderEvent, OrderSide, OrderType
from execution.slippage import ImpactLinear


@dataclass(slots=True)
class _StaticLoader:
    candles: Sequence[CandleEvent]

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        del start, end
        return [candle for candle in self.candles if candle.symbol == symbol]


class _RoundTripStrategy:
    def __init__(self) -> None:
        self._step = 0

    def generate_orders(self, candles: Sequence[CandleEvent]) -> list[OrderEvent]:
        candle = candles[0]
        orders: list[OrderEvent] = []
        if self._step == 0:
            orders.append(
                OrderEvent(
                    id="entry-buy",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.BUY,
                    qty=1.0,
                    type=OrderType.MARKET,
                    price=candle.close,
                    stop=None,
                    tif="GTC",
                )
            )
        elif self._step == 2:
            orders.append(
                OrderEvent(
                    id="exit-sell",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    qty=1.0,
                    type=OrderType.MARKET,
                    price=candle.close,
                    stop=None,
                    tif="GTC",
                    reduce_only=True,
                )
            )
        self._step += 1
        return orders


def _tca_digest(entries: Sequence[dict[str, float]]) -> str:
    payload = json.dumps(list(entries), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_engine_emits_tca_metrics(tiny_candles) -> None:
    candles = tiny_candles(n=40, symbol="BTCUSDT")
    loader = _StaticLoader(candles)
    strategy = _RoundTripStrategy()
    config = BacktestConfig(
        data_path=Path("./dummy.parquet"),
        symbol="BTCUSDT",
        start=candles[0].start,
        end=candles[-1].end,
        seed=1337,
        execution="sim",
        exec_params={
            "slippage": ImpactLinear(k=0.0),
            "taker_fee": 0.0003,
            "maker_fee": 0.0001,
            "latency_ms": 0,
            "jitter_ms": 0,
        },
        initial_cash=50_000.0,
    )

    engine = BacktestEngine(config, data_loader=loader, strategy=strategy)
    result = engine.run()

    assert result.tca, "Expected TCA metrics for completed trades"
    assert all(set(entry.keys()) == {"vwap", "is_bps", "adverse_bps"} for entry in result.tca)
    first_entry = result.tca[0]
    inferred_mid = first_entry["vwap"] / (1.0 + first_entry["is_bps"] / 10_000.0)
    if first_entry["is_bps"] >= 0:
        assert first_entry["vwap"] >= inferred_mid
    else:
        assert first_entry["vwap"] <= inferred_mid

    digest = _tca_digest(result.tca)
    assert digest == "d61df2f33935cda82bb52b1d236d56177c082d931dc35675d3ac380d25daee5a"

    taker_fills = [fill for fill in result.fills if fill.order_id == "entry-buy"]
    assert taker_fills and taker_fills[0].side is OrderSide.BUY
