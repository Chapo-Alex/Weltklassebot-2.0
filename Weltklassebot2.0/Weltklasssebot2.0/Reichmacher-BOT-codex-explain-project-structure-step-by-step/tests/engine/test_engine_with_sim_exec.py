"""Integration tests covering the simulated execution stack within the engine."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest

from core.engine import BacktestConfig, BacktestEngine
from core.events import CandleEvent, LiquidityFlag, OrderEvent, OrderSide, OrderType
from execution.slippage import SlippageModel


@dataclass(slots=True)
class _StaticLoader:
    """Deterministic data loader returning a pre-computed candle series."""

    candles: Sequence[CandleEvent]

    def load(self, symbol: str, start: datetime, end: datetime) -> list[CandleEvent]:
        del start, end
        return [c for c in self.candles if c.symbol == symbol]


class _TestSlippage(SlippageModel):
    """Slippage model with deterministic taker adjustment and maker certainty."""

    def __init__(self, impact: float, maker_prob: float) -> None:
        self._impact = impact
        self._maker_prob = maker_prob

    def taker_price(
        self, side: Literal["buy", "sell"], notional: float, mid: float
    ) -> float:
        del notional
        adjust = 1.0 + self._impact if side == "buy" else 1.0 - self._impact
        return max(mid * adjust, 0.0)

    def maker_fill_prob(self, queue_eta: float) -> float:
        del queue_eta
        return self._maker_prob

    def bind_rng(self, rng: Any) -> None:
        del rng


class _ScriptedStrategy:
    """Strategy emitting a fixed sequence of orders for deterministic validation."""

    def __init__(self) -> None:
        self._step = 0

    def generate_orders(self, candles: Sequence[CandleEvent]) -> list[OrderEvent]:
        candle = candles[0]
        orders: list[OrderEvent] = []
        if self._step == 0:
            orders.append(
                OrderEvent(
                    id="step-0-buy",
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
        elif self._step == 1:
            orders.append(
                OrderEvent(
                    id="step-1-maker",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    qty=0.4,
                    type=OrderType.LIMIT,
                    price=candle.close + 0.5,
                    stop=None,
                    tif="GTC",
                )
            )
        elif self._step == 2:
            orders.append(
                OrderEvent(
                    id="step-2-sell",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    qty=0.2,
                    type=OrderType.MARKET,
                    price=candle.close,
                    stop=None,
                    tif="GTC",
                )
            )
        elif self._step == 3:
            orders.append(
                OrderEvent(
                    id="step-3-reduce",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    qty=0.2,
                    type=OrderType.LIMIT,
                    price=candle.close + 0.75,
                    stop=None,
                    tif="GTC",
                    reduce_only=True,
                )
            )
        elif self._step == 4:
            orders.append(
                OrderEvent(
                    id="step-4-flatten",
                    ts=candle.end,
                    symbol=candle.symbol,
                    side=OrderSide.SELL,
                    qty=0.2,
                    type=OrderType.MARKET,
                    price=candle.close,
                    stop=None,
                    tif="GTC",
                )
            )
        self._step += 1
        return orders


def _expected_trade_digest(fills: Sequence[dict[str, object]]) -> str:
    payload = json.dumps(fills, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_engine_with_simulated_execution(tiny_candles):
    candles = tiny_candles(n=40, symbol="BTCUSDT")
    loader = _StaticLoader(candles)
    strategy = _ScriptedStrategy()
    slippage = _TestSlippage(impact=0.002, maker_prob=1.0)
    config = BacktestConfig(
        data_path=Path("./dummy.parquet"),
        symbol="BTCUSDT",
        start=candles[0].start,
        end=candles[-1].end,
        seed=1337,
        execution="sim",
        exec_params={
            "slippage": slippage,
            "taker_fee": 0.0003,
            "maker_fee": 0.0001,
            "latency_ms": 0,
            "jitter_ms": 0,
        },
        initial_cash=100_000.0,
    )

    engine = BacktestEngine(config, data_loader=loader, strategy=strategy)
    result = engine.run()

    assert len(result.fills) >= 5

    taker_buy = next(fill for fill in result.fills if fill.order_id == "step-0-buy")
    assert taker_buy.liquidity_flag is LiquidityFlag.TAKER
    assert taker_buy.price > candles[0].close

    taker_sell = next(fill for fill in result.fills if fill.order_id == "step-2-sell")
    assert taker_sell.liquidity_flag is LiquidityFlag.TAKER
    assert taker_sell.price < candles[2].close

    maker_fills = [fill for fill in result.fills if fill.liquidity_flag is LiquidityFlag.MAKER]
    assert maker_fills
    assert {fill.order_id for fill in maker_fills} == {"step-1-maker", "step-3-reduce"}

    position = 0.0
    for fill in result.fills:
        if fill.side is OrderSide.BUY:
            position += fill.qty
        else:
            prior = position
            position -= fill.qty
            if fill.order_id == "step-3-reduce":
                assert prior >= 0.0
                assert position >= -1e-9
    assert position == pytest.approx(0.0, abs=1e-9)

    digest = _expected_trade_digest(result.trades_records())
    assert digest == "d7399f75d5d313b7efea596ee18e9b56b6db21f9b582759f13a546eb4a7f08bd"
