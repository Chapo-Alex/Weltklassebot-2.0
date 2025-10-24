"""Deterministic walk-forward smoke test for the breakout strategy."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from execution.adapters import MarketExecution

from core.events import CandleEvent, OrderEvent, OrderSide, OrderType
from portfolio.risk import RiskContext, RiskManagerV2, RiskParameters
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig


@dataclass(slots=True)
class SliceResult:
    name: str
    trades: int
    hit_rate: float
    pnl: float
    max_drawdown: float


def _generate_candles(seed: int, length: int, start_price: float) -> list[CandleEvent]:
    rng = random.Random(seed)
    base = datetime(2024, 1, 1, tzinfo=UTC)
    price = start_price
    candles: list[CandleEvent] = []
    for index in range(length):
        drift = 0.0015 * (1 + (seed % 7) * 0.1)
        shock = rng.uniform(-0.003, 0.004)
        close = max(price * (1.0 + drift + shock), 1e-6)
        high = max(price, close) * (1.0 + rng.uniform(0.001, 0.004))
        low = min(price, close) * (1.0 - rng.uniform(0.001, 0.004))
        start = base + timedelta(minutes=index)
        candles.append(
            CandleEvent(
                symbol="BTCUSDT",
                open=price,
                high=high,
                low=max(low, 0.01),
                close=close,
                volume=1.0,
                start=start,
                end=start + timedelta(minutes=1),
            )
        )
        price = close
    return candles


def _run_slice(name: str, candles: list[CandleEvent], config: StrategyConfig) -> SliceResult:
    strategy = BreakoutBiasStrategy(config)
    executor = MarketExecution(fee_bps=0.5)
    orders = strategy.generate_orders(candles)

    position_qty = 0.0
    avg_price = 0.0
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    wins = 0
    trades = 0

    for order in orders:
        fill = executor.fill(order)
        if order.side == "buy":
            cumulative -= fill.fee
            notional = (avg_price * position_qty) + (fill.price * fill.qty)
            position_qty += fill.qty
            avg_price = notional / position_qty if position_qty else 0.0
        else:
            pnl = (fill.price - avg_price) * fill.qty - fill.fee
            cumulative += pnl
            wins += int(pnl > 0.0)
            trades += 1
            peak = max(peak, cumulative)
            max_drawdown = max(max_drawdown, peak - cumulative)
            position_qty = max(position_qty - fill.qty, 0.0)
            if position_qty == 0:
                avg_price = 0.0
    hit_rate = wins / trades if trades else 0.0
    return SliceResult(
        name=name,
        trades=trades,
        hit_rate=hit_rate,
        pnl=cumulative,
        max_drawdown=max_drawdown,
    )


def main() -> None:
    config = StrategyConfig(
        breakout_threshold=0.0,
        order_size=0.25,
        pyramid_steps=(1.0, 0.5),
        threshold_mode="percentile",
        threshold_percentile=0.7,
        threshold_lookback=8,
        take_profit_pct=0.02,
        exit_mode="chandelier",
        chandelier_lookback=5,
        chandelier_atr_mult=1.8,
    )

    slices = (
        ("train", 101, 32, 100.0),
        ("valid", 203, 24, 118.0),
        ("test", 307, 24, 115.0),
    )

    results: list[SliceResult] = []
    for name, seed, length, start_price in slices:
        candles = _generate_candles(seed, length, start_price)
        result = _run_slice(name, candles, config)
        results.append(result)
        print(
            f"[{result.name}] trades={result.trades} hit_rate={result.hit_rate:.2%} "
            f"pnl={result.pnl:.2f} max_dd={result.max_drawdown:.2f}"
        )

    _print_risk_transitions(results)


def _print_risk_transitions(results: list[SliceResult]) -> None:
    manager = RiskManagerV2(
        RiskParameters(
            max_drawdown=9_000.0,
            max_notional=150_000.0,
            max_trades_per_day=2,
            cooldown_minutes=20.0,
        )
    )
    base = datetime(2024, 1, 1, tzinfo=UTC)
    total_trades = sum(result.trades for result in results)
    total_pnl = sum(result.pnl for result in results)
    max_dd = max((result.max_drawdown for result in results), default=0.0)
    dummy_order = OrderEvent(
        id="risk-demo",
        ts=base,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        qty=0.0,
        type=OrderType.MARKET,
        price=30_000.0,
        stop=None,
        tif="GTC",
        reduce_only=False,
        post_only=False,
        client_tag=None,
    )
    contexts = (
        RiskContext(
            equity=100_000.0 + total_pnl,
            drawdown=max_dd,
            notional=25_000.0,
            trades_today=0,
            now=base,
            session="asia",
        ),
        RiskContext(
            equity=98_500.0,
            drawdown=1_500.0,
            notional=45_000.0,
            trades_today=max(3, total_trades // max(len(results), 1)),
            now=base + timedelta(hours=1),
            session="asia",
        ),
        RiskContext(
            equity=87_500.0,
            drawdown=12_500.0,
            notional=55_000.0,
            trades_today=1,
            now=base + timedelta(hours=2),
            session="europe",
        ),
        RiskContext(
            equity=105_000.0,
            drawdown=2_000.0,
            notional=15_000.0,
            trades_today=0,
            now=base + timedelta(hours=3),
            session="us",
        ),
    )

    transitions: list[str] = []
    previous = manager.state
    for ctx in contexts:
        decision = manager.allow(dummy_order, ctx)
        if manager.state is not previous:
            snapshot = manager.metrics_snapshot()
            transitions.append(
                f"{ctx.session}@{ctx.now.isoformat()} -> {manager.state.name} "
                f"(reason={snapshot['reason']}, decision={decision})"
            )
        previous = manager.state

    print("Risk manager transitions:")
    if transitions:
        for line in transitions:
            print(f"  {line}")
    else:
        print("  (no transitions)")
    print("Risk metrics snapshot:")
    for key, value in manager.metrics_snapshot().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
