"""Execution simulator with order handling and deterministic slippage."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal, cast

try:  # pragma: no cover - optional dependency guard
    from numpy.random import Generator, PCG64
except ModuleNotFoundError:  # pragma: no cover - deterministic fallback
    import random as _random

    class Generator:  # type: ignore[override]
        """Minimal RNG stub implementing the required numpy Generator API."""

        __slots__ = ("_rng",)

        def __init__(self, seed: int | None = None) -> None:
            self._rng = _random.Random(seed)

        def random(self) -> float:
            return self._rng.random()

        def uniform(self, low: float, high: float) -> float:
            return self._rng.uniform(low, high)

        def integers(
            self,
            low: int,
            high: int | None = None,
            size: None = None,
        ) -> int:
            if size is not None:
                msg = "stub generator does not support vectorised integers"
                raise NotImplementedError(msg)
            if high is None:
                high = low
                low = 0
            if high <= low:
                return low
            return self._rng.randrange(low, high)

    class PCG64:  # type: ignore[override]
        def __init__(self, seed: int | None = None) -> None:
            self._seed = seed

    def _build_generator(seed: int | None = None) -> Generator:
        return Generator(seed)
else:  # pragma: no cover - numpy path

    def _build_generator(seed: int | None = None) -> Generator:
        return Generator(PCG64(seed))

from core.events import (
    FillEvent,
    LiquidityFlag,
    OrderEvent,
    OrderSide,
    OrderType,
)
from execution.fees import FeeModel, FlatFee
from execution.orderbook import OrderBook
from execution.slippage import SlippageModel
from execution.normalizer import round_price, round_qty


@dataclass(frozen=True, slots=True)
class Rejection:
    """Rejected order outcome with a structured reason."""

    reason: str


@dataclass(slots=True)
class ExecConfig:
    """Configuration controlling the OMS simulator."""

    slippage: SlippageModel
    taker_fee: float = 0.0
    maker_fee: float = 0.0
    latency_ms: int = 0
    jitter_ms: int = 0
    fee_model: FeeModel | None = None
    venue: str = "default"
    symbol: str | None = None
    tick_size: float = 1e-8
    min_qty: float = 1e-6

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.jitter_ms < 0:
            msg = "latency and jitter must be non-negative"
            raise ValueError(msg)
        if self.taker_fee < 0 or self.maker_fee < 0:
            msg = "fees must be non-negative"
            raise ValueError(msg)
        if self.tick_size <= 0:
            msg = "tick_size must be positive"
            raise ValueError(msg)
        if self.min_qty <= 0:
            msg = "min_qty must be positive"
            raise ValueError(msg)
        if self.fee_model is None:
            maker_bps = abs(self.maker_fee) * 10_000.0
            taker_bps = abs(self.taker_fee) * 10_000.0
            self.fee_model = FlatFee(bps=maker_bps, taker_bps_override=taker_bps)
        self.fee_model = cast(FeeModel, self.fee_model)


@dataclass(slots=True)
class _RestingOrder:
    order: OrderEvent
    side: OrderSide
    price: float
    remaining: float
    queue_eta: float


class OmsSimulator:
    """Simulate an order management system against a level-2 book."""

    def __init__(
        self,
        book: OrderBook,
        cfg: ExecConfig,
        *,
        seed: int | None = None,
        rng: Generator | None = None,
    ) -> None:
        self._book = book
        self._cfg = cfg
        self._fee_model = cfg.fee_model
        self._rng = rng or _build_generator(seed)
        self._tick_size = cfg.tick_size
        self._min_qty = cfg.min_qty
        if hasattr(self._cfg.slippage, "bind_rng"):
            slippage_seed = (seed if seed is not None else 0) + 1
            self._cfg.slippage.bind_rng(_build_generator(slippage_seed))
        self._positions: dict[str, float] = {}
        self._resting: list[_RestingOrder] = []

    @property
    def tick_size(self) -> float:
        return self._tick_size

    @property
    def min_qty(self) -> float:
        return self._min_qty

    def normalise_order(self, order: OrderEvent) -> OrderEvent:
        price = round_price(order.price, self._tick_size)
        stop = round_price(order.stop, self._tick_size)
        qty = round_qty(order.qty, self._min_qty)
        return replace(order, price=price, stop=stop, qty=qty)

    def send(self, order: OrderEvent, now: datetime) -> list[FillEvent] | Rejection:
        try:
            order = self.normalise_order(order)
        except ValueError as exc:
            return Rejection(str(exc))
        if order.reduce_only and not self._can_reduce(order):
            return Rejection("reduce_only_violation")
        effective_type = order.type
        limit_price = order.price
        mid_before = self._book.mid()
        if order.type in {OrderType.STOP, OrderType.STOP_LIMIT}:
            if order.stop is None:
                return Rejection("missing_stop")
            if not self._stop_triggered(order):
                return Rejection("stop_not_triggered")
            if order.type is OrderType.STOP:
                effective_type = OrderType.MARKET
            else:
                effective_type = OrderType.LIMIT
                limit_price = order.price
        if effective_type is OrderType.MARKET:
            return self._execute_market(order, mid_before, now)
        if effective_type is OrderType.LIMIT:
            if limit_price is None:
                return Rejection("missing_price")
            return self._execute_limit(order, limit_price, mid_before, now)
        return Rejection("unsupported_type")

    def tick(self, now: datetime) -> list[FillEvent]:
        fills: list[FillEvent] = []
        remaining_orders: list[_RestingOrder] = []
        for resting in self._resting:
            probability = self._cfg.slippage.maker_fill_prob(resting.queue_eta)
            if probability <= 0.0:
                remaining_orders.append(resting)
                continue
            sample = self._rng.random()
            if sample > probability:
                remaining_orders.append(resting)
                continue
            book_side = "bid" if resting.side is OrderSide.BUY else "ask"
            removed = self._book.remove(book_side, resting.price, resting.remaining)
            if removed <= 0.0:
                continue
            fill_ts = self._fill_timestamp(now)
            notional = resting.price * removed
            fee = 0.0
            if self._fee_model is not None:
                fee = self._fee_model.apply(notional, is_maker=True)
            fill = FillEvent(
                order_id=resting.order.id,
                ts=fill_ts,
                qty=removed,
                price=resting.price,
                fee=fee,
                liquidity_flag=LiquidityFlag.MAKER,
                symbol=resting.order.symbol,
                side=resting.side,
            )
            fills.append(fill)
            self._update_position(resting.order.symbol, resting.side, removed)
        self._resting = remaining_orders
        return fills

    def _execute_market(
        self, order: OrderEvent, mid_before: float | None, now: datetime
    ) -> list[FillEvent]:
        book_side = "ask" if order.side is OrderSide.BUY else "bid"
        vwap, filled = self._book.sweep(book_side, order.qty)
        if filled <= 0.0:
            return []
        mid = mid_before or vwap
        notional = filled * mid
        price = self._cfg.slippage.taker_price(
            "buy" if order.side is OrderSide.BUY else "sell",
            notional,
            mid,
        )
        notional = price * filled
        fee = 0.0
        if self._fee_model is not None:
            fee = self._fee_model.apply(notional, is_maker=False)
        fill_ts = self._fill_timestamp(now)
        fill = FillEvent(
            order_id=order.id,
            ts=fill_ts,
            qty=filled,
            price=price,
            fee=fee,
            liquidity_flag=LiquidityFlag.TAKER,
            symbol=order.symbol,
            side=order.side,
        )
        self._update_position(order.symbol, order.side, filled)
        return [fill]

    def _execute_limit(
        self, order: OrderEvent, limit_price: float, mid_before: float | None, now: datetime
    ) -> list[FillEvent] | Rejection:
        crosses = self._would_cross(order.side, limit_price)
        fills: list[FillEvent] = []
        taker_notional = 0.0
        taker_filled = 0.0
        if crosses:
            if order.post_only:
                return Rejection("post_only_would_cross")
            _, preview_fill = self._preview_limit_fill(order.side, order.qty, limit_price)
            if order.tif == "FOK" and preview_fill < order.qty:
                return Rejection("fok_unfilled")
            taker_notional, taker_filled = self._execute_limit_cross(
                order.side, order.qty, limit_price
            )
            if taker_filled > 0:
                mid = mid_before or (taker_notional / taker_filled)
                notional = taker_filled * (mid if mid > 0 else limit_price)
                price = self._cfg.slippage.taker_price(
                    "buy" if order.side is OrderSide.BUY else "sell",
                    notional,
                    mid if mid > 0 else limit_price,
                )
                notional = price * taker_filled
                fee = 0.0
                if self._fee_model is not None:
                    fee = self._fee_model.apply(notional, is_maker=False)
                fill_ts = self._fill_timestamp(now)
                fills.append(
                    FillEvent(
                        order_id=order.id,
                        ts=fill_ts,
                        qty=taker_filled,
                        price=price,
                        fee=fee,
                        liquidity_flag=LiquidityFlag.TAKER,
                        symbol=order.symbol,
                        side=order.side,
                    )
                )
                self._update_position(order.symbol, order.side, taker_filled)
            if order.tif == "IOC":
                return fills
            remaining = order.qty - taker_filled
        else:
            if order.tif == "FOK":
                return Rejection("fok_unfilled")
            remaining = order.qty
        if remaining <= 0.0:
            return fills
        self._book.add("bid" if order.side is OrderSide.BUY else "ask", limit_price, remaining)
        self._resting.append(
            _RestingOrder(
                order=order,
                side=order.side,
                price=limit_price,
                remaining=remaining,
                queue_eta=remaining,
            )
        )
        return fills

    def _execute_limit_cross(
        self, side: OrderSide, qty: float, limit_price: float
    ) -> tuple[float, float]:
        remaining = qty
        notional = 0.0
        filled = 0.0
        while remaining > 0.0:
            best_price = self._book.best_ask() if side is OrderSide.BUY else self._book.best_bid()
            if best_price is None:
                break
            if side is OrderSide.BUY and best_price > limit_price:
                break
            if side is OrderSide.SELL and best_price < limit_price:
                break
            level_size = self._level_size("ask" if side is OrderSide.BUY else "bid", best_price)
            if level_size <= 0.0:
                break
            take = min(remaining, level_size)
            removed = self._book.remove("ask" if side is OrderSide.BUY else "bid", best_price, take)
            if removed <= 0.0:
                break
            notional += removed * best_price
            filled += removed
            remaining -= removed
        return notional, filled

    def _preview_limit_fill(
        self, side: OrderSide, qty: float, limit_price: float
    ) -> tuple[float, float]:
        remaining = qty
        notional = 0.0
        levels = self._book.asks if side is OrderSide.BUY else self._book.bids
        for level in levels:
            if side is OrderSide.BUY and level.price > limit_price:
                break
            if side is OrderSide.SELL and level.price < limit_price:
                break
            take = min(remaining, level.size)
            notional += take * level.price
            remaining -= take
            if remaining <= 0.0:
                break
        filled = qty - remaining
        return notional, filled

    def _fill_timestamp(self, now: datetime) -> datetime:
        if self._cfg.jitter_ms > 0:
            jitter_sample = self._rng.integers(0, self._cfg.jitter_ms + 1)
            jitter = int(jitter_sample)
        else:
            jitter = 0
        delay = self._cfg.latency_ms + jitter
        return now + timedelta(milliseconds=delay)

    def _can_reduce(self, order: OrderEvent) -> bool:
        current = self._positions.get(order.symbol, 0.0)
        if order.side is OrderSide.BUY:
            if current >= 0:
                return False
            max_qty = float(abs(current))
            return bool(order.qty <= max_qty)
        if current <= 0:
            return False
        return bool(order.qty <= current)

    def _update_position(self, symbol: str, side: OrderSide, qty: float) -> None:
        delta = qty if side is OrderSide.BUY else -qty
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta

    def _stop_triggered(self, order: OrderEvent) -> bool:
        reference = self._current_reference_price(order.side)
        if reference is None or order.stop is None:
            return False
        if order.side is OrderSide.BUY:
            return bool(reference >= order.stop)
        return bool(reference <= order.stop)

    def _current_reference_price(self, side: OrderSide) -> float | None:
        if side is OrderSide.BUY:
            best_ask = self._book.best_ask()
            if best_ask is not None:
                return float(best_ask)
            fallback_bid = cast(float | None, self._book.best_bid())
            return fallback_bid
        best_bid = self._book.best_bid()
        if best_bid is not None:
            return float(best_bid)
        fallback_ask = cast(float | None, self._book.best_ask())
        return fallback_ask

    def _would_cross(self, side: OrderSide, price: float) -> bool:
        best = self._book.best_ask() if side is OrderSide.BUY else self._book.best_bid()
        if best is None:
            return False
        if side is OrderSide.BUY:
            return bool(price >= best)
        return bool(price <= best)

    def _level_size(self, side: Literal["bid", "ask"], price: float) -> float:
        depth = self._book.depth(side)
        for level_price, level_size in depth:
            if level_price == price:
                return float(level_size)
        return 0.0


__all__ = ["Rejection", "ExecConfig", "OmsSimulator"]
