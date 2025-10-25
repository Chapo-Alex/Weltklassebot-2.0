"""Deterministic backtesting engine that coordinates strategy, OMS and risk."""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from time import perf_counter
from typing import Any, Literal, Protocol, TypeVar, cast

from backtest.primitives import BacktestResult, EquityPoint, MakerTakerFeeModel, ParquetDataLoader
from core.config import BacktestConfig
from core.events import CandleEvent, FillEvent, LiquidityFlag, OrderEvent, OrderSide
from core.metrics import LAT
from core.io.replay import ReplayLogs
from execution.orderbook import OrderBook
from execution.fees import FlatFee
from execution.oms_simulator import ExecConfig, OmsSimulator, Rejection
from execution.slippage import LinearSlippage, SlippageModel
from registry.venues import resolve_models
from portfolio.accounting import FeeModel, Portfolio
from portfolio.risk import RiskContext, RiskManagerV2, State
from portfolio.risk_state_store import JsonlStateStore
from strategy.breakout_bias import BreakoutBiasStrategy, StrategyConfig
from tca.metrics import adverse_selection, implementation_shortfall, vwap

try:  # pragma: no cover - numpy optional dependency
    from numpy.random import Generator as _NPGenerator, PCG64 as _NPPCG64
except ModuleNotFoundError:  # pragma: no cover - numpy not installed
    _NPGenerator = None
    _NPPCG64 = None


def _build_uniform_rng(seed: int | None) -> Any:
    """Return an RNG supporting ``uniform`` for deterministic jitter."""

    if _NPGenerator is not None and _NPPCG64 is not None:
        return _NPGenerator(_NPPCG64(seed))
    return _random.Random(seed)

T = TypeVar("T")
_TCA_EPS = 1e-9


class ExecutionClient(Protocol):
    """Interface for execution adapters used by the backtest engine."""

    def send(self, order: OrderEvent, now: datetime) -> list[FillEvent]:
        """Submit ``order`` and return the resulting fills (if any)."""

    def tick(self, now: datetime) -> None:
        """Advance time allowing the execution venue to release maker fills."""


@dataclass(slots=True)
class _OpenTrade:
    """Tracks entry metadata until the corresponding quantity is closed."""

    side: OrderSide
    entry_mid: float | None
    fills: list[FillEvent]
    remaining_qty: float


def _is_slippage(candidate: Any) -> bool:
    """Return ``True`` when ``candidate`` satisfies the slippage protocol."""

    return hasattr(candidate, "taker_price") and hasattr(candidate, "maker_fill_prob")



class BacktestEngine:
    """Replay candles through a strategy while applying risk controls."""

    def __init__(
        self,
        config: BacktestConfig,
        *,
        data_loader: ParquetDataLoader | None = None,
        strategy: BreakoutBiasStrategy | None = None,
        portfolio: Portfolio | None = None,
        risk_manager: RiskManagerV2 | None = None,
        execution_client: ExecutionClient | None = None,
        mode: Literal["backtest", "replay"] = "backtest",
        replay_logs: ReplayLogs | None = None,
    ) -> None:
        self._config = config
        if mode not in {"backtest", "replay"}:
            msg = f"Unsupported engine mode: {mode!r}"
            raise ValueError(msg)
        if mode == "replay" and replay_logs is None:
            msg = "Replay mode requires replay_logs"
            raise ValueError(msg)
        self._mode = mode
        self._replay_logs = replay_logs
        self._rng_seed = int(config.seed)
        self._data_loader = data_loader or ParquetDataLoader(config.data_path)
        strategy_cfg = config.strategy_config or StrategyConfig()
        self._strategy = strategy or BreakoutBiasStrategy(strategy_cfg)
        fee_model: FeeModel | None = None
        if config.execution != "sim" and (config.maker_fee > 0.0 or config.taker_fee > 0.0):
            fee_model = MakerTakerFeeModel(config.maker_fee, config.taker_fee)
        self._portfolio = portfolio or Portfolio(cash=config.initial_cash, fee_model=fee_model)
        self._risk = risk_manager or RiskManagerV2(config.risk)
        self._risk_store: JsonlStateStore | None = None
        self._persisted_risk_state: str | None = None
        if config.risk_store_dir is not None:
            store = JsonlStateStore(
                config.risk_store_dir,
                rotate_lines=config.risk_store_rotate_lines,
                rotate_mb=config.risk_store_rotate_mb,
                fsync=config.risk_store_fsync,
            )
            record = store.load_state()
            raw_state = str(record.get("state", "RUNNING"))
            state_name = raw_state if raw_state in State.__members__ else "RUNNING"
            target_state = State[state_name]
            if target_state is not self._risk.state:
                self._risk._set_state(target_state, reason="boot", ctx=None)
            self._risk_store = store
            self._persisted_risk_state = state_name
        self._marks: dict[str, float] = {}
        self._order_book = OrderBook()
        self._fills: list[FillEvent] = []
        self._orders: list[OrderEvent] = []
        self._events: list[dict[str, Any]] = []
        if self._mode == "replay":
            self._execution = execution_client
        else:
            self._execution = execution_client or self._build_execution()
        self._max_equity = config.initial_cash
        self._last_equity = config.initial_cash
        self._drawdown = 0.0
        self._trades_today = 0
        self._current_session_date: date | None = None
        self._open_trades: dict[str, list[_OpenTrade]] = {}
        self._tca_results: list[dict[str, float]] = []
        self._last_candle_mid: float | None = None

    def run(self) -> BacktestResult:
        if self._mode == "replay":
            return self._run_replay()
        return self._run_backtest()

    def _run_backtest(self) -> BacktestResult:
        self._orders = []
        self._events = []
        self._tca_results = []
        self._marks = {}
        self._fills = []
        self._order_book = OrderBook()
        self._open_trades = {}
        self._max_equity = self._config.initial_cash
        self._last_equity = self._config.initial_cash
        self._drawdown = 0.0
        self._trades_today = 0
        self._current_session_date = None
        candles = self._time_stage(
            "data_load",
            self._data_loader.load,
            self._config.symbol,
            self._config.start,
            self._config.end,
        )
        equity_curve: list[EquityPoint] = []
        for candle in candles:
            self._update_order_book(candle)
            self._marks[candle.symbol] = candle.close
            self._update_equity()
            self._max_equity = max(self._max_equity, self._last_equity)
            self._drawdown = max(0.0, self._max_equity - self._last_equity)
            self._maybe_reset_session(candle.end)
            self._update_risk_context(timestamp=candle.end)

            orders = self._time_stage("strategy", self._strategy.generate_orders, [candle])
            for order in orders:
                self._record_order(order)
                ctx = self._build_context(order.ts, symbol=order.symbol)
                decision = self._time_stage("risk_allow", self._risk.allow, order, ctx)
                self._record_risk_state_change("allow")
                if decision is not True:
                    continue
                exec_fills = self._time_stage("execution", self._execution.send, order, order.ts)
                for fill in exec_fills:
                    self._handle_fill(fill)

            self._update_equity()
            self._max_equity = max(self._max_equity, self._last_equity)
            self._drawdown = max(0.0, self._max_equity - self._last_equity)
            self._time_stage("execution_tick", self._execution.tick, candle.end)
            equity_curve.append(
                EquityPoint(
                    ts=candle.end,
                    equity=self._last_equity,
                    cash=self._portfolio.cash,
                    notional=self._current_notional(),
                    drawdown=self._drawdown,
                )
            )

        return BacktestResult(
            fills=list(self._fills),
            equity_curve=equity_curve,
            candles_processed=len(candles),
            config=self._config,
            tca=list(self._tca_results),
            orders=list(self._orders),
            events=list(self._events),
        )

    def _run_replay(self) -> BacktestResult:
        if self._replay_logs is None:
            msg = "Replay mode requires replay_logs"
            raise ValueError(msg)

        self._marks = {}
        self._fills = []
        self._orders = list(self._replay_logs.orders)
        self._events = list(self._replay_logs.events)
        self._tca_results = []
        self._order_book = OrderBook()
        self._open_trades = {}
        self._max_equity = self._config.initial_cash
        self._last_equity = self._config.initial_cash
        self._drawdown = 0.0
        self._trades_today = 0
        self._current_session_date = None

        candles = self._time_stage(
            "data_load",
            self._data_loader.load,
            self._config.symbol,
            self._config.start,
            self._config.end,
        )

        replay_fills = sorted(self._replay_logs.fills, key=lambda fill: fill.ts)
        fill_idx = 0
        total_fills = len(replay_fills)
        equity_curve: list[EquityPoint] = []

        for candle in candles:
            self._update_order_book(candle)
            self._marks[candle.symbol] = candle.close
            self._update_equity()
            self._max_equity = max(self._max_equity, self._last_equity)
            self._drawdown = max(0.0, self._max_equity - self._last_equity)
            self._maybe_reset_session(candle.end)
            self._update_risk_context(timestamp=candle.end)

            while fill_idx < total_fills and replay_fills[fill_idx].ts <= candle.end:
                self._handle_fill(replay_fills[fill_idx])
                fill_idx += 1

            self._update_equity()
            self._max_equity = max(self._max_equity, self._last_equity)
            self._drawdown = max(0.0, self._max_equity - self._last_equity)
            equity_curve.append(
                EquityPoint(
                    ts=candle.end,
                    equity=self._last_equity,
                    cash=self._portfolio.cash,
                    notional=self._current_notional(),
                    drawdown=self._drawdown,
                )
            )

        if fill_idx != total_fills:
            msg = "Not all replay fills were consumed"
            raise ValueError(msg)

        return BacktestResult(
            fills=list(self._fills),
            equity_curve=equity_curve,
            candles_processed=len(candles),
            config=self._config,
            tca=list(self._tca_results),
            orders=list(self._orders),
            events=list(self._events),
        )

    def _maybe_reset_session(self, timestamp: datetime) -> None:
        session_date = timestamp.date()
        if self._current_session_date != session_date:
            self._current_session_date = session_date
            self._trades_today = 0

    def _update_risk_context(self, timestamp: datetime) -> None:
        ctx = self._build_context(timestamp, symbol=self._config.symbol)
        self._time_stage("risk_transition", self._risk.transition, ctx)
        self._record_risk_state_change("transition")

    def _build_context(self, timestamp: datetime, symbol: str | None = None) -> RiskContext:
        orderbook_mid = self._order_book.mid()
        if orderbook_mid is None:
            orderbook_mid = self._last_candle_mid
        last_close = None
        if symbol is not None:
            last_close = self._marks.get(symbol)
        elif self._marks:
            last_close = self._marks.get(self._config.symbol)
        return RiskContext(
            equity=self._last_equity,
            drawdown=self._drawdown,
            notional=self._current_notional(),
            trades_today=self._trades_today,
            now=timestamp,
            session=self._config.session_name,
            orderbook_mid=orderbook_mid,
            last_close=last_close,
        )

    def _current_notional(self) -> float:
        notional = 0.0
        for symbol, qty in self._portfolio.exposure().items():
            price = self._marks.get(symbol)
            if price is None:
                continue
            notional += abs(qty) * price
        return notional

    def _handle_fill(self, fill: FillEvent) -> None:
        before_qty = self._position_qty(fill.symbol)
        self._time_stage("portfolio_apply", self._portfolio.apply_fill, fill)
        after_qty = self._position_qty(fill.symbol)
        self._fills.append(fill)
        self._log_fill(fill)
        self._trades_today += 1
        self._update_tca(fill, before_qty, after_qty)
        self._update_equity()
        self._max_equity = max(self._max_equity, self._last_equity)
        self._drawdown = max(0.0, self._max_equity - self._last_equity)
        transition_ctx = self._build_context(fill.ts, symbol=fill.symbol)
        self._time_stage("risk_transition", self._risk.transition, transition_ctx)
        self._record_risk_state_change("fill")

    def _position_qty(self, symbol: str) -> float:
        position = self._portfolio.positions.get(symbol)
        if position is None:
            return 0.0
        return position.qty

    def _update_tca(self, fill: FillEvent, before_qty: float, after_qty: float) -> None:
        sign_before = self._sign(before_qty)
        sign_after = self._sign(after_qty)
        before_abs = abs(before_qty)
        after_abs = abs(after_qty)

        closed_qty = 0.0
        entry_qty = 0.0

        if sign_before == 0 and sign_after != 0:
            entry_qty = after_abs
        elif sign_before != 0 and sign_after == 0:
            closed_qty = before_abs
        elif sign_before != 0 and sign_after == sign_before:
            delta = after_abs - before_abs
            if delta > _TCA_EPS:
                entry_qty = delta
            elif delta < -_TCA_EPS:
                closed_qty = -delta
        elif sign_before != 0 and sign_after != 0 and sign_before != sign_after:
            closed_qty = before_abs
            entry_qty = after_abs

        if closed_qty > _TCA_EPS:
            next_mid = self._current_mid(fill.symbol)
            self._finalise_trades(fill.symbol, closed_qty, next_mid)

        if entry_qty > _TCA_EPS:
            self._record_entry(fill, entry_qty)

    def _record_entry(self, fill: FillEvent, qty: float) -> None:
        portion = self._scaled_fill(fill, qty)
        bucket = self._open_trades.setdefault(fill.symbol, [])
        entry_mid = self._current_mid(fill.symbol)
        bucket.append(
            _OpenTrade(
                side=fill.side,
                entry_mid=entry_mid,
                fills=[portion],
                remaining_qty=portion.qty,
            )
        )

    def _finalise_trades(
        self, symbol: str, qty: float, next_mid: float | None
    ) -> None:
        queue = self._open_trades.get(symbol)
        if not queue:
            return
        remaining = qty
        while remaining > _TCA_EPS and queue:
            trade = queue[0]
            take = min(trade.remaining_qty, remaining)
            fills = self._consume_entry_fills(trade, take)
            consumed = sum(fill.qty for fill in fills)
            remaining -= consumed
            if trade.remaining_qty <= _TCA_EPS or not trade.fills:
                queue.pop(0)
            if not fills or consumed <= _TCA_EPS:
                continue
            entry_mid = trade.entry_mid
            if entry_mid is None or next_mid is None:
                continue
            metrics = {
                "vwap": vwap(fills),
                "is_bps": implementation_shortfall(entry_mid, fills, trade.side),
                "adverse_bps": adverse_selection(fills, next_mid, trade.side),
            }
            self._tca_results.append(metrics)
        if queue:
            self._open_trades[symbol] = queue
        elif symbol in self._open_trades:
            self._open_trades.pop(symbol, None)

    def _consume_entry_fills(self, trade: _OpenTrade, qty: float) -> list[FillEvent]:
        remaining = qty
        consumed: list[FillEvent] = []
        while remaining > _TCA_EPS and trade.fills:
            current = trade.fills[0]
            if current.qty <= remaining + _TCA_EPS:
                consumed.append(current)
                remaining -= current.qty
                trade.fills.pop(0)
            else:
                portion = self._scaled_fill(current, remaining)
                leftover_qty = current.qty - portion.qty
                leftover_fee = max(current.fee - portion.fee, 0.0)
                trade.fills[0] = replace(current, qty=leftover_qty, fee=leftover_fee)
                consumed.append(portion)
                remaining = 0.0
        consumed_qty = sum(fill.qty for fill in consumed)
        trade.remaining_qty = max(trade.remaining_qty - consumed_qty, 0.0)
        return consumed

    def _scaled_fill(self, fill: FillEvent, qty: float) -> FillEvent:
        if fill.qty <= _TCA_EPS:
            return fill
        ratio = qty / fill.qty
        fee = fill.fee * ratio
        return replace(fill, qty=qty, fee=fee)

    def _current_mid(self, symbol: str) -> float | None:
        mid = self._order_book.mid()
        if mid is not None:
            return mid
        return self._marks.get(symbol)

    @staticmethod
    def _sign(value: float) -> int:
        if abs(value) <= _TCA_EPS:
            return 0
        return 1 if value > 0 else -1

    def _record_risk_state_change(self, source: str) -> None:
        if self._risk_store is None:
            return
        snapshot = self._risk.metrics_snapshot()
        state_name = str(snapshot.get("state", "RUNNING"))
        if self._persisted_risk_state == state_name:
            return
        timestamp = self._risk_store.now()
        reason = str(snapshot.get("reason", ""))
        cooldown_until = snapshot.get("cooldown_until")
        meta: dict[str, Any] = {"reason": reason}
        if isinstance(cooldown_until, str) and cooldown_until:
            meta["cooldown_until"] = cooldown_until
        payload = {"state": state_name, "since": timestamp, "meta": meta}
        self._risk_store.save_state(payload)
        event: dict[str, Any] = {
            "ts": timestamp,
            "actor": "engine",
            "state": state_name,
            "reason": reason,
            "source": source,
        }
        if self._persisted_risk_state is not None:
            event["previous_state"] = self._persisted_risk_state
        if isinstance(cooldown_until, str) and cooldown_until:
            event["cooldown_until"] = cooldown_until
        self._risk_store.append_audit(event)
        self._persisted_risk_state = state_name
        self._log_event(
            {
                "ts": datetime.fromtimestamp(timestamp, tz=UTC).isoformat(),
                "type": "risk_state",
                "state": state_name,
                "source": source,
                "reason": reason,
            }
        )

    def _build_execution(self) -> ExecutionClient:
        if self._config.execution == "paper":
            return _PaperExecution(self._marks, self._config, seed=self._rng_seed)
        if self._config.execution != "sim":
            msg = f"Unsupported execution mode: {self._config.execution}"
            raise ValueError(msg)

        exec_params = dict(self._config.exec_params)
        profile = resolve_models(self._config.venue, self._config.symbol)
        slippage_candidate = exec_params.get("slippage")
        if _is_slippage(slippage_candidate):
            exec_params["slippage"] = cast(SlippageModel, slippage_candidate)
        else:
            coefficient = float(self._config.impact_coefficient)
            if coefficient > 0:
                exec_params["slippage"] = LinearSlippage(
                    bps_per_notional=coefficient,
                    seed=self._rng_seed,
                )
            else:
                exec_params["slippage"] = profile.slippage_model
        taker_default = self._config.taker_fee if self._config.taker_fee > 0 else 0.0004
        maker_default = self._config.maker_fee if self._config.maker_fee > 0 else 0.0002
        latency_default = (
            int(self._config.latency_ms)
            if self._config.latency_ms > 0
            else 30
        )
        fee_model = exec_params.get("fee_model")
        if fee_model is None:
            if self._config.maker_fee > 0 or self._config.taker_fee > 0:
                maker_bps = abs(self._config.maker_fee) * 10_000.0
                taker_bps = abs(self._config.taker_fee) * 10_000.0
                exec_params["fee_model"] = FlatFee(
                    bps=maker_bps,
                    taker_bps_override=taker_bps,
                )
                exec_params.setdefault("taker_fee", taker_default)
                exec_params.setdefault("maker_fee", maker_default)
            else:
                exec_params["fee_model"] = profile.fee_model
        exec_params.setdefault("latency_ms", latency_default)
        exec_params.setdefault("jitter_ms", 5)
        exec_seed = int(exec_params.pop("seed", self._rng_seed))
        if "taker_fee" in exec_params:
            exec_params["taker_fee"] = float(exec_params["taker_fee"])
        if "maker_fee" in exec_params:
            exec_params["maker_fee"] = float(exec_params["maker_fee"])
        exec_params["latency_ms"] = int(exec_params["latency_ms"])
        exec_params["jitter_ms"] = int(exec_params["jitter_ms"])
        exec_params.setdefault("venue", self._config.venue)
        exec_params.setdefault("symbol", self._config.symbol)
        exec_params.setdefault("tick_size", float(profile.tick_size))
        exec_params.setdefault("min_qty", float(profile.min_qty))
        cfg = ExecConfig(**exec_params)
        simulator = OmsSimulator(self._order_book, cfg, seed=exec_seed)
        return _SimulatorClient(simulator, self._handle_fill)

    def _update_order_book(self, candle: CandleEvent) -> None:
        spread = max(candle.close * 0.0005, 0.01)
        size = max(candle.volume * 0.05, 1.0)
        bid_price = max(candle.close - spread, 0.01)
        ask_price = candle.close + spread
        self._last_candle_mid = (candle.high + candle.low) / 2.0
        self._order_book.ts = candle.end
        self._order_book.add("bid", bid_price, size)
        self._order_book.add("ask", ask_price, size)
        self._order_book.add("bid", max(bid_price * 0.99, 0.01), size * 2)
        self._order_book.add("ask", ask_price * 1.01, size * 2)

    def _update_equity(self) -> None:
        if not self._marks:
            self._last_equity = self._portfolio.cash
            return
        self._last_equity = self._time_stage(
            "portfolio_equity",
            self._portfolio.equity,
            self._marks,
        )

    def _time_stage(self, stage: str, func: Callable[..., T], *args: Any) -> T:
        start = perf_counter()
        try:
            return func(*args)
        finally:
            duration = perf_counter() - start
            LAT.labels(stage=stage).observe(duration)

    def _record_order(self, order: OrderEvent) -> None:
        if self._mode == "replay":
            return
        self._orders.append(order)
        self._log_event(
            {
                "ts": order.ts.astimezone(UTC).isoformat(),
                "type": "order_submitted",
                "order_id": order.id,
                "symbol": order.symbol,
                "side": order.side.value,
                "qty": order.qty,
                "price": order.price,
                "tif": order.tif,
            }
        )

    def _log_fill(self, fill: FillEvent) -> None:
        if self._mode == "replay":
            return
        self._log_event(
            {
                "ts": fill.ts.astimezone(UTC).isoformat(),
                "type": "fill",
                "order_id": fill.order_id,
                "symbol": fill.symbol,
                "side": fill.side.value,
                "qty": fill.qty,
                "price": fill.price,
                "fee": fill.fee,
                "liquidity": fill.liquidity_flag.value,
            }
        )

    def _log_event(self, payload: dict[str, Any]) -> None:
        if self._mode == "replay":
            return
        self._events.append(payload)


class _SimulatorClient:
    """Adapter that exposes ``OmsSimulator`` via the ``ExecutionClient`` protocol."""

    __slots__ = ("_simulator", "_on_fill")

    def __init__(
        self, simulator: OmsSimulator, on_fill: Callable[[FillEvent], None] | None
    ) -> None:
        self._simulator = simulator
        self._on_fill = on_fill

    def send(self, order: OrderEvent, now: datetime) -> list[FillEvent]:
        result = self._simulator.send(order, now)
        if isinstance(result, Rejection):
            return []
        return list(result)

    def tick(self, now: datetime) -> None:
        fills = self._simulator.tick(now)
        if not fills or self._on_fill is None:
            return
        for fill in fills:
            self._on_fill(fill)


class _PaperExecution:
    """Deterministic execution fallback mirroring the legacy fill logic."""

    __slots__ = ("_marks", "_config", "_rng")

    def __init__(self, marks: dict[str, float], config: BacktestConfig, *, seed: int) -> None:
        self._marks = marks
        self._config = config
        self._rng = _build_uniform_rng(seed)

    def send(self, order: OrderEvent, now: datetime) -> list[FillEvent]:
        price = order.price if order.price is not None else self._marks.get(order.symbol, 0.0)
        notional = abs(order.qty) * price
        slippage = notional * self._config.impact_coefficient
        adjusted_price = price + slippage if order.side is OrderSide.BUY else price - slippage
        adjusted_price = max(adjusted_price, 0.0)
        latency_jitter = self._rng.uniform(
            -0.25 * self._config.latency_ms,
            0.25 * self._config.latency_ms,
        )
        latency_ms = max(0.0, self._config.latency_ms + latency_jitter)
        timestamp = now + timedelta(milliseconds=latency_ms)
        fill = FillEvent(
            order_id=order.id,
            ts=timestamp,
            qty=abs(order.qty),
            price=adjusted_price,
            fee=0.0,
            liquidity_flag=LiquidityFlag.TAKER,
            symbol=order.symbol,
            side=order.side,
        )
        return [fill]

    def tick(self, now: datetime) -> None:
        del now
        return None


__all__ = [
    "ExecutionClient",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "EquityPoint",
    "MakerTakerFeeModel",
    "ParquetDataLoader",
]
