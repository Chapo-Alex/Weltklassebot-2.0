"""Breakout plus bias strategy with pyramiding and stateful resets."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from typing import Any, Literal, cast

from core.events import CandleEvent, OrderEvent, OrderSide, OrderType

ThresholdMode = Literal["fixed", "percentile", "atr_k"]
ExitMode = Literal["atr_trail", "chandelier"]
PyramidMode = Literal["fixed", "geometric"]

try:  # pragma: no cover - exercised in integration tests
    from prometheus_client import Counter
except ModuleNotFoundError:  # pragma: no cover - offline fallback
    from portfolio._prometheus_stub import Counter

try:  # pragma: no cover - optional dependency for offline test environments
    from pydantic import BaseModel, Field
    try:  # pragma: no cover - pydantic v2
        from pydantic import ConfigDict
    except ImportError:  # pragma: no cover - compatibility for v1
        ConfigDict = None  # type: ignore[assignment]
except ModuleNotFoundError:  # pragma: no cover - basic stub for tests
    class _StubBaseModel:  # type: ignore[override]
        """Lightweight stub mimicking a minimal pydantic interface."""

        def __init__(self, **data: Any) -> None:
            for key, value in data.items():
                setattr(self, key, value)

    def Field(
        *, default: Any | None = None, default_factory: Any | None = None, alias: str | None = None
    ) -> Any:  # type: ignore[override]
        if default_factory is not None:
            return default_factory()
        return default

    BaseModel = cast(type, _StubBaseModel)
    ConfigDict = None  # type: ignore[assignment]

PYRAMID_ADDS = Counter(
    "pyramid_adds_total",
    "Number of pyramiding adds emitted",
    labelnames=("symbol",),
)
PYRAMID_RESETS = Counter(
    "pyramid_resets_total",
    "Number of pyramiding resets triggered",
    labelnames=("symbol", "reason"),
)
ENTRIES_SKIPPED = Counter(
    "entries_skipped_total",
    "Number of potential entries skipped by session or news filters",
    labelnames=("symbol", "reason"),
)


class StrategyConfig(BaseModel):
    """Configuration for :class:`BreakoutBiasStrategy`."""

    order_size: float = 0.1
    breakout_threshold: float = 0.01
    lookback: int = 20
    bias_lookback: int = 10
    bias_min_slope: float = 0.0
    bias_vol_ratio: float = 0.05
    bias_overrides: Mapping[str, float] = Field(default_factory=dict)
    threshold_mode: ThresholdMode = "fixed"
    threshold_lookback: int = 20
    threshold_percentile: float = 0.75
    atr_lookback: int = 14
    atr_k_low: float = 1.0
    atr_k_high: float = 1.5
    atr_percentile_split: float = 0.5
    exit_mode: ExitMode = "atr_trail"
    atr_trailing_multiplier: float | None = 1.2
    chandelier_lookback: int = 22
    chandelier_atr_mult: float = 3.0
    take_profit_pct: float | None = None
    stop_loss_pct: float | None = None
    pyramid_mode: PyramidMode = "fixed"
    pyramid_steps: tuple[float, ...] = Field(default_factory=lambda: (1.0,))
    pyramid_scale: float = 0.5
    max_pyramids: int = 3
    pyramid_dd_reset_pct: float = 0.03
    pyramid_stagnation_bars: int = 0
    vol_shock_multiple: float = 4.0
    flatten_on_session_close: bool = True
    allow_weekends: bool = True
    session: tuple[tuple[time, time, bool], ...] = Field(default_factory=tuple)
    news_blackout: tuple[tuple[datetime, datetime], ...] = Field(default_factory=tuple)
    trade_sessions: tuple[tuple[time, time], ...] = Field(default_factory=tuple)
    news_freeze: tuple[tuple[datetime, datetime], ...] = Field(default_factory=tuple)

    if "ConfigDict" in globals() and ConfigDict is not None:  # pragma: no cover - pydantic v2
        model_config = ConfigDict(arbitrary_types_allowed=True)  # type: ignore[call-arg]

    class Config:  # pragma: no cover - pydantic v1
        arbitrary_types_allowed = True

    def __init__(self, **data: Any) -> None:  # pragma: no cover - direct alias normalisation
        if "trade_sessions" in data and "session" not in data:
            data["session"] = data.pop("trade_sessions")
        if "news_freeze" in data and "news_blackout" not in data:
            data["news_blackout"] = data.pop("news_freeze")
        if "pyramid_max_steps" in data and "max_pyramids" not in data:
            data["max_pyramids"] = data.pop("pyramid_max_steps")
        super().__init__(**data)
        self._normalise()

    def _normalise(self) -> None:
        lookup: dict[str, Any] = self.__dict__  # type: ignore[attr-defined]
        lookup["lookback"] = max(int(lookup.get("lookback", 1)), 1)
        lookup["bias_lookback"] = max(int(lookup.get("bias_lookback", lookup["lookback"])), 1)
        lookup["threshold_lookback"] = max(
            int(lookup.get("threshold_lookback", lookup["lookback"])), 1
        )
        lookup["atr_lookback"] = max(
            int(lookup.get("atr_lookback", lookup["lookback"])), 1
        )
        lookup["chandelier_lookback"] = max(
            int(lookup.get("chandelier_lookback", lookup["lookback"])), 1
        )
        lookup["threshold_percentile"] = min(
            max(float(lookup.get("threshold_percentile", 0.5)), 0.0),
            1.0,
        )
        lookup["atr_percentile_split"] = min(
            max(float(lookup.get("atr_percentile_split", 0.5)), 0.0),
            1.0,
        )
        lookup["max_pyramids"] = max(int(lookup.get("max_pyramids", 1)), 1)
        lookup["pyramid_stagnation_bars"] = max(int(lookup.get("pyramid_stagnation_bars", 0)), 0)
        lookup["pyramid_dd_reset_pct"] = max(float(lookup.get("pyramid_dd_reset_pct", 0.0)), 0.0)
        lookup["vol_shock_multiple"] = max(float(lookup.get("vol_shock_multiple", 0.0)), 0.0)
        lookup["pyramid_scale"] = max(float(lookup.get("pyramid_scale", 0.0)), 0.0)
        lookup["order_size"] = max(float(lookup.get("order_size", 0.0)), 0.0)
        lookup["breakout_threshold"] = max(float(lookup.get("breakout_threshold", 0.0)), 0.0)
        lookup["bias_vol_ratio"] = max(float(lookup.get("bias_vol_ratio", 0.0)), 0.0)
        lookup["bias_min_slope"] = max(float(lookup.get("bias_min_slope", 0.0)), 0.0)
        lookup["session"] = tuple(self._parse_session_windows(lookup.get("session", ())))
        lookup["news_blackout"] = tuple(self._parse_blackouts(lookup.get("news_blackout", ())))

    @staticmethod
    def _parse_session_windows(raw: Sequence[tuple[Any, ...]]) -> list[tuple[time, time, bool]]:
        windows: list[tuple[time, time, bool]] = []
        for entry in raw:
            if len(entry) == 3:
                start_raw, end_raw, crosses = entry
                start = StrategyConfig._coerce_time(start_raw)
                end = StrategyConfig._coerce_time(end_raw)
                crosses_midnight = bool(crosses)
            elif len(entry) == 2:
                start_raw, end_raw = entry
                start = StrategyConfig._coerce_time(start_raw)
                end = StrategyConfig._coerce_time(end_raw)
                crosses_midnight = start >= end
            else:
                raise ValueError(f"invalid session window: {entry!r}")
            windows.append((start, end, crosses_midnight))
        return windows

    @staticmethod
    def _parse_blackouts(raw: Sequence[tuple[Any, Any]]) -> list[tuple[datetime, datetime]]:
        windows: list[tuple[datetime, datetime]] = []
        for start_raw, end_raw in raw:
            start = StrategyConfig._coerce_datetime(start_raw)
            end = StrategyConfig._coerce_datetime(end_raw)
            if end < start:
                start, end = end, start
            windows.append((start, end))
        return windows

    @staticmethod
    def _coerce_time(value: Any) -> time:
        if isinstance(value, time):
            return value.replace(tzinfo=None)
        if isinstance(value, str):
            parsed = time.fromisoformat(value)
            return parsed.replace(tzinfo=None)
        raise TypeError(f"unsupported time value: {value!r}")

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value.astimezone(UTC)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        raise TypeError(f"unsupported datetime value: {value!r}")


@dataclass(slots=True)
class _PositionState:
    """State tracking helper for pyramiding layers and exits."""

    layers: int = 0
    qty: float = 0.0
    average_price: float = 0.0
    high_water_mark: float = 0.0
    chandelier_stop: float | None = None
    entry_peak_price: float = 0.0
    stagnation_bars: int = 0
    adds_blocked: bool = False
    pending_flatten: bool = False

    def register_fill(self, qty: float, price: float) -> None:
        total_qty = self.qty + qty
        if total_qty <= 0:
            self.reset()
            return
        if self.qty > 0:
            self.average_price = (self.average_price * self.qty + price * qty) / total_qty
        else:
            self.average_price = price
        self.qty = total_qty
        self.layers += 1
        self.high_water_mark = max(self.high_water_mark, price)
        self.entry_peak_price = max(self.entry_peak_price, price)
        self.stagnation_bars = 0
        self.adds_blocked = False
        self.pending_flatten = False

    def reset(self) -> None:
        self.layers = 0
        self.qty = 0.0
        self.average_price = 0.0
        self.high_water_mark = 0.0
        self.chandelier_stop = None
        self.entry_peak_price = 0.0
        self.stagnation_bars = 0
        self.adds_blocked = False
        self.pending_flatten = False

    def flatten(self) -> float:
        qty = self.qty
        self.reset()
        return qty


class BreakoutBiasStrategy:
    """Generate deterministic orders when price breaks out under a bias regime."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self._config = config or StrategyConfig()
        self._pyramid_steps = self._build_pyramid_steps()
        self._history_capacity = max(
            1,
            self._config.lookback,
            self._config.bias_lookback,
            self._config.threshold_lookback,
            self._config.atr_lookback,
            self._config.chandelier_lookback,
        )
        self._positions: dict[str, _PositionState] = {}
        self._history: dict[str, deque[CandleEvent]] = {}
        self._highs: dict[str, deque[float]] = {}
        self._lows: dict[str, deque[float]] = {}
        self._closes: dict[str, deque[float]] = {}

    @property
    def config(self) -> StrategyConfig:
        return self._config

    def notify_risk_drawdown(self, symbol: str) -> None:
        position = self._positions.setdefault(symbol, _PositionState())
        position.pending_flatten = True

    def generate_orders(self, candles: Sequence[CandleEvent]) -> list[OrderEvent]:
        orders: list[OrderEvent] = []
        for index, candle in enumerate(candles):
            position = self._positions.setdefault(candle.symbol, _PositionState())
            history = self._history.setdefault(
                candle.symbol, deque(maxlen=self._history_capacity)
            )
            highs = self._highs.setdefault(candle.symbol, deque(maxlen=self._config.lookback))
            lows = self._lows.setdefault(candle.symbol, deque(maxlen=self._config.lookback))
            closes = self._closes.setdefault(
                candle.symbol, deque(maxlen=self._config.lookback)
            )

            skip_reason = self._entry_filter(candle)
            bias_ok = self._bias_allows_entry(list(history) + [candle])
            previous_high = max(highs) if highs else None
            threshold = self._compute_breakout_threshold(list(history), previous_high)
            entry_allowed = (
                skip_reason is None
                and bias_ok
                and previous_high is not None
                and candle.close >= previous_high + threshold
                and position.layers < len(self._pyramid_steps)
                and not position.adds_blocked
                and not position.pending_flatten
            )

            if entry_allowed:
                qty = self._entry_quantity(position.layers, candle.symbol)
                if qty > 0:
                    orders.append(
                        OrderEvent(
                            id=f"breakout-{candle.symbol}-{index}-layer{position.layers + 1}",
                            ts=candle.end,
                            symbol=candle.symbol,
                            side=OrderSide.BUY,
                            qty=qty,
                            type=OrderType.MARKET,
                            price=candle.close,
                            stop=None,
                            tif="GTC",
                        )
                    )
                    position.register_fill(qty=qty, price=candle.close)
                    PYRAMID_ADDS.labels(symbol=candle.symbol).inc()
            elif skip_reason is not None:
                ENTRIES_SKIPPED.labels(symbol=candle.symbol, reason=skip_reason).inc()

            combined_history = list(history) + [candle]
            exit_qty, exit_reason = self._check_exit_conditions(candle, position, combined_history)
            if exit_qty <= 0:
                exit_qty, exit_reason = self._check_forced_resets(
                    candle, position, combined_history
                )

            if exit_qty > 0:
                orders.append(
                    OrderEvent(
                        id=f"exit-{candle.symbol}-{index}",
                        ts=candle.end,
                        symbol=candle.symbol,
                        side=OrderSide.SELL,
                        qty=round(exit_qty, 6),
                        type=OrderType.MARKET,
                        price=candle.close,
                        stop=None,
                        tif="GTC",
                        reduce_only=True,
                        client_tag=exit_reason,
                    )
                )

            self._update_pyramiding_state(candle, position)
            history.append(candle)
            highs.append(candle.high)
            lows.append(candle.low)
            closes.append(candle.close)
        return orders

    def _entry_filter(self, candle: CandleEvent) -> str | None:
        if not self._config.allow_weekends and not self._is_weekday(candle.end):
            return "weekend"
        if not self._in_session(candle.end):
            return "session"
        if self._in_blackout(candle.end):
            return "news"
        return None

    def _entry_quantity(self, layer: int, symbol: str) -> float:
        multiplier = self._pyramid_steps[layer]
        bias_weight = self._config.bias_overrides.get(symbol, 1.0)
        qty = max(self._config.order_size * multiplier * bias_weight, 0.0)
        return round(qty, 6)

    def _compute_breakout_threshold(
        self, history: Sequence[CandleEvent], reference_high: float | None
    ) -> float:
        if reference_high is None or reference_high <= 0:
            reference_high = 0.0
        mode = self._config.threshold_mode
        if mode == "fixed":
            return reference_high * self._config.breakout_threshold
        if mode == "percentile":
            window = history[-self._config.threshold_lookback :]
            spans = [
                c.high - c.low for c in window if c.high >= c.low
            ]
            if spans:
                ratio = self._percentile(spans, self._config.threshold_percentile)
                return ratio
            return reference_high * self._config.breakout_threshold
        if mode == "atr_k":
            window = history[-self._config.atr_lookback :]
            if not window:
                return reference_high * self._config.breakout_threshold
            tr_ratios = self._true_range_ratios(window)
            if not tr_ratios:
                return reference_high * self._config.breakout_threshold
            current = tr_ratios[-1]
            split = self._percentile(tr_ratios, self._config.atr_percentile_split)
            multiplier = self._config.atr_k_high if current >= split else self._config.atr_k_low
            ref_price = window[-1].close if window[-1].close > 0 else reference_high or 1.0
            return ref_price * multiplier * current
        return reference_high * self._config.breakout_threshold

    def _bias_allows_entry(self, candles: Sequence[CandleEvent]) -> bool:
        window = self._config.bias_lookback
        if len(candles) < window:
            return False
        recent = candles[-window:]
        closes = [c.close for c in recent if c.close > 0]
        if len(closes) < window:
            return False
        sma = sum(closes) / len(closes)
        last = closes[-1]
        if last <= sma:
            return False
        slope_base = closes[0]
        slope = (last - slope_base) / slope_base if slope_base > 0 else 0.0
        if slope < self._config.bias_min_slope:
            return False
        if self._config.bias_vol_ratio > 0:
            atr = self._compute_atr(recent)
            if atr is None or atr <= 0:
                return False
            if atr / last > self._config.bias_vol_ratio:
                return False
        return True

    def _check_exit_conditions(
        self,
        candle: CandleEvent,
        position: _PositionState,
        history: Sequence[CandleEvent],
    ) -> tuple[float, str | None]:
        if position.qty <= 0:
            return 0.0, None

        tp_pct = self._config.take_profit_pct
        if tp_pct is not None and position.average_price > 0:
            tp_price = position.average_price * (1 + tp_pct)
            if candle.high >= tp_price:
                return position.flatten(), "take_profit"

        sl_pct = self._config.stop_loss_pct
        if sl_pct is not None and position.average_price > 0:
            sl_price = position.average_price * (1 - sl_pct)
            if candle.low <= sl_price:
                return position.flatten(), "stop_loss"

        if (
            self._config.exit_mode == "atr_trail"
            and self._config.atr_trailing_multiplier is not None
        ):
            atr_window = history[-self._config.atr_lookback :]
            atr = self._compute_atr(atr_window)
            if atr is not None and atr > 0:
                position.high_water_mark = max(position.high_water_mark, candle.high)
                trailing_stop = (
                    position.high_water_mark
                    - atr * self._config.atr_trailing_multiplier
                )
                if candle.close <= trailing_stop:
                    return position.flatten(), "atr_trail"

        if self._config.exit_mode == "chandelier":
            lookback = self._config.chandelier_lookback
            window = history[-lookback:]
            highest_high = max((c.high for c in window), default=candle.high)
            atr = self._compute_atr(window)
            if atr is not None and atr > 0:
                stop = highest_high - atr * self._config.chandelier_atr_mult
                if position.chandelier_stop is None or stop > position.chandelier_stop:
                    position.chandelier_stop = stop
                if (
                    position.chandelier_stop is not None
                    and candle.close <= position.chandelier_stop
                ):
                    return position.flatten(), "chandelier"

        return 0.0, None

    def _check_forced_resets(
        self,
        candle: CandleEvent,
        position: _PositionState,
        history: Sequence[CandleEvent],
    ) -> tuple[float, str | None]:
        if position.qty <= 0:
            return 0.0, None

        if position.pending_flatten:
            qty = position.flatten()
            PYRAMID_RESETS.labels(symbol=candle.symbol, reason="risk").inc()
            return qty, "risk"

        if self._config.flatten_on_session_close and not self._in_session(candle.end):
            qty = position.flatten()
            PYRAMID_RESETS.labels(symbol=candle.symbol, reason="session").inc()
            return qty, "session"

        if self._config.vol_shock_multiple > 0:
            atr = self._compute_atr(history[-self._config.atr_lookback :])
            if atr is not None and atr > 0:
                if (candle.high - candle.low) / atr >= self._config.vol_shock_multiple:
                    qty = position.flatten()
                    PYRAMID_RESETS.labels(symbol=candle.symbol, reason="vol_shock").inc()
                    return qty, "vol_shock"

        return 0.0, None

    def _update_pyramiding_state(self, candle: CandleEvent, position: _PositionState) -> None:
        if position.qty <= 0:
            position.entry_peak_price = 0.0
            position.stagnation_bars = 0
            position.adds_blocked = False
            return

        if position.entry_peak_price <= 0:
            position.entry_peak_price = candle.close

        if candle.close > position.entry_peak_price:
            position.entry_peak_price = candle.close
            position.stagnation_bars = 0
            position.adds_blocked = False
        else:
            position.stagnation_bars += 1

        reset_reason: str | None = None
        dd_threshold = self._config.pyramid_dd_reset_pct
        if (
            position.layers > 0
            and dd_threshold > 0
            and position.entry_peak_price > 0
            and candle.low < position.entry_peak_price
        ):
            drawdown = (position.entry_peak_price - candle.low) / position.entry_peak_price
            if drawdown >= dd_threshold:
                reset_reason = "dd"

        if (
            reset_reason is None
            and self._config.pyramid_stagnation_bars > 0
            and position.stagnation_bars >= self._config.pyramid_stagnation_bars
        ):
            reset_reason = "stagnation"

        if reset_reason is not None:
            position.layers = 0
            position.chandelier_stop = None
            position.entry_peak_price = candle.close
            position.stagnation_bars = 0
            position.adds_blocked = True
            PYRAMID_RESETS.labels(symbol=candle.symbol, reason=reset_reason).inc()

    def _build_pyramid_steps(self) -> tuple[float, ...]:
        limit = max(self._config.max_pyramids, 1)
        if self._config.pyramid_mode == "fixed":
            steps = [step for step in self._config.pyramid_steps if step > 0]
            if not steps:
                steps = [1.0]
            return tuple(steps[:limit])
        geometric_steps: list[float] = []
        scale = 1.0
        for _ in range(limit):
            geometric_steps.append(scale)
            scale *= max(self._config.pyramid_scale, 0.0)
        return tuple(geometric_steps)

    def _in_session(self, ts: datetime) -> bool:
        if not self._config.session:
            return True
        current_time = ts.astimezone(UTC).time()
        for start, end, crosses_midnight in self._config.session:
            if crosses_midnight:
                if current_time >= start or current_time < end:
                    return True
            else:
                if start <= current_time < end:
                    return True
        return False

    @staticmethod
    def _is_weekday(ts: datetime) -> bool:
        return ts.astimezone(UTC).weekday() < 5

    def _in_blackout(self, ts: datetime) -> bool:
        if not self._config.news_blackout:
            return False
        current = ts.astimezone(UTC)
        return any(start <= current <= end for start, end in self._config.news_blackout)

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        bounded = min(max(percentile, 0.0), 1.0)
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = bounded * (len(sorted_values) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)
        weight = rank - lower_index
        lower = sorted_values[lower_index]
        upper = sorted_values[upper_index]
        return lower * (1 - weight) + upper * weight

    @staticmethod
    def _true_range_ratios(candles: Sequence[CandleEvent]) -> list[float]:
        if not candles:
            return []
        ratios: list[float] = []
        prev_close = candles[0].open
        for candle in candles:
            tr = BreakoutBiasStrategy._true_range(candle, prev_close)
            ref_price = candle.close if candle.close > 0 else candle.open
            if ref_price <= 0:
                ratios.append(0.0)
            else:
                ratios.append(tr / ref_price)
            prev_close = candle.close
        return ratios

    @staticmethod
    def _compute_atr(candles: Sequence[CandleEvent]) -> float | None:
        if not candles:
            return None
        trs: list[float] = []
        prev_close = candles[0].open
        for candle in candles:
            trs.append(BreakoutBiasStrategy._true_range(candle, prev_close))
            prev_close = candle.close
        if not trs:
            return None
        return sum(trs) / len(trs)

    @staticmethod
    def _true_range(candle: CandleEvent, prev_close: float) -> float:
        high_low = candle.high - candle.low
        high_close = abs(candle.high - prev_close)
        low_close = abs(candle.low - prev_close)
        return max(high_low, high_close, low_close)


__all__ = ["BreakoutBiasStrategy", "StrategyConfig"]
