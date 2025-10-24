from __future__ import annotations

import importlib.util
import pathlib
import sys
from datetime import UTC, datetime, timedelta

import pytest

try:
    from hypothesis import HealthCheck, given, settings, strategies as st
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    stub_name = "_hypothesis_stub"
    if stub_name in sys.modules:
        module = sys.modules[stub_name]
    else:
        stub_path = pathlib.Path(__file__).resolve().parents[1] / "_hypothesis_stub.py"
        spec = importlib.util.spec_from_file_location(stub_name, stub_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[stub_name] = module
        spec.loader.exec_module(module)
    given = module.given

    class _ExtendedStrategies(module._Strategies):  # type: ignore[misc]
        @staticmethod
        def integers(
            min_value: int | None = None,
            max_value: int | None = None,
        ) -> module._Strategy:
            lower = 0 if min_value is None else min_value
            upper = lower if max_value is None else max_value
            value = (lower + upper) // 2 if upper >= lower else lower

            def _factory() -> int:
                return int(value)

            return module._Strategy(factory=_factory)

        @staticmethod
        def sampled_from(values: list[object]) -> module._Strategy:
            sequence = list(values)
            chosen = sequence[0]

            def _factory() -> object:
                return chosen

            return module._Strategy(factory=_factory)

        @staticmethod
        def builds(func, *args: module._Strategy, **kwargs: module._Strategy) -> module._Strategy:
            def _factory() -> object:
                positional = [strategy.example() for strategy in args]
                named = {name: strategy.example() for name, strategy in kwargs.items()}
                return func(*positional, **named)

            return module._Strategy(factory=_factory)

    st = _ExtendedStrategies()

    def _map(self, fn):  # pragma: no cover - stub helper
        return module._Strategy(factory=lambda: fn(self.example()))

    module._Strategy.map = _map  # type: ignore[attr-defined]

    class _HealthCheck:  # pragma: no cover - stub compatibility
        too_slow = object()

    def settings(*args, **kwargs):  # pragma: no cover - stub compatibility
        def _decorator(func):
            return func

        return _decorator

    HealthCheck = _HealthCheck

from core.events import FillEvent, LiquidityFlag, OrderSide
from portfolio.accounting import Portfolio, Position, fifo_pnl

BASE_TS = datetime(2024, 1, 1, tzinfo=UTC)


class LinearFeeModel:
    """Simple deterministic fee model used in tests."""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def fee(self, qty: float, price: float, taker: bool) -> float:  # pragma: no cover - trivial
        multiplier = 1.5 if taker else 1.0
        return abs(qty) * price * self.rate * multiplier


def make_fill(
    *,
    order_id: int | str,
    side: OrderSide,
    qty: float,
    price: float,
    fee: float,
    ts_offset: int,
    symbol: str = "BTCUSDT",
    liquidity: LiquidityFlag = LiquidityFlag.TAKER,
) -> FillEvent:
    return FillEvent(
        order_id=str(order_id),
        ts=BASE_TS + timedelta(seconds=ts_offset),
        qty=qty,
        price=price,
        fee=fee,
        liquidity_flag=liquidity,
        symbol=symbol,
        side=side,
    )


@given(
    sell_qty=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
    extra=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    buy_price=st.floats(
        min_value=10.0,
        max_value=100_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
    sell_price=st.floats(
        min_value=10.0,
        max_value=100_000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
def test_fifo_matches_manual_calculation(
    sell_qty: float, extra: float, buy_price: float, sell_price: float
) -> None:
    buy_qty = sell_qty + extra
    fills = [
        make_fill(
            order_id="buy",
            ts_offset=0,
            qty=buy_qty,
            price=buy_price,
            fee=0.0,
            side=OrderSide.BUY,
        ),
        make_fill(
            order_id="sell",
            ts_offset=1,
            qty=sell_qty,
            price=sell_price,
            fee=0.0,
            side=OrderSide.SELL,
        ),
    ]

    expected = (sell_price - buy_price) * sell_qty
    assert fifo_pnl(fills) == pytest.approx(expected)


def test_fifo_handles_short_covering() -> None:
    fills = [
        make_fill(
            order_id="open-short",
            ts_offset=0,
            qty=1.0,
            price=150.0,
            fee=0.0,
            side=OrderSide.SELL,
        ),
        make_fill(
            order_id="cover",
            ts_offset=1,
            qty=1.0,
            price=120.0,
            fee=0.0,
            side=OrderSide.BUY,
        ),
    ]

    assert fifo_pnl(fills) == pytest.approx(30.0)


def test_position_partial_fill_fifo() -> None:
    position = Position("BTCUSDT")
    buy_fill = make_fill(
        order_id="buy-1",
        ts_offset=0,
        qty=2.0,
        price=100.0,
        fee=0.2,
        side=OrderSide.BUY,
    )
    position.apply_fill(buy_fill)

    assert position.qty == pytest.approx(2.0)
    assert position.avg_price == pytest.approx(100.0)
    assert position.fees_accum == pytest.approx(-0.2)
    assert position.last_ts == buy_fill.ts

    sell_fill = make_fill(
        order_id="sell-1",
        ts_offset=5,
        qty=1.0,
        price=110.0,
        fee=0.1,
        side=OrderSide.SELL,
    )
    position.apply_fill(sell_fill)

    assert position.qty == pytest.approx(1.0)
    assert position.avg_price == pytest.approx(100.0)
    assert position.realized_pnl == pytest.approx(10.0)
    assert position.fees_accum == pytest.approx(-0.3)
    assert position.last_ts == sell_fill.ts
    assert position.mark_to_market(120.0) == pytest.approx(29.7)


def test_position_flips_direction_after_over_sell() -> None:
    position = Position("ETHUSDT")
    buy_fill = make_fill(
        order_id="buy-eth",
        ts_offset=0,
        qty=1.0,
        price=50.0,
        fee=0.05,
        side=OrderSide.BUY,
        symbol="ETHUSDT",
    )
    position.apply_fill(buy_fill)

    sell_fill = make_fill(
        order_id="sell-eth",
        ts_offset=1,
        qty=2.0,
        price=55.0,
        fee=0.1,
        side=OrderSide.SELL,
        symbol="ETHUSDT",
    )
    position.apply_fill(sell_fill)

    assert position.qty == pytest.approx(-1.0)
    assert position.avg_price == pytest.approx(55.0)
    assert position.realized_pnl == pytest.approx(5.0)
    assert position.fees_accum == pytest.approx(-0.15000000000000002)


def test_position_zero_cross_flat() -> None:
    position = Position("SOLUSDT")
    open_short = make_fill(
        order_id="sell-sol",
        ts_offset=0,
        qty=1.5,
        price=200.0,
        fee=0.15,
        side=OrderSide.SELL,
        symbol="SOLUSDT",
    )
    position.apply_fill(open_short)

    close_short = make_fill(
        order_id="buy-sol",
        ts_offset=1,
        qty=1.5,
        price=150.0,
        fee=0.12,
        side=OrderSide.BUY,
        symbol="SOLUSDT",
    )
    position.apply_fill(close_short)

    assert position.qty == 0.0
    assert position.avg_price == 0.0
    assert position.realized_pnl == pytest.approx(75.0)
    assert position.fees_accum == pytest.approx(-0.27)


def test_position_rejects_mismatched_symbol() -> None:
    position = Position("BTCUSDT")
    fill = make_fill(
        order_id="bad",
        ts_offset=0,
        qty=1.0,
        price=30_000.0,
        fee=0.0,
        side=OrderSide.BUY,
        symbol="ETHUSDT",
    )

    with pytest.raises(ValueError):
        position.apply_fill(fill)


def test_position_rejects_negative_fee_override() -> None:
    position = Position("BTCUSDT")
    fill = make_fill(
        order_id="buy",
        ts_offset=0,
        qty=1.0,
        price=10_000.0,
        fee=0.0,
        side=OrderSide.BUY,
    )

    with pytest.raises(ValueError):
        position.apply_fill(fill, fee_override=-0.1)


def test_position_rejects_non_positive_qty() -> None:
    position = Position("BTCUSDT")
    zero_qty_fill = make_fill(
        order_id="zero",
        ts_offset=0,
        qty=0.0,
        price=10_000.0,
        fee=0.0,
        side=OrderSide.BUY,
    )

    with pytest.raises(ValueError):
        position.apply_fill(zero_qty_fill)


def test_portfolio_equity_aggregates_positions() -> None:
    portfolio = Portfolio(cash=10_000.0)
    fills = [
        make_fill(
            order_id="btc-buy",
            ts_offset=0,
            qty=1.0,
            price=20_000.0,
            fee=10.0,
            side=OrderSide.BUY,
        ),
        make_fill(
            order_id="eth-sell",
            ts_offset=1,
            qty=2.0,
            price=1_500.0,
            fee=6.0,
            side=OrderSide.SELL,
            symbol="ETHUSDT",
        ),
    ]

    for fill in fills:
        portfolio.apply_fill(fill)

    price_map = {"BTCUSDT": 21_000.0, "ETHUSDT": 1_600.0}
    equity = portfolio.equity(price_map)

    manual = portfolio.cash
    for symbol, position in portfolio.positions.items():
        manual += position.qty * price_map[symbol] + position.realized_pnl + position.fees_accum

    assert equity == pytest.approx(manual)
    exposure = portfolio.exposure()
    assert exposure.keys() == {"BTCUSDT", "ETHUSDT"}
    assert exposure["BTCUSDT"] == pytest.approx(1.0)
    assert exposure["ETHUSDT"] == pytest.approx(-2.0)


def test_portfolio_uses_fee_model_override() -> None:
    fee_model = LinearFeeModel(rate=0.001)
    portfolio = Portfolio(cash=0.0, fee_model=fee_model)

    fill = make_fill(
        order_id="btc-buy",
        ts_offset=0,
        qty=0.5,
        price=30_000.0,
        fee=0.0,
        side=OrderSide.BUY,
        liquidity=LiquidityFlag.MAKER,
    )

    portfolio.apply_fill(fill)

    position = portfolio.positions["BTCUSDT"]
    expected_fee = 0.5 * 30_000.0 * 0.001
    assert position.fees_accum == pytest.approx(-expected_fee)


def test_equity_requires_mark_for_open_position() -> None:
    portfolio = Portfolio()
    fill = make_fill(
        order_id="btc-buy",
        ts_offset=0,
        qty=1.0,
        price=20_000.0,
        fee=0.0,
        side=OrderSide.BUY,
    )
    portfolio.apply_fill(fill)

    with pytest.raises(KeyError):
        portfolio.equity({})


def test_equity_allows_missing_mark_for_flat_position() -> None:
    portfolio = Portfolio()
    buysell = [
        make_fill(
            order_id="buy",
            ts_offset=0,
            qty=1.0,
            price=20_000.0,
            fee=0.0,
            side=OrderSide.BUY,
        ),
        make_fill(
            order_id="sell",
            ts_offset=1,
            qty=1.0,
            price=20_500.0,
            fee=0.0,
            side=OrderSide.SELL,
        ),
    ]

    for fill in buysell:
        portfolio.apply_fill(fill)

    assert portfolio.positions["BTCUSDT"].qty == 0.0
    equity = portfolio.equity({})
    btc_position = portfolio.positions["BTCUSDT"]
    manual = portfolio.cash + btc_position.realized_pnl + btc_position.fees_accum
    assert equity == pytest.approx(manual)


def test_portfolio_clears_inventory_when_flat() -> None:
    portfolio = Portfolio()
    fills = [
        make_fill(
            order_id="buy",
            ts_offset=0,
            qty=0.25,
            price=40_000.0,
            fee=0.0,
            side=OrderSide.BUY,
        ),
        make_fill(
            order_id="sell",
            ts_offset=1,
            qty=0.25,
            price=41_000.0,
            fee=0.0,
            side=OrderSide.SELL,
        ),
    ]

    for fill in fills:
        portfolio.apply_fill(fill)

    position = portfolio.positions["BTCUSDT"]
    assert position.qty == 0.0
    assert list(position._lots) == []  # noqa: SLF001 - intentional internal check for coverage

fill_strategy = st.builds(
    make_fill,
    order_id=st.integers(min_value=0, max_value=1_000_000),
    ts_offset=st.integers(min_value=0, max_value=86_400),
    qty=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    price=st.floats(min_value=10.0, max_value=5_000.0, allow_nan=False, allow_infinity=False),
    fee=st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    liquidity=st.sampled_from(list(LiquidityFlag)),
    symbol=st.sampled_from(["BTCUSDT", "ETHUSDT", "SOLUSDT"]),
    side=st.sampled_from(list(OrderSide)),
)


@settings(max_examples=200, seed=12345, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(st.lists(fill_strategy, min_size=1, max_size=15))
def test_equity_invariant_holds(fills: list[FillEvent]) -> None:
    portfolio = Portfolio(cash=50_000.0)
    last_price: dict[str, float] = {}

    for fill in fills:
        portfolio.apply_fill(fill)
        last_price[fill.symbol] = fill.price

    price_map: dict[str, float] = {}
    for symbol, position in portfolio.positions.items():
        fallback_price = position.avg_price if position.avg_price else 1.0
        price_map[symbol] = last_price.get(symbol, fallback_price)

    if not price_map:
        price_map = {symbol: 1.0 for symbol in last_price or {"BTCUSDT": 1.0}}

    equity = portfolio.equity(price_map)

    manual = portfolio.cash
    for symbol, position in portfolio.positions.items():
        price = price_map[symbol]
        manual += position.qty * price + position.realized_pnl + position.fees_accum

    assert equity == pytest.approx(manual)

