"""Order book behaviour and invariants."""

from __future__ import annotations

from datetime import datetime

import pytest

from execution.orderbook import OrderBook


def _build_book() -> OrderBook:
    book = OrderBook(ts=datetime(2024, 1, 1))
    book.add("bid", 99.0, 2.0)
    book.add("bid", 98.0, 3.0)
    book.add("ask", 101.0, 1.5)
    book.add("ask", 102.0, 2.5)
    return book


def test_best_prices_and_midpoint() -> None:
    book = _build_book()
    assert book.best_bid() == pytest.approx(99.0)
    assert book.best_ask() == pytest.approx(101.0)
    assert book.mid() == pytest.approx(100.0)
    assert book.depth("bid") == [(99.0, 2.0), (98.0, 3.0)]
    assert book.depth("ask") == [(101.0, 1.5), (102.0, 2.5)]


def test_sweep_consumes_levels_fifo() -> None:
    book = _build_book()
    vwap, filled = book.sweep("ask", 2.0)
    expected_vwap = ((1.5 * 101.0) + (0.5 * 102.0)) / 2.0
    assert filled == pytest.approx(2.0)
    assert vwap == pytest.approx(expected_vwap)
    assert book.best_ask() == pytest.approx(102.0)
    ask_price, ask_size = book.depth("ask")[0]
    assert ask_price == pytest.approx(102.0)
    assert ask_size == pytest.approx(2.0)


def test_remove_reduces_and_cleans_levels() -> None:
    book = _build_book()
    removed = book.remove("bid", 99.0, 1.0)
    assert removed == pytest.approx(1.0)
    bid_price, bid_size = book.depth("bid")[0]
    assert bid_price == pytest.approx(99.0)
    assert bid_size == pytest.approx(1.0)
    removed_all = book.remove("bid", 99.0, 5.0)
    assert removed_all == pytest.approx(1.0)
    assert book.best_bid() == pytest.approx(98.0)
    assert book.remove("ask", 105.0, 1.0) == pytest.approx(0.0)
