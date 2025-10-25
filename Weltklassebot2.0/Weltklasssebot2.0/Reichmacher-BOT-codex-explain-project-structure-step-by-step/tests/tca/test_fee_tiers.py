"""Fee model regression tests covering tiering and caps."""

from __future__ import annotations

import pytest

from execution.fees import FlatFee, TieredFee


def test_flat_fee_applies_distinct_maker_taker_rates() -> None:
    model = FlatFee(bps=5.0, taker_bps_override=7.5)
    notional = 250_000.0
    maker_fee = model.apply(notional, is_maker=True)
    taker_fee = model.apply(notional, is_maker=False)
    assert maker_fee == pytest.approx(notional * 0.0005)
    assert taker_fee == pytest.approx(notional * 0.00075)


def test_tiered_fee_selects_correct_bps_for_thresholds() -> None:
    model = TieredFee(
        tiers=[(50_000.0, 8.0), (250_000.0, 6.0), (1_000_000.0, 4.0)],
    )
    low_notional = 25_000.0
    mid_notional = 200_000.0
    high_notional = 900_000.0
    assert model.apply(low_notional, is_maker=True) == pytest.approx(low_notional * 0.0008)
    assert model.apply(mid_notional, is_maker=False) == pytest.approx(mid_notional * 0.0006)
    assert model.apply(high_notional, is_maker=True) == pytest.approx(high_notional * 0.0004)


def test_tiered_fee_respects_absolute_cap() -> None:
    model = TieredFee(tiers=[(100_000.0, 2.5)], cap_abs=40.0)
    uncapped = model.apply(50_000.0, is_maker=False)
    capped = model.apply(500_000.0, is_maker=False)
    assert uncapped == pytest.approx(12.5)
    assert capped == pytest.approx(40.0)
