from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from research.splits import time_kfold


def _build_index(count: int) -> list[datetime]:
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(count)]


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_time_kfold_embargo_exclusion(mode: str) -> None:
    index = _build_index(120)
    embargo = timedelta(hours=6)
    folds = 6
    splits = time_kfold(index, folds=folds, mode=mode, embargo=embargo)

    assert len(splits) == folds

    test_windows = [set(test_idx) for _, test_idx in splits]
    for i, window in enumerate(test_windows):
        assert window  # each fold must provide a test window
        for other in test_windows[:i]:
            assert window.isdisjoint(other)

    embargo_intervals: list[tuple[datetime, datetime]] = []
    for train_idx, test_idx in splits:
        if not test_idx:
            continue
        test_start = index[test_idx[0]]
        test_end = index[test_idx[-1]]
        embargo_intervals.append((test_start, test_end + embargo))
        if not train_idx:
            continue
        for ts in (index[i] for i in train_idx):
            for prev_start, prev_end in embargo_intervals[:-1]:
                assert not (prev_start <= ts <= prev_end)


def test_rolling_window_size_matches_test_length() -> None:
    index = _build_index(24)
    embargo = timedelta(hours=1)
    splits = time_kfold(index, folds=4, mode="rolling", embargo=embargo)

    for train_idx, test_idx in splits[1:]:  # first fold may have smaller train set
        if not test_idx or not train_idx:
            continue
        assert len(train_idx) <= len(test_idx)
