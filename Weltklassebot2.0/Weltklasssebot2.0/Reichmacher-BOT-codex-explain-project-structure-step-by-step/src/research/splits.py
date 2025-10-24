"""Time-based cross-validation splits with embargo handling."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal


def time_kfold(
    df_index: Sequence[datetime],
    folds: int,
    mode: Literal["expanding", "rolling"],
    embargo: timedelta,
) -> list[tuple[list[int], list[int]]]:
    """Return deterministic train/test index splits for time-series data.

    The function divides ``df_index`` into ``folds`` contiguous segments that act as
    the test windows.  Training indices are drawn from the data strictly prior to a
    fold's test range.  For ``expanding`` splits the training window grows with each
    fold while ``rolling`` keeps a sliding window whose length matches the
    associated test span.  After a test window completes, its range – extended by
    ``embargo`` – is excluded from the training data of subsequent folds.
    """

    if folds < 2:
        msg = "folds must be at least 2 for cross validation"
        raise ValueError(msg)
    if embargo < timedelta(0):
        msg = "embargo must not be negative"
        raise ValueError(msg)

    timeline = list(df_index)
    total = len(timeline)
    if total == 0:
        msg = "df_index must contain at least one timestamp"
        raise ValueError(msg)
    if folds > total:
        msg = "folds must not exceed the number of samples"
        raise ValueError(msg)

    for earlier, later in zip(timeline, timeline[1:], strict=False):
        if later < earlier:
            msg = "df_index must be sorted in non-decreasing order"
            raise ValueError(msg)

    base = total // folds
    remainder = total % folds
    boundaries: list[tuple[int, int]] = []
    start = 0
    for fold in range(folds):
        size = base + (1 if fold < remainder else 0)
        if size == 0:
            continue
        end = start + size
        boundaries.append((start, end))
        start = end

    if not boundaries or boundaries[-1][1] != total:
        msg = "failed to build contiguous folds"
        raise ValueError(msg)

    splits: list[tuple[list[int], list[int]]] = []
    embargo_windows: list[tuple[datetime, datetime]] = []

    for start_idx, end_idx in boundaries:
        test_idx = list(range(start_idx, end_idx))
        if not test_idx:
            continue
        test_start = timeline[test_idx[0]]
        test_end = timeline[test_idx[-1]]

        if mode == "expanding":
            candidates = list(range(0, start_idx))
        elif mode == "rolling":
            window = end_idx - start_idx
            train_start = max(0, start_idx - window)
            candidates = list(range(train_start, start_idx))
        else:  # pragma: no cover - validated by type hints
            msg = f"unsupported mode {mode!r}"
            raise ValueError(msg)

        train_idx: list[int] = []
        for idx in candidates:
            ts = timeline[idx]
            skip = False
            for embargo_start, embargo_end in embargo_windows:
                if embargo_start <= ts <= embargo_end:
                    skip = True
                    break
            if not skip:
                train_idx.append(idx)

        splits.append((train_idx, test_idx))

        embargo_end = test_end + embargo if embargo else test_end
        embargo_windows.append((test_start, embargo_end))

    return splits


__all__ = ["time_kfold"]

