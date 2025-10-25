from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data.loader_parquet import ParquetDataLoader


def test_parquet_preserves_utc_timestamp(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    symbol = "BTC-USD"
    start_one = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    end_one = start_one + timedelta(minutes=1)
    start_two = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
    end_two = start_two + timedelta(minutes=1)
    table = pa.table(
        {
            "symbol": pa.array([symbol, symbol], type=pa.string()),
            "open": pa.array([1.0, 2.0], type=pa.float64()),
            "high": pa.array([1.5, 2.5], type=pa.float64()),
            "low": pa.array([0.5, 1.5], type=pa.float64()),
            "close": pa.array([1.2, 2.2], type=pa.float64()),
            "volume": pa.array([10.0, 20.0], type=pa.float64()),
            "start": pa.array([start_one, start_two], type=pa.timestamp("us", tz="UTC")),
            "end": pa.array([end_one, end_two], type=pa.timestamp("us", tz="UTC")),
        }
    )
    dataset = tmp_path / "candles.parquet"
    pq.write_table(table, dataset)

    loader = ParquetDataLoader(dataset)
    events = loader.load(symbol, start_one - timedelta(minutes=1), end_two + timedelta(minutes=1))

    assert [event.start for event in events] == [start_one, start_two]
    assert [event.end for event in events] == [end_one, end_two]
    for event in events:
        assert event.start.tzinfo is not None
        assert event.end.tzinfo is not None
        assert event.start.tzinfo.utcoffset(event.start) == timedelta(0)
        assert event.end.tzinfo.utcoffset(event.end) == timedelta(0)


def test_csv_naive_timestamps_are_promoted_to_utc(tmp_path) -> None:
    symbol = "ETH-USD"
    start = datetime(2024, 2, 2, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(minutes=5)
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "\n".join(
            [
                "symbol,open,high,low,close,volume,start,end",
                ",".join(
                    [
                        symbol,
                        "100.0",
                        "101.0",
                        "99.0",
                        "100.5",
                        "42.0",
                        start.replace(tzinfo=None).isoformat(),
                        end.replace(tzinfo=None).isoformat(),
                    ]
                ),
            ]
        ),
        encoding="utf-8",
    )

    loader = ParquetDataLoader(csv_path)
    events = loader.load(symbol, start - timedelta(minutes=10), end + timedelta(minutes=10))

    assert len(events) == 1
    event = events[0]
    assert event.start == start
    assert event.end == end
    assert event.start.tzinfo is not None
    assert event.end.tzinfo is not None
    assert event.start.tzinfo.utcoffset(event.start) == timedelta(0)
    assert event.end.tzinfo.utcoffset(event.end) == timedelta(0)
