from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from leadlag.utils import NoPeekError, ensure_strictly_increasing


def test_ensure_strictly_increasing_rejects_mixed_timezones() -> None:
    mixed = [
        pd.Timestamp("2024-01-01 00:00"),
        pd.Timestamp("2024-01-01 00:15", tz="UTC"),
    ]
    with pytest.raises(NoPeekError):
        ensure_strictly_increasing(mixed, name="mixed_series")


def test_ensure_strictly_increasing_detects_dst_backwards_step() -> None:
    tz = pd.Timestamp("2020-11-01", tz="America/New_York").tz
    assert tz is not None
    times = [
        tz.localize(dt.datetime(2020, 11, 1, 0, 30), is_dst=True),
        tz.localize(dt.datetime(2020, 11, 1, 1, 30), is_dst=False),
        tz.localize(dt.datetime(2020, 11, 1, 1, 0), is_dst=False),
    ]
    with pytest.raises(NoPeekError):
        ensure_strictly_increasing(times, name="dst_series")


def test_ensure_strictly_increasing_detects_duplicates() -> None:
    duplicates = pd.to_datetime(["2024-02-01", "2024-02-02", "2024-02-02"])
    with pytest.raises(NoPeekError):
        ensure_strictly_increasing(duplicates, name="duplicate_series")
