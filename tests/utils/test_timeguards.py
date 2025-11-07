import pandas as pd
import pytest

from leadlag.utils import NoPeekError, assert_no_peek, ensure_strictly_increasing


def _make_index(start: str = "2025-01-01", periods: int = 4, freq: str = "1h") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq)


def test_assert_no_peek_accepts_monotonic_sequences():
    feature_times = _make_index(periods=4)
    decision_times = feature_times.shift(1, freq="1h")
    assert_no_peek(feature_times[:-1], decision_times[1:])


def test_assert_no_peek_raises_on_equal_or_future_times():
    feature_times = _make_index(periods=3)
    decision_times = feature_times  # identical => should raise
    with pytest.raises(NoPeekError):
        assert_no_peek(feature_times, decision_times)

    future_decisions = feature_times - pd.Timedelta(minutes=10)
    with pytest.raises(NoPeekError):
        assert_no_peek(feature_times, future_decisions)


def test_assert_no_peek_respects_min_gap_requirement():
    feature_times = _make_index(periods=4)
    decision_times = feature_times + pd.Timedelta(hours=1)
    with pytest.raises(NoPeekError):
        assert_no_peek(feature_times, decision_times, min_gap=pd.Timedelta(hours=2))


def test_ensure_strictly_increasing_detects_duplicates():
    index = _make_index(periods=4)
    duplicated = index.insert(2, index[2])
    with pytest.raises(NoPeekError):
        ensure_strictly_increasing(duplicated, name="duplicated_index")
