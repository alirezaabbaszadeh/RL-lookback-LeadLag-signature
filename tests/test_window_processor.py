import pytest

try:
    import pandas as pd
except ValueError:
    pytest.skip("Pandas binary incompatible with current numpy", allow_module_level=True)

from models.leadlag import WindowProcessor


def test_window_processor_returns_cached_values():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [2, 3, 4, 5, 6]}, index=dates)
    processor = WindowProcessor()

    first = processor.get_log_returns(data, dates[0], dates[4])
    second = processor.get_log_returns(data, dates[0], dates[4])

    pd.testing.assert_frame_equal(first, second)


def test_window_processor_handles_empty_universe():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    data = pd.DataFrame({"A": [1.0, 1.1, 1.2]}, index=dates)
    processor = WindowProcessor(df_universe=None)

    result = processor.get_log_returns(data, dates[0], dates[-1])

    assert not result.empty
