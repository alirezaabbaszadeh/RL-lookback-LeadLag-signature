from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leadlag.pipelines import run_full_suite
from leadlag.utils import NoPeekError
from leadlag.utils.feature_frame_guard import inspect_feature_frame


def test_construct_feature_frame_wide_layout() -> None:
    price_index = pd.date_range("2024-01-01", periods=6, freq="h")
    returns = pd.DataFrame(
        {
            "AssetA": [0.1, 0.2, -0.05, 0.03, 0.01],
            "AssetB": [0.05, -0.02, 0.04, 0.01, -0.03],
        },
        index=price_index[1:],
    )
    stack = {
        "returns": returns.to_numpy(dtype=float),
        "time_channel": np.linspace(0.0, 1.0, num=returns.shape[0], dtype=float),
    }

    frame = run_full_suite._construct_feature_frame(price_index, returns, stack)

    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.equals(price_index[:-1])
    expected_t = pd.Series(price_index[:-1], index=price_index[:-1], name="t_feat")
    pd.testing.assert_series_equal(frame["t_feat"], expected_t)
    assert sorted(col for col in frame.columns if col.startswith("returns::")) == [
        "returns::AssetA",
        "returns::AssetB",
    ]

    meta = inspect_feature_frame(
        frame,
        decision_times=price_index[1:],
        feature_time_col="t_feat",
    )
    assert meta["checked_rows"] == len(frame)
    assert meta["min_lag_ns"] is not None
    assert meta["max_lag_ns"] is not None


def test_inspect_feature_frame_rejects_irregular_sampling() -> None:
    irregular_index = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 02:00"])
    frame = pd.DataFrame(
        {
            "t_feat": irregular_index,
            "returns::AssetA": [0.1, 0.2, -0.3],
        },
        index=irregular_index,
    )
    decisions = irregular_index + pd.to_timedelta(["1h", "1h", "1h"])

    with pytest.raises(NoPeekError):
        inspect_feature_frame(
            frame,
            decision_times=decisions,
            feature_time_col="t_feat",
        )

    meta = inspect_feature_frame(
        frame,
        decision_times=decisions,
        feature_time_col="t_feat",
        allow_irregular=True,
    )
    assert meta["freq_hint"] is None
