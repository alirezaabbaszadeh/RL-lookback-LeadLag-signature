from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .timeguards import NoPeekError, assert_no_peek, ensure_strictly_increasing

__all__ = ["inspect_feature_frame"]


def _normalise_timezone(timestamps: pd.DatetimeIndex) -> str | None:
    tz_info = timestamps.tz
    if tz_info is None:
        return None
    tz_name = getattr(tz_info, "key", None) or getattr(tz_info, "zone", None)
    return str(tz_name) if tz_name else str(tz_info)


def inspect_feature_frame(
    frame: pd.DataFrame,
    *,
    decision_times: Sequence[pd.Timestamp] | pd.Index | None = None,
    feature_time_col: str = "t_feat",
    min_gap: pd.Timedelta | None = pd.Timedelta("1ns"),
    allow_irregular: bool = False,
) -> Mapping[str, object]:
    """Validate the canonical feature-frame contract and return timing metadata."""

    if frame is None or frame.empty:
        return {
            "checked_rows": 0,
            "min_lag_ns": None,
            "max_lag_ns": None,
            "tz": None,
            "freq_hint": None,
        }

    if feature_time_col not in frame:
        raise NoPeekError(f"Feature frame missing required column: {feature_time_col}")

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise NoPeekError("Feature frame index must be a DatetimeIndex")

    ensure_strictly_increasing(frame.index, name="feature_frame_index")

    feature_times = frame[feature_time_col]
    if not isinstance(feature_times, (pd.Series, pd.Index)):
        feature_times = pd.Series(feature_times)
    feature_times = pd.DatetimeIndex(feature_times.dropna())
    ensure_strictly_increasing(feature_times, name=feature_time_col)

    index_view = frame.index[: len(feature_times)]
    if not index_view.equals(feature_times):
        raise NoPeekError("Feature frame index must align with feature timestamps")

    meta: dict[str, object] = {
        "checked_rows": int(feature_times.size),
        "min_lag_ns": None,
        "max_lag_ns": None,
        "tz": _normalise_timezone(feature_times),
        "freq_hint": None,
    }

    if feature_times.size > 1:
        deltas = np.diff(feature_times.view("i8"))
        unique_deltas = np.unique(deltas)
        if unique_deltas.size > 1 and not allow_irregular:
            raise NoPeekError(
                "Feature frame has irregular sampling intervals; set allow_irregular=True to permit"
            )
        if unique_deltas.size == 1:
            try:
                inferred = feature_times.inferred_freq  # type: ignore[attr-defined]
            except (AttributeError, ValueError, TypeError):
                inferred = None
            meta["freq_hint"] = inferred

    if decision_times is not None and feature_times.size:
        decision_index = pd.DatetimeIndex(decision_times)
        assert_no_peek(frame, decision_index, feature_time_col=feature_time_col, min_gap=min_gap)

        comparison = feature_times[: len(decision_index)]
        aligned = decision_index[: len(comparison)]
        if aligned.size:
            lags = aligned.view("i8") - comparison.view("i8")
            meta["min_lag_ns"] = int(lags.min())
            meta["max_lag_ns"] = int(lags.max())

    return meta
