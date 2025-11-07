"""Utilities to guard against temporal leakage in feature pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = ["NoPeekError", "assert_no_peek", "ensure_strictly_increasing"]


class NoPeekError(ValueError):
    """Raised when feature timestamps peek at or exceed their execution times."""


@dataclass(frozen=True)
class _AlignmentView:
    feature_times: pd.DatetimeIndex
    decision_times: pd.DatetimeIndex


def _coerce_datetime_index(data: object, *, name: str) -> pd.DatetimeIndex:
    if isinstance(data, pd.Series):
        values = data.dropna().to_numpy()
    elif isinstance(data, pd.DataFrame):
        raise TypeError(
            "DataFrame provided where a column selection is required for timing checks"
        )
    elif isinstance(data, (pd.Index, np.ndarray)):
        values = np.asarray(data)
    elif isinstance(data, Iterable):
        values = list(data)
    else:
        raise TypeError(f"Unsupported type for {name}: {type(data)!r}")

    try:
        index = pd.DatetimeIndex(values)
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive
        raise NoPeekError(f"{name} could not be coerced to datetimes") from exc

    if index.isna().any():
        index = index[~index.isna()]

    return index


def _normalise_feature_times(
    features: Sequence[pd.Timestamp] | pd.Series | pd.Index | pd.DataFrame,
    *,
    feature_time_col: str | None,
) -> pd.DatetimeIndex:
    if feature_time_col is not None:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("feature_time_col specified but features is not a DataFrame")
        if feature_time_col not in features:
            raise NoPeekError(f"Missing feature time column: {feature_time_col}")
        column = features[feature_time_col]
        if not isinstance(column, (pd.Series, pd.Index)):
            column = pd.Series(column)
        return _coerce_datetime_index(column, name=feature_time_col)

    return _coerce_datetime_index(features, name="feature_times")


def _build_alignment(
    features: Sequence[pd.Timestamp] | pd.Series | pd.Index | pd.DataFrame,
    decisions: Sequence[pd.Timestamp] | pd.Series | pd.Index,
    *,
    feature_time_col: str | None,
) -> _AlignmentView:
    feature_times = _normalise_feature_times(features, feature_time_col=feature_time_col)
    decision_times = _coerce_datetime_index(decisions, name="decision_times")

    if feature_times.empty or decision_times.empty:
        return _AlignmentView(feature_times=feature_times, decision_times=decision_times)

    diff = len(feature_times) - len(decision_times)
    if abs(diff) > 1:
        raise NoPeekError(
            "Feature/decision sequences have incompatible lengths for alignment: "
            f"features={len(feature_times)} decisions={len(decision_times)}"
        )

    if diff > 0:
        feature_times = feature_times[: len(decision_times)]
    elif diff < 0:
        decision_times = decision_times[: len(feature_times)]

    return _AlignmentView(feature_times=feature_times, decision_times=decision_times)


def ensure_strictly_increasing(index: Sequence[pd.Timestamp] | pd.Index, *, name: str) -> None:
    dt_index = _coerce_datetime_index(index, name=name)
    if dt_index.empty:
        return
    deltas = np.diff(dt_index.view("i8"))
    if np.any(deltas <= 0):
        raise NoPeekError(f"{name} must be strictly increasing without duplicates")


def assert_no_peek(
    features: Sequence[pd.Timestamp] | pd.Series | pd.Index | pd.DataFrame,
    decisions: Sequence[pd.Timestamp] | pd.Series | pd.Index,
    *,
    feature_time_col: str | None = None,
    min_gap: pd.Timedelta | None = None,
) -> None:
    alignment = _build_alignment(
        features,
        decisions,
        feature_time_col=feature_time_col,
    )

    feature_times = alignment.feature_times
    decision_times = alignment.decision_times

    if feature_times.empty or decision_times.empty:
        return

    ensure_strictly_increasing(feature_times, name="feature_times")
    ensure_strictly_increasing(decision_times, name="decision_times")

    comparison = feature_times.to_series().reset_index(drop=True)
    target = decision_times.to_series().reset_index(drop=True)
    if (comparison >= target).any():
        raise NoPeekError(
            "Feature timestamps must be strictly earlier than execution times"
        )

    if min_gap is not None:
        if min_gap < pd.Timedelta(0):
            raise ValueError("min_gap must be non-negative")
        gaps = target - comparison
        if (gaps < min_gap).any():
            raise NoPeekError(
                f"Feature timestamps violate the minimum lag requirement of {min_gap}."
            )
