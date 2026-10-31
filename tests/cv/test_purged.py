"""Unit tests for the purged walk-forward splitter and frequency-aware KPI calculations."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from leadlag.cv.purged import purged_kfold, walk_forward_purged
from leadlag.evaluation.finance_kpis import ANNUALIZATION_FACTOR, compute_kpis


def test_embargo_removes_indices_around_each_test_window() -> None:
    total_samples = 12
    n_splits = 3
    embargo_frac = 0.5

    splits = list(walk_forward_purged(total_samples, n_splits, embargo_frac))
    for split in splits:
        test_size = len(split.test_indices)
        embargo = int(np.ceil(test_size * embargo_frac))
        test_start = split.test_start
        test_end = split.test_end

        for idx in split.train_indices:
            assert not (test_start - embargo <= idx < test_start)
            assert not (test_end <= idx < test_end + embargo)


def test_embargo_keeps_remote_training_indices() -> None:
    splits = list(walk_forward_purged(total_samples=12, n_splits=3, embargo_frac=0.5))

    expected_trains = [
        np.array([6, 7, 8, 9, 10, 11]),
        np.array([0, 1, 10, 11]),
        np.array([0, 1, 2, 3, 4, 5]),
    ]

    for split, expected in zip(splits, expected_trains):
        assert np.array_equal(split.train_indices, expected)


def test_walk_forward_purged_covers_all_indices_with_remainder() -> None:
    total_samples = 10
    n_splits = 3
    splits = list(walk_forward_purged(total_samples=total_samples, n_splits=n_splits))

    # Every fold should have contiguous, unique test indices.
    seen_test_indices = []
    for split in splits:
        if len(split.test_indices) > 1:
            assert np.all(np.diff(split.test_indices) == 1)
        seen_test_indices.extend(split.test_indices.tolist())

    assert sorted(seen_test_indices) == list(range(total_samples))

    # The combination of all train/test indices across folds should cover the full range.
    combined = np.concatenate(
        [
            np.concatenate([split.train_indices, split.test_indices])
            for split in splits
        ]
    )
    assert set(combined.tolist()) == set(range(total_samples))


def test_train_and_test_sets_remain_disjoint_with_embargo() -> None:
    total_samples = 24
    n_splits = 4
    embargo_frac = 0.25

    for split in walk_forward_purged(total_samples, n_splits, embargo_frac):
        assert split.embargo == int(np.ceil(len(split.test_indices) * embargo_frac))
        assert set(split.train_indices.tolist()).isdisjoint(
            split.test_indices.tolist()
        )


def test_min_embargo_enforces_minimum_when_embargo_frac_smaller() -> None:
    """Verify min_embargo enforces minimum embargo size even when embargo_frac * test_size is smaller."""
    total_samples = 30
    n_splits = 3
    # Each fold test_size = 10.
    # With embargo_frac = 0.05, ceil(10 * 0.05) = 1.
    # With min_embargo = 4, embargo must equal max(4, 1) = 4.
    embargo_frac = 0.05
    min_embargo = 4

    splits = list(
        walk_forward_purged(
            total_samples=total_samples,
            n_splits=n_splits,
            embargo_frac=embargo_frac,
            min_embargo=min_embargo,
        )
    )

    assert len(splits) == n_splits
    for split in splits:
        assert split.embargo == min_embargo
        test_start = split.test_start
        test_end = split.test_end
        for idx in split.train_indices:
            assert not (test_start - min_embargo <= idx < test_start)
            assert not (test_end <= idx < test_end + min_embargo)


def test_purged_kfold_propagates_min_embargo() -> None:
    """Verify purged_kfold helper passes min_embargo through."""
    splits = list(
        purged_kfold(
            total_samples=20,
            n_splits=4,
            embargo_frac=0.0,
            min_embargo=3,
        )
    )
    assert len(splits) == 4
    for split in splits:
        assert split.embargo == 3


def test_min_embargo_negative_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_embargo must be non-negative"):
        list(walk_forward_purged(total_samples=20, n_splits=4, min_embargo=-1))


def test_compute_kpis_default_and_explicit_periods() -> None:
    """Verify compute_kpis uses ANNUALIZATION_FACTOR (252) by default and respects periods_per_year."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.012])

    # Default daily (252)
    kpis_default = compute_kpis(returns)
    assert "annualized_return" in kpis_default
    assert "sharpe_ratio" in kpis_default
    assert "annualized_volatility" in kpis_default
    assert "max_drawdown" in kpis_default

    # Explicit 252 should match default
    kpis_252 = compute_kpis(returns, periods_per_year=252)
    assert pytest.approx(kpis_default["sharpe_ratio"], rel=1e-7) == kpis_252["sharpe_ratio"]
    assert pytest.approx(kpis_default["annualized_return"], rel=1e-7) == kpis_252["annualized_return"]
    assert pytest.approx(kpis_default["annualized_volatility"], rel=1e-7) == kpis_252["annualized_volatility"]

    # Hourly scaling (8760)
    kpis_hourly = compute_kpis(returns, periods_per_year=8760)
    expected_sharpe_ratio = math.sqrt(8760 / 252)
    assert pytest.approx(kpis_hourly["sharpe_ratio"] / kpis_default["sharpe_ratio"], rel=1e-5) == expected_sharpe_ratio
    assert pytest.approx(kpis_hourly["annualized_volatility"] / kpis_default["annualized_volatility"], rel=1e-5) == expected_sharpe_ratio


def test_compute_kpis_auto_detects_hourly_frequency() -> None:
    """Verify compute_kpis automatically detects hourly series and uses 8760."""
    hourly_index = pd.date_range("2023-01-01", periods=100, freq="h")
    returns = pd.Series(np.random.RandomState(42).normal(0.0005, 0.01, size=100), index=hourly_index)

    kpis_auto = compute_kpis(returns)
    kpis_explicit_8760 = compute_kpis(returns, periods_per_year=8760)

    assert pytest.approx(kpis_auto["sharpe_ratio"], rel=1e-7) == kpis_explicit_8760["sharpe_ratio"]
    assert pytest.approx(kpis_auto["annualized_return"], rel=1e-7) == kpis_explicit_8760["annualized_return"]
    assert pytest.approx(kpis_auto["annualized_volatility"], rel=1e-7) == kpis_explicit_8760["annualized_volatility"]


def test_compute_kpis_auto_detects_hourly_from_deltas_or_attrs() -> None:
    """Verify hourly detection works from raw timestamps without explicit freq or from attrs."""
    # DatetimeIndex without explicit freq but with 1-hour deltas
    timestamps = [pd.Timestamp("2024-01-01 00:00") + pd.Timedelta(hours=i) for i in range(50)]
    returns = pd.Series([0.001] * 50, index=pd.DatetimeIndex(timestamps))
    kpis_delta = compute_kpis(returns)
    kpis_8760 = compute_kpis(returns, periods_per_year=8760)
    assert pytest.approx(kpis_delta["annualized_return"], rel=1e-7) == kpis_8760["annualized_return"]

    # Series with attrs
    daily_returns = pd.Series([0.01, -0.005, 0.02])
    daily_returns.attrs["freq"] = "1h"
    kpis_attr = compute_kpis(daily_returns)
    assert pytest.approx(kpis_attr["annualized_return"], rel=1e-7) == compute_kpis(daily_returns, periods_per_year=8760)["annualized_return"]
