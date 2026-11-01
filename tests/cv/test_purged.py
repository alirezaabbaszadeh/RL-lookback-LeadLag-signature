"""Unit tests for the purged walk-forward splitter."""

from __future__ import annotations

import numpy as np

from leadlag.cv.purged import walk_forward_purged


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
    embargo_frac = 0.05
    min_embargo = 4
    splits = list(walk_forward_purged(total_samples=total_samples, n_splits=n_splits, embargo_frac=embargo_frac, min_embargo=min_embargo))
    assert len(splits) == n_splits
    for split in splits:
        assert split.embargo == min_embargo
        test_start = split.test_start
        test_end = split.test_end
        for idx in split.train_indices:
            assert not (test_start - min_embargo <= idx < test_start)
            assert not (test_end <= idx < test_end + min_embargo)


def test_purged_kfold_propagates_min_embargo() -> None:
    from leadlag.cv.purged import purged_kfold
    splits = list(purged_kfold(total_samples=20, n_splits=4, embargo_frac=0.0, min_embargo=3))
    assert len(splits) == 4
    for split in splits:
        assert split.embargo == 3


def test_min_embargo_negative_raises_value_error() -> None:
    import pytest
    with pytest.raises(ValueError, match="min_embargo must be non-negative"):
        list(walk_forward_purged(total_samples=20, n_splits=4, min_embargo=-1))


def test_purged_kfold_default_min_embargo_zero() -> None:
    from leadlag.cv.purged import purged_kfold
    splits = list(purged_kfold(total_samples=20, n_splits=4))
    assert len(splits) == 4
    for split in splits:
        assert split.embargo >= 0
