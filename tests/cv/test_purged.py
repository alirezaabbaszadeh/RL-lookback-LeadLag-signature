"""Unit tests for the purged walk-forward splitter."""

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
        test_start = split.test_indices[0]
        test_end = split.test_indices[-1] + 1

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
