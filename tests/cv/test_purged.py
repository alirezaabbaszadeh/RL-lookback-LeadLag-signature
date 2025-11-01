"""Unit tests for the purged walk-forward splitter."""

import numpy as np

from leadlag.cv.purged import walk_forward_purged


def test_embargo_removes_indices_around_each_test_window() -> None:
    total_samples = 12
    n_splits = 3
    embargo_frac = 0.5

    splits = list(walk_forward_purged(total_samples, n_splits, embargo_frac))
    test_size = total_samples // n_splits
    embargo = int(np.ceil(test_size * embargo_frac))

    for split in splits:
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
