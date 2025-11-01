"""Unit tests for the purged walk-forward splitter."""

import numpy as np

from leadlag.cv.purged import WalkForwardPurged, walk_forward_purged


def test_no_overlap_with_embargo() -> None:
    splitter = WalkForwardPurged(n_splits=5, embargo_frac=0.02)
    for train_indices, test_indices in (
        (split.train_indices, split.test_indices) for split in splitter.split(1_000)
    ):
        assert not np.intersect1d(train_indices, test_indices).size


def test_embargo_removes_indices_close_to_test_window() -> None:
    total_samples = 64
    embargo_frac = 0.1
    splitter = WalkForwardPurged(n_splits=4, embargo_frac=embargo_frac)
    embargo_width = int(np.ceil(total_samples * embargo_frac))

    for split in splitter.split(total_samples):
        if split.test_indices.size == 0:
            continue
        buffer_start = max(0, split.test_indices.min() - embargo_width)
        buffer_end = min(total_samples, split.test_indices.max() + 1 + embargo_width)

        for idx in split.train_indices:
            assert idx < buffer_start or idx >= buffer_end


def test_walk_forward_purged_covers_all_indices_with_remainder() -> None:
    total_samples = 10
    n_splits = 3
    splits = list(walk_forward_purged(total_samples=total_samples, n_splits=n_splits))

    seen_test_indices = []
    for split in splits:
        if split.test_indices.size > 1:
            assert np.all(np.diff(split.test_indices) == 1)
        seen_test_indices.extend(split.test_indices.tolist())

    assert sorted(seen_test_indices) == list(range(total_samples))

    combined = np.concatenate(
        [
            np.concatenate([split.train_indices, split.test_indices])
            for split in splits
        ]
    )
    assert set(combined.tolist()) == set(range(total_samples))
