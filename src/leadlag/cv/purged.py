"""Purged walk-forward cross validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _apply_embargo(
    train_indices: np.ndarray,
    test_start: int,
    test_end: int,
    embargo: int,
) -> np.ndarray:
    """Remove indices that fall within the embargo around the test window."""

    if embargo <= 0:
        return train_indices

    left_start = test_start - embargo
    left_end = test_start
    right_start = test_end
    right_end = test_end + embargo

    mask = np.ones_like(train_indices, dtype=bool)
    if left_start < left_end:
        mask &= ~((train_indices >= left_start) & (train_indices < left_end))
    if right_start < right_end:
        mask &= ~((train_indices >= right_start) & (train_indices < right_end))

    return train_indices[mask]


def walk_forward_purged(
    total_samples: int,
    n_splits: int,
    embargo_frac: float = 0.0,
) -> Iterator[PurgedSplit]:
    """Yield purged walk-forward train/test splits.

    The splits are constructed by iteratively selecting contiguous test windows of
    equal length (``total_samples // n_splits``).  An embargo is optionally applied
    around each window: any training index that lies within ``embargo`` steps
    immediately before ``test_start`` or immediately after ``test_end`` is removed.
    The embargo therefore depends on the absolute position of each test slice and
    never trims unrelated regions of the training set.
    """
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")
    indices = np.arange(total_samples)
    test_size = total_samples // n_splits
    embargo = int(np.ceil(test_size * embargo_frac))
    for split in range(n_splits):
        test_start = split * test_size
        test_end = test_start + test_size
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        train_indices = _apply_embargo(train_indices, test_start, test_end, embargo)
        yield PurgedSplit(train_indices=train_indices, test_indices=test_indices)


def purged_kfold(total_samples: int, n_splits: int, embargo_frac: float = 0.0) -> Iterator[PurgedSplit]:
    return walk_forward_purged(total_samples, n_splits, embargo_frac)


__all__ = ["PurgedSplit", "walk_forward_purged", "purged_kfold"]
