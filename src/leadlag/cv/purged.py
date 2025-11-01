"""Purged walk-forward cross validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedSplit:
    """Container for a single purged split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    embargo: int = 0

    def __post_init__(self) -> None:
        """Normalise storage to integer ``ndarray`` instances."""

        self.train_indices = np.asarray(self.train_indices, dtype=int)
        self.test_indices = np.asarray(self.test_indices, dtype=int)

    @property
    def test_start(self) -> int | None:
        if self.test_indices.size == 0:
            return None
        return int(self.test_indices[0])

    @property
    def test_end(self) -> int | None:
        if self.test_indices.size == 0:
            return None
        return int(self.test_indices[-1]) + 1


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

    The splits are constructed by iteratively selecting contiguous test windows
    whose lengths differ by at most one sample.  Any remainder after dividing the
    dataset across ``n_splits`` windows is distributed across the earliest folds,
    ensuring that every index appears in exactly one test window.  An embargo is
    optionally applied around each window: any training index that lies within
    ``embargo`` steps immediately before ``test_start`` or immediately after
    ``test_end`` is removed.  The embargo therefore depends on the absolute
    position of each test slice and never trims unrelated regions of the training
    set.
    """
    if n_splits <= 1:
        raise ValueError("n_splits must be greater than 1")
    if total_samples < n_splits:
        raise ValueError("total_samples must be greater than or equal to n_splits")
    if embargo_frac < 0:
        raise ValueError("embargo_frac must be non-negative")

    indices = np.arange(total_samples)
    base_size = total_samples // n_splits
    remainder = total_samples % n_splits

    test_start = 0
    for split in range(n_splits):
        test_size = base_size + (1 if split < remainder else 0)
        test_end = test_start + test_size
        test_indices = indices[test_start:test_end]

        embargo = int(np.ceil(test_size * embargo_frac))
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        train_indices = _apply_embargo(train_indices, test_start, test_end, embargo)

        yield PurgedSplit(
            train_indices=train_indices,
            test_indices=test_indices,
            embargo=int(embargo),
        )
        test_start = test_end


def purged_kfold(total_samples: int, n_splits: int, embargo_frac: float = 0.0) -> Iterator[PurgedSplit]:
    return walk_forward_purged(total_samples, n_splits, embargo_frac)


__all__ = ["PurgedSplit", "walk_forward_purged", "purged_kfold"]
