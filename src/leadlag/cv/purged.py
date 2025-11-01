"""Purged walk-forward cross validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class PurgedSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _apply_embargo(indices: np.ndarray, embargo: int) -> np.ndarray:
    if embargo <= 0:
        return indices
    mask = np.ones_like(indices, dtype=bool)
    for i in range(len(indices)):
        if i < embargo or i >= len(indices) - embargo:
            mask[i] = False
    return indices[mask]


def walk_forward_purged(
    total_samples: int,
    n_splits: int,
    embargo_frac: float = 0.0,
) -> Iterator[PurgedSplit]:
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
        train_indices = _apply_embargo(train_indices, embargo)
        yield PurgedSplit(train_indices=train_indices, test_indices=test_indices)


def purged_kfold(total_samples: int, n_splits: int, embargo_frac: float = 0.0) -> Iterator[PurgedSplit]:
    return walk_forward_purged(total_samples, n_splits, embargo_frac)


__all__ = ["PurgedSplit", "walk_forward_purged", "purged_kfold"]
