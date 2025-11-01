"""Purged walk-forward cross validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class SplitWindow:
    """Container for a single purged train/test window."""

    train_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class PurgedSplit:
    """Configuration for generating purged walk-forward splits."""

    n_splits: int
    embargo_frac: float = 0.01

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.embargo_frac < 0:
            raise ValueError("embargo_frac must be non-negative")

    def split(self, n: int) -> Iterator[SplitWindow]:
        if n < self.n_splits:
            raise ValueError("total samples must be >= n_splits")

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1

        indices = np.arange(n)
        embargo_width = int(np.ceil(self.embargo_frac * n))

        test_start = 0
        for size in fold_sizes:
            test_end = test_start + size
            test_indices = indices[test_start:test_end]

            mask = np.ones(n, dtype=bool)
            left = max(0, test_start - embargo_width)
            right = min(n, test_end + embargo_width)
            mask[left:right] = False

            train_indices = indices[mask]

            yield SplitWindow(train_indices=train_indices, test_indices=test_indices)
            test_start = test_end


class WalkForwardPurged:
    """Convenience wrapper that exposes a scikit-learn like API."""

    def __init__(self, n_splits: int = 6, embargo_frac: float = 0.01) -> None:
        self.n_splits = int(n_splits)
        self.embargo_frac = float(embargo_frac)
        self._splitter = PurgedSplit(self.n_splits, self.embargo_frac)

    def split(self, n: int) -> Iterator[SplitWindow]:
        """Yield purged train/test indices for ``n`` samples."""

        yield from self._splitter.split(n)


def walk_forward_purged(
    total_samples: int,
    n_splits: int,
    embargo_frac: float = 0.0,
) -> Iterator[SplitWindow]:
    """Yield purged walk-forward splits as :class:`SplitWindow` objects."""

    splitter = WalkForwardPurged(n_splits=n_splits, embargo_frac=embargo_frac)
    yield from splitter.split(total_samples)


def purged_kfold(total_samples: int, n_splits: int, embargo_frac: float = 0.0) -> Iterator[SplitWindow]:
    """Backward compatible alias for :func:`walk_forward_purged`."""

    yield from walk_forward_purged(total_samples, n_splits, embargo_frac)


__all__ = [
    "SplitWindow",
    "PurgedSplit",
    "WalkForwardPurged",
    "walk_forward_purged",
    "purged_kfold",
]
