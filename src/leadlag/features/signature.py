"""Signature feature computation stubs."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


@lru_cache(maxsize=16)
def signature_basis(depth: int) -> np.ndarray:
    """Return a deterministic basis matrix for a given signature depth."""

    rng = np.random.default_rng(depth)
    return rng.standard_normal((depth, depth))


def compute_signature_features(series: Iterable[float], depth: int) -> np.ndarray:
    data = np.asarray(list(series), dtype=float)
    if data.size == 0:
        return np.zeros(depth)
    basis = signature_basis(depth)
    # Simple linear projection as a lightweight placeholder for true signatures.
    padded = np.pad(data, (0, max(0, depth - data.size)))[:depth]
    return basis @ padded


__all__ = ["compute_signature_features"]
