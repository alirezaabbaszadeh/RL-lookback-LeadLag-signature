"""Signature feature computation using deterministic path moments."""

from __future__ import annotations

from typing import Iterable
import numpy as np


def compute_signature_features(series: Iterable[float], depth: int) -> np.ndarray:
    """Compute deterministic path signature features for a 1D return series."""
    data = np.asarray(list(series), dtype=float)
    if data.size == 0 or depth <= 0:
        return np.zeros(depth, dtype=float)

    clean = np.nan_to_num(data, nan=0.0)
    L = max(1, clean.size)
    mean = float(np.mean(clean))
    std = float(np.std(clean)) if clean.size > 1 else 1.0
    std = 1.0 if std == 0.0 or np.isnan(std) else std

    normed = (clean - mean) / std
    cum_path = np.cumsum(normed) / np.sqrt(float(L))

    features = np.zeros(depth, dtype=float)
    features[0] = mean
    if depth > 1:
        features[1] = std
    if depth > 2:
        features[2] = float(cum_path[-1])  # total displacement
    if depth > 3:
        features[3] = float(np.sum(normed ** 2) / float(L))  # realized quadratic variation
    if depth > 4:
        for k in range(4, depth):
            power = k - 2
            features[k] = float(np.mean(cum_path ** power))

    return features


__all__ = ["compute_signature_features"]
