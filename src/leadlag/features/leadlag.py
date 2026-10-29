"""Lead-lag transform implementation."""

from __future__ import annotations

from typing import Iterable
import numpy as np


def compute_lead_lag(series: Iterable[float]) -> np.ndarray:
    """Construct canonical (lead, lag) 2D path from a 1D sequence."""
    data = np.asarray(list(series), dtype=float)
    if data.size < 2:
        return np.zeros((2, max(1, data.size)), dtype=float)
    clean = np.nan_to_num(data, nan=0.0)
    lead = clean[1:]
    lag = clean[:-1]
    return np.vstack([lead, lag])


__all__ = ["compute_lead_lag"]
