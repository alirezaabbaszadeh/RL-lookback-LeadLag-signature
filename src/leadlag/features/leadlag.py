"""Lead-lag transform placeholder implementation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def compute_lead_lag(series: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(series), dtype=float)
    if data.size < 2:
        return np.zeros((2, max(1, data.size)))
    lead = data[1:]
    lag = data[:-1]
    return np.vstack([lead, lag])


__all__ = ["compute_lead_lag"]
