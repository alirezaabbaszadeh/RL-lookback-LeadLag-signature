"""Signature feature extraction helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

try:
    import iisignature

    SIGNATURE_AVAILABLE = True
except ImportError:  # pragma: no cover - handled via availability flag
    SIGNATURE_AVAILABLE = False


@dataclass
class SignatureConfig:
    order: int = 2
    scaling_method: str = "mean-centering"
    sig_method: str = "custom"
    allow_fallback: bool = True


class SignatureExtractor:
    """Compute lead-lag signature-based measures for data pairs."""

    def __init__(self, config: Optional[SignatureConfig] = None, cache_size: int = 64) -> None:
        self.config = config or SignatureConfig()
        if not SIGNATURE_AVAILABLE and not self.config.allow_fallback:
            raise ImportError("iisignature package is required for signature-based methods")
        self.cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[bytes, float]" = OrderedDict()

    def compute(self, data_pair: np.ndarray) -> float:
        """Return signature-based lead-lag score for a 2D array of shape (n_samples, 2)."""
        if data_pair.ndim != 2 or data_pair.shape[1] != 2:
            raise ValueError("data_pair must have shape (n_samples, 2)")

        contiguous = np.ascontiguousarray(data_pair, dtype=float)
        cache_key = contiguous.tobytes()
        if self.cache_size > 0:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._cache.move_to_end(cache_key)
                return cached

        standardized = self._standardize(contiguous)
        L = max(1, standardized.shape[0])

        if SIGNATURE_AVAILABLE:
            # Construct zero-anchored cumulative path scaled by 1/sqrt(L) for scale invariance
            cum_path = np.vstack([
                np.zeros((1, 2), dtype=float),
                np.cumsum(standardized, axis=0) / np.sqrt(float(L)),
            ])
            signature = iisignature.sig(cum_path, self.config.order, 1)

            if self.config.sig_method == "levy":
                value = float((signature[1][1] - signature[1][2]) * 0.5)
            else:
                value = float(signature[1][1] - signature[1][2])
        else:
            # Discrete Lévy area fallback:
            # A_{1,2} = 0.5 * sum_{1 <= u < v <= L} (r_{u,1} * r_{v,2} - r_{u,2} * r_{v,1}) / L
            r0 = standardized[:, 0]
            r1 = standardized[:, 1]
            outer_diff = np.outer(r0, r1) - np.outer(r1, r0)
            levy_area = 0.5 * float(np.sum(np.triu(outer_diff, k=1))) / float(L)
            value = float(levy_area) if self.config.sig_method == "levy" else float(2.0 * levy_area)

        if self.cache_size > 0:
            self._cache[cache_key] = value
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value

    def compute_batch(self, batch: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Compute signatures for a batch of 2D arrays."""
        if isinstance(batch, np.ndarray) and batch.ndim == 3 and batch.shape[2] == 2:
            iterable: Iterable[np.ndarray] = [batch[i] for i in range(batch.shape[0])]
        else:
            iterable = batch  # type: ignore[assignment]

        results: list[float] = []
        for item in iterable:
            results.append(self.compute(np.asarray(item)))
        return np.asarray(results, dtype=float)

    def _standardize(self, array: np.ndarray) -> np.ndarray:
        if array.ndim != 2:
            raise ValueError("Input array must be 2D (n_samples, n_features)")

        mean = np.nanmean(array, axis=0)
        std = np.nanstd(array, axis=0, ddof=1)
        std = np.where((std == 0) | np.isnan(std), 1.0, std)
        arr_clean = np.nan_to_num(array, nan=0.0)

        if self.config.scaling_method == "by_std":
            return (arr_clean - mean) / std
        if self.config.scaling_method == "mean-centering":
            # Normalize variance while preserving zero-mean centering
            return (arr_clean - mean) / std
        return arr_clean / std
