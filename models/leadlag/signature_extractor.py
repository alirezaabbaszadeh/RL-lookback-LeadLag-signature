"""Signature feature extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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


class SignatureExtractor:
    """Compute lead-lag signature-based measures for data pairs."""

    def __init__(self, config: Optional[SignatureConfig] = None) -> None:
        self.config = config or SignatureConfig()
        if not SIGNATURE_AVAILABLE:
            raise ImportError("iisignature package is required for signature-based methods")

    def compute(self, data_pair: np.ndarray) -> float:
        """Return signature-based lead-lag score for a 2D array of shape (n_samples, 2)."""
        if data_pair.ndim != 2 or data_pair.shape[1] != 2:
            raise ValueError("data_pair must have shape (n_samples, 2)")

        standardized = self._standardize(data_pair)
        signature = iisignature.sig(standardized, self.config.order, 1)

        if self.config.sig_method == "levy":
            return (signature[1][1] - signature[1][2]) * 0.5
        return signature[1][1] - signature[1][2]

    def _standardize(self, array: np.ndarray) -> np.ndarray:
        if array.ndim != 2:
            raise ValueError("Input array must be 2D (n_samples, n_features)")

        mean = np.mean(array, axis=0)
        std = np.std(array, axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)

        if self.config.scaling_method == 'by_std':
            return (array - mean) / std
        if self.config.scaling_method == 'mean-centering':
            return array - mean
        return array / std

