"""SOTA Rough-Path Directed-Laplacian & Counterfactual Mirror-Descent Alpha Package."""

from .sota_engine import (
    SOTABacktestResult,
    SOTAConfig,
    SOTALeadLagEngine,
    compute_directed_laplacian_signal,
)

__all__ = [
    "SOTABacktestResult",
    "SOTAConfig",
    "SOTALeadLagEngine",
    "compute_directed_laplacian_signal",
]
