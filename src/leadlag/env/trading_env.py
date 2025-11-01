"""Synthetic trading environment placeholder.

This module provides a minimal, reproducible market simulator that generates
stochastic returns while respecting t->t+1 execution semantics. The goal is not
high fidelity but to allow the GPU-first training pipeline to run end-to-end in
continuous integration and documentation contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TradeMetrics:
    pnl: float
    turnover: float
    exposure: float


class SyntheticTradingEnvironment:
    """Generate synthetic return paths for experimentation."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        fee_bps: float,
        slippage_bps: float,
        seed: int,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.rng = np.random.default_rng(seed)

    def simulate_returns(self, steps: int) -> np.ndarray:
        """Simulate ``steps`` of fractional returns."""

        drift = 0.0002
        vol = 0.01
        shocks = self.rng.normal(loc=drift, scale=vol, size=steps)
        costs = (self.fee_bps + self.slippage_bps) / 10000.0
        signed_turnover = self.rng.uniform(-1, 1, size=steps)
        turnover = np.abs(signed_turnover)
        pnl = shocks - turnover * costs
        return pnl

    def summarize_trades(self, returns: Iterable[float]) -> TradeMetrics:
        arr = np.asarray(list(returns), dtype=float)
        turnover = float(np.mean(np.abs(arr)))
        exposure = float(np.mean(np.maximum.accumulate(np.abs(arr))))
        pnl = float(arr.sum())
        return TradeMetrics(pnl=pnl, turnover=turnover, exposure=exposure)


__all__ = ["SyntheticTradingEnvironment", "TradeMetrics"]
