"""Synthetic trading environment placeholder.

This module provides a minimal, reproducible market simulator that generates
stochastic returns while respecting t->t+1 execution semantics. The goal is not
high fidelity but to allow the GPU-first training pipeline to run end-to-end in
continuous integration and documentation contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


@dataclass
class TradeMetrics:
    pnl: float
    turnover: float
    exposure: float
    env_steps: int


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

    def simulate_returns(self, total_steps: int, n_envs: int = 1) -> np.ndarray:
        """Simulate ``total_steps`` of fractional returns across ``n_envs`` envs."""

        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        n_envs = max(1, int(n_envs))

        steps_per_env, remainder = divmod(total_steps, n_envs)
        counts: List[int] = [steps_per_env + (1 if idx < remainder else 0) for idx in range(n_envs)]

        sequences: List[np.ndarray] = []
        drift = 0.0002
        vol = 0.01
        costs = (self.fee_bps + self.slippage_bps) / 10000.0

        for count in counts:
            if count <= 0:
                continue
            shocks = self.rng.normal(loc=drift, scale=vol, size=count)
            signed_turnover = self.rng.uniform(-1, 1, size=count)
            turnover = np.abs(signed_turnover)
            pnl = shocks - turnover * costs
            sequences.append(pnl)

        if not sequences:
            return np.zeros(0, dtype=float)

        # Interleave the sequences to mimic vectorised environment stepping.
        interleaved: List[float] = []
        max_len = max(len(seq) for seq in sequences)
        for step_idx in range(max_len):
            for seq in sequences:
                if step_idx < len(seq):
                    interleaved.append(float(seq[step_idx]))

        return np.asarray(interleaved, dtype=float)

    def summarize_trades(self, returns: Iterable[float]) -> TradeMetrics:
        arr = np.asarray(list(returns), dtype=float)
        turnover = float(np.mean(np.abs(arr)))
        exposure = float(np.mean(np.maximum.accumulate(np.abs(arr))))
        pnl = float(arr.sum())
        env_steps = int(arr.size)
        return TradeMetrics(pnl=pnl, turnover=turnover, exposure=exposure, env_steps=env_steps)


__all__ = ["SyntheticTradingEnvironment", "TradeMetrics"]
