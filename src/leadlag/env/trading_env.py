"""Synthetic trading environment placeholder.

This module provides a minimal, reproducible market simulator that generates
stochastic returns while respecting t->t+1 execution semantics. The goal is not
high fidelity but to allow the GPU-first training pipeline to run end-to-end in
continuous integration and documentation contexts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class TradeMetrics:
    pnl: float
    turnover: float
    exposure: float
    env_steps: int
    costs: float = 0.0


@dataclass
class TradePath:
    """Container describing a single simulated trading path."""

    returns: np.ndarray
    positions: np.ndarray
    trades: np.ndarray
    costs: np.ndarray


class SyntheticTradingEnvironment:
    """Generate synthetic return paths for experimentation."""

    def __init__(
        self,
        lookback: int,
        horizon: int,
        fee_bps: float,
        slippage_bps: float,
        seed: int,
        max_abs_position: float = 1.0,
        initial_position: float = 0.0,
        allow_short: bool = True,
    ) -> None:
        self.lookback = lookback
        self.horizon = horizon
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.rng = np.random.default_rng(seed)
        self.allow_short = bool(allow_short)
        self.max_abs_position = float(max(0.0, max_abs_position))
        if self.max_abs_position == 0.0:
            self.max_abs_position = 1.0
        self._min_position = -self.max_abs_position if self.allow_short else 0.0
        clipped_initial = np.clip(initial_position, self._min_position, self.max_abs_position)
        self.initial_position = float(clipped_initial)
        self._last_metrics: List[TradeMetrics] = []
        self._last_paths: List[TradePath] = []

    def _normalise_parameter(
        self,
        parameter: Optional[Sequence[Iterable[float]] | Iterable[float] | np.ndarray],
        counts: Sequence[int],
        *,
        name: str,
    ) -> List[Optional[np.ndarray]]:
        if parameter is None:
            return [None for _ in counts]

        if isinstance(parameter, np.ndarray):
            if parameter.ndim == 1:
                if len(counts) != 1:
                    raise ValueError(f"{name} must provide per-env sequences when n_envs > 1")
                return [parameter.astype(float, copy=False)]
            if parameter.ndim == 2:
                if parameter.shape[0] != len(counts):
                    raise ValueError(
                        f"{name} with ndim=2 must have shape[0] == n_envs ({len(counts)})"
                    )
                return [parameter[idx].astype(float, copy=False) for idx in range(parameter.shape[0])]
            raise ValueError(f"{name} expects 1D or 2D numpy array, got ndim={parameter.ndim}")

        if isinstance(parameter, Sequence) and parameter and isinstance(parameter[0], Iterable):
            if len(parameter) != len(counts):
                raise ValueError(f"{name} length must match number of envs ({len(counts)})")
            return [np.asarray(seq, dtype=float) for seq in parameter]  # type: ignore[arg-type]

        if len(counts) != 1:
            raise ValueError(f"{name} must specify per-env sequences when n_envs > 1")
        return [np.asarray(parameter, dtype=float)]  # type: ignore[arg-type]

    def _run_episode(
        self,
        steps: int,
        *,
        actions: Optional[np.ndarray] = None,
        base_returns: Optional[np.ndarray] = None,
        initial_position: Optional[float] = None,
    ) -> Tuple[np.ndarray, TradeMetrics, TradePath]:
        drift = 0.0002
        vol = 0.01
        cost_rate = (self.fee_bps + self.slippage_bps) / 10000.0
        if base_returns is None:
            base_returns = self.rng.normal(loc=drift, scale=vol, size=steps).astype(float)
        else:
            base_returns = np.asarray(base_returns, dtype=float)
            if base_returns.size < steps:
                raise ValueError("base_returns must have at least `steps` elements")
            base_returns = base_returns[:steps]

        episode_actions = None
        if actions is not None:
            episode_actions = np.asarray(actions, dtype=float)

        initial_pos = self.initial_position if initial_position is None else float(initial_position)
        initial_pos = float(np.clip(initial_pos, self._min_position, self.max_abs_position))

        positions: List[float] = []
        trades: List[float] = []
        costs: List[float] = []
        realised_returns: List[float] = []

        prev_position = initial_pos
        current_position = initial_pos
        pending_position = initial_pos

        for step in range(steps):
            current_position = pending_position
            current_position = float(np.clip(current_position, self._min_position, self.max_abs_position))
            trade_size = current_position - prev_position
            trade_cost = abs(trade_size) * cost_rate
            pnl = current_position * float(base_returns[step]) - trade_cost

            positions.append(current_position)
            trades.append(trade_size)
            costs.append(trade_cost)
            realised_returns.append(pnl)

            prev_position = current_position
            if episode_actions is not None and step < episode_actions.size:
                target = float(episode_actions[step])
            else:
                target = float(self.rng.uniform(self._min_position, self.max_abs_position))
            pending_position = float(np.clip(target, self._min_position, self.max_abs_position))

        returns_arr = np.asarray(realised_returns, dtype=float)
        positions_arr = np.asarray(positions, dtype=float)
        trades_arr = np.asarray(trades, dtype=float)
        costs_arr = np.asarray(costs, dtype=float)

        metrics = self.summarize_trades(
            returns_arr,
            positions=positions_arr,
            trades=trades_arr,
            costs=costs_arr,
        )
        path = TradePath(
            returns=returns_arr,
            positions=positions_arr,
            trades=trades_arr,
            costs=costs_arr,
        )
        return returns_arr, metrics, path

    def simulate_returns(
        self,
        total_steps: int,
        n_envs: int = 1,
        *,
        actions: Optional[Sequence[Iterable[float]] | Iterable[float] | np.ndarray] = None,
        base_returns: Optional[Sequence[Iterable[float]] | Iterable[float] | np.ndarray] = None,
        initial_position: Optional[float] = None,
    ) -> np.ndarray:
        """Simulate ``total_steps`` of fractional returns across ``n_envs`` envs."""

        total_steps = int(total_steps)
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        n_envs = max(1, int(n_envs))

        steps_per_env, remainder = divmod(total_steps, n_envs)
        counts: List[int] = [steps_per_env + (1 if idx < remainder else 0) for idx in range(n_envs)]

        sequences: List[np.ndarray] = []
        metrics_list: List[TradeMetrics] = []
        paths_list: List[TradePath] = []

        actions_per_env = self._normalise_parameter(actions, counts, name="actions")
        returns_per_env = self._normalise_parameter(base_returns, counts, name="base_returns")

        for idx, count in enumerate(counts):
            if count <= 0:
                continue
            sequence, metrics, path = self._run_episode(
                count,
                actions=actions_per_env[idx],
                base_returns=returns_per_env[idx],
                initial_position=initial_position,
            )
            sequences.append(sequence)
            metrics_list.append(metrics)
            paths_list.append(path)

        if not sequences:
            return np.zeros(0, dtype=float)

        # Interleave the sequences to mimic vectorised environment stepping.
        interleaved: List[float] = []
        max_len = max(len(seq) for seq in sequences)
        for step_idx in range(max_len):
            for seq in sequences:
                if step_idx < len(seq):
                    interleaved.append(float(seq[step_idx]))

        self._last_metrics = metrics_list
        self._last_paths = paths_list
        return np.asarray(interleaved, dtype=float)

    def summarize_trades(
        self,
        returns: Iterable[float],
        *,
        positions: Optional[Iterable[float]] = None,
        trades: Optional[Iterable[float]] = None,
        costs: Optional[Iterable[float]] = None,
    ) -> TradeMetrics:
        arr = np.asarray(list(returns), dtype=float)
        if arr.size == 0:
            return TradeMetrics(pnl=0.0, turnover=0.0, exposure=0.0, env_steps=0, costs=0.0)

        turnover_arr = np.asarray(list(trades), dtype=float) if trades is not None else None
        if turnover_arr is not None and turnover_arr.size:
            turnover = float(np.mean(np.abs(turnover_arr)))
        else:
            turnover = float(np.mean(np.abs(arr)))

        position_arr = np.asarray(list(positions), dtype=float) if positions is not None else None
        if position_arr is not None and position_arr.size:
            exposure = float(np.mean(np.abs(position_arr)))
        else:
            exposure = float(np.mean(np.maximum.accumulate(np.abs(arr))))

        pnl = float(arr.sum())
        env_steps = int(arr.size)
        cost_arr = np.asarray(list(costs), dtype=float) if costs is not None else None
        total_costs = float(np.sum(cost_arr)) if cost_arr is not None else 0.0
        return TradeMetrics(
            pnl=pnl,
            turnover=turnover,
            exposure=exposure,
            env_steps=env_steps,
            costs=total_costs,
        )

    @property
    def last_metrics(self) -> List[TradeMetrics]:
        """Return metrics for the most recent ``simulate_returns`` call."""

        return list(self._last_metrics)

    @property
    def last_paths(self) -> List[TradePath]:
        """Return trade paths from the most recent simulation."""

        return list(self._last_paths)


__all__ = ["SyntheticTradingEnvironment", "TradeMetrics", "TradePath"]
