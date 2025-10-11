from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import gym
import numpy as np
import pandas as pd
from gym import spaces

from models.LeadLag_main import LeadLagAnalyzer, LeadLagConfig


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.05


class LeadLagEnv(gym.Env):
    """Gym environment for adaptive lookback selection using LeadLagAnalyzer."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        price_df: pd.DataFrame,
        leadlag_config: LeadLagConfig,
        min_lookback: int = 10,
        max_lookback: int = 120,
        discrete_actions: bool = True,
        reward_weights: Optional[Dict[str, float]] = None,
        penalty_same: float = 0.05,
        penalty_step: int = 10,
    ) -> None:
        super().__init__()
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise ValueError("price_df must have a DatetimeIndex")
        self.price_df = price_df.sort_index()
        self.analyzer = LeadLagAnalyzer(leadlag_config)
        self.min_lookback = int(min_lookback)
        self.max_lookback = int(max_lookback)
        if self.min_lookback < 2:
            raise ValueError("min_lookback must be >= 2")
        if self.max_lookback <= self.min_lookback:
            raise ValueError("max_lookback must be greater than min_lookback")
        self.discrete_actions = discrete_actions
        self.penalty_same = penalty_same
        self.penalty_step = penalty_step

        self.weights = RewardWeights(**(reward_weights or {}))

        # Action space: discrete lookback choices or Box for continuous
        if self.discrete_actions:
            self.action_space = spaces.Discrete(self.max_lookback - self.min_lookback + 1)
        else:
            self.action_space = spaces.Box(
                low=np.array([self.min_lookback], dtype=np.float32),
                high=np.array([self.max_lookback], dtype=np.float32),
                dtype=np.float32,
            )

        # Observation space: vector of engineered features (8 dims)
        self.obs_dim = 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._dates = self.price_df.index
        self._episode_start_index = max(self.max_lookback, leadlag_config.lookback or self.max_lookback)
        self._current_index = self._episode_start_index
        self._current_lookback = self.max_lookback
        self._prev_lookback = self.max_lookback
        self._prev_matrix: Optional[pd.DataFrame] = None
        self._prev_metrics: Optional[Dict[str, float]] = None
        self.history: List[Dict[str, Any]] = []

    # ---------- Core Gym API ----------

    def reset(self):
        self._current_index = self._episode_start_index
        self._current_lookback = max(self.min_lookback, min(self.max_lookback, int((self.min_lookback + self.max_lookback) / 2)))
        self._prev_lookback = self._current_lookback
        self._prev_matrix = None
        self._prev_metrics = None
        self.history = []

        matrix, metrics = self._compute_matrix_and_metrics(self._current_index, self._current_lookback)
        self._prev_matrix = matrix
        self._prev_metrics = metrics

        obs = self._make_observation(metrics, self._current_lookback)
        return obs

    def step(self, action):
        lookback = self._action_to_lookback(action)
        lookback = int(np.clip(lookback, self.min_lookback, self.max_lookback))
        self._prev_lookback = self._current_lookback
        self._current_lookback = lookback

        matrix, metrics = self._compute_matrix_and_metrics(self._current_index, lookback)

        reward = self._compute_reward(metrics, lookback)

        obs = self._make_observation(metrics, lookback)

        current_date = self._dates[self._current_index]
        self.history.append({
            'date': current_date,
            'lookback': lookback,
            'metrics': metrics,
            'matrix': matrix,
        })

        self._prev_matrix = matrix
        self._prev_metrics = metrics

        self._current_index += 1
        done = self._current_index >= len(self._dates)

        info = {
            'date': current_date,
            'lookback': lookback,
            'reward_components': self._reward_components(metrics, lookback),
        }

        return obs, reward, done, info

    # ---------- Helpers ----------

    def _action_to_lookback(self, action) -> int:
        if self.discrete_actions:
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            return self.min_lookback + int(action)
        else:
            return int(np.round(float(action)))

    def _compute_matrix_and_metrics(self, index: int, lookback: int) -> tuple[pd.DataFrame, Dict[str, float]]:
        end_ts = self._dates[index]
        start_idx = max(0, index - lookback + 1)
        start_ts = self._dates[start_idx]

        log_returns = self.analyzer._compute_log_returns_for_window(self.price_df, start_ts, end_ts)
        if log_returns.empty or log_returns.shape[1] < 2:
            return pd.DataFrame(), self._zero_metrics()

        data = log_returns.values
        valid_columns = log_returns.columns.tolist()
        n_assets = len(valid_columns)
        lead_lag_matrix = np.zeros((n_assets, n_assets))
        for i, j in itertools.combinations(range(n_assets), 2):
            pair_data = data[:, [i, j]]
            if np.all(np.isnan(pair_data[:, 0])) or np.all(np.isnan(pair_data[:, 1])):
                continue
            val = self.analyzer._compute_lead_lag_measure_optimized(pair_data)
            if not np.isnan(val):
                lead_lag_matrix[i, j] = val
                lead_lag_matrix[j, i] = -val

        matrix_df = pd.DataFrame(lead_lag_matrix, index=valid_columns, columns=valid_columns)
        metrics = self._compute_metrics(matrix_df)
        return matrix_df, metrics

    def _compute_metrics(self, matrix: pd.DataFrame) -> Dict[str, float]:
        if matrix.empty:
            return self._zero_metrics()
        arr = matrix.values.astype(float)
        n = arr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off = arr[mask]
        mean_abs = np.nanmean(np.abs(off)) if off.size else 0.0
        max_abs = np.nanmax(np.abs(off)) if off.size else 0.0
        row_sums = np.nansum(arr, axis=1)
        row_range = np.nanmax(row_sums) - np.nanmin(row_sums) if n > 0 else 0.0
        row_std = np.nanstd(row_sums) if n > 0 else 0.0

        stability_matrix = 0.0
        stability_rows = 0.0
        if self._prev_matrix is not None and not self._prev_matrix.empty:
            common_cols = matrix.columns.intersection(self._prev_matrix.columns)
            if len(common_cols) >= 2:
                cur = matrix.loc[common_cols, common_cols].values.astype(float)
                prev = self._prev_matrix.loc[common_cols, common_cols].values.astype(float)
                c_mask = ~np.eye(len(common_cols), dtype=bool)
                cur_off = cur[c_mask]
                prev_off = prev[c_mask]
                if cur_off.size and prev_off.size:
                    try:
                        stability_matrix = float(np.corrcoef(np.nan_to_num(cur_off), np.nan_to_num(prev_off))[0, 1])
                    except Exception:
                        stability_matrix = 0.0
                cur_rows = np.nansum(cur, axis=1)
                prev_rows = np.nansum(prev, axis=1)
                try:
                    stability_rows = float(np.corrcoef(np.nan_to_num(cur_rows), np.nan_to_num(prev_rows))[0, 1])
                except Exception:
                    stability_rows = 0.0

        prev_mean = self._prev_metrics['mean_abs'] if self._prev_metrics else 0.0
        delta_mean = mean_abs - prev_mean

        return {
            'mean_abs': float(mean_abs),
            'max_abs': float(max_abs),
            'row_range': float(row_range),
            'row_std': float(row_std),
            'stability_matrix': float(stability_matrix),
            'stability_rows': float(stability_rows),
            'delta_mean': float(delta_mean),
        }

    def _reward_components(self, metrics: Dict[str, float], lookback: int) -> Dict[str, float]:
        # S: combine mean_abs and row_range
        S = 0.5 * metrics['mean_abs'] + 0.5 * metrics['row_range']
        # C: average of stability metrics
        C = 0.5 * metrics['stability_matrix'] + 0.5 * metrics['stability_rows']
        # E: penalty for extremes or large jumps
        edge_penalty = 1.0 if lookback in (self.min_lookback, self.max_lookback) else 0.0
        jump_penalty = 0.0
        if self._prev_lookback is not None:
            jump = abs(lookback - self._prev_lookback)
            if jump > self.penalty_step:
                jump_penalty = jump / self.penalty_step
            if jump == 0:
                edge_penalty += self.penalty_same
        E = edge_penalty + jump_penalty
        return {'S': S, 'C': C, 'E': E}

    def _compute_reward(self, metrics: Dict[str, float], lookback: int) -> float:
        comps = self._reward_components(metrics, lookback)
        reward = (
            self.weights.alpha * comps['S']
            + self.weights.beta * comps['C']
            - self.weights.gamma * comps['E']
        )
        return float(reward)

    def _make_observation(self, metrics: Dict[str, float], lookback: int) -> np.ndarray:
        norm_lookback = (lookback - self.min_lookback) / (self.max_lookback - self.min_lookback)
        prev_mean = self._prev_metrics['mean_abs'] if self._prev_metrics else 0.0
        prev_row_range = self._prev_metrics['row_range'] if self._prev_metrics else 0.0
        obs = np.array([
            norm_lookback,
            metrics['mean_abs'],
            metrics['max_abs'],
            metrics['row_range'],
            metrics['row_std'],
            metrics['stability_matrix'],
            metrics['stability_rows'],
            metrics['mean_abs'] - prev_mean,
        ], dtype=np.float32)
        return obs

    def _zero_metrics(self) -> Dict[str, float]:
        return {
            'mean_abs': 0.0,
            'max_abs': 0.0,
            'row_range': 0.0,
            'row_std': 0.0,
            'stability_matrix': 0.0,
            'stability_rows': 0.0,
            'delta_mean': 0.0,
        }

    # ---------- Utility ----------

    def get_history_matrices(self) -> pd.Series:
        if not self.history:
            return pd.Series(dtype=object)
        data = {entry['date']: entry['matrix'] for entry in self.history if isinstance(entry['matrix'], pd.DataFrame)}
        series = pd.Series(data)
        if not series.empty:
            series.index = pd.DatetimeIndex(series.index)
        return series.sort_index()

    def get_history_dataframe(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        rows = []
        for entry in self.history:
            row = {'date': entry['date'], 'lookback': entry['lookback']}
            row.update(entry['metrics'])
            rows.append(row)
        df = pd.DataFrame(rows).set_index('date').sort_index()
        return df
