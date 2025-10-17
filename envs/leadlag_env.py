from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import gym
import numpy as np
import pandas as pd
from gym import spaces

from models.LeadLag_main import LeadLagAnalyzer, LeadLagConfig
from training.rewards import RewardContext, RewardTemplate, load_reward_template


@dataclass
class RewardWeights:
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 0.05


class LeadLagEnv(gym.Env):
    """Gym environment for adaptive lookback selection with random starts and flexible actions."""

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
        action_mode: Literal["absolute", "relative", "hybrid"] = "absolute",
        relative_step: int = 5,
        episode_length: Optional[int] = None,
        random_start: bool = True,
        random_seed: Optional[int] = None,
        ema_alpha: Optional[float] = None,
        reward_template: Optional[Any] = None,
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
        self.action_mode = action_mode
        if self.action_mode not in {"absolute", "relative", "hybrid"}:
            raise ValueError("action_mode must be one of {'absolute', 'relative', 'hybrid'}")
        self.relative_step = max(1, int(relative_step))
        self.random_start = bool(random_start)
        self.episode_length = int(episode_length) if episode_length else None
        self._rng = np.random.default_rng(random_seed)
        self.ema_alpha = None if ema_alpha is None else float(ema_alpha)

        self.weights = RewardWeights(**(reward_weights or {}))
        self._reward_template: Optional[RewardTemplate] = load_reward_template(reward_template)

        # Action space depends on action mode
        if self.action_mode == "absolute":
            if self.discrete_actions:
                self.action_space = spaces.Discrete(self.max_lookback - self.min_lookback + 1)
            else:
                self.action_space = spaces.Box(
                    low=np.array([self.min_lookback], dtype=np.float32),
                    high=np.array([self.max_lookback], dtype=np.float32),
                    dtype=np.float32,
                )
        elif self.action_mode == "relative":
            if self.discrete_actions:
                self.action_space = spaces.Discrete(3)  # {-step, 0, +step}
            else:
                self.action_space = spaces.Box(
                    low=np.array([-1.0], dtype=np.float32),
                    high=np.array([1.0], dtype=np.float32),
                    dtype=np.float32,
                )
        else:  # hybrid
            if self.discrete_actions:
                self._hybrid_actions: List[Any] = ["min", -self.relative_step, 0, self.relative_step, "max"]
                self.action_space = spaces.Discrete(len(self._hybrid_actions))
            else:
                self.action_space = spaces.Box(
                    low=np.array([self.min_lookback, -1.0], dtype=np.float32),
                    high=np.array([self.max_lookback, 1.0], dtype=np.float32),
                    dtype=np.float32,
                )

        # Observation space: enriched features (14 dims)
        self.obs_dim = 14
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._dates = self.price_df.index
        self._min_start_index = max(self.max_lookback, leadlag_config.lookback or self.max_lookback)
        self._episode_length = self.episode_length
        self._episode_start_index = self._min_start_index
        self._episode_end_index = len(self._dates)
        self._current_index = self._episode_start_index
        self._current_lookback = self.max_lookback
        self._prev_lookback = self.max_lookback
        self._prev_matrix: Optional[pd.DataFrame] = None
        self._prev_metrics: Optional[Dict[str, float]] = None
        self._ema_state: Optional[Dict[str, float]] = None
        self._ema_prev_mean: float = 0.0
        self._prev_reward: float = 0.0
        self._last_delta_norm: float = 0.0
        self._prev_regime: Optional[Dict[str, float]] = None
        self.history: List[Dict[str, Any]] = []

    # ---------- Core Gym API ----------

    def reset(self):
        self._episode_start_index = self._select_start_index()
        self._episode_end_index = self._determine_episode_end(self._episode_start_index)
        self._current_index = self._episode_start_index
        midpoint = int((self.min_lookback + self.max_lookback) / 2)
        self._current_lookback = max(self.min_lookback, min(self.max_lookback, midpoint))
        self._prev_lookback = self._current_lookback
        self._prev_matrix = None
        self._prev_metrics = None
        self._prev_regime = None
        self._ema_state = None
        self._ema_prev_mean = 0.0
        self._prev_reward = 0.0
        self._last_delta_norm = 0.0
        self.history = []

        matrix, metrics, extras = self._compute_matrix_and_metrics(self._current_index, self._current_lookback)
        metrics = self._apply_smoothing(metrics)
        self._prev_matrix = matrix
        self._prev_metrics = metrics
        self._prev_regime = extras.get('regime')

        obs = self._make_observation(metrics, extras, self._current_lookback)
        return obs

    def step(self, action):
        prev_lookback = self._current_lookback
        target_lookback = self._resolve_action(action)
        lookback = int(np.clip(target_lookback, self.min_lookback, self.max_lookback))
        self._prev_lookback = prev_lookback
        self._current_lookback = lookback

        matrix, metrics, extras = self._compute_matrix_and_metrics(self._current_index, lookback)
        metrics = self._apply_smoothing(metrics)

        reward, reward_components = self._compute_reward(metrics, extras, lookback)
        self._prev_reward = reward

        delta_norm = 0.0
        if self.max_lookback != self.min_lookback:
            delta_norm = (lookback - prev_lookback) / (self.max_lookback - self.min_lookback)
        self._last_delta_norm = float(delta_norm)

        obs = self._make_observation(metrics, extras, lookback)

        current_date = self._dates[self._current_index]
        history_entry = {
            'date': current_date,
            'lookback': lookback,
            'metrics': metrics,
            'matrix': matrix,
            'regime': extras.get('regime'),
            'signature': extras.get('signature'),
            'delta_norm': self._last_delta_norm,
            'reward': reward,
            'action_mode': self.action_mode,
            'reward_components': reward_components,
        }
        self.history.append(history_entry)

        self._prev_matrix = matrix
        self._prev_metrics = metrics
        self._prev_regime = extras.get('regime')

        self._current_index += 1
        done = self._current_index >= self._episode_end_index

        info = {
            'date': current_date,
            'lookback': lookback,
            'reward_components': reward_components,
            'episode_end': self._episode_end_index,
        }

        return obs, reward, done, info

    # ---------- Helpers ----------

    def _select_start_index(self) -> int:
        min_idx = self._min_start_index
        max_idx = len(self._dates) - 1
        if self._episode_length is not None:
            max_idx = max(min_idx, len(self._dates) - self._episode_length)
        if max_idx <= min_idx or not self.random_start:
            return min_idx
        return int(self._rng.integers(min_idx, max_idx + 1))

    def _determine_episode_end(self, start_index: int) -> int:
        if self._episode_length is None:
            return len(self._dates)
        return min(len(self._dates), start_index + self._episode_length)

    def _resolve_action(self, action) -> int:
        if self.action_mode == "absolute":
            return self._resolve_absolute(action)
        if self.action_mode == "relative":
            return self._resolve_relative(action)
        return self._resolve_hybrid(action)

    def _resolve_absolute(self, action) -> int:
        if self.discrete_actions:
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            return self.min_lookback + int(action)
        return int(np.round(float(action)))

    def _resolve_relative(self, action) -> int:
        current = self._current_lookback
        if self.discrete_actions:
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            idx = int(action)
            if idx not in (0, 1, 2):
                raise ValueError("Relative discrete action must be in {0,1,2}")
            delta = (-self.relative_step, 0, self.relative_step)[idx]
            return current + delta
        delta = float(action[0] if isinstance(action, (np.ndarray, list)) else action)
        return int(current + delta * self.relative_step)

    def _resolve_hybrid(self, action) -> int:
        current = self._current_lookback
        if self.discrete_actions:
            if isinstance(action, (np.ndarray, list)):
                action = int(action[0])
            token = self._hybrid_actions[int(action)]
            if token == "min":
                return self.min_lookback
            if token == "max":
                return self.max_lookback
            return current + int(token)
        if not isinstance(action, (np.ndarray, list)) or len(action) != 2:
            raise ValueError("Hybrid continuous action expects [absolute_target, relative_factor]")
        absolute_target = float(action[0])
        relative_factor = float(action[1])
        relative_component = current + relative_factor * self.relative_step
        return int(0.5 * (absolute_target + relative_component))

    def _compute_matrix_and_metrics(self, index: int, lookback: int) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Dict[str, float]]]:
        end_ts = self._dates[index]
        start_idx = max(0, index - lookback + 1)
        start_ts = self._dates[start_idx]

        log_returns = self.analyzer._compute_log_returns_for_window(self.price_df, start_ts, end_ts)
        if log_returns.empty or log_returns.shape[1] < 2:
            return pd.DataFrame(), self._zero_metrics(), {
                'regime': self._zero_regime_stats(),
                'signature': self._zero_signature_features(),
            }

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
        extras = {
            'regime': self._compute_regime_stats(log_returns),
            'signature': self._compute_signature_features(matrix_df),
        }
        return matrix_df, metrics, extras

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

        metrics = {
            'mean_abs': float(mean_abs),
            'max_abs': float(max_abs),
            'row_range': float(row_range),
            'row_std': float(row_std),
            'stability_matrix': float(stability_matrix),
            'stability_rows': float(stability_rows),
            'delta_mean': float(delta_mean),
        }
        return metrics

    def _compute_regime_stats(self, log_returns: pd.DataFrame) -> Dict[str, float]:
        if log_returns.empty:
            return self._zero_regime_stats()
        stacked = log_returns.stack()
        returns_mean = float(stacked.mean()) if not stacked.empty else 0.0
        returns_vol = float(stacked.std()) if stacked.size else 0.0
        cross_sectional_vol = float(log_returns.std(axis=1, ddof=0).mean()) if not log_returns.empty else 0.0
        dispersion = float(log_returns.std(axis=0, ddof=0).mean()) if not log_returns.empty else 0.0
        return {
            'returns_mean': returns_mean,
            'returns_vol': returns_vol,
            'cross_sectional_vol': cross_sectional_vol,
            'dispersion': dispersion,
        }

    def _compute_signature_features(self, matrix: pd.DataFrame) -> Dict[str, float]:
        if matrix.empty:
            return self._zero_signature_features()
        row_sums = matrix.sum(axis=1)
        leader_score = float(row_sums.max()) if not row_sums.empty else 0.0
        laggard_score = float(row_sums.min()) if not row_sums.empty else 0.0
        spread = leader_score - laggard_score
        arr = matrix.values.astype(float)
        if arr.size:
            off_diag = arr[np.triu_indices_from(arr, k=1)]
            top_pair = float(np.nanmax(np.abs(off_diag))) if off_diag.size else 0.0
        else:
            top_pair = 0.0
        return {
            'leader_score': leader_score,
            'laggard_score': laggard_score,
            'spread': spread,
            'top_pair': top_pair,
        }

    def _apply_smoothing(self, metrics: Dict[str, float]) -> Dict[str, float]:
        metrics = metrics.copy()
        if self.ema_alpha is None:
            metrics['ema_mean_abs'] = metrics['mean_abs']
            return metrics
        if self._ema_state is None:
            self._ema_state = metrics.copy()
            self._ema_prev_mean = self._ema_state['mean_abs']
        else:
            for key, value in metrics.items():
                prev = self._ema_state.get(key, value)
                self._ema_state[key] = self.ema_alpha * value + (1.0 - self.ema_alpha) * prev
        smoothed = {key: self._ema_state[key] for key in metrics}
        smoothed['delta_mean'] = self._ema_state['mean_abs'] - self._ema_prev_mean
        smoothed['ema_mean_abs'] = self._ema_state['mean_abs']
        self._ema_prev_mean = self._ema_state['mean_abs']
        return smoothed

    def _legacy_reward_components(self, metrics: Dict[str, float], lookback: int) -> Dict[str, float]:
        S = 0.5 * metrics.get('mean_abs', 0.0) + 0.5 * metrics.get('row_range', 0.0)
        C = 0.5 * metrics.get('stability_matrix', 0.0) + 0.5 * metrics.get('stability_rows', 0.0)
        edge_penalty = 1.0 if lookback in (self.min_lookback, self.max_lookback) else 0.0
        jump_penalty = 0.0
        if self._prev_lookback is not None:
            jump = abs(lookback - self._prev_lookback)
            if jump > self.penalty_step:
                jump_penalty = jump / self.penalty_step
            if jump == 0:
                edge_penalty += self.penalty_same
        delta_mean = float(metrics.get('delta_mean', 0.0))
        return {
            'S': float(S),
            'C': float(C),
            'E': float(edge_penalty + jump_penalty),
            'Sharpe': 0.0,
            'LookbackPenalty': abs(delta_mean),
        }

    def _compute_reward(
        self,
        metrics: Dict[str, float],
        extras: Dict[str, Dict[str, float]],
        lookback: int,
    ) -> tuple[float, Dict[str, float]]:
        if self._reward_template is not None:
            context = RewardContext(
                lookback=lookback,
                prev_lookback=self._prev_lookback,
                min_lookback=self.min_lookback,
                max_lookback=self.max_lookback,
            )
            reward, components = self._reward_template.compute(metrics, extras, context)
            return float(reward), {k: float(v) for k, v in components.items()}

        comps = self._legacy_reward_components(metrics, lookback)
        reward = (
            self.weights.alpha * comps['S']
            + self.weights.beta * comps['C']
            - self.weights.gamma * comps['E']
        )
        return float(reward), comps

    def _make_observation(self, metrics: Dict[str, float], extras: Dict[str, Dict[str, float]], lookback: int) -> np.ndarray:
        range_span = self.max_lookback - self.min_lookback
        norm_lookback = 0.0 if range_span == 0 else (lookback - self.min_lookback) / range_span
        regime = extras.get('regime', self._zero_regime_stats())
        signature = extras.get('signature', self._zero_signature_features())
        ema_mean = metrics.get('ema_mean_abs', metrics['mean_abs'])
        obs_values = np.array([
            float(norm_lookback),
            float(metrics['mean_abs']),
            float(metrics['row_range']),
            float(metrics['stability_matrix']),
            float(metrics['stability_rows']),
            float(metrics['delta_mean']),
            float(regime['returns_mean']),
            float(regime['returns_vol']),
            float(regime['cross_sectional_vol']),
            float(signature['leader_score']),
            float(signature['laggard_score']),
            float(signature['spread']),
            float(self._last_delta_norm),
            float(ema_mean),
        ], dtype=np.float32)
        return obs_values

    def _zero_metrics(self) -> Dict[str, float]:
        return {
            'mean_abs': 0.0,
            'max_abs': 0.0,
            'row_range': 0.0,
            'row_std': 0.0,
            'stability_matrix': 0.0,
            'stability_rows': 0.0,
            'delta_mean': 0.0,
            'ema_mean_abs': 0.0,
        }

    def _zero_regime_stats(self) -> Dict[str, float]:
        return {
            'returns_mean': 0.0,
            'returns_vol': 0.0,
            'cross_sectional_vol': 0.0,
            'dispersion': 0.0,
        }

    def _zero_signature_features(self) -> Dict[str, float]:
        return {
            'leader_score': 0.0,
            'laggard_score': 0.0,
            'spread': 0.0,
            'top_pair': 0.0,
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
            row = {
                'date': entry['date'],
                'lookback': entry['lookback'],
                'reward': entry.get('reward', 0.0),
                'delta_norm': entry.get('delta_norm', 0.0),
            }
            metrics = entry.get('metrics', {})
            row.update({f"metric_{k}": v for k, v in metrics.items()})
            regime = entry.get('regime') or {}
            row.update({f"regime_{k}": v for k, v in regime.items()})
            signature = entry.get('signature') or {}
            row.update({f"signature_{k}": v for k, v in signature.items()})
            rows.append(row)
        df = pd.DataFrame(rows).set_index('date').sort_index()
        return df
