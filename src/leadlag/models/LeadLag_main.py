from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from leadlag.models.config import (
    LeadLagConfig,
    LeaderFollowerConfig,
    coerce_lead_lag_config,
    coerce_leader_follower_config,
)
from leadlag.models.leadlag import WindowProcessor
from leadlag.models.leadlag.matrix_builder import build_matrices_batch, build_matrix
from leadlag.models.strategies import LeadLagStrategy, LeadLagStrategyFactory

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when tqdm missing
    TQDM_AVAILABLE = False

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


class LeadLagAnalyzer:
    """Lead-lag analysis orchestrator that delegates heavy lifting to helper utilities."""

    def __init__(self, config: Union[LeadLagConfig, Dict[str, Any], Any], df_universe: pd.Series | None = None):
        self.config = coerce_lead_lag_config(config)
        self.strategy: LeadLagStrategy = LeadLagStrategyFactory(self.config).create()
        self.lead_lag_matrix_rolling: Optional[pd.Series] = None
        self.df_universe = df_universe
        scaling = getattr(self.config, "Scaling_Method", "mean-centering")
        self.window_processor = WindowProcessor(df_universe=df_universe, scaling_method=scaling)
        self._validate_config()
        self.selected_window_info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        if not isinstance(self.config, LeadLagConfig):
            raise TypeError("config must be a LeadLagConfig instance")

    def _validate_data(self, data: pd.DataFrame, allow_partial_nan: bool = True) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")
        if data.shape[1] < 2:
            raise ValueError("DataFrame must have at least 2 columns")

        if not allow_partial_nan:
            if data.isnull().all().any():
                raise ValueError("Some columns contain only NaN values")
        else:
            non_nan_counts = (~data.isnull()).sum(axis=1)
            if (non_nan_counts < 2).all():
                raise ValueError(
                    "Insufficient non-NaN data for analysis (need at least 2 assets per time period)"
                )

    # ------------------------------------------------------------------
    # Window processing convenience wrappers
    # ------------------------------------------------------------------
    def _get_universe_coins_for_date(self, date: pd.Timestamp) -> List[str]:
        return self.window_processor._get_universe_coins_for_date(date)

    def _preprocess_window_data(
        self, price_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> pd.DataFrame:
        return self.window_processor._preprocess_window_data(price_df, window_start, window_end)

    def _compute_log_returns_for_window(
        self, price_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp
    ) -> pd.DataFrame:
        return self.window_processor.get_log_returns(price_df, window_start, window_end)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(self, price_df: pd.DataFrame, return_rolling: bool = False) -> Union[pd.DataFrame, pd.Series]:
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")

        if return_rolling:
            return self._compute_rolling_lead_lag_matrix(price_df)
        return self._compute_single_lead_lag_matrix(price_df)

    def compute_matrices_batch(
        self,
        price_df: pd.DataFrame,
        windows: Sequence[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> Dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pandas DataFrame")
        if not isinstance(price_df.index, pd.DatetimeIndex):
            raise TypeError("price_df must have a DatetimeIndex")
        if not windows:
            return {}

        normalized_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        for window_start, window_end in windows:
            normalized_windows.append((pd.Timestamp(window_start), pd.Timestamp(window_end)))

        log_returns_batch = self.window_processor.get_log_returns_batch(price_df, normalized_windows)
        return build_matrices_batch(log_returns_batch, self._compute_lead_lag_measure_optimized)

    # ------------------------------------------------------------------
    # Rolling / single window computation
    # ------------------------------------------------------------------
    def _compute_rolling_lead_lag_matrix(self, price_df: pd.DataFrame) -> pd.Series:
        if self.config.lookback is None:
            raise ValueError("lookback must be specified for rolling computation")
        if self.config.lookback >= len(price_df):
            raise ValueError("lookback period must be less than data length")

        self.column_names = price_df.columns.tolist()
        date_index = price_df.index
        n_data = len(price_df)
        result_dict: Dict[pd.Timestamp, pd.DataFrame] = {}

        update_freq = self.config.update_freq or 1
        iterator = (
            tqdm(range(self.config.lookback, n_data, update_freq), desc="Computing rolling lead-lag matrices", unit="window")
            if self.config.show_progress and TQDM_AVAILABLE
            else range(self.config.lookback, n_data, update_freq)
        )

        for i in iterator:
            current_date = date_index[i]
            window_start = date_index[i - self.config.lookback + 1]
            window_log_returns = self._compute_log_returns_for_window(price_df, window_start, current_date)
            if window_log_returns.empty or window_log_returns.shape[1] < 2:
                continue
            matrix_df = build_matrix(window_log_returns, self._compute_lead_lag_measure_optimized)
            result_dict[current_date] = matrix_df

        result_series = pd.Series(result_dict)
        result_series.index = pd.DatetimeIndex(result_series.index)
        return result_series

    def _compute_single_lead_lag_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        return build_matrix(data, self._compute_lead_lag_measure_optimized)

    def _compute_lead_lag_measure_optimized(self, data_pair: np.ndarray) -> float:
        return self.strategy.compute(data_pair)

    # ------------------------------------------------------------------
    # Leader/follower detection helpers
    # ------------------------------------------------------------------
    def apply_detector(self, config: Union[Dict[str, Any], LeaderFollowerConfig]) -> pd.Series:
        config = coerce_leader_follower_config(config)
        if self.lead_lag_matrix_rolling is None:
            raise ValueError(
                "lead_lag_matrix_rolling is None. Please call the 'leader_follower_detector' method first."
            )

        leaders_followers_dict: Dict[pd.Timestamp, pd.DataFrame] = {}
        for date, lead_lag_matrix in self.lead_lag_matrix_rolling.items():
            if config.method == "percentile":
                leaders, followers = self._identify_leaders_followers_percentile(lead_lag_matrix, config)
            else:  # pragma: no cover - defensive guard
                raise ValueError(f"Unknown method: {config.method}")

            temp_df = pd.DataFrame({"leaders": pd.Series(leaders), "followers": pd.Series(followers)})
            leaders_followers_dict[date] = temp_df.dropna()

        return pd.Series(leaders_followers_dict)

    def leader_follower_detector(
        self,
        lead_lag_matrix_rolling: pd.Series,
        method_config: Union[Dict[str, Any], LeaderFollowerConfig],
    ) -> pd.Series:
        config = coerce_leader_follower_config(method_config)
        self.lead_lag_matrix_rolling = lead_lag_matrix_rolling
        return self.apply_detector(config)

    def _identify_leaders_followers_percentile(
        self, lead_lag_matrix: pd.DataFrame, config: LeaderFollowerConfig
    ) -> Tuple[pd.Index, pd.Index]:
        return self.identify_quantiles(
            lead_lag_matrix,
            upper_perc=config.top_percentile,
            lower_perc=config.bottom_percentile,
            config=config,
        )

    def identify_quantiles(
        self, lead_lag_matrix: pd.DataFrame, upper_perc: float, lower_perc: float, config: LeaderFollowerConfig
    ) -> Tuple[pd.Index, pd.Index]:
        if config.agg_func == "sum":
            row_sums = lead_lag_matrix.sum(axis=1)
        elif config.agg_func == "mean":
            row_sums = lead_lag_matrix.mean(axis=1)
        else:
            raise ValueError(
                f"Invalid agg_func: '{config.agg_func}'. Supported values are 'sum' and 'mean'."
            )

        row_sums = row_sums[row_sums != 0]
        leaders_threshold = np.percentile(row_sums, upper_perc)
        followers_threshold = np.percentile(row_sums, lower_perc)

        leaders = row_sums[row_sums > leaders_threshold].index
        followers = row_sums[row_sums < followers_threshold].index
        return leaders, followers


__all__ = ["LeadLagAnalyzer", "LeadLagConfig", "LeaderFollowerConfig"]
