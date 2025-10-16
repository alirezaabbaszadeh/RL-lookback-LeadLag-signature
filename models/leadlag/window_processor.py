from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowCache:
    """Lightweight cache for the most recent window log-returns."""

    key: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    values: Optional[pd.DataFrame] = None


class WindowProcessor:
    """Handle universe filtering, preprocessing and log-return computation for windows."""

    def __init__(self, df_universe: Optional[pd.Series] = None, scaling_method: str = "mean-centering") -> None:
        self.df_universe = df_universe
        self.scaling_method = scaling_method
        self._cache = WindowCache()

    def get_log_returns(
        self,
        price_df: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> pd.DataFrame:
        cache_key = (pd.Timestamp(window_start), pd.Timestamp(window_end))
        if self._cache.key == cache_key and self._cache.values is not None:
            return self._cache.values

        processed = self._preprocess_window_data(price_df, window_start, window_end)
        if processed.empty or processed.shape[1] == 0:
            log_returns = pd.DataFrame()
        else:
            log_returns = self._compute_log_returns(processed)

        self._cache = WindowCache(key=cache_key, values=log_returns)
        return log_returns

    # ------------------------------------------------------------------
    # Internal helpers (logic migrated from LeadLagAnalyzer)
    # ------------------------------------------------------------------

    def _preprocess_window_data(
        self,
        price_df: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> pd.DataFrame:
        if not isinstance(window_start, pd.Timestamp):
            window_start = pd.Timestamp(window_start)
        if not isinstance(window_end, pd.Timestamp):
            window_end = pd.Timestamp(window_end)

        price_index = pd.to_datetime(price_df.index)
        window_mask = (price_index >= window_start) & (price_index <= window_end)

        if self.df_universe is None:
            window_data = price_df.loc[window_mask].copy()
            return window_data.ffill()

        universe_coins = self._get_universe_coins_for_date(window_end)
        if not universe_coins:
            return pd.DataFrame(index=price_df.index[window_mask])

        available_coins = [coin for coin in universe_coins if coin in price_df.columns]
        if not available_coins:
            return pd.DataFrame(index=price_df.index[window_mask])

        preprocessed_data = price_df.loc[window_mask, available_coins].copy()
        return preprocessed_data.ffill()

    def _compute_log_returns(self, window_prices: pd.DataFrame) -> pd.DataFrame:
        log_returns = np.log(window_prices).diff()
        log_returns = log_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return log_returns

    def _get_universe_coins_for_date(self, target_date: pd.Timestamp) -> list[str]:
        if self.df_universe is None:
            return []
        if target_date not in self.df_universe.index:
            # fallback to the latest available date before target
            prior_dates = self.df_universe.index[self.df_universe.index <= target_date]
            if len(prior_dates) == 0:
                return []
            target_date = prior_dates.max()
        universe_value = self.df_universe.loc[target_date]
        if isinstance(universe_value, (list, tuple)):
            return list(universe_value)
        if isinstance(universe_value, str):
            return [coin.strip() for coin in universe_value.split(',') if coin.strip()]
        if isinstance(universe_value, pd.Series):
            return universe_value.dropna().astype(str).tolist()
        return []

