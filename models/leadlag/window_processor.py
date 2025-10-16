from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


class WindowProcessor:
    """Handle universe filtering, preprocessing and log-return computation for windows."""

    def __init__(
        self,
        df_universe: Optional[pd.Series] = None,
        scaling_method: str = "mean-centering",
        cache_size: int = 8,
    ) -> None:
        self.df_universe = df_universe
        self.scaling_method = scaling_method
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]" = OrderedDict()

    def get_log_returns(
        self,
        price_df: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> pd.DataFrame:
        cache_key = (pd.Timestamp(window_start), pd.Timestamp(window_end))
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Promote hit to the end of the LRU structure
            self._cache.move_to_end(cache_key)
            return cached.copy()

        processed = self._preprocess_window_data(price_df, window_start, window_end)
        if processed.empty or processed.shape[1] == 0:
            log_returns = pd.DataFrame()
        else:
            log_returns = self._compute_log_returns(processed)

        stored = log_returns.copy()
        self._cache[cache_key] = stored
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return stored.copy()

    def get_log_returns_batch(
        self,
        price_df: pd.DataFrame,
        windows: Iterable[Tuple[pd.Timestamp, pd.Timestamp]],
    ) -> dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame]:
        """
        Compute log-returns for a batch of windows while reusing the internal cache.

        The result dict is keyed by normalized (start, end) timestamps. Each
        value is a fresh DataFrame instance to avoid accidental mutation.
        """
        results: dict[Tuple[pd.Timestamp, pd.Timestamp], pd.DataFrame] = {}
        for window_start, window_end in windows:
            key = (pd.Timestamp(window_start), pd.Timestamp(window_end))
            log_returns = self.get_log_returns(price_df, key[0], key[1])
            # Return a copy to prevent external mutation from polluting the cache.
            results[key] = log_returns.copy()
        return results

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
