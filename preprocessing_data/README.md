# Preprocessing Utilities

The `preprocessing_data` package exposes helpers used by training and research notebooks to reshape raw crypto price panels.
The wrappers under `preprocessing_data/preprocessing.py` simply re-export the implementations in
`src/leadlag/preprocessing_data/preprocessing.py` so the functions are available both from the installed package and from local
scripts.

## Available functions

### `resample_crypto_data(df, timeframe, price_type="close", method=None)`
Resamples OHLC data to a coarser timeframe with explicit aggregation semantics. The default aggregation depends on the
selected `price_type` (`last` for close, `first` for open, etc.). Validation follows the runtime checks in the source module:
`df` must be a non-empty `DataFrame` with a `DatetimeIndex`, otherwise informative errors are raised.

### `selected_uni(close_price, df_universe, maximum_coin=50, window_size=210)`
Builds a rolling list of tradable symbols by combining the historical close-price panel with a month-by-month inclusion list.
The helper ensures:
- only unique symbols are kept;
- the list respects the `maximum_coin` cap and enforces an even count;
- assets with excessive missing data in the lookback window are dropped.

Both utilities are deterministic given their inputs and contain the same guard rails used by the production runners (e.g.
`training/run_support.read_prices`). When new preprocessing logic is added, document it here so notebooks and reports stay in
sync with the code.
