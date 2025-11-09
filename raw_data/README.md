# Raw Data Catalogue

The training runners expect a wide CSV of daily crypto prices. By default the pipeline looks for
`raw_data/daily_price.csv`; if the file is missing it scans for `raw_data/daily_prices_*.csv`. When neither path exists the code
falls back to generating a seeded synthetic panel (three assets, 300 daily points) so smoke tests remain reproducible.

## Included files

- `daily_price.csv` – canonical panel used by most scenarios.
- `daily_prices_*.csv` – alternate exports captured during historical backfills.
- `1H_prices_*.csv`, `volume_data.csv`, `Market_Turnover.csv`, etc. – auxiliary sources consumed by research notebooks. These are
  not loaded automatically but remain available for ad-hoc feature engineering.
- `universe_data.csv` – month-by-month inclusion lists consumed by `selected_uni` when building dynamic universes.

The loader (`training/run_support.read_prices`) converts the first timestamp column to a `DatetimeIndex`, sorts rows
chronologically, applies optional truncation (`data.limit_days`), and supports a placebo shuffle (`data.placebo_shuffle`) for
leakage probes. Any run that reads from this directory also records `data_manifest.json` and quality flags under the run output
so provenance is traceable.
