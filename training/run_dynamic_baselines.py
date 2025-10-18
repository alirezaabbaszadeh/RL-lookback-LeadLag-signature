from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

import itertools

from models.LeadLag_main import LeadLagConfig, LeadLagAnalyzer
from training.run_scenario import (
    _merge_extends,
    _read_prices,
    _config_to_leadlag,
    _env_info,
    _detect_git,
    _set_seed,
)
from evaluation.metrics import compute_metrics_timeseries, summarize_metrics, plot_signal_strength, plot_stability
from governance.dataset import build_manifest, record_manifest, run_quality_checks
from reporting.logging_utils import get_logger, setup_logging


def _compute_matrix_for_window(analyzer: LeadLagAnalyzer, price_df: pd.DataFrame,
                               window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    """Compute lead-lag matrix for an arbitrary window using analyzer's internal primitives."""
    # preprocess and get log returns
    window_log_returns = analyzer._compute_log_returns_for_window(price_df, window_start, window_end)
    if window_log_returns.empty or window_log_returns.shape[1] < 2:
        return pd.DataFrame()
    valid_columns = window_log_returns.columns.tolist()
    window_data = window_log_returns.values
    n_assets = len(valid_columns)
    lead_lag_matrix = np.zeros((n_assets, n_assets))
    pairs = list(itertools.combinations(range(n_assets), 2))
    for i1, j1 in pairs:
        pair_data = window_data[:, [i1, j1]]
        if np.all(np.isnan(pair_data[:, 0])) or np.all(np.isnan(pair_data[:, 1])):
            continue
        val = analyzer._compute_lead_lag_measure_optimized(pair_data)
        if not np.isnan(val):
            lead_lag_matrix[i1, j1] = val
            lead_lag_matrix[j1, i1] = -val
    matrix_df = pd.DataFrame(lead_lag_matrix, index=valid_columns, columns=valid_columns)
    return matrix_df


def _signal_strength(mat: pd.DataFrame) -> float:
    if mat.empty:
        return np.nan
    arr = mat.values.astype(float)
    n = arr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = arr[mask]
    if off.size == 0:
        return np.nan
    return float(np.nanmean(np.abs(off)))


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_dynamic(config_path: str, out_root: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Path:
    overrides = dict(overrides or {})
    raw_cfg = overrides.pop('_raw_config', None)

    if raw_cfg is not None:
        cfg = raw_cfg
        cfg_path = Path(config_path)
        if overrides:
            cfg = _deep_update(cfg, overrides)
    else:
        cfg_path = Path(config_path)
        cfg = _merge_extends(cfg_path)
        if overrides:
            cfg = _deep_update(cfg, overrides)

    # dynamic params
    dyn = cfg.get('dynamic', {})
    L_min = int(dyn.get('min_lookback', 10))
    L_max = int(dyn.get('max_lookback', 120))
    step = int(dyn.get('step', 5))

    # seeds and output dir
    _set_seed(int(cfg['run'].get('seed', 42)))
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = cfg['run'].get('run_name', 'dynamic_adaptive')
    out_root = out_root or cfg['run'].get('output_root', 'results')
    out_dir = Path(out_root) / f"{run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_context = {
        "module": "dynamic",
        "run_name": run_name,
        "seed": cfg["run"].get("seed"),
        "scenario": cfg_path.stem,
    }
    try:
        setup_logging(
            out_dir / "run.log",
            level="INFO",
            config_path=Path("logging_config.yaml"),
            context=logging_context,
        )
    except Exception:
        setup_logging(out_dir / "run.log", level="INFO", context=logging_context)
    logger = get_logger("run_dynamic", context=logging_context)
    logger.info("Starting dynamic baseline run", context={"output_dir": str(out_dir)})

    # write merged config snapshot
    if yaml is not None:
        (out_dir / 'config_merged.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding='utf-8')
    else:
        (out_dir / 'config_merged.yaml').write_text(str(cfg), encoding='utf-8')

    prices, resolved_price_path = _read_prices(cfg)
    manifest = build_manifest(
        prices,
        source_path=resolved_price_path,
        extras={"quality": run_quality_checks(prices)},
    )
    manifest_path = record_manifest(manifest, out_dir)
    meta = {
        'config_path': str(cfg_path.resolve()),
        'created_at': ts,
        'git': _detect_git(),
        'env': _env_info(),
        'data_source_config': cfg.get('data', {}).get('price_csv', ''),
        'data_manifest': str(manifest_path),
    }
    (out_dir / 'run_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    logger.info("Dataset manifest captured", context={"manifest": str(manifest_path)})

    # load data and analyzer
    ll_cfg = _config_to_leadlag(cfg)
    analyzer = LeadLagAnalyzer(ll_cfg)

    # time index
    dates = prices.index
    # start after max lookback so that any choice is feasible
    start_i = max(L_max, ll_cfg.lookback or L_max)

    decisions: list[dict] = []
    rolling: dict[pd.Timestamp, pd.DataFrame] = {}

    # initialize lookback at mid
    L = max(L_min, min(L_max, int((L_min + L_max) / 2)))

    for i in range(start_i, len(dates)):
        end_ts = dates[i]
        # candidate lookbacks: L-step, L, L+step within bounds
        candidates = sorted({
            max(L_min, min(L_max, L - step)),
            max(L_min, min(L_max, L)),
            max(L_min, min(L_max, L + step)),
        })
        best_L = L
        best_S = -np.inf
        best_mat: pd.DataFrame | None = None
        for cL in candidates:
            start_ts = dates[i - cL + 1]
            mat = _compute_matrix_for_window(analyzer, prices, start_ts, end_ts)
            S = _signal_strength(mat)
            if np.isnan(S):
                continue
            if S > best_S:
                best_S = S
                best_L = cL
                best_mat = mat
        # update state
        L = best_L
        if best_mat is None:
            # fallback to previous or skip
            continue
        rolling[end_ts] = best_mat
        decisions.append({'date': end_ts, 'lookback': L, 'signal': best_S})

    # metrics
    rolling_series = pd.Series(rolling)
    metrics_df = compute_metrics_timeseries(rolling_series)
    decisions_df = pd.DataFrame(decisions).set_index('date')
    metrics_df = metrics_df.join(decisions_df, how='left')
    metrics_df.to_csv(out_dir / 'metrics_timeseries.csv', index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    # plots
    plot_signal_strength(metrics_df, out_dir / 'fig_signal_strength.png')
    plot_stability(metrics_df, out_dir / 'fig_stability.png')

    logger.info(
        "Dynamic baseline completed",
        context={
            "decisions": len(decisions),
            "metrics_path": str(out_dir / 'summary.csv'),
        },
    )

    return out_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to scenario YAML (extends base)')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()
    out = run_dynamic(args.config, args.out)
    print(f"Saved dynamic baseline to: {out}")
