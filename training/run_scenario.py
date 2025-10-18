import os
import json
import time
import hashlib
import argparse
import logging
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

try:  # optional MLflow
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

from models.LeadLag_main import LeadLagConfig, LeadLagAnalyzer
from evaluation.metrics import (
    compute_metrics_timeseries,
    summarize_metrics,
    plot_signal_strength,
    plot_stability,
)
from reporting.logging_utils import setup_logging
from reporting.profiling import profile_to


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs. Install with: pip install pyyaml")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _merge_extends(cfg_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(cfg_path)
    if 'extends' in cfg and cfg['extends']:
        base_path = (cfg_path.parent / cfg['extends']).resolve()
        base = _load_yaml(base_path)
        # shallow merge: base <- cfg
        def deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(a.get(k), dict):
                    a[k] = deep_update(a[k], v)
                else:
                    a[k] = v
            return a
        merged = deep_update(base, {k: v for k, v in cfg.items() if k != 'extends'})
        return merged
    return cfg


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def _detect_git() -> Dict[str, Any]:
    import subprocess
    meta = {}
    try:
        meta['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        meta['git_branch'] = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode()
        meta['git_dirty'] = len(status.strip()) > 0
    except Exception:
        meta['git_commit'] = None
        meta['git_branch'] = None
        meta['git_dirty'] = None
    return meta


def _env_info() -> Dict[str, Any]:
    import sys, platform
    info = {
        'python_version': sys.version,
        'platform': platform.platform(),
    }
    # capture key libs if available
    def v(pkg):
        try:
            mod = __import__(pkg)
            return getattr(mod, '__version__', 'unknown')
        except Exception:
            return None
    for pkg in ['numpy', 'pandas', 'scipy', 'sklearn', 'tqdm', 'iisignature', 'dcor']:
        info[f'{pkg}_version'] = v(pkg)
    return info


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)


def _read_prices(cfg: Dict[str, Any]) -> pd.DataFrame:
    price_path = Path(cfg['data'].get('price_csv', 'raw_data/daily_price.csv'))
    if not price_path.exists():
        # try to autodetect a daily_prices_* file
        candidates = list(Path('raw_data').glob('daily_prices_*.csv'))
        if candidates:
            price_path = candidates[0]
    if not price_path.exists():
        # generate synthetic random-walk data for demonstration/testing
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        rng = np.random.default_rng(seed=cfg['run'].get('seed', 42))
        data = rng.normal(0, 0.01, size=(len(dates), 3)).cumsum(axis=0) + 100
        df = pd.DataFrame(data, index=dates, columns=['AssetA', 'AssetB', 'AssetC'])
        return df

    df = pd.read_csv(price_path)
    if 'date' in df.columns:
        idx = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])
    elif 'Date' in df.columns:
        idx = pd.to_datetime(df['Date'])
        df = df.drop(columns=['Date'])
    else:
        idx = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]
    df.index = idx
    # Baseline: sort chronologically
    df = df.sort_index()

    # Optional: limit rows for walk-forward verification
    try:
        limit_days = cfg.get('data', {}).get('limit_days', None)
    except Exception:
        limit_days = None
    if limit_days is not None:
        try:
            n = int(limit_days)
            if n > 0:
                df = df.iloc[:n]
        except Exception:
            pass

    # Optional: placebo shuffle to probe leakage (destroys chronological order)
    placebo = False
    try:
        placebo = bool(cfg.get('data', {}).get('placebo_shuffle', False))
    except Exception:
        placebo = False
    if placebo and len(df) > 1:
        rng = np.random.default_rng(seed=cfg.get('run', {}).get('seed', 42))
        idx_perm = rng.permutation(len(df))
        df = df.iloc[idx_perm]
        # keep the shuffled order (do not sort back)

    return df


def _config_to_leadlag(cfg: Dict[str, Any]) -> LeadLagConfig:
    # Flatten config into the dict expected by LeadLagConfig.from_dict
    a = cfg['analysis']
    merged = dict(a)
    # carry method-specific block under the same key name
    if a.get('method') and a.get(a['method']):
        merged[a['method']] = a[a['method']]
    return LeadLagConfig.from_dict(merged)


def run_scenario(
    config_path: str,
    out_root: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    overrides = deepcopy(overrides) if overrides else {}
    raw_cfg = overrides.pop('_raw_config', None)

    if raw_cfg is not None:
        cfg = deepcopy(raw_cfg)
        cfg_path = Path(config_path)
    else:
        cfg_path = Path(config_path)
        cfg = _merge_extends(cfg_path)
        if overrides:
            cfg = _deep_update(cfg, overrides)

    if overrides and raw_cfg is not None:
        cfg = _deep_update(cfg, overrides)

    # seeds and output dir
    _set_seed(int(cfg['run'].get('seed', 42)))
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_name = cfg['run'].get('run_name', 'auto')
    out_root = out_root or cfg['run'].get('output_root', 'results')
    out_dir = Path(out_root) / f"{run_name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # logging setup
    try:
        setup_logging(out_dir / "run.log", level="INFO", config_path=Path("logging_config.yaml"))
    except Exception:
        # fallback handled in setup_logging
        pass
    logger = logging.getLogger("run_scenario")
    logger.info("Starting scenario run: %s", run_name)

    # snapshot configs and metadata
    with open(out_dir / 'config_merged.yaml', 'w', encoding='utf-8') as f:
        if yaml is None:
            f.write(json.dumps(cfg, indent=2))
        else:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    # compute dataset file hash if file exists
    data_price_path = cfg['data'].get('price_csv', '')
    price_path_resolved = Path(data_price_path)
    data_price_hash = None
    try:
        if price_path_resolved.exists():
            data_price_hash = _hash_file(price_path_resolved)
    except Exception:
        data_price_hash = None

    meta = {
        'config_path': str(cfg_path.resolve()),
        'data_price_path': data_price_path,
        'data_price_hash': data_price_hash,
        'created_at': ts,
        'git': _detect_git(),
        'env': _env_info(),
    }
    with open(out_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    # load data
    with profile_to(out_dir, label="load_data"):
        prices = _read_prices(cfg)
    # optional universe is not wired due to format variability; pass None

    # build analyzer
    ll_cfg = _config_to_leadlag(cfg)
    analyzer = LeadLagAnalyzer(ll_cfg)

    # compute rolling matrices
    with profile_to(out_dir, label="analyze"):
        rolling = analyzer.analyze(prices, return_rolling=True)
    # compute metrics
    with profile_to(out_dir, label="metrics"):
        metrics_df = compute_metrics_timeseries(rolling)
    metrics_df.to_csv(out_dir / 'metrics_timeseries.csv', index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    # Log summary metrics to MLflow if available and enabled via env
    import os as _os  # local import to avoid polluting namespace
    mlflow_enabled_env = _os.getenv('MLFLOW_ENABLED', '1').lower() in ('1', 'true', 'yes')
    if MLFLOW_AVAILABLE and mlflow_enabled_env:
        try:  # pragma: no cover - integration path
            with mlflow.start_run(run_name=run_name, nested=False):
                for _, row in summary.iterrows():
                    metric_name = row.get('metric', 'metric')
                    for col, val in row.items():
                        if col == 'metric':
                            continue
                        if isinstance(val, (int, float)) and not (val != val):  # NaN check
                            mlflow.log_metric(f"{metric_name}_{col}", float(val))
                # log artifacts
                mlflow.log_artifact(str(out_dir / 'config_merged.yaml'))
                mlflow.log_artifact(str(out_dir / 'summary.csv'))
                if (out_dir / 'metrics_timeseries.csv').exists():
                    mlflow.log_artifact(str(out_dir / 'metrics_timeseries.csv'))
                for plot in ['fig_signal_strength.png', 'fig_stability.png']:
                    p = out_dir / plot
                    if p.exists():
                        mlflow.log_artifact(str(p))
        except Exception:
            logger.warning("MLflow logging failed; continuing without MLflow.")

    # plots
    headless = False
    try:
        headless = bool(cfg.get('metrics', {}).get('headless', False))
    except Exception:
        headless = False
    if (not headless) and 'metrics' in cfg and 'plots' in cfg['metrics']:
        if 'signal_strength' in cfg['metrics']['plots']:
            try:
                plot_signal_strength(metrics_df, out_dir / 'fig_signal_strength.png')
            except Exception:
                logger.warning("Plot generation failed: signal_strength")
        if 'stability' in cfg['metrics']['plots']:
            try:
                plot_stability(metrics_df, out_dir / 'fig_stability.png')
            except Exception:
                logger.warning("Plot generation failed: stability")

    # save a small sample matrix for inspection
    if len(rolling) > 0:
        first_date = rolling.index[0]
        last_date = rolling.index[-1]
        rolling[first_date].to_csv(out_dir / f"matrix_{first_date.date()}.csv")
        rolling[last_date].to_csv(out_dir / f"matrix_{last_date.date()}.csv")

    logger.info("Scenario run completed: %s", out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to scenario YAML')
    ap.add_argument('--out', type=str, default=None, help='Output root directory')
    args = ap.parse_args()
    out_dir = run_scenario(args.config, args.out)
    print(f"Saved results to: {out_dir}")


if __name__ == '__main__':
    main()

