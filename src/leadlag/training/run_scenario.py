import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

try:  # optional MLflow
    import mlflow  # type: ignore

    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

from leadlag.evaluation.metrics import (
    compute_metrics_timeseries,
    plot_signal_strength,
    plot_stability,
    summarize_metrics,
)
from leadlag.models.LeadLag_main import LeadLagAnalyzer, LeadLagConfig
from leadlag.reporting.profiling import profile_to
from leadlag.utils.config import deep_update
from leadlag.utils.yaml import load_yaml
from leadlag.training.run_support import (
    prepare_run_environment,
    read_prices as _read_prices,
    _detect_git,
    _env_info,
    _set_seed,
)

def _merge_extends(cfg_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(cfg_path)
    if "extends" in cfg and cfg["extends"]:
        base_path = (cfg_path.parent / cfg["extends"]).resolve()
        base = load_yaml(base_path)

        # shallow merge: base <- cfg
        merged = deep_update(base, {k: v for k, v in cfg.items() if k != "extends"})
        return merged
    return cfg


def _validate_scenario_schema(cfg: Dict[str, Any], *, scenario: str) -> None:
    """Ensure the merged scenario config contains required sections."""

    required_sections = ("run", "data", "analysis")
    missing = [section for section in required_sections if section not in cfg]
    if missing:
        raise ValueError(f"Scenario '{scenario}' missing sections: {', '.join(missing)}")

    run_section = cfg["run"]
    if not isinstance(run_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'run' must be a mapping")

    data_section = cfg["data"]
    if not isinstance(data_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'data' must be a mapping")
    price_csv = data_section.get("price_csv")
    if not isinstance(price_csv, str) or not price_csv:
        raise ValueError(f"Scenario '{scenario}' must define data.price_csv as a string path")

    analysis_section = cfg["analysis"]
    if not isinstance(analysis_section, dict):
        raise TypeError(f"Scenario '{scenario}' section 'analysis' must be a mapping")

    method = analysis_section.get("method")
    if not isinstance(method, str) or not method:
        raise ValueError(f"Scenario '{scenario}' must define analysis.method as a string")
    lookback = analysis_section.get("lookback")
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError(
            f"Scenario '{scenario}' must define analysis.lookback as a positive integer"
        )

    metrics_cfg = cfg.get("metrics")
    if metrics_cfg is not None and not isinstance(metrics_cfg, dict):
        raise TypeError(f"Scenario '{scenario}' section 'metrics' must be a mapping when provided")
def _config_to_leadlag(cfg: Dict[str, Any]) -> LeadLagConfig:
    # Flatten config into the dict expected by LeadLagConfig.from_dict
    a = cfg["analysis"]
    merged = dict(a)
    # carry method-specific block under the same key name
    if a.get("method") and a.get(a["method"]):
        merged[a["method"]] = a[a["method"]]
    return LeadLagConfig.from_dict(merged)


def run_scenario(
    config_path: str,
    out_root: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Path:
    overrides = deepcopy(overrides) if overrides else {}
    raw_cfg = overrides.pop("_raw_config", None)

    if raw_cfg is not None:
        cfg = deepcopy(raw_cfg)
        cfg_path = Path(config_path)
    else:
        cfg_path = Path(config_path)
        cfg = _merge_extends(cfg_path)
        if overrides:
            cfg = deep_update(cfg, overrides)

    if overrides and raw_cfg is not None:
        cfg = deep_update(cfg, overrides)

    _validate_scenario_schema(cfg, scenario=cfg["run"].get("run_name", Path(config_path).stem))

    run_name = cfg["run"].get("run_name", "auto")
    preparation = prepare_run_environment(
        cfg,
        cfg_path=cfg_path,
        module="scenario",
        logger_name="run_scenario",
        out_root=out_root,
        run_name=run_name,
        profile_label="load_data",
    )
    out_dir = preparation.out_dir
    logger = preparation.logger
    logger.info("Starting scenario run: %s", preparation.run_name)
    prices = preparation.prices

    # build analyzer
    ll_cfg = _config_to_leadlag(cfg)
    analyzer = LeadLagAnalyzer(ll_cfg)

    # compute rolling matrices
    with profile_to(out_dir, label="analyze"):
        rolling = analyzer.analyze(prices, return_rolling=True)
    # compute metrics
    with profile_to(out_dir, label="metrics"):
        metrics_df = compute_metrics_timeseries(rolling)
    metrics_df.to_csv(out_dir / "metrics_timeseries.csv", index=True)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # Log summary metrics to MLflow if available and enabled via env
    import os as _os  # local import to avoid polluting namespace

    mlflow_enabled_env = _os.getenv("MLFLOW_ENABLED", "1").lower() in ("1", "true", "yes")
    if MLFLOW_AVAILABLE and mlflow_enabled_env:
        try:  # pragma: no cover - integration path
            with mlflow.start_run(run_name=run_name, nested=False):
                for _, row in summary.iterrows():
                    metric_name = row.get("metric", "metric")
                    for col, val in row.items():
                        if col == "metric":
                            continue
                        if isinstance(val, (int, float)) and not (val != val):  # NaN check
                            mlflow.log_metric(f"{metric_name}_{col}", float(val))
                # log artifacts
                mlflow.log_artifact(str(out_dir / "config_merged.yaml"))
                mlflow.log_artifact(str(out_dir / "summary.csv"))
                if (out_dir / "metrics_timeseries.csv").exists():
                    mlflow.log_artifact(str(out_dir / "metrics_timeseries.csv"))
                for plot in ["fig_signal_strength.png", "fig_stability.png"]:
                    p = out_dir / plot
                    if p.exists():
                        mlflow.log_artifact(str(p))
        except Exception:
            logger.warning("MLflow logging failed; continuing without MLflow.")

    # plots
    headless = False
    try:
        headless = bool(cfg.get("metrics", {}).get("headless", False))
    except Exception:
        headless = False
    if (not headless) and "metrics" in cfg and "plots" in cfg["metrics"]:
        if "signal_strength" in cfg["metrics"]["plots"]:
            try:
                plot_signal_strength(metrics_df, out_dir / "fig_signal_strength.png")
            except Exception:
                logger.warning("Plot generation failed: signal_strength")
        if "stability" in cfg["metrics"]["plots"]:
            try:
                plot_stability(metrics_df, out_dir / "fig_stability.png")
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
    ap.add_argument("--config", type=str, required=True, help="Path to scenario YAML")
    ap.add_argument("--out", type=str, default=None, help="Output root directory")
    args = ap.parse_args()
    out_dir = run_scenario(args.config, args.out)
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
