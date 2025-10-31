from __future__ import annotations

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
from leadlag.models.LeadLag_main import LeadLagAnalyzer
from leadlag.models.config import LeadLagConfig
from leadlag.training.pipeline import (
    PipelineContext,
    ScenarioPipeline,
    ScenarioPipelineHooks,
)
from leadlag.training.run_support import prepare_run_environment
from leadlag.training.scenario_config import _merge_extends, _validate_scenario_schema
from leadlag.utils.config import deep_update


def _config_to_leadlag(cfg: Dict[str, Any]) -> LeadLagConfig:
    """Flatten a scenario configuration into a LeadLag config."""

    analysis_cfg = cfg["analysis"]
    merged: Dict[str, Any] = dict(analysis_cfg)
    method = analysis_cfg.get("method")
    if method and analysis_cfg.get(method):
        merged[method] = analysis_cfg[method]
    return LeadLagConfig.from_dict(merged)


def _build_analyzer(cfg: Dict[str, Any]) -> LeadLagAnalyzer:
    return LeadLagAnalyzer(_config_to_leadlag(cfg))


def _run_analysis(analyzer: LeadLagAnalyzer, prices, cfg: Dict[str, Any]):  # pragma: no cover - thin wrapper
    return analyzer.analyze(prices, return_rolling=True)


def _format_label(label: Any) -> str:
    try:
        value = label.date()
    except Exception:  # pragma: no cover - fallback for non-date indices
        value = label
    return str(value)


def _sample_matrix_artifact(context: PipelineContext) -> None:
    """Persist a couple of rolling matrices for manual inspection."""

    rolling = context.rolling
    try:
        length = len(rolling)
    except Exception:  # pragma: no cover - guardrail
        return

    if length <= 0:
        return

    try:
        index = rolling.index
    except Exception:  # pragma: no cover - guardrail
        return

    try:
        first = index[0]
        last = index[-1]
    except Exception:  # pragma: no cover - guardrail
        return

    out_dir = context.preparation.out_dir
    for label in (first, last):
        try:
            matrix = rolling[label]
            to_csv = getattr(matrix, "to_csv", None)
            if callable(to_csv):
                to_csv(out_dir / f"matrix_{_format_label(label)}.csv")
        except Exception:  # pragma: no cover - best effort
            continue


def _mlflow_hook(context: PipelineContext) -> None:
    if not MLFLOW_AVAILABLE:  # pragma: no cover - requires optional dependency
        return

    import os as _os

    enabled = _os.getenv("MLFLOW_ENABLED", "1").lower() in {"1", "true", "yes"}
    if not enabled:
        return

    logger = context.preparation.logger
    run_name = context.preparation.run_name
    summary = context.summary

    try:  # pragma: no cover - integration path
        with mlflow.start_run(run_name=run_name, nested=False):
            for _, row in summary.iterrows():
                metric_name = row.get("metric", "metric")
                for col, val in row.items():
                    if col == "metric":
                        continue
                    if isinstance(val, (int, float)) and not (val != val):
                        mlflow.log_metric(f"{metric_name}_{col}", float(val))

            mlflow.log_artifact(str(context.preparation.out_dir / "config_merged.yaml"))
            mlflow.log_artifact(str(context.summary_path))
            if context.metrics_path.exists():
                mlflow.log_artifact(str(context.metrics_path))

            for plot_name in ("fig_signal_strength.png", "fig_stability.png"):
                plot_path = context.preparation.out_dir / plot_name
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path))
    except Exception:
        logger.warning("MLflow logging failed; continuing without MLflow.")


def _plotting_hook(context: PipelineContext) -> None:
    cfg = context.cfg
    metrics_cfg = cfg.get("metrics", {}) if isinstance(cfg.get("metrics"), dict) else {}

    try:
        headless = bool(metrics_cfg.get("headless", False))
    except Exception:  # pragma: no cover - guardrail
        headless = False

    if headless:
        return

    plots_cfg = metrics_cfg.get("plots")
    if not plots_cfg:
        return

    logger = context.preparation.logger
    metrics_df = context.metrics
    out_dir = context.preparation.out_dir

    def _is_requested(name: str) -> bool:
        if isinstance(plots_cfg, dict):
            return bool(plots_cfg.get(name))
        try:
            return name in plots_cfg
        except Exception:  # pragma: no cover - guardrail
            return False

    if _is_requested("signal_strength"):
        try:
            plot_signal_strength(metrics_df, out_dir / "fig_signal_strength.png")
        except Exception:
            logger.warning("Plot generation failed: signal_strength")

    if _is_requested("stability"):
        try:
            plot_stability(metrics_df, out_dir / "fig_stability.png")
        except Exception:
            logger.warning("Plot generation failed: stability")


def _create_pipeline() -> ScenarioPipeline:
    hooks = ScenarioPipelineHooks(
        mlflow=_mlflow_hook if MLFLOW_AVAILABLE else None,
        plotting=_plotting_hook,
    )
    return ScenarioPipeline(
        analyzer_factory=_build_analyzer,
        analysis_runner=_run_analysis,
        metrics_computer=compute_metrics_timeseries,
        metrics_summarizer=summarize_metrics,
        artifact_generators=[_sample_matrix_artifact],
        hooks=hooks,
    )


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

    scenario_name = cfg["run"].get("run_name", Path(config_path).stem)
    _validate_scenario_schema(cfg, scenario=scenario_name)

    preparation = prepare_run_environment(
        cfg,
        cfg_path=cfg_path,
        module="scenario",
        logger_name="run_scenario",
        out_root=out_root,
        run_name=scenario_name,
        profile_label="load_data",
    )

    pipeline = _create_pipeline()
    result = pipeline.run(cfg, preparation)

    preparation.logger.info("Scenario run completed", extra={"out_dir": str(result.out_dir)})
    return result.out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to scenario YAML")
    ap.add_argument("--out", type=str, default=None, help="Output root directory")
    args = ap.parse_args()
    run_scenario(args.config, args.out)


__all__ = [
    "run_scenario",
    "_merge_extends",
    "_validate_scenario_schema",
]

