"""CLI entry point for orchestrating RL + signature experiments."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import logging
import platform
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import pandas as pd
import yaml

from leadlag.cli.errors import ERROR_DEPENDENCY, ERROR_VALUE, emit_error
from leadlag.cli.formatters import (
    add_format_flags,
    emit_formatted_output,
    finalize_format_args,
)
from leadlag.evaluation.finance_kpis import CANDIDATE_RETURN_COLUMNS, compute_kpis
from leadlag.reporting.logging_utils import ContextFilter, StructuredAdapter, get_logger, setup_logging
from leadlag.training.run_rl import run_rl
from leadlag.utils.repro import collect_environment_manifest
from leadlag.utils.resources import resolve_path
from leadlag import hydra_main

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None  # type: ignore[assignment]

SCENARIO_NAMES = ["rl_ppo", "rl_ppo_sharpe", "rl_ppo_drawdown", "rl_ppo_lstm"]
DEFAULT_RESULTS_ROOT = Path("/kaggle/working/results_rl_signature")
DEFAULT_BUNDLE_ROOT = Path("/kaggle/working/rl_signature_bundle")


@dataclass
class TrainingProfile:
    name: str
    seeds: List[int]
    config_path: Path
    total_env_steps: int | None = None
    windows: int | None = None
    periods_per_year: int | None = None
    overrides: Dict[str, object] = field(default_factory=dict)


@dataclass
class ScenarioPlan:
    name: str
    config_path: Path
    seeds: List[int]
    overrides: Dict[str, object] = field(default_factory=dict)


@dataclass
class RunRecord:
    scenario: str
    seed: int
    status: str
    run_dir: Path | None = None
    error: str | None = None
    returns_column: str | None = None
    metrics: Dict[str, float] = field(default_factory=dict)
    summary_path: Path | None = None
    metrics_path: Path | None = None
    scenario_log: Path | None = None
    plots: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScenarioOutcome:
    name: str
    duration_seconds: float
    attempted: bool
    failed: bool
    log_path: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL + signature training bundle with standardized outputs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory to store run outputs and metrics.",
    )
    parser.add_argument(
        "--bundle-root",
        "--out",
        dest="bundle_root",
        type=Path,
        default=DEFAULT_BUNDLE_ROOT,
        help="Directory for aggregated artifacts (reports, bundles).",
    )
    parser.add_argument(
        "--training-profile",
        choices=["smoke", "paper"],
        default="paper",
        help="Selects the training preset to execute (smoke or paper scale).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional path to write pipeline logs.",
    )
    add_format_flags(parser, default="text")
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalize_paths(args: argparse.Namespace) -> None:
    args.results_root = Path(args.results_root).expanduser().resolve()
    args.bundle_root = Path(args.bundle_root).expanduser().resolve()
    if args.log_path is not None:
        args.log_path = Path(args.log_path).expanduser().resolve()


def _ensure_directories(results_root: Path, bundle_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)


def _prepare_bundle_dirs(bundle_root: Path) -> Dict[str, Path]:
    summary_dir = bundle_root / "summary"
    plots_dir = bundle_root / "plots"
    logs_dir = bundle_root / "logs"
    metadata_dir = bundle_root / "metadata"
    for path in (summary_dir, plots_dir, logs_dir, metadata_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "summary": summary_dir,
        "plots": plots_dir,
        "logs": logs_dir,
        "metadata": metadata_dir,
    }


def _check_dependencies(args, logger) -> tuple[bool, str | None]:
    details = {
        "kaggle": "pip install --no-binary iisignature --no-build-isolation iisignature",
        "local": "PIP_NO_BUILD_ISOLATION=1 pip install --no-binary iisignature iisignature",
    }

    try:
        iisignature = importlib.import_module("iisignature")  # type: ignore
    except ImportError:
        emit_error(
            args,
            code=ERROR_DEPENDENCY,
            message="iisignature is not installed",
            details=details,
        )
        return False, None
    except Exception as exc:  # pragma: no cover - defensive import guard
        emit_error(
            args,
            code=ERROR_DEPENDENCY,
            message="Failed to import iisignature",
            details={"error": str(exc), **details},
        )
        return False, None

    version = getattr(iisignature, "__version__", None)
    logger.info("iisignature available", context={"version": version})
    return True, version


def _resolve_scenario_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    resolved = resolve_path("leadlag.configs", f"scenarios/{candidate.stem}.yaml")
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Scenario config not found for '{name}'")
    return resolved


def _load_effective_config(plan: ScenarioPlan) -> Dict[str, Any]:
    raw_cfg = hydra_main._load_scenario_cfg(plan.name)  # pylint: disable=protected-access
    merged_cfg: Mapping[str, Any] = raw_cfg if isinstance(raw_cfg, Mapping) else {}
    merged_cfg = _merge_overrides(dict(merged_cfg), plan.overrides)
    if isinstance(raw_cfg, Mapping):
        merged_cfg.setdefault("runner", raw_cfg.get("runner"))
        analysis_cfg = raw_cfg.get("analysis") if isinstance(raw_cfg.get("analysis"), Mapping) else {}
        if isinstance(analysis_cfg, Mapping):
            merged_cfg.setdefault("analysis", dict(analysis_cfg))
    return merged_cfg


def _validate_scenarios(plans: List[ScenarioPlan], args, logger) -> bool:
    violations: List[Dict[str, Any]] = []
    for plan in plans:
        cfg = _load_effective_config(plan)
        runner = cfg.get("runner")
        analysis = cfg.get("analysis", {}) if isinstance(cfg, Mapping) else {}
        method = analysis.get("method") if isinstance(analysis, Mapping) else None
        if runner != "rl" or method != "signature":
            violations.append(
                {
                    "scenario": plan.name,
                    "runner": runner,
                    "method": method,
                }
            )
    if violations:
        details = {"violations": violations}
        logger.error(
            "Scenario configuration failed validation", context=details
        )
        emit_error(
            args,
            code="invalid_config",
            message="One or more scenarios violated RL+Signature invariants.",
            details=details,
        )
        return False
    return True


def _merge_overrides(*layers: Mapping[str, object]) -> Dict[str, object]:
    merged: Dict[str, object] = {}
    for layer in layers:
        for key, value in layer.items():
            if isinstance(value, Mapping):
                existing = merged.get(key, {})
                new_value: MutableMapping[str, object] = dict(existing) if isinstance(existing, Mapping) else {}
                new_value.update({k: v for k, v in value.items()})
                merged[key] = new_value
            else:
                merged[key] = value
    return merged


def _load_training_profile(profile_name: str) -> TrainingProfile:
    config_path = resolve_path("leadlag.configs", f"training/{profile_name}.yaml")
    if config_path is None:
        raise FileNotFoundError(f"Training profile not found: {profile_name}")

    raw_cfg = yaml.safe_load(config_path.read_text()) or {}
    seeds = [int(seed) for seed in raw_cfg.get("seeds", [0])]
    total_env_steps = raw_cfg.get("total_env_steps")
    windows = raw_cfg.get("windows")
    periods_per_year = raw_cfg.get("periods_per_year")

    overrides: Dict[str, object] = {
        "training": {k: v for k, v in raw_cfg.items() if v is not None},
        "run": {"training_profile": profile_name},
    }
    if total_env_steps is not None:
        overrides.setdefault("rl", {})
        overrides["rl"]["total_timesteps"] = int(total_env_steps)

    return TrainingProfile(
        name=profile_name,
        seeds=seeds,
        config_path=config_path,
        total_env_steps=int(total_env_steps) if total_env_steps is not None else None,
        windows=int(windows) if windows is not None else None,
        periods_per_year=int(periods_per_year) if periods_per_year is not None else None,
        overrides=overrides,
    )


def _build_scenario_plans(profile: TrainingProfile) -> List[ScenarioPlan]:
    plans: List[ScenarioPlan] = []
    for scenario in SCENARIO_NAMES:
        cfg_path = _resolve_scenario_path(scenario)
        plans.append(
            ScenarioPlan(
                name=Path(scenario).stem,
                config_path=cfg_path,
                seeds=list(profile.seeds),
                overrides=dict(profile.overrides),
            )
        )
    return plans


def _scenario_logger(log_dir: Path, scenario: str, level: str) -> StructuredAdapter:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{scenario}.log"
    logger = logging.getLogger(f"pipelines.rl_signature.{scenario}")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(context)s | %(message)s")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter({"scenario": scenario}))
    logger.addHandler(handler)
    return StructuredAdapter(logger, {"context_map": {"scenario": scenario, "log_path": str(log_path)}})


def _derive_returns(df: pd.DataFrame) -> tuple[pd.Series, str] | None:
    for col in CANDIDATE_RETURN_COLUMNS:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 2:
                return series, col
    if "mean_abs_matrix" in df.columns:
        fallback = pd.to_numeric(df["mean_abs_matrix"], errors="coerce").dropna()
        if len(fallback) > 5:
            returns = fallback.pct_change().dropna()
            if len(returns) > 2:
                return returns, "mean_abs_matrix_pct_change"
    return None


def _plot_equity_curve(returns: pd.Series, path: Path, title: str, logger) -> str | None:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        logger.warning("Skipping equity curve plot; matplotlib not available")
        return None
    cumulative = (1 + returns).cumprod()
    plt.figure(figsize=(9, 4))
    try:
        plt.plot(cumulative.index, cumulative.values)
    except Exception:
        plt.plot(range(len(cumulative)), cumulative.values)
    plt.title(title)
    plt.xlabel("index")
    plt.ylabel("equity")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def _plot_histogram(returns: pd.Series, path: Path, title: str, logger) -> str | None:
    if not MATPLOTLIB_AVAILABLE or plt is None:
        logger.warning("Skipping histogram plot; matplotlib not available")
        return None
    plt.figure(figsize=(6, 4))
    plt.hist(returns, bins=min(30, max(5, len(returns) // 4)), alpha=0.8)
    plt.title(title)
    plt.xlabel("returns")
    plt.ylabel("frequency")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def _prepare_run_overrides(plan: ScenarioPlan, seed: int) -> Dict[str, object]:
    run_name = f"{plan.name}_s{seed:02d}"
    seed_override: Dict[str, object] = {"run": {"seed": seed, "run_name": run_name}}
    return _merge_overrides(plan.overrides, seed_override)


def _execute_runs(
    plans: Iterable[ScenarioPlan],
    *,
    results_root: Path,
    bundle_dirs: Mapping[str, Path],
    log_level: str,
    logger,
) -> tuple[List[RunRecord], Dict[str, ScenarioOutcome]]:
    records: List[RunRecord] = []
    outcomes: Dict[str, ScenarioOutcome] = {}
    abort_execution = False
    for plan in plans:
        scenario_start = time.perf_counter()
        scenario_root = results_root / plan.name
        scenario_root.mkdir(parents=True, exist_ok=True)
        scenario_log_path = bundle_dirs["logs"] / f"{plan.name}.log"
        scenario_logger = _scenario_logger(bundle_dirs["logs"], plan.name, log_level)
        scenario_logger.info(
            "Executing scenario",
            context={"scenario": plan.name, "config": str(plan.config_path), "seeds": plan.seeds},
        )
        logger.info(
            "Scenario started",
            context={"scenario": plan.name, "config": str(plan.config_path), "seeds": plan.seeds},
        )
        scenario_attempted = False
        scenario_failed = False
        for seed in plan.seeds:
            scenario_attempted = True
            overrides = _prepare_run_overrides(plan, seed)
            scenario_logger.info(
                "Launching RL runner",
                context={"seed": seed, "overrides": list(overrides.keys())},
            )
            try:
                run_dir = run_rl(str(plan.config_path), str(scenario_root), overrides)
                record = RunRecord(
                    scenario=plan.name,
                    seed=int(seed),
                    status="success",
                    run_dir=Path(run_dir),
                    metrics_path=Path(run_dir) / "metrics_timeseries.csv",
                    summary_path=Path(run_dir) / "summary.csv",
                    scenario_log=scenario_log_path,
                )
                records.append(record)
                scenario_logger.info(
                    "Scenario run completed",
                    context={"seed": seed, "run_dir": str(run_dir)},
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                scenario_logger.exception("Scenario run failed", context={"seed": seed})
                scenario_failed = True
                records.append(
                    RunRecord(
                        scenario=plan.name,
                        seed=int(seed),
                        status="error",
                        error=str(exc),
                        scenario_log=scenario_log_path,
                    )
                )
                abort_execution = True
                break
        scenario_successes = sum(rec.status == "success" and rec.scenario == plan.name for rec in records)
        logger.info(
            "Scenario completed",
            context={"scenario": plan.name, "runs": len(plan.seeds), "successes": scenario_successes},
        )
        outcomes[plan.name] = ScenarioOutcome(
            name=plan.name,
            duration_seconds=time.perf_counter() - scenario_start,
            attempted=scenario_attempted,
            failed=scenario_failed,
            log_path=scenario_log_path,
        )
        if abort_execution:
            logger.warning("Aborting suite after failure", context={"scenario": plan.name})
            break
    logger.info("Finished executing scenarios", context={"count": len(records)})
    for plan in plans:
        if plan.name not in outcomes:
            outcomes[plan.name] = ScenarioOutcome(
                name=plan.name,
                duration_seconds=0.0,
                attempted=False,
                failed=True,
                log_path=bundle_dirs["logs"] / f"{plan.name}.log",
            )
    return records, outcomes


def _process_run_output(
    record: RunRecord,
    *,
    summary_dir: Path,
    plots_dir: Path,
    logger,
) -> RunRecord:
    if record.run_dir is None:
        return record

    metrics_path = record.metrics_path or (record.run_dir / "metrics_timeseries.csv")
    summary_path = record.summary_path or (record.run_dir / "summary.csv")
    record.metrics_path = metrics_path
    record.summary_path = summary_path

    if not metrics_path.exists():
        logger.warning(
            "Missing metrics_timeseries.csv; skipping metrics extraction",
            context={"run_dir": str(record.run_dir)},
        )
        return record

    try:
        metrics_df = pd.read_csv(metrics_path)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        logger.warning(
            "Failed to read metrics_timeseries.csv",
            context={"run_dir": str(record.run_dir), "error": str(exc)},
        )
        return record

    if "date" in metrics_df.columns:
        try:
            metrics_df["date"] = pd.to_datetime(metrics_df["date"])
            metrics_df = metrics_df.sort_values("date")
        except Exception:
            pass

    returns_info = _derive_returns(metrics_df)
    if returns_info is None:
        logger.warning(
            "No return-like column found for KPI computation",
            context={"run_dir": str(record.run_dir)},
        )
        return record

    returns, column_used = returns_info
    record.returns_column = column_used
    kpis = compute_kpis(returns)
    record.metrics.update(kpis)
    record.metrics["total_return"] = float((1 + returns).prod() - 1)

    equity_path = plots_dir / f"{record.run_dir.name}_equity.png"
    hist_path = plots_dir / f"{record.run_dir.name}_hist.png"
    eq_plot = _plot_equity_curve(returns, equity_path, f"Equity curve: {record.run_dir.name}", logger)
    hist_plot = _plot_histogram(returns, hist_path, f"Returns histogram: {record.run_dir.name}", logger)
    if eq_plot:
        record.plots["equity"] = eq_plot
    if hist_plot:
        record.plots["histogram"] = hist_plot

    if summary_path.exists():
        record.summary_path = summary_path

    run_summary_path = summary_dir / f"{record.run_dir.name}_kpis.json"
    run_summary_path.write_text(json.dumps({"metrics": record.metrics, "returns_column": column_used}, indent=2))
    return record


def _summarize_runs(
    records: List[RunRecord],
    *,
    summary_dir: Path,
    plots_dir: Path,
    logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed: List[RunRecord] = []
    for record in records:
        if record.status != "success":
            processed.append(record)
            continue
        processed.append(_process_run_output(record, summary_dir=summary_dir, plots_dir=plots_dir, logger=logger))

    run_rows: List[Dict[str, object]] = []
    for record in processed:
        if record.status != "success" or record.run_dir is None:
            continue
        row: Dict[str, object] = {
            "scenario": record.scenario,
            "seed": record.seed,
            "run_dir": str(record.run_dir),
            "returns_column": record.returns_column,
        }
        row.update(record.metrics)
        if "sharpe_ratio" in record.metrics:
            row.setdefault("sharpe", record.metrics["sharpe_ratio"])
        run_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    if not run_df.empty:
        run_df.to_csv(summary_dir / "rl_signature_runs.csv", index=False)
        logger.info(
            "Wrote run-level KPI summary",
            context={"path": str(summary_dir / "rl_signature_runs.csv")},
        )
    else:
        logger.warning("No successful runs to summarize")

    scenario_df = pd.DataFrame()
    if not run_df.empty:
        metric_mapping: Dict[str, str] = {}
        preferred_metrics = {
            "sharpe": ["sharpe", "sharpe_ratio"],
            "max_drawdown": ["max_drawdown"],
            "total_return": ["total_return"],
            "annualized_return": ["annualized_return"],
        }
        for metric, candidates in preferred_metrics.items():
            for candidate in candidates:
                if candidate in run_df.columns:
                    metric_mapping[metric] = candidate
                    break

        agg: Dict[str, tuple[str, str]] = {}
        for metric, column in metric_mapping.items():
            agg[f"{metric}_mean"] = (column, "mean")
            agg[f"{metric}_std"] = (column, "std")
        if agg:
            grouped = run_df.groupby("scenario").agg(**agg)
            grouped["run_count"] = run_df.groupby("scenario")["seed"].size()
            scenario_df = grouped.reset_index()
            scenario_df.to_csv(summary_dir / "rl_signature_scenarios.csv", index=False)
            logger.info(
                "Wrote scenario-level KPI summary",
                context={"path": str(summary_dir / "rl_signature_scenarios.csv")},
            )
    return run_df, scenario_df


def _plot_comparisons(scenario_df: pd.DataFrame, plots_dir: Path, logger) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    if scenario_df.empty:
        logger.warning("Skipping comparison plots; no scenario summary data")
        return plots

    metric_mapping = {
        "sharpe": "sharpe_mean",
        "max_drawdown": "max_drawdown_mean",
        "total_return": "total_return_mean",
    }
    for label, column in metric_mapping.items():
        if column not in scenario_df.columns:
            logger.warning("Metric missing for comparison plot", context={"metric": column})
            continue
        if not MATPLOTLIB_AVAILABLE or plt is None:
            logger.warning("Skipping comparison plots; matplotlib not available")
            break
        path = plots_dir / f"scenario_{label}.png"
        plt.figure(figsize=(7, 4))
        plt.bar(scenario_df["scenario"], scenario_df[column])
        plt.title(f"Scenario comparison: {label}")
        plt.xlabel("scenario")
        plt.ylabel(label)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        plots[label] = str(path)
    return plots


def _write_metadata(
    *,
    metadata_dir: Path,
    results_root: Path,
    bundle_root: Path,
    profile: TrainingProfile,
    plans: List[ScenarioPlan],
    run_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    scenario_statuses: List[Dict[str, Any]] | None = None,
    scenario_outcomes: Mapping[str, ScenarioOutcome] | None = None,
    env_diagnostics: Mapping[str, Any] | None = None,
    runtime_seconds: float | None = None,
    logger,
) -> Dict[str, str]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    env_path = metadata_dir / "environment.json"
    environment_payload = collect_environment_manifest()
    if env_diagnostics:
        environment_payload["rl_signature"] = env_diagnostics
    env_path.write_text(json.dumps(environment_payload, indent=2))

    scenarios_path = metadata_dir / "scenarios.json"
    status_lookup = {entry.get("scenario"): entry for entry in (scenario_statuses or [])}
    scenarios_payload = []
    for plan in plans:
        effective_cfg = _load_effective_config(plan)
        analysis = effective_cfg.get("analysis", {}) if isinstance(effective_cfg, Mapping) else {}
        method = analysis.get("method") if isinstance(analysis, Mapping) else None
        scenario_runs = run_df[run_df["scenario"] == plan.name] if not run_df.empty else pd.DataFrame()
        outcome = (scenario_outcomes or {}).get(plan.name)
        scenario_entry = {
            "name": plan.name,
            "config_path": str(plan.config_path),
            "runner": effective_cfg.get("runner"),
            "analysis_method": method,
            "seeds": plan.seeds,
            "overrides": plan.overrides,
            "run_dirs": scenario_runs["run_dir"].tolist() if not scenario_runs.empty else [],
            "duration_seconds": outcome.duration_seconds if outcome else None,
            "log_path": str(outcome.log_path) if outcome and outcome.log_path else None,
        }
        status_entry = status_lookup.get(plan.name) or {}
        scenario_entry["status"] = status_entry.get(
            "status",
            "success" if len(scenario_entry["run_dirs"]) == len(plan.seeds) and scenario_entry["run_dirs"] else "failed",
        )
        scenarios_payload.append(scenario_entry)
    scenarios_path.write_text(json.dumps(scenarios_payload, indent=2))

    summary_path = metadata_dir / "rl_signature_summary.json"
    summary_payload: Dict[str, object] = {
        "profile": profile.name,
        "results_root": str(results_root),
        "bundle_root": str(bundle_root),
        "runs": len(run_df),
        "scenarios": len(scenario_df) if not scenario_df.empty else 0,
        "metrics_available": not run_df.empty,
        "runtime_seconds": runtime_seconds,
        "environment": env_diagnostics,
    }
    if not scenario_df.empty:
        summary_payload["scenario_metrics"] = scenario_df.to_dict(orient="list")
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    summary_md = metadata_dir / "summary.md"
    lines = [f"# RL + signature bundle ({profile.name})", ""]
    if run_df.empty:
        lines.append("No successful runs were completed.")
    else:
        lines.append(f"Completed {len(run_df)} run(s) across {run_df['scenario'].nunique()} scenario(s).")
        if not scenario_df.empty:
            lines.append("")
            lines.append("## Scenario KPIs")
            lines.append("")
            for _, row in scenario_df.iterrows():
                lines.append(
                    f"- **{row['scenario']}**: sharpe={row.get('sharpe_mean', float('nan')):.3f}, "
                    f"max_drawdown={row.get('max_drawdown_mean', float('nan')):.3f}, "
                    f"total_return={row.get('total_return_mean', float('nan')):.3f}"
                )
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Metadata written", context={"metadata_dir": str(metadata_dir)})
    return {
        "environment": str(env_path),
        "scenarios": str(scenarios_path),
        "summary": str(summary_path),
        "summary_md": str(summary_md),
    }


def _gather_environment_diagnostics() -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        diagnostics["leadlag_version"] = importlib_metadata.version("leadlag")
    except importlib_metadata.PackageNotFoundError:
        diagnostics["leadlag_version"] = None

    try:
        diagnostics["iisignature_version"] = importlib_metadata.version("iisignature")
    except importlib_metadata.PackageNotFoundError:
        diagnostics["iisignature_version"] = None

    if importlib.util.find_spec("torch") is not None:
        import importlib

        torch = importlib.import_module("torch")
        diagnostics["torch_version"] = getattr(torch, "__version__", None)
        diagnostics["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics["cuda_device"] = torch.cuda.get_device_name(0)
            diagnostics["cuda_device_count"] = torch.cuda.device_count()
    else:
        diagnostics["torch_version"] = None
        diagnostics["cuda_available"] = False
    return diagnostics


def _log_environment_diagnostics(logger) -> Dict[str, Any]:
    diagnostics = _gather_environment_diagnostics()
    logger.info("Environment diagnostics", context=diagnostics)
    return diagnostics


def _build_scenario_statuses(
    plans: List[ScenarioPlan],
    run_records: List[RunRecord],
    scenario_df: pd.DataFrame,
    scenario_outcomes: Mapping[str, ScenarioOutcome],
) -> List[Dict[str, Any]]:
    metrics_lookup: Dict[str, Dict[str, Any]] = {}
    if not scenario_df.empty:
        for row in scenario_df.to_dict(orient="records"):
            scenario_name = row.pop("scenario", None)
            if scenario_name:
                metrics_lookup[scenario_name] = row

    scenario_statuses: List[Dict[str, Any]] = []
    for plan in plans:
        scenario_runs = [rec for rec in run_records if rec.scenario == plan.name]
        successes = sum(rec.status == "success" for rec in scenario_runs)
        failures = sum(rec.status != "success" for rec in scenario_runs)
        outcome = scenario_outcomes.get(plan.name)
        if outcome and not outcome.attempted:
            status = "skipped"
        elif failures:
            status = "failed"
        elif successes > 0:
            status = "success"
        else:
            status = "pending"
        scenario_statuses.append(
            {
                "scenario": plan.name,
                "status": status,
                "planned_seeds": plan.seeds,
                "successful_runs": successes,
                "failed_runs": failures,
                "metrics": metrics_lookup.get(plan.name),
                "duration_seconds": outcome.duration_seconds if outcome else None,
                "log_path": str(outcome.log_path) if outcome and outcome.log_path else None,
            }
        )
    return scenario_statuses


def _build_bundle(bundle_root: Path, artifacts: Iterable[Path]) -> Path:
    zip_path = bundle_root / "rl_signature_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in artifacts:
            if not item.exists():
                continue
            if item.is_dir():
                for child in item.rglob("*"):
                    if child.is_file():
                        try:
                            arcname = child.relative_to(bundle_root)
                        except ValueError:
                            arcname = child.name
                        zf.write(child, arcname=str(arcname))
            else:
                try:
                    arcname = item.relative_to(bundle_root)
                except ValueError:
                    arcname = item.name
                zf.write(item, arcname=str(arcname))
    return zip_path


def main(argv: Sequence[str] | None = None) -> int:
    start_time = time.perf_counter()
    args = parse_args(argv)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-rl-signature"
    if argv:
        command = "leadlag-rl-signature " + " ".join(argv)

    _normalize_paths(args)
    _ensure_directories(args.results_root, args.bundle_root)

    bundle_dirs = _prepare_bundle_dirs(args.bundle_root)
    log_path = args.log_path or bundle_dirs["logs"] / "rl_signature.log"
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "rl_signature"})
    logger = get_logger(
        "pipelines.rl_signature",
        context={
            "results_root": args.results_root,
            "bundle_root": args.bundle_root,
            "profile": args.training_profile,
            "log_path": log_path,
        },
    )
    logger.info(
        "Starting RL + signature suite",
        context={"log_path": log_path, "logs_root": bundle_dirs["logs"]},
    )
    env_diagnostics = _log_environment_diagnostics(logger)

    deps_ok, iisig_version = _check_dependencies(args, logger)
    if not deps_ok:
        return 1

    try:
        profile = _load_training_profile(args.training_profile)
    except Exception as exc:  # pragma: no cover - defensive
        emit_error(
            args,
            code=ERROR_VALUE,
            message="Unsupported training profile.",
            details={"profile": args.training_profile, "error": str(exc)},
        )
        return 1

    logger.info(
        "Resolved training profile",
        context={"profile": profile.name, "seeds": profile.seeds, "scenarios": SCENARIO_NAMES},
    )

    plans = _build_scenario_plans(profile)
    if not _validate_scenarios(plans, args, logger):
        return 1

    run_records, scenario_outcomes = _execute_runs(
        plans,
        results_root=args.results_root,
        bundle_dirs=bundle_dirs,
        log_level=str(args.log_level),
        logger=logger,
    )

    run_df, scenario_df = _summarize_runs(
        run_records,
        summary_dir=bundle_dirs["summary"],
        plots_dir=bundle_dirs["plots"],
        logger=logger,
    )

    scenario_statuses = _build_scenario_statuses(plans, run_records, scenario_df, scenario_outcomes)

    artifact_warnings: List[dict[str, Any]] = []
    try:
        comparison_plots = _plot_comparisons(scenario_df, bundle_dirs["plots"], logger)
    except Exception as exc:  # pragma: no cover - defensive bundle guard
        comparison_plots = {}
        logger.warning("Failed to generate comparison plots", context={"error": str(exc)})
        artifact_warnings.append(
            {
                "code": "artifact_generation_failed",
                "message": "Failed to generate comparison plots.",
                "details": {"error": str(exc)},
            }
        )

    bundle_artifacts = [args.bundle_root, Path(log_path)]
    bundle_path: Path | None = None
    try:
        bundle_path = _build_bundle(args.bundle_root, bundle_artifacts)
    except Exception as exc:  # pragma: no cover - defensive bundle guard
        logger.warning("Failed to build bundle", context={"error": str(exc)})
        artifact_warnings.append(
            {
                "code": "artifact_generation_failed",
                "message": "Failed to build bundle archive.",
                "details": {"error": str(exc)},
            }
        )

    runtime_seconds = time.perf_counter() - start_time

    try:
        metadata_paths = _write_metadata(
            metadata_dir=bundle_dirs["metadata"],
            results_root=args.results_root,
            bundle_root=args.bundle_root,
            profile=profile,
            plans=plans,
            run_df=run_df,
            scenario_df=scenario_df,
            scenario_statuses=scenario_statuses,
            scenario_outcomes=scenario_outcomes,
            env_diagnostics=env_diagnostics,
            runtime_seconds=runtime_seconds,
            logger=logger,
        )
    except Exception as exc:  # pragma: no cover - defensive bundle guard
        metadata_paths = {}
        logger.warning("Failed to write metadata", context={"error": str(exc)})
        artifact_warnings.append(
            {
                "code": "artifact_generation_failed",
                "message": "Failed to write metadata.",
                "details": {"error": str(exc)},
            }
        )

    artifacts: Dict[str, Any] = {
        "results_root": str(args.results_root),
        "bundle_root": str(args.bundle_root),
        "log_path": str(log_path),
        "logs_root": str(bundle_dirs["logs"]),
        "summary_dir": str(bundle_dirs["summary"]),
        "plots_dir": str(bundle_dirs["plots"]),
        "logs_dir": str(bundle_dirs["logs"]),
        "metadata_dir": str(bundle_dirs["metadata"]),
        "bundle_zip": str(bundle_path) if bundle_path else None,
        "runs_summary_csv": str(bundle_dirs["summary"] / "rl_signature_runs.csv") if not run_df.empty else None,
        "scenarios_summary_csv": str(bundle_dirs["summary"] / "rl_signature_scenarios.csv") if not scenario_df.empty else None,
    }
    artifacts.update(metadata_paths)
    artifacts.update(comparison_plots)

    run_errors = [
        {"code": "run_failed", "message": rec.error, "details": {"scenario": rec.scenario, "seed": rec.seed}}
        for rec in run_records
        if rec.status != "success" and rec.error
    ]

    data = {
        "status": "success" if not run_errors else "failed",
        "training_profile": profile.name,
        "environment": env_diagnostics,
        "paths": {
            "results_root": str(args.results_root),
            "bundle_root": str(args.bundle_root),
            "log_path": str(log_path),
            "logs_root": str(bundle_dirs["logs"]),
        },
        "dependencies": {
            "required": ["iisignature"],
            "iisignature_version": iisig_version,
            "validated": deps_ok,
        },
        "runs": [
            {
                "scenario": rec.scenario,
                "seed": rec.seed,
                "status": rec.status,
                "run_dir": str(rec.run_dir) if rec.run_dir else None,
                "metrics": rec.metrics,
                "returns_column": rec.returns_column,
                "log_path": str(rec.scenario_log) if rec.scenario_log else None,
            }
            for rec in run_records
        ],
        "scenarios": scenario_statuses,
        "summaries": {
            "run_level": str(bundle_dirs["summary"] / "rl_signature_runs.csv") if not run_df.empty else None,
            "scenario_level": str(bundle_dirs["summary"] / "rl_signature_scenarios.csv") if not scenario_df.empty else None,
            "comparison_plots": comparison_plots,
            "scenario_metrics": scenario_df.to_dict(orient="records") if not scenario_df.empty else [],
        },
        "runtime_seconds": runtime_seconds,
    }

    errors = run_errors + artifact_warnings

    text_lines = [
        f"Profile: {profile.name}",
        f"Results root: {args.results_root}",
        f"Bundle root: {args.bundle_root}",
        f"Runs executed: {len(run_records)} (success={sum(rec.status == 'success' for rec in run_records)})",
        f"Runtime: {runtime_seconds:.2f}s",
    ]
    if run_df.empty:
        text_lines.append("No metrics available for aggregation.")
    else:
        text_lines.append(f"Aggregated {len(run_df)} run(s) into summaries.")
    if artifact_warnings:
        text_lines.append("Some artifacts could not be generated; see logs for details.")

    overall_success = not run_errors
    message = "RL+Signature suite completed." if overall_success else "RL+Signature suite completed with errors."

    emit_formatted_output(
        args,
        data=data,
        artifacts=artifacts,
        errors=errors if errors else None,
        text="\n".join(text_lines),
        message=message,
        pretty=True,
        command=command,
        success=overall_success,
    )
    return 0 if overall_success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
