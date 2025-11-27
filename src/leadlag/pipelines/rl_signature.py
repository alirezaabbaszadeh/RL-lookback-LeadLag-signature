"""CLI entry point for orchestrating RL + signature experiments."""

from __future__ import annotations

import argparse
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

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None
if MATPLOTLIB_AVAILABLE:
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None  # type: ignore[assignment]


@dataclass
class TrainingProfile:
    name: str
    seeds: List[int]
    scenarios: List[str]
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


TRAINING_PROFILES: Dict[str, TrainingProfile] = {
    "smoke": TrainingProfile(
        name="smoke",
        seeds=[7],
        scenarios=["rl_ppo"],
        overrides={
            "training": {"preset_name": "smoke"},
            "run": {"training_preset": "smoke"},
            "rl": {"total_timesteps": 2000, "n_steps": 128, "batch_size": 128},
        },
    ),
    "paper": TrainingProfile(
        name="paper",
        seeds=[7, 13, 23],
        scenarios=["rl_ppo", "rl_ppo_lstm"],
        overrides={"training": {"preset_name": "paper"}, "run": {"training_preset": "paper"}},
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL + signature training bundle with standardized outputs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory to store run outputs and metrics.",
    )
    parser.add_argument(
        "--bundle-root",
        "--out",
        dest="bundle_root",
        type=Path,
        default=Path("results/bundles"),
        help="Directory for aggregated artifacts (reports, bundles).",
    )
    parser.add_argument(
        "--training-profile",
        choices=["smoke", "paper"],
        default="smoke",
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


def _check_dependencies(logger) -> tuple[bool, list[str]]:
    required = ["stable_baselines3", "torch", "iisignature"]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        logger.error(
            "Missing required dependencies",
            context={"missing": ", ".join(missing)},
        )
        return False, missing
    logger.info("All required dependencies available")
    return True, []


def _resolve_scenario_path(name: str) -> Path:
    candidate = Path(name)
    if candidate.exists():
        return candidate
    resolved = resolve_path("leadlag.configs", f"scenarios/{candidate.stem}.yaml")
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Scenario config not found for '{name}'")
    return resolved


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


def _plan_profile(profile_name: str) -> TrainingProfile | None:
    return TRAINING_PROFILES.get(profile_name)


def _build_scenario_plans(profile: TrainingProfile) -> List[ScenarioPlan]:
    plans: List[ScenarioPlan] = []
    for scenario in profile.scenarios:
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
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for plan in plans:
        scenario_logger = _scenario_logger(bundle_dirs["logs"], plan.name, log_level)
        scenario_logger.info(
            "Executing scenario",
            context={"scenario": plan.name, "config": str(plan.config_path), "seeds": plan.seeds},
        )
        logger.info(
            "Scenario started",
            context={"scenario": plan.name, "config": str(plan.config_path), "seeds": plan.seeds},
        )
        for seed in plan.seeds:
            overrides = _prepare_run_overrides(plan, seed)
            scenario_logger.info(
                "Launching RL runner",
                context={"seed": seed, "overrides": list(overrides.keys())},
            )
            try:
                run_dir = run_rl(str(plan.config_path), str(results_root), overrides)
                record = RunRecord(
                    scenario=plan.name,
                    seed=int(seed),
                    status="success",
                    run_dir=Path(run_dir),
                    metrics_path=Path(run_dir) / "metrics_timeseries.csv",
                    summary_path=Path(run_dir) / "summary.csv",
                    scenario_log=Path(bundle_dirs["logs"]) / f"{plan.name}.log",
                )
                records.append(record)
                scenario_logger.info(
                    "Scenario run completed",
                    context={"seed": seed, "run_dir": str(run_dir)},
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                scenario_logger.exception("Scenario run failed", context={"seed": seed})
                records.append(
                    RunRecord(
                        scenario=plan.name,
                        seed=int(seed),
                        status="error",
                        error=str(exc),
                        scenario_log=Path(bundle_dirs["logs"]) / f"{plan.name}.log",
                    )
                )
        scenario_successes = sum(rec.status == "success" and rec.scenario == plan.name for rec in records)
        logger.info(
            "Scenario completed",
            context={"scenario": plan.name, "runs": len(plan.seeds), "successes": scenario_successes},
        )
    logger.info("Finished executing scenarios", context={"count": len(records)})
    return records


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
        run_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    if not run_df.empty:
        run_df.to_csv(summary_dir / "runs.csv", index=False)
        logger.info("Wrote run-level KPI summary", context={"path": str(summary_dir / "runs.csv")})
    else:
        logger.warning("No successful runs to summarize")

    scenario_df = pd.DataFrame()
    metric_cols = ["annualized_return", "sharpe_ratio", "max_drawdown", "total_return"]
    if not run_df.empty:
        agg: Dict[str, str] = {col: "mean" for col in metric_cols if col in run_df.columns}
        if agg:
            scenario_df = run_df.groupby("scenario").agg(agg).reset_index()
            scenario_df["num_runs"] = run_df.groupby("scenario")["seed"].size().values
            scenario_df.to_csv(summary_dir / "scenarios.csv", index=False)
            logger.info("Wrote scenario-level KPI summary", context={"path": str(summary_dir / "scenarios.csv")})
    return run_df, scenario_df


def _plot_comparisons(scenario_df: pd.DataFrame, plots_dir: Path, logger) -> Dict[str, str]:
    plots: Dict[str, str] = {}
    if scenario_df.empty:
        logger.warning("Skipping comparison plots; no scenario summary data")
        return plots

    for metric in ("sharpe_ratio", "max_drawdown", "total_return"):
        if metric not in scenario_df.columns:
            logger.warning("Metric missing for comparison plot", context={"metric": metric})
            continue
        if not MATPLOTLIB_AVAILABLE or plt is None:
            logger.warning("Skipping comparison plots; matplotlib not available")
            break
        path = plots_dir / f"scenario_{metric}.png"
        plt.figure(figsize=(7, 4))
        plt.bar(scenario_df["scenario"], scenario_df[metric])
        plt.title(f"Scenario comparison: {metric}")
        plt.xlabel("scenario")
        plt.ylabel(metric)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        plt.close()
        plots[metric] = str(path)
    return plots


def _write_metadata(
    *,
    metadata_dir: Path,
    profile: TrainingProfile,
    plans: List[ScenarioPlan],
    run_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    logger,
) -> Dict[str, str]:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    env_path = metadata_dir / "environment.json"
    env_path.write_text(json.dumps(collect_environment_manifest(), indent=2))

    scenarios_path = metadata_dir / "scenarios.json"
    scenarios_payload = [
        {"name": plan.name, "config_path": str(plan.config_path), "seeds": plan.seeds, "overrides": plan.overrides}
        for plan in plans
    ]
    scenarios_path.write_text(json.dumps(scenarios_payload, indent=2))

    summary_path = metadata_dir / "rl_signature_summary.json"
    summary_payload: Dict[str, object] = {
        "profile": profile.name,
        "runs": len(run_df),
        "scenarios": len(scenario_df) if not scenario_df.empty else 0,
        "metrics_available": not run_df.empty,
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
                    f"- **{row['scenario']}**: sharpe={row.get('sharpe_ratio', float('nan')):.3f}, "
                    f"max_drawdown={row.get('max_drawdown', float('nan')):.3f}, "
                    f"total_return={row.get('total_return', float('nan')):.3f}"
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

    try:
        import torch

        diagnostics["torch_version"] = torch.__version__
        diagnostics["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            diagnostics["cuda_device"] = torch.cuda.get_device_name(0)
            diagnostics["cuda_device_count"] = torch.cuda.device_count()
    except Exception:
        diagnostics["torch_version"] = None
        diagnostics["cuda_available"] = False
    return diagnostics


def _log_environment_diagnostics(logger) -> Dict[str, Any]:
    diagnostics = _gather_environment_diagnostics()
    logger.info("Environment diagnostics", context=diagnostics)
    return diagnostics


def _build_scenario_statuses(
    plans: List[ScenarioPlan], run_records: List[RunRecord], scenario_df: pd.DataFrame
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
        status = "success" if failures == 0 and successes > 0 else "error" if failures else "pending"
        scenario_statuses.append(
            {
                "scenario": plan.name,
                "status": status,
                "planned_seeds": plan.seeds,
                "successful_runs": successes,
                "failed_runs": failures,
                "metrics": metrics_lookup.get(plan.name),
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
        "Starting RL + signature pipeline",
        context={"log_path": log_path, "logs_root": bundle_dirs["logs"]},
    )
    env_diagnostics = _log_environment_diagnostics(logger)

    deps_ok, missing = _check_dependencies(logger)
    if not deps_ok:
        emit_error(
            args,
            code=ERROR_DEPENDENCY,
            message="Missing required dependencies for RL + signature pipeline.",
            details={"missing": missing},
        )
        return 1

    profile = _plan_profile(args.training_profile)
    if profile is None:
        emit_error(
            args,
            code=ERROR_VALUE,
            message="Unsupported training profile.",
            details={"profile": args.training_profile},
        )
        return 2

    logger.info(
        "Resolved training profile",
        context={"profile": profile.name, "seeds": profile.seeds, "scenarios": profile.scenarios},
    )

    plans = _build_scenario_plans(profile)
    run_records = _execute_runs(
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

    try:
        metadata_paths = _write_metadata(
            metadata_dir=bundle_dirs["metadata"],
            profile=profile,
            plans=plans,
            run_df=run_df,
            scenario_df=scenario_df,
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
        "runs_summary_csv": str(bundle_dirs["summary"] / "runs.csv") if not run_df.empty else None,
        "scenarios_summary_csv": str(bundle_dirs["summary"] / "scenarios.csv") if not scenario_df.empty else None,
    }
    artifacts.update(metadata_paths)
    artifacts.update(comparison_plots)

    run_errors = [
        {"code": "run_failed", "message": rec.error, "details": {"scenario": rec.scenario, "seed": rec.seed}}
        for rec in run_records
        if rec.status != "success" and rec.error
    ]

    scenario_statuses = _build_scenario_statuses(plans, run_records, scenario_df)
    runtime_seconds = time.perf_counter() - start_time

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
            "required": ["stable_baselines3", "torch", "iisignature"],
            "missing": missing,
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
            "run_level": str(bundle_dirs["summary"] / "runs.csv") if not run_df.empty else None,
            "scenario_level": str(bundle_dirs["summary"] / "scenarios.csv") if not scenario_df.empty else None,
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
    message = (
        "RL + signature pipeline completed." if overall_success else "RL + signature pipeline completed with errors."
    )

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
