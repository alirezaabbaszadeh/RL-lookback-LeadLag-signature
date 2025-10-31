from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ in {None, ""}:
    # When executed as ``python src/leadlag/pipelines/run_full_suite.py`` the
    # ``leadlag`` package is not importable because the ``src`` directory is
    # missing from ``sys.path``. Add it lazily instead of mutating the import
    # path at module import time when the module is imported regularly.
    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from leadlag import hydra_main  # type: ignore
from leadlag.cli.formatters import (
    add_format_flags,
    emit_formatted_output,
    finalize_format_args,
)
from leadlag.reporting.logging_utils import get_logger, setup_logging

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    """Return the repository root (where ``pyproject.toml`` lives)."""

    marker = Path(__file__).resolve()
    for parent in marker.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to the src directory when running from an unpacked wheel.
    return marker.parents[2]


PROJECT_ROOT = _project_root()


def run_command(cmd: Sequence[str], logger, dry_run: bool = False) -> None:
    logger.info("Executing command", context={"cmd": " ".join(str(c) for c in cmd)})
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class ScenarioRequirements:
    name: str
    requires_signature: bool
    requires_sb3: bool
    requires_sb3_contrib: bool


@dataclass(frozen=True)
class FullSuiteCLIOptions:
    output_root: Path
    data_path: Path
    scenario: str
    baseline_scenarios: tuple[str, ...] | None
    meta_samples: int
    offline_episodes: int
    baseline_seeds: tuple[int, ...]
    baseline_single_seed: bool
    leakage_limit_days: int
    skip_ablation: bool
    skip_audit: bool
    skip_report: bool
    skip_meta_offline: bool
    skip_baseline: bool
    skip_optional_deps: bool
    ablation_scenarios: tuple[str, ...] | None
    ablation_single_seed: bool
    max_missing_ratio: float
    max_zero_variance: int
    fail_on_quality: bool
    skip_schema_check: bool
    log_level: str
    log_path: Path | None
    output_format: str
    wants_json: bool
    dry_run: bool
    command: str

    def to_namespace(self) -> argparse.Namespace:
        namespace = argparse.Namespace(
            output_root=self.output_root,
            data_path=self.data_path,
            scenario=self.scenario,
            baseline_scenarios=list(self.baseline_scenarios) if self.baseline_scenarios else None,
            meta_samples=self.meta_samples,
            offline_episodes=self.offline_episodes,
            baseline_seeds=list(self.baseline_seeds),
            baseline_single_seed=self.baseline_single_seed,
            leakage_limit_days=self.leakage_limit_days,
            skip_ablation=self.skip_ablation,
            skip_audit=self.skip_audit,
            skip_report=self.skip_report,
            skip_meta_offline=self.skip_meta_offline,
            skip_baseline=self.skip_baseline,
            skip_optional_deps=self.skip_optional_deps,
            ablation_scenarios=list(self.ablation_scenarios) if self.ablation_scenarios else None,
            ablation_single_seed=self.ablation_single_seed,
            max_missing_ratio=self.max_missing_ratio,
            max_zero_variance=self.max_zero_variance,
            fail_on_quality=self.fail_on_quality,
            skip_schema_check=self.skip_schema_check,
            log_level=self.log_level,
            log_path=self.log_path,
            format=self.output_format,
            json=self.wants_json,
            dry_run=self.dry_run,
        )
        setattr(namespace, "_leadlag_command", self.command)
        return namespace

    def serialize(self) -> dict[str, object]:
        serialized: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Path):
                serialized[field.name] = str(value)
            elif isinstance(value, tuple):
                serialized[field.name] = [
                    str(item) if isinstance(item, Path) else item for item in value
                ]
            else:
                serialized[field.name] = value
        return serialized


@dataclass(frozen=True)
class FullSuitePaths:
    output_root: Path
    logs_dir: Path
    baseline_root: Path
    robustness_root: Path
    ablation_root: Path
    meta_root: Path
    offline_root: Path


@dataclass
class FullSuiteContext:
    config: FullSuiteCLIOptions
    logger: Any
    dependency_status: Dict[str, bool]
    paths: FullSuitePaths
    python_executable: str
    run_log: Dict[str, object]
    start_time: float


@dataclass(frozen=True)
class WorkflowResult:
    success: bool
    run_log: dict[str, object]
    error_message: str | None = None

    @property
    def errors(self) -> list[dict[str, object]] | None:
        if self.success:
            return None
        entry: dict[str, object] = {
            "code": "full_suite_failed",
            "message": "Full-suite pipeline failed",
        }
        if self.error_message:
            entry["details"] = {"error": self.error_message}
        return [entry]


def inspect_scenario(name: str) -> ScenarioRequirements:
    cfg = hydra_main._load_scenario_cfg(name)  # pylint: disable=protected-access
    runner = cfg.get("runner", "scenario")
    rl_cfg = cfg.get("rl", {})
    policy = str(rl_cfg.get("policy", "")).lower() if isinstance(rl_cfg, dict) else ""
    random_policy = bool(rl_cfg.get("random_policy")) if isinstance(rl_cfg, dict) else False
    requires_sb3 = runner == "rl" and not (policy == "random" or random_policy)
    requires_sb3_contrib = requires_sb3 and ("lstm" in policy or "recurrent" in policy)

    analysis_cfg = cfg.get("analysis", {})
    method = str(analysis_cfg.get("method", "signature")).lower()
    requires_signature = method == "signature"

    return ScenarioRequirements(
        name=name,
        requires_signature=requires_signature,
        requires_sb3=requires_sb3,
        requires_sb3_contrib=requires_sb3_contrib,
    )


def dependency_preflight(skip_optional: bool, logger) -> Dict[str, bool]:
    modules = {
        "iisignature": importlib.util.find_spec("iisignature") is not None,
        "dcor": importlib.util.find_spec("dcor") is not None,
        "gym": importlib.util.find_spec("gym") is not None,
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "sb3_contrib": importlib.util.find_spec("sb3_contrib") is not None,
    }
    logger.info("Dependency preflight start")
    for module, available in modules.items():
        status = "OK" if available else "MISSING"
        logger.info("Dependency status", context={"module": module, "status": status})
    if not modules["gym"]:
        logger.warning("gym is required for environment execution")
    if not modules["iisignature"]:
        logger.warning("iisignature absent; signature-based scenarios will be skipped")
    if not modules["stable_baselines3"] or not modules["torch"]:
        logger.warning(
            "stable-baselines3/torch absent; RL scenarios will be skipped unless "
            "--skip-optional-deps is omitted after installing them",
        )
    if modules["stable_baselines3"] and not modules["sb3_contrib"]:
        logger.warning("sb3-contrib absent; PPO-LSTM scenarios will be skipped")
    return modules


def check_optional_dependencies(
    requirements: ScenarioRequirements,
    dependency_status: Dict[str, bool],
    skip_optional: bool,
    logger,
) -> bool:
    if requirements.requires_signature and not dependency_status.get("iisignature", False):
        message = (
            f"Scenario '{requirements.name}' requires the 'iisignature' package. "
            "Install via:\n  pip install iisignature"
        )
        if skip_optional:
            logger.warning(
                "Skipping scenario due to missing signature dependencies",
                context={"scenario": requirements.name},
            )
            return False
        raise SystemExit(message)

    if not requirements.requires_sb3:
        return True

    has_sb3 = dependency_status.get("stable_baselines3", False)
    has_torch = dependency_status.get("torch", False)
    has_contrib = dependency_status.get("sb3_contrib", False)

    if has_sb3 and has_torch and (has_contrib or not requirements.requires_sb3_contrib):
        return True

    missing = []
    if not has_sb3:
        missing.append("stable-baselines3")
    if not has_torch:
        missing.append("torch")
    if requirements.requires_sb3_contrib and not has_contrib:
        missing.append("sb3-contrib")

    message = (
        f"Scenario '{requirements.name}' requires optional dependencies: {', '.join(missing)}. "
        "Install via:\n"
        "  pip install stable-baselines3 torch --extra-index-url https://download.pytorch.org/whl/cu118\n"
        "  pip install sb3-contrib   # for PPO-LSTM variants"
    )
    if skip_optional:
        logger.warning(
            "Skipping scenario due to missing RL dependencies",
            context={"scenario": requirements.name, "missing": ", ".join(missing)},
        )
        return False
    raise SystemExit(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete experiment + audit suite for Kaggle or local CI.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/full_suite"),
        help="Directory to collect experiment outputs.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("raw_data/daily_price.csv"),
        help="Primary price CSV for dataset audit (synthetic data will be generated if missing).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="fixed_30",
        help="Scenario preset for baseline, leakage, and walk-forward probes.",
    )
    parser.add_argument(
        "--baseline-scenarios",
        nargs="*",
        default=None,
        help="List of scenarios to execute as core baselines; defaults to [--scenario].",
    )
    parser.add_argument(
        "--meta-samples",
        type=int,
        default=300,
        help="Samples per regime when generating meta-RL datasets.",
    )
    parser.add_argument(
        "--offline-episodes",
        type=int,
        default=3,
        help="Episodes to log for offline RL baseline.",
    )
    parser.add_argument(
        "--baseline-seeds",
        nargs="*",
        type=int,
        default=[42, 52, 62],
        help="Seeds for the primary scenario run (ignored when --baseline-single-seed is set).",
    )
    parser.add_argument(
        "--baseline-single-seed",
        action="store_true",
        help="Run the primary scenario once instead of multi-seed aggregation.",
    )
    parser.add_argument(
        "--leakage-limit-days",
        type=int,
        default=180,
        help="Length of truncated history for leakage and walk-forward probes.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the ablation pipeline (useful when optional RL dependencies are unavailable).",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip dataset-quality and robustness audits (leakage + walk-forward).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip report generation.",
    )
    parser.add_argument(
        "--skip-meta-offline",
        action="store_true",
        help="Skip meta-RL and offline RL baselines.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip core scenario execution (only run audits/ablation/report).",
    )
    parser.add_argument(
        "--skip-optional-deps",
        action="store_true",
        help=(
            "Pass --skip-missing-deps to the ablation pipeline (skip RL presets if SB3/Torch "
            "missing)."
        ),
    )
    parser.add_argument(
        "--ablation-scenarios",
        nargs="*",
        default=None,
        help="Override default scenarios for the ablation pipeline.",
    )
    parser.add_argument(
        "--ablation-single-seed",
        action="store_true",
        help="Run ablation scenarios with a single seed instead of multi-seed aggregation.",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.01,
        help="Maximum allowed missing-value ratio in dataset_quality (default 1%).",
    )
    parser.add_argument(
        "--max-zero-variance",
        type=int,
        default=0,
        help="Maximum number of zero-variance assets tolerated (default 0, negative to ignore).",
    )
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Exit if dataset quality checks report issues.",
    )
    parser.add_argument(
        "--skip-schema-check",
        action="store_true",
        help="Skip final artifact schema validation step.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional path for the full-suite log file.",
    )
    add_format_flags(parser, default="text")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview commands without executing subprocesses.",
    )
    return parser


def parse_args(
    argv: Iterable[str] | None = None,
    *,
    parser: argparse.ArgumentParser | None = None,
) -> FullSuiteCLIOptions:
    parser = parser or build_parser()
    raw_argv = list(argv) if argv is not None else None
    namespace = parser.parse_args(raw_argv if raw_argv is not None else None)
    finalize_format_args(namespace, remove_in="0.2.0")
    command = "leadlag-full-suite"
    if raw_argv:
        command = "leadlag-full-suite " + " ".join(str(arg) for arg in raw_argv)
    return FullSuiteCLIOptions(
        output_root=namespace.output_root.resolve(),
        data_path=namespace.data_path,
        scenario=namespace.scenario,
        baseline_scenarios=tuple(namespace.baseline_scenarios) if namespace.baseline_scenarios else None,
        meta_samples=namespace.meta_samples,
        offline_episodes=namespace.offline_episodes,
        baseline_seeds=tuple(namespace.baseline_seeds),
        baseline_single_seed=bool(namespace.baseline_single_seed),
        leakage_limit_days=namespace.leakage_limit_days,
        skip_ablation=bool(namespace.skip_ablation),
        skip_audit=bool(namespace.skip_audit),
        skip_report=bool(namespace.skip_report),
        skip_meta_offline=bool(namespace.skip_meta_offline),
        skip_baseline=bool(namespace.skip_baseline),
        skip_optional_deps=bool(namespace.skip_optional_deps),
        ablation_scenarios=tuple(namespace.ablation_scenarios) if namespace.ablation_scenarios else None,
        ablation_single_seed=bool(namespace.ablation_single_seed),
        max_missing_ratio=float(namespace.max_missing_ratio),
        max_zero_variance=int(namespace.max_zero_variance),
        fail_on_quality=bool(namespace.fail_on_quality),
        skip_schema_check=bool(namespace.skip_schema_check),
        log_level=str(namespace.log_level),
        log_path=namespace.log_path,
        output_format=str(getattr(namespace, "format", "text")),
        wants_json=bool(getattr(namespace, "json", False)),
        dry_run=bool(namespace.dry_run),
        command=command,
    )


def _run_dataset_audit(context: FullSuiteContext) -> None:
    config = context.config
    if config.skip_audit:
        return
    quality_cmd: list[str] = [
        context.python_executable,
        str(PROJECT_ROOT / "scripts" / "audit" / "dataset_quality.py"),
        "--path",
        str(config.data_path),
        "--missing-tolerance",
        str(config.max_missing_ratio),
        "--zero-variance-limit",
        str(config.max_zero_variance),
        "--output",
        str(context.paths.logs_dir / "dataset_manifest.json"),
    ]
    if config.fail_on_quality:
        quality_cmd.append("--exit-on-fail")
    try:
        run_command(quality_cmd, context.logger, config.dry_run)
    except subprocess.CalledProcessError as exc:
        context.logger.warning(
            "dataset_quality failed",
            context={"exit_code": exc.returncode, "fail_on_quality": config.fail_on_quality},
        )
        if config.fail_on_quality:
            raise


def _prepare_baselines(context: FullSuiteContext) -> list[str]:
    config = context.config
    baseline_scenarios = list(config.baseline_scenarios) if config.baseline_scenarios else [config.scenario]
    validated: list[str] = []
    for scenario_name in baseline_scenarios:
        requirements = inspect_scenario(scenario_name)
        if not check_optional_dependencies(
            requirements,
            context.dependency_status,
            config.skip_optional_deps,
            context.logger,
        ):
            continue
        try:
            cfg = hydra_main._load_scenario_cfg(scenario_name)  # pylint: disable=protected-access
            hydra_main.validate_scenario_cfg(cfg)
            validated.append(scenario_name)
        except Exception as exc:  # pragma: no cover - validation guard
            context.logger.warning(
                "Baseline validation failed",
                context={"scenario": scenario_name, "error": repr(exc)},
            )
    context.run_log["baseline_scenarios_requested"] = baseline_scenarios
    context.run_log["validated_baselines"] = validated
    if not validated:
        context.logger.warning("No baseline scenarios scheduled for execution")
    return validated


def _run_baselines(context: FullSuiteContext, validated: list[str]) -> None:
    config = context.config
    if config.skip_baseline:
        return
    for scenario_name in validated:
        requirements = inspect_scenario(scenario_name)
        if not check_optional_dependencies(
            requirements,
            context.dependency_status,
            config.skip_optional_deps,
            context.logger,
        ):
            continue
        baseline_cmd: list[str] = [
            context.python_executable,
            str(PACKAGE_ROOT / "hydra_main.py"),
            "--scenario",
            scenario_name,
            "--output_root",
            str(context.paths.baseline_root),
        ]
        if not config.baseline_single_seed and config.baseline_seeds:
            baseline_cmd.append("--multi_seed_enabled")
            baseline_cmd.append("--seeds")
            baseline_cmd.extend(str(seed) for seed in config.baseline_seeds)
        run_command(baseline_cmd, context.logger, config.dry_run)


def _run_finance_kpis(context: FullSuiteContext) -> None:
    finance_output = ensure_path(context.paths.baseline_root / "evaluation") / "finance_kpis.csv"
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "evaluation" / "finance_kpis.py"),
            "--results-root",
            str(context.paths.baseline_root),
            "--output",
            str(finance_output),
        ],
        context.logger,
        context.config.dry_run,
    )


def _run_meta_offline(context: FullSuiteContext) -> None:
    config = context.config
    if config.skip_meta_offline:
        return
    ensure_path(context.paths.meta_root)
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "research" / "meta_rl" / "run_meta_rl.py"),
            "--output-root",
            str(context.paths.meta_root),
            "--samples",
            str(config.meta_samples),
        ],
        context.logger,
        config.dry_run,
    )

    ensure_path(context.paths.offline_root)
    dataset_path = context.paths.offline_root / "offline_dataset.csv"
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "research" / "offline_rl" / "log_trajectories.py"),
            "--episodes",
            str(config.offline_episodes),
            "--output",
            str(dataset_path),
        ],
        context.logger,
        config.dry_run,
    )
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "research" / "offline_rl" / "train_offline.py"),
            "--dataset",
            str(dataset_path),
            "--output-root",
            str(context.paths.offline_root),
        ],
        context.logger,
        config.dry_run,
    )


def _run_ablation(context: FullSuiteContext) -> list[str]:
    config = context.config
    if config.skip_ablation:
        context.run_log["validated_ablation"] = list(config.ablation_scenarios or [])
        return []
    scenarios = list(config.ablation_scenarios) if config.ablation_scenarios else None
    validated: list[str] = []
    if scenarios:
        for scenario_name in scenarios:
            requirements = inspect_scenario(scenario_name)
            if not check_optional_dependencies(
                requirements,
                context.dependency_status,
                config.skip_optional_deps,
                context.logger,
            ):
                continue
            try:
                cfg = hydra_main._load_scenario_cfg(scenario_name)  # pylint: disable=protected-access
                hydra_main.validate_scenario_cfg(cfg)
                validated.append(scenario_name)
            except Exception as exc:
                context.logger.warning(
                    "Ablation validation failed",
                    context={"scenario": scenario_name, "error": repr(exc)},
                )
    context.run_log["validated_ablation"] = validated if scenarios else []
    ablation_cmd: List[str] = [
        context.python_executable,
        str(PACKAGE_ROOT / "pipelines" / "run_ablation.py"),
        "--output-root",
        str(context.paths.ablation_root),
    ]
    if config.skip_optional_deps:
        ablation_cmd.append("--skip-missing-deps")
    if config.ablation_single_seed:
        ablation_cmd.append("--single-seed")
    selected = validated if scenarios else None
    if selected:
        ablation_cmd.append("--scenarios")
        ablation_cmd.extend(selected)
    run_command(ablation_cmd, context.logger, config.dry_run)
    return validated


def _run_audits(context: FullSuiteContext) -> None:
    config = context.config
    if config.skip_audit:
        return
    run_command(
        [
            context.python_executable,
            str(PROJECT_ROOT / "scripts" / "audit" / "leakage_probes.py"),
            "--scenario",
            config.scenario,
            "--seed",
            "7",
            "--limit_days",
            str(config.leakage_limit_days),
            "--out",
            str(context.paths.robustness_root),
        ],
        context.logger,
        config.dry_run,
    )
    run_command(
        [
            context.python_executable,
            str(PROJECT_ROOT / "scripts" / "audit" / "check_walk_forward.py"),
            "--scenario",
            config.scenario,
            "--seed",
            "13",
            "--limit_days",
            str(config.leakage_limit_days),
            "--output-root",
            str(context.paths.robustness_root),
        ],
        context.logger,
        config.dry_run,
    )


def _run_reporting(context: FullSuiteContext) -> None:
    config = context.config
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "reporting" / "compare_scenarios.py"),
            "--results_root",
            str(context.paths.output_root),
            "--out",
            str(context.paths.output_root / "aggregate_comparison"),
        ],
        context.logger,
        config.dry_run,
    )
    if config.skip_report:
        return
    report_dir = ensure_path(context.paths.output_root / "reports")
    report_start = time.time()
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "reporting" / "generate_report.py"),
            "--results-root",
            str(context.paths.output_root),
            "--output-dir",
            str(report_dir),
        ],
        context.logger,
        config.dry_run,
    )
    elapsed = time.time() - report_start
    context.logger.info(
        "Report generated",
        context={"elapsed_seconds": round(elapsed, 1), "report_dir": str(report_dir)},
    )
    run_command(
        [
            context.python_executable,
            str(PACKAGE_ROOT / "reporting" / "plot_balance_history.py"),
            "--results-root",
            str(context.paths.output_root),
            "--out",
            str(context.paths.output_root / "evaluation" / "plots" / "balance"),
        ],
        context.logger,
        config.dry_run,
    )


def _run_schema_validation(context: FullSuiteContext) -> None:
    config = context.config
    if config.skip_schema_check:
        return
    audit_dir = ensure_path(context.paths.output_root / "audit")
    run_command(
        [
            context.python_executable,
            str(PROJECT_ROOT / "scripts" / "audit" / "validate_artifacts.py"),
            "--root",
            str(context.paths.output_root),
            "--out",
            str(audit_dir),
        ],
        context.logger,
        config.dry_run,
    )


class FullSuiteCoordinator:
    def __init__(self, config: FullSuiteCLIOptions) -> None:
        self.config = config

    def _initialise_context(self, start_time: float) -> FullSuiteContext:
        output_root = ensure_path(self.config.output_root)
        logs_dir = ensure_path(output_root / "logs")
        log_path = self.config.log_path or logs_dir / "full_suite.log"
        setup_logging(
            Path(log_path),
            level=str(self.config.log_level).upper(),
            context={"module": "full_suite"},
        )
        logger = get_logger(
            "pipelines.run_full_suite",
            context={"output_root": output_root, "dry_run": self.config.dry_run},
        )
        dependency_status = dependency_preflight(self.config.skip_optional_deps, logger)
        run_log: dict[str, object] = {
            "command": " ".join([sys.executable, *sys.argv[1:]]) if sys.argv else sys.executable,
            "args": self.config.serialize(),
            "dependency_status": dependency_status,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "output_root": str(output_root),
            "logs_dir": str(logs_dir),
            "dry_run": bool(self.config.dry_run),
        }
        paths = FullSuitePaths(
            output_root=output_root,
            logs_dir=logs_dir,
            baseline_root=ensure_path(output_root / "core"),
            robustness_root=ensure_path(output_root / "robustness"),
            ablation_root=output_root / "ablations",
            meta_root=output_root / "meta_rl",
            offline_root=output_root / "offline",
        )
        return FullSuiteContext(
            config=self.config,
            logger=logger,
            dependency_status=dependency_status,
            paths=paths,
            python_executable=sys.executable,
            run_log=run_log,
            start_time=start_time,
        )

    def _finalise(self, context: FullSuiteContext, success: bool, error_message: str | None) -> None:
        end_time = time.time()
        context.run_log["success"] = success
        context.run_log["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time))
        context.run_log["elapsed_seconds"] = round(end_time - context.start_time, 3)
        if error_message:
            context.run_log["error"] = error_message
        try:
            context.paths.logs_dir.mkdir(parents=True, exist_ok=True)
            (context.paths.logs_dir / f"run_summary_{int(end_time)}.json").write_text(
                json.dumps(context.run_log, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
        if success:
            context.logger.info("Pipeline completed successfully")

    def run(self) -> WorkflowResult:
        start_time = time.time()
        context = self._initialise_context(start_time)
        context.logger.info("Starting full-suite pipeline")
        success = False
        error_message: str | None = None
        try:
            _run_dataset_audit(context)
            validated_baselines = _prepare_baselines(context)
            _run_baselines(context, validated_baselines)
            _run_meta_offline(context)
            _run_finance_kpis(context)
            _run_ablation(context)
            _run_audits(context)
            _run_reporting(context)
            _run_schema_validation(context)
            success = True
        except Exception as exc:  # pragma: no cover - defensive guard
            error_message = repr(exc)
            success = False
        finally:
            self._finalise(context, success, error_message)
        return WorkflowResult(success=success, run_log=context.run_log, error_message=error_message)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    config = parse_args(list(argv) if argv is not None else None, parser=parser)
    coordinator = FullSuiteCoordinator(config)
    result = coordinator.run()
    args_namespace = config.to_namespace()
    text_message = (
        "Full-suite pipeline completed successfully."
        if result.success
        else "Full-suite pipeline failed; see logs for details."
    )
    emit_formatted_output(
        args_namespace,
        data=result.run_log,
        text=text_message,
        message="Full-suite pipeline completed." if result.success else "Full-suite pipeline failed.",
        errors=result.errors,
        success=result.success,
        pretty=True,
        command=config.command,
    )
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
