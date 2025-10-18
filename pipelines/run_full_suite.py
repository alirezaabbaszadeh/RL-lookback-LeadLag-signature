from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hydra_main  # type: ignore


def run_command(cmd: Sequence[str]) -> None:
    print(f"[full-suite] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class ScenarioRequirements:
    name: str
    requires_signature: bool
    requires_sb3: bool
    requires_sb3_contrib: bool


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


def dependency_preflight(skip_optional: bool) -> Dict[str, bool]:
    modules = {
        "iisignature": importlib.util.find_spec("iisignature") is not None,
        "dcor": importlib.util.find_spec("dcor") is not None,
        "gym": importlib.util.find_spec("gym") is not None,
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "sb3_contrib": importlib.util.find_spec("sb3_contrib") is not None,
    }
    print("[full-suite] Dependency preflight:")
    for module, available in modules.items():
        status = "OK" if available else "MISSING"
        print(f"  - {module}: {status}")
    if not modules["gym"]:
        print("[full-suite] WARN: gym is required for environment execution.")
    if not modules["iisignature"]:
        print("[full-suite] WARN: iisignature absent; signature-based scenarios will be skipped.")
    if not modules["stable_baselines3"] or not modules["torch"]:
        print(
            "[full-suite] WARN: stable-baselines3/torch absent; RL scenarios will be skipped unless "
            "--skip-optional-deps is omitted after installing them."
        )
    if modules["stable_baselines3"] and not modules["sb3_contrib"]:
        print("[full-suite] WARN: sb3-contrib absent; PPO-LSTM scenarios will be skipped.")
    return modules


def check_optional_dependencies(
    requirements: ScenarioRequirements,
    dependency_status: Dict[str, bool],
    skip_optional: bool,
) -> bool:
    if requirements.requires_signature and not dependency_status.get("iisignature", False):
        message = (
            f"Scenario '{requirements.name}' requires the 'iisignature' package. "
            "Install via:\n  pip install iisignature"
        )
        if skip_optional:
            print(f"[full-suite] Skipping {requirements.name}: {message}")
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
        print(f"[full-suite] Skipping {requirements.name}: {message}")
        return False
    raise SystemExit(message)



def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete experiment + audit suite for Kaggle or local CI.")
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
        help="Pass --skip-missing-deps to the ablation pipeline (skip RL presets if SB3/Torch missing).",
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
    args = parser.parse_args()

    dependency_status = dependency_preflight(args.skip_optional_deps)

    output_root = ensure_path(args.output_root.resolve())
    baseline_root = ensure_path(output_root / "core")
    robustness_root = ensure_path(output_root / "robustness")
    ablation_root = output_root / "ablations"
    meta_root = output_root / "meta_rl"
    offline_root = output_root / "offline"

    python_exe = sys.executable

    # Dataset audit
    if not args.skip_audit:
        quality_cmd = [
            python_exe,
            str(ROOT / "scripts" / "audit" / "dataset_quality.py"),
            "--path",
            str(args.data_path),
            "--missing-tolerance",
            str(args.max_missing_ratio),
            "--zero-variance-limit",
            str(args.max_zero_variance),
        ]
        if args.fail_on_quality:
            quality_cmd.append("--exit-on-fail")
        run_command(quality_cmd)

    # Determine baseline scenarios and validate configs
    baseline_scenarios = args.baseline_scenarios or [args.scenario]
    validated_baselines: List[str] = []
    for scenario_name in baseline_scenarios:
        requirements = inspect_scenario(scenario_name)
        if not check_optional_dependencies(requirements, dependency_status, args.skip_optional_deps):
            continue
        try:
            cfg = hydra_main._load_scenario_cfg(scenario_name)  # pylint: disable=protected-access
            hydra_main.validate_scenario_cfg(cfg)
            validated_baselines.append(scenario_name)
        except Exception as exc:  # pragma: no cover - validation guard
            print(f"[full-suite] Skipping {scenario_name}: validation failed ({exc})")
    if not validated_baselines:
        print("[full-suite] WARN: no baseline scenarios scheduled for execution.")

    # Validate ablation scenarios (if provided)
    ablation_scenarios = args.ablation_scenarios
    if ablation_scenarios:
        validated_ablation: List[str] = []
        for scenario_name in list(ablation_scenarios):
            requirements = inspect_scenario(scenario_name)
            if not check_optional_dependencies(requirements, dependency_status, args.skip_optional_deps):
                continue
            try:
                cfg = hydra_main._load_scenario_cfg(scenario_name)  # pylint: disable=protected-access
                hydra_main.validate_scenario_cfg(cfg)
                validated_ablation.append(scenario_name)
            except Exception as exc:
                print(f"[full-suite] Skipping ablation scenario {scenario_name}: validation failed ({exc})")
        ablation_scenarios = validated_ablation or None

    if not args.skip_baseline:
        for scenario_name in validated_baselines:
            requirements = inspect_scenario(scenario_name)
            if not check_optional_dependencies(requirements, dependency_status, args.skip_optional_deps):
                continue
            baseline_cmd: List[str] = [
                python_exe,
                str(ROOT / "hydra_main.py"),
                "--scenario",
                scenario_name,
                "--output_root",
                str(baseline_root),
            ]
            if not args.baseline_single_seed and args.baseline_seeds:
                baseline_cmd.append("--multi_seed_enabled")
                baseline_cmd.append("--seeds")
                baseline_cmd.extend(str(seed) for seed in args.baseline_seeds)
            run_command(baseline_cmd)

        if not args.skip_meta_offline:
            ensure_path(meta_root)
        run_command(
            [
                python_exe,
                str(ROOT / "research" / "meta_rl" / "run_meta_rl.py"),
                "--output-root",
                str(meta_root),
                "--samples",
                str(args.meta_samples),
            ]
        )
        ensure_path(offline_root)
        dataset_path = offline_root / "offline_dataset.csv"
        run_command(
            [
                python_exe,
                str(ROOT / "research" / "offline_rl" / "log_trajectories.py"),
                "--episodes",
                    str(args.offline_episodes),
                    "--output",
                    str(dataset_path),
                ]
            )
            run_command(
                [
                    python_exe,
                    str(ROOT / "research" / "offline_rl" / "train_offline.py"),
                    "--dataset",
                    str(dataset_path),
                    "--output-root",
                    str(offline_root),
                ]
            )

        # Finance KPIs on baseline runs
        finance_output = ensure_path(baseline_root / "evaluation") / "finance_kpis.csv"
        run_command(
            [
                python_exe,
                str(ROOT / "evaluation" / "finance_kpis.py"),
                "--results-root",
                str(baseline_root),
                "--output",
                str(finance_output),
            ]
        )

    # Ablation pipeline
    if not args.skip_ablation:
        ablation_cmd: List[str] = [
            python_exe,
            str(ROOT / "pipelines" / "run_ablation.py"),
            "--output-root",
            str(ablation_root),
        ]
        if args.skip_optional_deps:
            ablation_cmd.append("--skip-missing-deps")
        if args.ablation_single_seed:
            ablation_cmd.append("--single-seed")
        if args.ablation_scenarios:
            ablation_cmd.append("--scenarios")
            ablation_cmd.extend(args.ablation_scenarios)
        run_command(ablation_cmd)

    # Leakage probes and walk-forward verification
    if not args.skip_audit:
        run_command(
            [
                python_exe,
                str(ROOT / "scripts" / "audit" / "leakage_probes.py"),
                "--scenario",
                args.scenario,
                "--seed",
                "7",
                "--limit_days",
                str(args.leakage_limit_days),
                "--out",
                str(robustness_root),
            ]
        )
        run_command(
            [
                python_exe,
                str(ROOT / "scripts" / "audit" / "check_walk_forward.py"),
                "--scenario",
                args.scenario,
                "--seed",
                "13",
                "--limit_days",
                str(args.leakage_limit_days),
                "--output-root",
                str(robustness_root),
            ]
        )

    # Generate unified comparison plots for the entire output root
    run_command(
        [
            python_exe,
            str(ROOT / "reporting" / "compare_scenarios.py"),
            "--results_root",
            str(output_root),
            "--out",
            str(output_root / "aggregate_comparison"),
        ]
    )

    if not args.skip_report:
        report_dir = ensure_path(output_root / "reports")
        report_start = time.time()
        run_command(
            [
                python_exe,
                str(ROOT / "reporting" / "generate_report.py"),
                "--results-root",
                str(output_root),
                "--output-dir",
                str(report_dir),
            ]
        )
        elapsed = time.time() - report_start
        print(f"[full-suite] Report generated in {elapsed:.1f}s at {report_dir}")

    if not args.skip_schema_check:
        audit_dir = ensure_path(output_root / "audit")
        run_command(
            [
                python_exe,
                str(ROOT / "scripts" / "audit" / "validate_artifacts.py"),
                "--root",
                str(output_root),
                "--out",
                str(audit_dir),
            ]
        )

    print("[full-suite] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
