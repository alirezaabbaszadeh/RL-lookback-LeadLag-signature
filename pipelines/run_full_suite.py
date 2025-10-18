from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: Sequence[str]) -> None:
    print(f"[full-suite] {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    args = parser.parse_args()

    output_root = ensure_path(args.output_root.resolve())
    baseline_root = ensure_path(output_root / "core")
    robustness_root = ensure_path(output_root / "robustness")
    ablation_root = output_root / "ablations"
    meta_root = output_root / "meta_rl"
    offline_root = output_root / "offline"

    python_exe = sys.executable

    # Dataset audit
    if not args.skip_audit:
        run_command(
            [
                python_exe,
                str(ROOT / "scripts" / "audit" / "dataset_quality.py"),
                "--path",
                str(args.data_path),
            ]
        )

    # Core scenario + optional meta/offline baselines
    if not args.skip_baseline:
        starter_cmd: List[str] = [
            python_exe,
            str(ROOT / "kaggle" / "starter.py"),
            "--scenario",
            args.scenario,
            "--output-root",
            str(baseline_root),
        ]
        if not args.skip_meta_offline:
            starter_cmd.extend(
                [
                    "--run-meta-rl",
                    "--meta-samples",
                    str(args.meta_samples),
                    "--run-offline",
                    "--offline-episodes",
                    str(args.offline_episodes),
                ]
            )
        run_command(starter_cmd)

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
        run_command(
            [
                python_exe,
                str(ROOT / "reporting" / "generate_report.py"),
                "--results-root",
                str(output_root),
                "--output-dir",
                str(ensure_path(output_root / "reports")),
            ]
        )

    print("[full-suite] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
