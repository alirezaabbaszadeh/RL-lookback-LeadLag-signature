"""
Orchestrate multiple RL stacks sequentially in a Kaggle notebook.

The script installs each stack with an isolated ``pip`` transaction, runs its
stage script in a fresh subprocess, captures artifacts, and uninstalls the
packages before proceeding. This prevents long-lived dependency conflicts while
still allowing reproducible multi-stack experiments within a single runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ORCHESTRATOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ORCHESTRATOR_DIR.parent


@dataclass(frozen=True)
class StageDefinition:
    name: str
    requirements: Sequence[str]
    entrypoint: Path
    uninstall: Sequence[str]
    description: str


def discover_artifact_root(custom_root: Path | None) -> Path:
    if custom_root is not None:
        return custom_root
    kaggle_root = Path("/kaggle/working")
    if kaggle_root.exists():
        return kaggle_root / "multi_stage_artifacts"
    return Path.cwd() / "multi_stage_artifacts"


def pip_install(requirements: Iterable[str]) -> None:
    if not requirements:
        return
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--upgrade",
        "--force-reinstall",
        *requirements,
    ]
    subprocess.run(cmd, check=True)


def pip_check() -> None:
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


def pip_uninstall(packages: Iterable[str]) -> None:
    packages = list(packages)
    if not packages:
        return
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", *packages]
    subprocess.run(cmd, check=False)


def run_stage(stage: StageDefinition, output_dir: Path) -> tuple[str, str, float]:
    """Execute stage entrypoint synchronously and capture logs."""
    stage_dir = output_dir / stage.name
    if stage_dir.exists():
        if stage_dir.is_dir():
            shutil.rmtree(stage_dir)
        else:
            stage_dir.unlink()
    stage_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = stage_dir / "stdout.log"
    stderr_path = stage_dir / "stderr.log"
    requirements_path = stage_dir / "requirements.txt"
    requirements_path.write_text("\n".join(stage.requirements) + "\n", encoding="utf-8")

    command = [
        sys.executable,
        str(stage.entrypoint),
        "--output-dir",
        str(stage_dir),
    ]

    start = time.time()
    result = subprocess.run(
        command,
        cwd=stage.entrypoint.parent,
        text=True,
        capture_output=True,
    )
    duration = time.time() - start

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage {stage.name} failed (exit={result.returncode}); see {stderr_path}"
        )

    return stdout_path.as_posix(), stderr_path.as_posix(), duration


def build_argument_parser(stage_registry: dict[str, StageDefinition]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple RL stacks sequentially with dependency isolation. "
            "By default all registered stages execute in the predefined order."
        )
    )
    parser.add_argument(
        "--stage",
        dest="stages",
        action="append",
        help="Stage name to execute (can be provided multiple times).",
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        help="Directory for collected artifacts (defaults to Kaggle working directory).",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep packages installed after each stage (not recommended).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print registered stages and exit.",
    )
    return parser


def list_stages(stage_registry: dict[str, StageDefinition]) -> None:
    payload = [
        {"name": stage.name, "description": stage.description}
        for stage in stage_registry.values()
    ]
    print(json.dumps(payload, indent=2))


def main() -> None:
    stage_registry: dict[str, StageDefinition] = {
        "sb3_kaggle": StageDefinition(
            name="sb3_kaggle",
            requirements=[
                "gymnasium==0.29.0",
                "stable-baselines3==2.1.0",
                "sb3-contrib==2.1.0",
                "kaggle-environments==1.18.0",
            ],
            entrypoint=Path(__file__).parent / "stages" / "sb3_kaggle.py",
            uninstall=[
                "gymnasium",
                "stable-baselines3",
                "sb3-contrib",
                "kaggle-environments",
            ],
            description="Stable-Baselines3 PPO smoke test compatible with Kaggle stack.",
        ),
        "dopamine": StageDefinition(
            name="dopamine",
            requirements=[
                "gymnasium>=1.0.0",
                "dopamine-rl==4.1.2",
            ],
            entrypoint=Path(__file__).parent / "stages" / "dopamine.py",
            uninstall=[
                "gymnasium",
                "dopamine-rl",
            ],
            description="Random-policy dopamine rollouts on Gymnasium 1.x.",
        ),
        "sb3_leadlag": StageDefinition(
            name="sb3_leadlag",
            requirements=[
                # Classic Gym + SB3 1.x stack for project RL training
                "-r",
                str(REPO_ROOT / "requirements-kaggle.txt"),
                "gym==0.26.2",
                "stable-baselines3==1.8.0",
                "sb3-contrib==1.8.0",
                "torch>=2.0",
            ],
            entrypoint=Path(__file__).parent / "stages" / "sb3_leadlag.py",
            uninstall=[],
            description="Train PPO/variants on LeadLag env via training.run_rl with overrides.",
        ),
        "full_suite": StageDefinition(
            name="full_suite",
            requirements=[
                "-r",
                str(REPO_ROOT / "requirements-kaggle.txt"),
                "gym==0.26.2",
                "stable-baselines3==1.8.0",
                "sb3-contrib==1.8.0",
                "torch>=2.0",
            ],
            entrypoint=Path(__file__).parent / "stages" / "full_suite.py",
            uninstall=[],
            description="Run pipelines/run_full_suite.py (includes ablation, audits, reports).",
        ),
        "leadlag_hydra": StageDefinition(
            name="leadlag_hydra",
            requirements=[
                "-r",
                str(REPO_ROOT / "requirements-kaggle.txt"),
                # Add production RL stack compatible with classic Gym API
                "gym==0.26.2",
                "stable-baselines3==1.8.0",
                "sb3-contrib==1.8.0",
                "torch>=2.0",
            ],
            entrypoint=ORCHESTRATOR_DIR / "stages" / "leadlag_hydra.py",
            uninstall=[],
            description="Execute lead-lag Hydra scenarios with optional multi-seed aggregation.",
        ),
    }

    parser = build_argument_parser(stage_registry)
    args = parser.parse_args()

    if args.list:
        list_stages(stage_registry)
        return

    selected_stages: list[StageDefinition]
    if args.stages:
        missing = [name for name in args.stages if name not in stage_registry]
        if missing:
            valid = ", ".join(stage_registry)
            raise SystemExit(f"Unknown stage(s): {missing}. Valid options: {valid}")
        selected_stages = [stage_registry[name] for name in args.stages]
    else:
        selected_stages = list(stage_registry.values())

    artifact_root = discover_artifact_root(args.artifacts_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    summary_path = artifact_root / "summary.json"

    for stage in selected_stages:
        pip_uninstall(stage.uninstall)
        pip_install(stage.requirements)
        pip_check()

        try:
            stdout_path, stderr_path, duration = run_stage(stage, artifact_root)
            status = "success"
            error = None
        except RuntimeError as exc:
            status = "failed"
            duration = 0.0
            stdout_path = (artifact_root / stage.name / "stdout.log").as_posix()
            stderr_path = (artifact_root / stage.name / "stderr.log").as_posix()
            error = str(exc)

        summary_record = {
            "stage": stage.name,
            "status": status,
            "duration_seconds": duration,
            "stdout_log": stdout_path,
            "stderr_log": stderr_path,
            "requirements": stage.requirements,
            "timestamp": time.time(),
            "description": stage.description,
        }
        if error:
            summary_record["error"] = error
            summary.append(summary_record)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            break

        summary.append(summary_record)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if not args.skip_cleanup:
            pip_uninstall(stage.uninstall)

    # Optionally keep site-packages clean even if the final stage fails.
    if not args.skip_cleanup:
        residual_packages = {
            package
            for stage in selected_stages
            for package in stage.uninstall
        }
        pip_uninstall(sorted(residual_packages))


if __name__ == "__main__":
    main()
