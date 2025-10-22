"""Run lead-lag Hydra scenarios inside the Kaggle orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
HYDRA_ENTRY = REPO_ROOT / "hydra_main.py"


def run_hydra(
    output_root: Path,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    multi_seed_enabled: bool,
) -> subprocess.CompletedProcess[str]:
    """Invoke hydra_main.py with CLI overrides."""
    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one scenario must be provided.")

    command = [
        sys.executable,
        str(HYDRA_ENTRY),
        "--scenario",
        scenario_list[0],
        "--scenarios",
        *scenario_list,
        "--output_root",
        str(output_root),
    ]
    if multi_seed_enabled and seeds:
        command.append("--multi_seed_enabled")
    if seeds:
        command.extend(["--seeds", *map(str, seeds)])

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def collect_metadata(
    output_dir: Path,
    hydra_result: subprocess.CompletedProcess[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    multi_seed_enabled: bool,
) -> None:
    """Persist metadata about the Hydra run and copy hydra summary if present."""
    hydra_runs_path = output_dir / "runs" / "hydra_runs.json"
    metadata = {
        "stage": "leadlag_hydra",
        "timestamp": time.time(),
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "multi_seed_enabled": multi_seed_enabled,
        "returncode": hydra_result.returncode,
        "hydra_runs_path": str(hydra_runs_path) if hydra_runs_path.exists() else None,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    source_summary = output_dir / "runs" / "hydra_runs.json"
    target_summary = output_dir / "hydra_runs.json"
    if source_summary.exists():
        target_summary.write_text(source_summary.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute lead-lag Hydra scenarios and capture outputs."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["fixed_30"],
        help="Scenario names packaged with LeadLag.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[],
        help="Seed values for multi-seed aggregation.",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Enable multi-seed runner regardless of scenario defaults.",
    )
    args = parser.parse_args()

    # Environment variable overrides allow configuring the stage from the orchestrator caller
    # without changing the orchestrator CLI. Useful in Kaggle one-cell runs.
    env_scenarios = os.getenv("LEADLAG_SCENARIOS")
    if env_scenarios:
        args.scenarios = [s for s in env_scenarios.split(",") if s]
    env_seeds = os.getenv("LEADLAG_SEEDS")
    if env_seeds:
        try:
            args.seeds = [int(s) for s in env_seeds.split(",") if s]
        except ValueError:
            print("Warning: LEADLAG_SEEDS is not a comma-separated list of ints; ignoring.")
    env_ms = os.getenv("LEADLAG_MULTI_SEED")
    if env_ms:
        args.multi_seed = env_ms.strip().lower() in {"1", "true", "yes", "on"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    multi_seed_enabled = args.multi_seed or len(args.seeds) > 1
    result = run_hydra(run_root, args.scenarios, args.seeds, multi_seed_enabled)

    if result.returncode != 0:
        # Preserve metadata before raising so orchestrator surfaces the failure.
        collect_metadata(args.output_dir, result, args.scenarios, args.seeds, multi_seed_enabled)
        raise SystemExit(result.returncode)

    collect_metadata(args.output_dir, result, args.scenarios, args.seeds, multi_seed_enabled)


if __name__ == "__main__":
    main()
