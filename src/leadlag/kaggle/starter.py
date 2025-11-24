from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print(f"[kaggle-starter] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convenience runner for Kaggle notebooks.")
    parser.add_argument(
        "--scenario",
        default="fixed_30",
        help="Scenario to execute via hydra_main.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working"),
        help="Directory where results/meta artifacts are written.",
    )
    parser.add_argument(
        "--run-offline",
        action="store_true",
        help=(
            "Capture trajectories and train offline behaviour cloning baseline"
            " under <output-root>/offline."
        ),
    )
    parser.add_argument(
        "--offline-episodes",
        type=int,
        default=3,
        help="Episodes to record when --run-offline is enabled.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_dir = output_root / "results"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "-m",
            "leadlag.hydra_main",
            "--scenario",
            args.scenario,
            "--output_root",
            str(scenario_dir),
        ]
    )

    if args.run_offline:
        offline_dir = output_root / "offline"
        offline_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = offline_dir / "offline_dataset.csv"
        _run(
            [
                "python",
                "-m",
                "leadlag.research.offline_rl.log_trajectories",
                "--episodes",
                str(args.offline_episodes),
                "--output",
                str(dataset_path),
            ]
        )
        _run(
            [
                "python",
                "-m",
                "leadlag.research.offline_rl.train_offline",
                "--dataset",
                str(dataset_path),
                "--output-root",
                str(offline_dir),
            ]
        )

    print("[kaggle-starter] Completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
