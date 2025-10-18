from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    print(f"[smoke-kaggle] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run minimal end-to-end smoke tests for Kaggle packaging.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dist/kaggle_smoke"),
        help="Directory to store smoke test artifacts.",
    )
    parser.add_argument("--keep-meta-rl", action="store_true", help="Include meta-RL smoke run.")
    parser.add_argument("--keep-offline", action="store_true", help="Include offline RL smoke run.")
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    scenario_dir = root / "results"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "hydra_main.py",
            "--scenario",
            "fast_smoke",
            "--output_root",
            str(scenario_dir),
        ]
    )

    if args.keep_meta_rl:
        meta_dir = root / "meta_rl"
        meta_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "python",
                "research/meta_rl/run_meta_rl.py",
                "--output-root",
                str(meta_dir),
                "--samples",
                "120",
            ]
        )

    if args.keep_offline:
        offline_dir = root / "offline"
        offline_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = offline_dir / "offline_dataset.h5"
        _run(
            [
                "python",
                "research/offline_rl/log_trajectories.py",
                "--episodes",
                "1",
                "--output",
                str(dataset_path),
            ]
        )
        _run(
            [
                "python",
                "research/offline_rl/train_offline.py",
                "--dataset",
                str(dataset_path),
                "--output-root",
                str(offline_dir),
            ]
        )

    print(f"[smoke-kaggle] Artifacts saved under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
