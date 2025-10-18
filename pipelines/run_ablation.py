from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra_main  # type: ignore


@dataclass
class ScenarioInfo:
    name: str
    runner: str
    requires_sb3: bool


def load_scenario(name: str) -> ScenarioInfo:
    cfg = hydra_main._load_scenario_cfg(name)  # pylint: disable=protected-access
    runner = cfg.get("runner", "scenario")
    rl_cfg = cfg.get("rl", {})
    policy = str(rl_cfg.get("policy", "")).lower() if isinstance(rl_cfg, dict) else ""
    random_policy = bool(rl_cfg.get("random_policy")) if isinstance(rl_cfg, dict) else False
    requires_sb3 = runner == "rl" and not (policy == "random" or random_policy)
    return ScenarioInfo(name=name, runner=runner, requires_sb3=requires_sb3)


def run_command(cmd: Sequence[str]) -> None:
    print(f"[ablation] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_dependencies(info: ScenarioInfo, skip_missing: bool) -> bool:
    if not info.requires_sb3:
        return True
    has_sb3 = importlib.util.find_spec("stable_baselines3") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    if has_sb3 and has_torch:
        return True
    message = (
        f"Scenario '{info.name}' requires stable-baselines3 and torch. "
        "Install optional dependencies via:\n"
        "  pip install stable-baselines3 torch --extra-index-url https://download.pytorch.org/whl/cu118"
    )
    if skip_missing:
        print(f"[ablation] Skipping {info.name}: {message}")
        return False
    raise SystemExit(message)


def run_scenario(
    info: ScenarioInfo,
    output_root: Path,
    seeds: Iterable[int],
    single_seed: bool,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(ROOT / "hydra_main.py"),
        "--scenario",
        info.name,
        "--output_root",
        str(output_root),
    ]
    seed_list = list(seeds)
    if not single_seed and seed_list:
        cmd.append("--multi_seed_enabled")
        cmd.append("--seeds")
        cmd.extend(str(s) for s in seed_list)
    run_command(cmd)


def run_comparison(results_root: Path, out_dir: Path, metric: str) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "reporting" / "compare_scenarios.py"),
        "--results_root",
        str(results_root),
        "--out",
        str(out_dir),
        "--metric",
        metric,
    ]
    run_command(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation scenarios end-to-end.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/kaggle/working/ablations"),
        help="Directory to write scenario outputs and aggregates.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=[
            "fixed_30",
            "fixed_90",
            "dynamic_adaptive",
            "abl_smoke",
            "abl_lite_gpu",
            "abl_server",
            "abl_random",
        ],
        help="Scenario names to execute (defaults cover signature, dynamic, RL, and random controls).",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[42, 52, 62],
        help="Seeds for multi-seed aggregation.",
    )
    parser.add_argument(
        "--single-seed",
        action="store_true",
        help="Run each scenario once (skip multi-seed aggregation).",
    )
    parser.add_argument(
        "--skip-missing-deps",
        action="store_true",
        help="Skip scenarios whose optional dependencies are unavailable instead of exiting.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_abs_matrix",
        help="Metric name to highlight in the comparison plots/CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root: Path = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    executed: List[str] = []

    for name in args.scenarios:
        info = load_scenario(name)
        if not ensure_dependencies(info, args.skip_missing_deps):
            continue
        try:
            run_scenario(info, output_root, args.seeds, args.single_seed)
            executed.append(name)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Scenario '{name}' failed with exit code {exc.returncode}") from exc

    if not executed:
        print("[ablation] No scenarios were executed; nothing to compare.")
        return 0

    comparison_out = output_root / "ablation_comparison"
    comparison_out.mkdir(parents=True, exist_ok=True)
    run_comparison(output_root, comparison_out, args.metric)

    print("[ablation] Completed scenarios:", ", ".join(executed))
    print(f"[ablation] Comparison artifacts saved under: {comparison_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
