"""Train SB3 algorithms on the LeadLag environment for production runs.

This stage bypasses Hydra and calls the project's "training.run_rl" API directly,
so we can inject overrides (e.g. total_timesteps, device) without editing YAML.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from leadlag.training.run_rl import run_rl
from leadlag.utils.resources import resolve_path

DEFAULT_SCENARIOS = ["rl_ppo", "rl_ppo_lstm", "rl_ppo_sharpe"]


def parse_env_overrides() -> Dict[str, object]:
    o: Dict[str, object] = {}
    seed = os.getenv("SB3_SEED")
    if seed:
        try:
            o.setdefault("run", {})["seed"] = int(seed)
        except ValueError:
            print("Warning: SB3_SEED must be int; ignoring.")
    ts = os.getenv("SB3_TIMESTEPS")
    if ts:
        try:
            o.setdefault("rl", {})["total_timesteps"] = int(ts)
        except ValueError:
            print("Warning: SB3_TIMESTEPS must be int; ignoring.")
    device = os.getenv("SB3_DEVICE")
    if device:
        o.setdefault("rl", {})["device"] = device
    nsteps = os.getenv("SB3_N_STEPS")
    if nsteps:
        try:
            o.setdefault("rl", {})["n_steps"] = int(nsteps)
        except ValueError:
            print("Warning: SB3_N_STEPS must be int; ignoring.")
    batch = os.getenv("SB3_BATCH_SIZE")
    if batch:
        try:
            o.setdefault("rl", {})["batch_size"] = int(batch)
        except ValueError:
            print("Warning: SB3_BATCH_SIZE must be int; ignoring.")
    lr = os.getenv("SB3_LR")
    if lr:
        try:
            o.setdefault("rl", {})["learning_rate"] = float(lr)
        except ValueError:
            print("Warning: SB3_LR must be float; ignoring.")
    eval_freq = os.getenv("SB3_EVAL_FREQ")
    if eval_freq:
        try:
            o.setdefault("rl", {})["eval_freq"] = int(eval_freq)
        except ValueError:
            print("Warning: SB3_EVAL_FREQ must be int; ignoring.")
    verbose = os.getenv("SB3_VERBOSE")
    if verbose:
        o.setdefault("rl", {})["verbose"] = verbose.strip().lower() in {"1", "true", "yes", "on"}
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description="LeadLag SB3 production stage")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="Scenario names packaged with LeadLag (or explicit YAML paths)",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = parse_env_overrides()

    summary: List[Dict[str, object]] = []
    for scn in args.scenarios:
        scn_path = Path(scn)
        if not scn_path.exists():
            resource_name = (
                scn_path.name if scn_path.suffix == ".yaml" else f"{scn_path.name}.yaml"
            )
            resolved = resolve_path("leadlag.configs", f"scenarios/{resource_name}")
            if resolved is None or not resolved.exists():
                raise FileNotFoundError(f"Packaged scenario not found: {scn}")
            scn_path = resolved
        out_root = args.output_dir / scn_path.stem
        out_root.mkdir(parents=True, exist_ok=True)
        out_dir = run_rl(str(scn_path), str(out_root), overrides)
        summary.append(
            {
                "scenario": scn_path.stem,
                "path": str(out_dir),
                "timestamp": time.time(),
            }
        )

    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "stage": "sb3_leadlag",
                "timestamp": time.time(),
                "scenarios": [Path(s).stem for s in args.scenarios],
                "overrides": overrides,
                "items": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
