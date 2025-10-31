from __future__ import annotations

import argparse

import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    _SRC_ROOT = Path(__file__).resolve().parents[2]
    if str(_SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(_SRC_ROOT))

from leadlag import hydra_main  # type: ignore
from leadlag.cli.formatters import add_format_flags, emit_formatted_output, finalize_format_args
from leadlag.reporting.logging_utils import get_logger, setup_logging


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ScenarioInfo:
    name: str
    runner: str
    requires_sb3: bool
    requires_sb3_contrib: bool
    requires_signature: bool


def load_scenario(name: str) -> ScenarioInfo:
    cfg = hydra_main._load_scenario_cfg(name)  # pylint: disable=protected-access
    runner = cfg.get("runner", "scenario")
    rl_cfg = cfg.get("rl", {})
    policy = str(rl_cfg.get("policy", "")).lower() if isinstance(rl_cfg, dict) else ""
    random_policy = bool(rl_cfg.get("random_policy")) if isinstance(rl_cfg, dict) else False
    requires_sb3 = runner == "rl" and not (policy == "random" or random_policy)
    requires_contrib = requires_sb3 and ("lstm" in policy or "recurrent" in policy)
    analysis_cfg = cfg.get("analysis", {})
    method = str(analysis_cfg.get("method", "signature")).lower()
    requires_signature = method == "signature"
    return ScenarioInfo(
        name=name,
        runner=runner,
        requires_sb3=requires_sb3,
        requires_sb3_contrib=requires_contrib,
        requires_signature=requires_signature,
    )


def run_command(cmd: Sequence[str], logger) -> None:
    logger.info("Executing command", context={"cmd": " ".join(cmd)})
    subprocess.run(cmd, check=True)


def ensure_dependencies(info: ScenarioInfo, skip_missing: bool, logger) -> Tuple[bool, Optional[str]]:
    if info.requires_signature and importlib.util.find_spec("iisignature") is None:
        message = (
            f"Scenario '{info.name}' requires the 'iisignature' package. "
            "Install via:\n  pip install iisignature"
        )
        if skip_missing:
            logger.warning(
                "Skipping scenario due to missing signature deps",
                context={"scenario": info.name},
            )
            return False, "missing optional dependency: iisignature"
        raise SystemExit(message)

    if not info.requires_sb3:
        return True, None
    has_sb3 = importlib.util.find_spec("stable_baselines3") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    has_contrib = True
    if info.requires_sb3_contrib:
        has_contrib = importlib.util.find_spec("sb3_contrib") is not None
    if has_sb3 and has_torch and has_contrib:
        return True, None
    missing = []
    if not has_sb3:
        missing.append("stable-baselines3")
    if not has_torch:
        missing.append("torch")
    if not has_contrib:
        missing.append("sb3-contrib")
    message = (
        f"Scenario '{info.name}' requires optional dependencies: {', '.join(missing)}. "
        "Install via:\n"
        "  pip install stable-baselines3 torch --extra-index-url https://download.pytorch.org/whl/cu118\n"
        "  pip install sb3-contrib   # for PPO-LSTM"
    )
    if skip_missing:
        logger.warning(
            "Skipping scenario due to missing RL deps",
            context={"scenario": info.name, "missing": ", ".join(missing)},
        )
        return False, "missing optional dependencies: " + ", ".join(missing)
    raise SystemExit(message)


def run_scenario(
    info: ScenarioInfo,
    output_root: Path,
    seeds: Iterable[int],
    single_seed: bool,
    logger,
) -> None:
    cmd: List[str] = [
        sys.executable,
        str(PACKAGE_ROOT / "hydra_main.py"),
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
    run_command(cmd, logger)


def run_comparison(results_root: Path, out_dir: Path, metric: str, logger) -> None:
    cmd = [
        sys.executable,
        str(PACKAGE_ROOT / "reporting" / "compare_scenarios.py"),
        "--results_root",
        str(results_root),
        "--out",
        str(out_dir),
        "--metric",
        metric,
    ]
    run_command(cmd, logger)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
            "ccf_fixed",
            "dynamic_adaptive",
            "abl_smoke",
            "abl_lite_gpu",
            "abl_server",
            "abl_random",
            "rl_ppo",
            "rl_ppo_sharpe",
            "rl_ppo_drawdown",
            "rl_ppo_lstm",
        ],
        help=(
            "Scenario names to execute (defaults cover signature, dynamic, RL, and random "
            "controls)."
        ),
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
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional path for the ablation log file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List scenarios after dependency checks without executing them.",
    )
    add_format_flags(parser, default="text")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-ablation"
    if argv:
        command = "leadlag-ablation " + " ".join(argv)
    output_root: Path = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or logs_dir / "ablation.log"
    setup_logging(Path(log_path), level=str(args.log_level).upper(), context={"module": "ablation"})
    logger = get_logger("pipelines.run_ablation", context={"output_root": output_root})

    executed: List[str] = []
    skipped: List[Dict[str, str]] = []

    for name in args.scenarios:
        info = load_scenario(name)
        ok, reason = ensure_dependencies(info, args.skip_missing_deps, logger)
        if not ok:
            skipped.append({"scenario": name, "reason": reason or "optional dependencies unavailable"})
            continue
        logger.info("Scenario ready", context={"scenario": name, "runner": info.runner})
        if args.dry_run:
            executed.append(name)
            continue
        try:
            run_scenario(info, output_root, args.seeds, args.single_seed, logger)
            executed.append(name)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Scenario '{name}' failed with exit code {exc.returncode}") from exc

    summary: Dict[str, object] = {
        "output_root": str(output_root),
        "dry_run": bool(args.dry_run),
        "executed": executed,
        "skipped": skipped,
    }
    artifacts: Dict[str, object] = {}

    text_lines: List[str] = []
    if executed:
        text_lines.append("Executed scenarios: " + ", ".join(executed))
    if skipped:
        text_lines.append("Skipped scenarios:")
        for entry in skipped:
            text_lines.append(f"  - {entry['scenario']}: {entry['reason']}")

    if not executed:
        logger.warning("No scenarios executed; skipping comparison")
        if not text_lines:
            text_lines.append("No scenarios executed; skipping comparison")
        emit_formatted_output(
            args,
            data=summary,
            text="\n".join(text_lines),
            message="No scenarios executed; skipping comparison.",
            pretty=True,
            command=command,
        )
        return 0

    comparison_out = output_root / "ablation_comparison"
    comparison_out.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        run_comparison(output_root, comparison_out, args.metric, logger)
        summary["comparison_output"] = str(comparison_out)
        artifacts["comparison_output"] = str(comparison_out)
        text_lines.append(f"Comparison artifacts: {comparison_out}")
    else:
        text_lines.append(f"[dry-run] Comparison would be written to: {comparison_out}")

    logger.info(
        "Completed scenarios",
        context={"scenarios": ", ".join(executed), "comparison": str(comparison_out)},
    )

    message = "Ablation run completed."
    if args.dry_run:
        message = "Ablation dry-run completed."
    emit_formatted_output(
        args,
        data=summary,
        text="\n".join(text_lines),
        message=message,
        artifacts=artifacts or None,
        pretty=True,
        command=command,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
