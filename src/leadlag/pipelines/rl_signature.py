"""CLI entry point for orchestrating RL + signature experiments."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Sequence

from leadlag.cli.errors import ERROR_DEPENDENCY, ERROR_VALUE, emit_error
from leadlag.cli.formatters import (
    add_format_flags,
    emit_formatted_output,
    finalize_format_args,
)
from leadlag.reporting.logging_utils import get_logger, setup_logging


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RL + signature training bundle with standardized outputs.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Directory to store run outputs and metrics.",
    )
    parser.add_argument(
        "--bundle-root",
        "--out",
        dest="bundle_root",
        type=Path,
        default=Path("results/bundles"),
        help="Directory for aggregated artifacts (reports, bundles).",
    )
    parser.add_argument(
        "--training-profile",
        choices=["smoke", "paper"],
        default="smoke",
        help="Selects the training preset to execute (smoke or paper scale).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="Optional path to write pipeline logs.",
    )
    add_format_flags(parser, default="text")
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalize_paths(args: argparse.Namespace) -> None:
    args.results_root = Path(args.results_root).expanduser().resolve()
    args.bundle_root = Path(args.bundle_root).expanduser().resolve()
    if args.log_path is not None:
        args.log_path = Path(args.log_path).expanduser().resolve()


def _ensure_directories(results_root: Path, bundle_root: Path) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)


def _check_dependencies(logger) -> tuple[bool, list[str]]:
    required = [
        "stable_baselines3",
        "torch",
        "iisignature",
    ]
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        logger.error(
            "Missing required dependencies",
            context={"missing": ", ".join(missing)},
        )
        return False, missing
    logger.info("All required dependencies available")
    return True, []


def _plan_scenarios(profile: str) -> Dict[str, Iterable[str]]:
    if profile == "paper":
        return {
            "rl": ["ppo", "ppo_lstm"],
            "signature": ["full_signature_eval"],
            "analysis": ["report_bundle"],
        }
    return {
        "rl": ["ppo_smoke"],
        "signature": ["signature_smoke"],
        "analysis": ["report_bundle"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    finalize_format_args(args, remove_in="0.2.0")
    command = "leadlag-rl-signature"
    if argv:
        command = "leadlag-rl-signature " + " ".join(argv)

    _normalize_paths(args)
    _ensure_directories(args.results_root, args.bundle_root)

    log_path = args.log_path or args.bundle_root / "rl_signature.log"
    setup_logging(log_path, level=str(args.log_level).upper(), context={"module": "rl_signature"})
    logger = get_logger(
        "pipelines.rl_signature",
        context={
            "results_root": args.results_root,
            "bundle_root": args.bundle_root,
            "profile": args.training_profile,
        },
    )
    logger.info("Starting RL + signature pipeline")

    deps_ok, missing = _check_dependencies(logger)
    if not deps_ok:
        emit_error(
            args,
            code=ERROR_DEPENDENCY,
            message="Missing required dependencies for RL + signature pipeline.",
            details={"missing": missing},
        )
        return 1

    if args.training_profile not in {"smoke", "paper"}:
        emit_error(
            args,
            code=ERROR_VALUE,
            message="Unsupported training profile.",
            details={"profile": args.training_profile},
        )
        return 2

    plan = _plan_scenarios(args.training_profile)
    logger.info("Planned execution", context={"profile": args.training_profile, "plan": plan})

    artifacts: Dict[str, str] = {
        "results_root": str(args.results_root),
        "bundle_root": str(args.bundle_root),
        "log_path": str(log_path),
    }

    data = {
        "profile": args.training_profile,
        "paths": artifacts,
        "dependencies": {
            "required": ["stable_baselines3", "torch", "iisignature"],
            "missing": missing,
            "validated": deps_ok,
        },
        "execution_plan": plan,
        "status": "planned",
    }

    text_lines = [
        f"Profile: {args.training_profile}",
        f"Results root: {args.results_root}",
        f"Bundle root: {args.bundle_root}",
        "Dependencies: OK",
    ]
    message = "RL + signature pipeline planned."

    emit_formatted_output(
        args,
        data=data,
        artifacts=artifacts,
        text="\n".join(text_lines),
        message=message,
        pretty=True,
        command=command,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
