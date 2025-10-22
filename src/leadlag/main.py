from __future__ import annotations

import argparse
import json
import os
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List

from leadlag.evaluation.aggregate import aggregate
from leadlag.reporting.logging_utils import get_logger, setup_logging
from leadlag.training.run_scenario import _merge_extends, run_scenario
from leadlag.utils.resources import resolve_path


def discover_scenarios() -> list[Path]:
    scenarios: list[Path] = []
    try:
        base = resources.files("leadlag.configs").joinpath("scenarios")
        for entry in base.iterdir():
            if entry.name.endswith(".yaml"):
                resolved = resolve_path("leadlag.configs", f"scenarios/{entry.name}")
                if resolved:
                    scenarios.append(resolved)
    except (ModuleNotFoundError, AttributeError):
        pass

    if not scenarios:
        fallback_dir = resolve_path("leadlag.configs", "scenarios")
        if fallback_dir and fallback_dir.is_dir():
            scenarios.extend(sorted(fallback_dir.glob("*.yaml")))

    return sorted({path.resolve() for path in scenarios})


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured LeadLag scenarios and aggregate results.",
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Directory where scenario outputs and aggregates are stored "
            "(default: LEADLAG_RESULTS_ROOT or 'results')."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Only run scenarios whose filename contains any of the provided substrings.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Skip scenarios whose filename contains any of the provided substrings.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        help="Maximum number of scenarios to execute after filtering.",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "scenario", "dynamic", "rl"],
        default="auto",
        help="Force a specific runner or let it auto-detect based on config blocks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected scenarios without executing them.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort immediately if a scenario fails to load or execute.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available packaged scenarios and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON for listings and run summaries.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    parser.add_argument(
        "--log-path",
        help="Optional path for the driver log file (defaults to <results-root>/main.log).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.results_root is None:
        args.results_root = os.environ.get("LEADLAG_RESULTS_ROOT", "results")
    return args


def _match_filters(name: str, include: Iterable[str] | None, exclude: Iterable[str] | None) -> bool:
    if include:
        if not any(token.lower() in name.lower() for token in include):
            return False
    if exclude:
        if any(token.lower() in name.lower() for token in exclude):
            return False
    return True


def _pick_runner(preference: str, config: Dict[str, object]) -> str:
    if preference in {"scenario", "dynamic", "rl"}:
        return preference

    if "dynamic" in config:
        return "dynamic"
    if "rl" in config:
        return "rl"
    return "scenario"


def _execute_runner(runner: str, scenario_path: Path, results_root: Path) -> Path:
    if runner == "scenario":
        return run_scenario(str(scenario_path), str(results_root))

    if runner == "dynamic":
        try:
            from leadlag.training.run_dynamic_baselines import run_dynamic
        except ImportError as exc:  # pragma: no cover - optional dependency path
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(
                "Dynamic baseline runner unavailable. Install optional dependencies for dynamic "
                f"baselines (missing module: {missing})."
            ) from exc
        return run_dynamic(str(scenario_path), str(results_root))

    if runner == "rl":
        try:
            from leadlag.training.run_rl import run_rl
        except ImportError as exc:  # pragma: no cover - optional dependency path
            missing = getattr(exc, "name", None) or str(exc)
            raise RuntimeError(
                "RL runner unavailable. Install the RL extras (pip install -r requirements-rl.txt) "
                f"(missing module: {missing})."
            ) from exc
        return run_rl(str(scenario_path), str(results_root))

    raise ValueError(f"Unknown runner '{runner}'")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    scenarios = discover_scenarios()
    if not scenarios:
        if args.list and args.json:
            print(json.dumps({"scenarios": []}))
        else:
            print("No scenarios found in packaged scenarios (leadlag.configs.scenarios)")
        return 1

    scenario_names = [path.stem for path in scenarios]
    if args.list:
        if args.json:
            print(json.dumps({"scenarios": scenario_names}))
        else:
            print("\n".join(scenario_names))
        return 0

    selected = [sc for sc in scenarios if _match_filters(sc.stem, args.include, args.exclude)]
    if args.max_scenarios is not None:
        selected = selected[: max(args.max_scenarios, 0)]

    results_root = Path(args.results_root).resolve()

    if not selected:
        if args.json:
            payload = {
                "selected": [],
                "summary": [],
                "aggregate": None,
                "results_root": str(results_root),
            }
            print(json.dumps(payload))
        else:
            print("No scenarios match the provided filters.")
        return 1

    results_root.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_path).resolve() if args.log_path else results_root / "main.log"
    setup_logging(log_path, level=args.log_level.upper(), context={"module": "driver"})
    logger = get_logger("leadlag.main", context={"results_root": results_root})

    logger.info(
        "Discovered %s scenario(s); %s selected after filtering.",
        len(scenarios),
        len(selected),
    )
    if args.dry_run:
        for sc in selected:
            logger.info("[dry-run] %s", sc)
        if args.json:
            payload = {
                "selected": [sc.stem for sc in selected],
                "summary": [],
                "aggregate": None,
                "results_root": str(results_root),
            }
            print(json.dumps(payload))
        return 0

    summary: list[dict[str, object]] = []
    aggregate_path: Path | None = None
    exit_code = 0
    aborted = False

    for sc in selected:
        name = sc.stem
        try:
            config = _merge_extends(sc)
        except Exception as exc:
            logger.exception("Failed to load scenario config", context={"scenario": name})
            summary.append(
                {
                    "scenario": name,
                    "status": "load_failed",
                    "runner": None,
                    "error": str(exc),
                }
            )
            if args.stop_on_error:
                exit_code = 1
                aborted = True
                break
            continue

        runner = _pick_runner(args.runner, config)
        logger.info("Running scenario", context={"scenario": name, "runner": runner})
        try:
            out_dir = _execute_runner(runner, sc, results_root)
            summary.append(
                {
                    "scenario": name,
                    "status": "success",
                    "output": str(out_dir),
                    "runner": runner,
                }
            )
            logger.info("Scenario completed", context={"scenario": name, "output": out_dir})
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Scenario execution failed", context={"scenario": name})
            summary.append(
                {
                    "scenario": name,
                    "status": "error",
                    "runner": runner,
                    "error": str(exc),
                }
            )
            if args.stop_on_error:
                exit_code = 1
                aborted = True
                break

    if not aborted:
        successes = [row for row in summary if row.get("status") == "success"]
        if successes:
            try:
                aggregate_path = aggregate(str(results_root))
                logger.info("Aggregated comparison complete", context={"aggregate": aggregate_path})
            except Exception:  # pragma: no cover - aggregate failures should not hide scenario results
                logger.exception("Aggregation failed", context={"results_root": results_root})
                if args.stop_on_error:
                    exit_code = 1
                    aborted = True

    failures = [row for row in summary if row.get("status") != "success"]
    if failures:
        logger.warning(
            "Some scenarios did not complete successfully",
            context={"failures": len(failures)},
        )
        if args.stop_on_error and exit_code == 0:
            exit_code = 1

    if aborted and args.stop_on_error:
        # ensure non-zero exit when aborted early
        exit_code = 1

    if args.json:
        payload = {
            "selected": [sc.stem for sc in selected],
            "summary": summary,
            "aggregate": str(aggregate_path) if aggregate_path else None,
            "results_root": str(results_root),
        }
        print(json.dumps(payload))

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
