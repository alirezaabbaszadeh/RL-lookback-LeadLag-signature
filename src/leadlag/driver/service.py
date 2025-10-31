"""Service-layer logic for discovering, selecting, and executing scenarios."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

from leadlag.evaluation.aggregate import aggregate
from leadlag.training.run_scenario import (
    _merge_extends,
    _validate_scenario_schema,
    run_scenario,
)
from leadlag.utils.resources import resolve_path


class _NullLogger:
    """Fallback logger that accepts the structured logging interface."""

    def info(self, *_args, **_kwargs) -> None:
        return None

    def warning(self, *_args, **_kwargs) -> None:
        return None

    def exception(self, *_args, **_kwargs) -> None:
        return None


@dataclass(slots=True)
class ScenarioSelection:
    """Structured representation of a selected scenario reference."""

    name: str
    display: str
    path: str

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-serialisable payload for the selection."""

        return asdict(self)


@dataclass(slots=True)
class ScenarioResult:
    """Scenario execution outcome captured by the driver."""

    scenario: str
    status: str
    runner: str | None
    output: str | None = None
    error: str | None = None
    reason: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload matching the historical JSON schema."""

        data = asdict(self)
        payload: dict[str, object] = {
            "scenario": data["scenario"],
            "status": data["status"],
            "runner": data["runner"],
        }
        for optional in ("output", "error", "reason"):
            value = data[optional]
            if value is not None:
                payload[optional] = value
        return payload


@dataclass(slots=True)
class RunStatusEntry:
    """Metadata describing the status of a run discovered on disk."""

    run_dir: str
    status: str
    scenario: str | None = None
    path: str | None = None
    summary_path: str | None = None
    metadata_path: str | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload mirroring the legacy run status mapping."""

        data = asdict(self)
        payload: dict[str, object] = {
            "run_dir": data["run_dir"],
            "status": data["status"],
        }
        if data["scenario"] is not None:
            payload["scenario"] = data["scenario"]
        if data["status"] == "aggregate" and data["path"] is not None:
            payload["path"] = data["path"]
        if data["summary_path"] is not None:
            payload["summary_path"] = data["summary_path"]
        if data["metadata_path"] is not None:
            payload["metadata_path"] = data["metadata_path"]
        return payload


@dataclass(slots=True)
class DriverSummary:
    """Serializable payload summarising the driver execution."""

    selected: list[str]
    results_root: str
    summary: list[ScenarioResult]
    aggregate: str | None
    dry_run: bool
    dry_run_entries: list[ScenarioSelection] | None = None

    def to_payload(self) -> dict[str, object]:
        """Return a payload preserving historical JSON structure."""

        payload: dict[str, object] = {
            "selected": list(self.selected),
            "results_root": self.results_root,
            "summary": [entry.to_payload() for entry in self.summary],
            "aggregate": self.aggregate,
            "dry_run": self.dry_run,
        }
        if self.dry_run_entries:
            payload["dry_run_entries"] = [entry.to_payload() for entry in self.dry_run_entries]
        return payload


@dataclass(slots=True)
class ExecutionOptions:
    """Configuration for executing scenarios."""

    results_root: Path
    runner_preference: str = "auto"
    skip_existing: bool = False
    stop_on_error: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class ExecutionResult:
    """Structured result returned from scenario execution."""

    summary: list[ScenarioResult] = field(default_factory=list)
    errors: list[dict[str, object]] = field(default_factory=list)
    aggregate: Path | None = None
    exit_code: int = 0
    aborted: bool = False
    dry_run: bool = False
    dry_run_entries: list[ScenarioSelection] = field(default_factory=list)


def discover_scenarios() -> list[Path]:
    """Return all discoverable scenario configuration files."""

    scenarios: list[Path] = []
    local_dir = Path("configs") / "scenarios"
    if local_dir.exists():
        local_scenarios = sorted(local_dir.glob("*.yaml"))
        if local_scenarios:
            return [p.resolve() for p in local_scenarios]

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


def matches_filters(name: str, include: Iterable[str] | None, exclude: Iterable[str] | None) -> bool:
    """Return ``True`` when *name* matches the include/exclude filters."""

    if include:
        if not any(token.lower() in name.lower() for token in include):
            return False
    if exclude:
        if any(token.lower() in name.lower() for token in exclude):
            return False
    return True


def filter_scenarios(
    scenarios: Sequence[Path],
    include: Iterable[str] | None,
    exclude: Iterable[str] | None,
) -> list[Path]:
    """Filter scenarios by name using include/exclude tokens."""

    return [sc for sc in scenarios if matches_filters(sc.stem, include, exclude)]


def resolve_scenario_reference(entry: str) -> Path:
    """Resolve a user-supplied scenario reference to a concrete path."""

    candidate = Path(entry)
    if candidate.exists():
        return candidate.resolve()

    name = candidate.name
    resource = name if name.endswith(".yaml") else f"{name}.yaml"
    resolved = resolve_path("leadlag.configs", f"scenarios/{resource}")
    if resolved is not None and resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Scenario '{entry}' not found in packaged resources or filesystem paths."
    )


def resolve_scenario_references(entries: Sequence[str]) -> tuple[list[Path], list[str]]:
    """Resolve multiple scenario references, returning successes and failures."""

    resolved: list[Path] = []
    errors: list[str] = []
    for entry in entries:
        try:
            resolved.append(resolve_scenario_reference(entry))
        except FileNotFoundError as exc:
            errors.append(str(exc))
    return resolved, errors


def has_successful_run(run_name: str, results_root: Path) -> bool:
    """Return ``True`` if a prior successful run exists for *run_name*."""

    if not results_root.exists():
        return False

    prefix = f"{run_name}_"
    for child in results_root.iterdir():
        if child.is_dir() and child.name.startswith(prefix):
            if (child / "summary.csv").exists():
                return True
    return False


def collect_status(results_root: Path) -> list[RunStatusEntry]:
    """Collect execution status metadata under *results_root*."""

    runs: list[RunStatusEntry] = []
    if not results_root.exists():
        return runs

    for child in sorted(results_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue

        if child.name == "aggregate":
            runs.append(
                RunStatusEntry(run_dir=str(child), status="aggregate", path=str(child))
            )
            continue

        entry = RunStatusEntry(run_dir=str(child), status="empty")
        metadata_path = child / "run_metadata.json"
        summary_path = child / "summary.csv"

        scenario_name: str | None = None
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                config_path = meta.get("config_path")
                if isinstance(config_path, str) and config_path:
                    scenario_name = Path(config_path).stem
                scenario_name = scenario_name or meta.get("scenario") or meta.get("run_name")
            except Exception:
                scenario_name = None
        if scenario_name:
            entry.scenario = scenario_name

        if summary_path.exists():
            entry.status = "success"
            entry.summary_path = str(summary_path)
        elif metadata_path.exists():
            entry.status = "incomplete"
            entry.metadata_path = str(metadata_path)

        runs.append(entry)

    return runs


def _pick_runner(preference: str, config: dict[str, object]) -> str:
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


def execute_scenarios(
    selected: Sequence[Path],
    options: ExecutionOptions,
    *,
    logger=None,
) -> ExecutionResult:
    """Execute *selected* scenarios with the provided options."""

    if logger is None:
        logger = _NullLogger()
    results_root = options.results_root

    if options.dry_run:
        dry_entries: list[ScenarioSelection] = []
        for sc in selected:
            sc_path = Path(sc)
            try:
                display = sc_path.relative_to(Path.cwd())
            except ValueError:
                display = sc_path
            logger.info(f"[dry-run] {display}")
            dry_entries.append(
                ScenarioSelection(
                    name=sc_path.stem,
                    display=str(display),
                    path=str(sc_path),
                )
            )
        return ExecutionResult(dry_run=True, dry_run_entries=dry_entries)

    results_root.mkdir(parents=True, exist_ok=True)

    summary: list[ScenarioResult] = []
    errors_list: list[dict[str, object]] = []
    aggregate_path: Path | None = None
    exit_code = 0
    aborted = False

    for sc in selected:
        name = sc.stem
        if options.skip_existing and has_successful_run(name, results_root):
            logger.info(
                "Skipping scenario with existing successful run",
                context={"scenario": name},
            )
            summary.append(
                ScenarioResult(
                    scenario=name,
                    status="skipped",
                    runner=None,
                    reason="existing_results",
                )
            )
            continue
        try:
            config = _merge_extends(sc)
            _validate_scenario_schema(config, scenario=name)
        except Exception as exc:
            logger.exception("Failed to load scenario config", context={"scenario": name})
            summary.append(
                ScenarioResult(
                    scenario=name,
                    status="load_failed",
                    runner=None,
                    error=str(exc),
                )
            )
            errors_list.append(
                {
                    "code": "scenario_load_failed",
                    "message": "Scenario load failed",
                    "details": {"scenario": name, "error": str(exc)},
                }
            )
            if options.stop_on_error:
                exit_code = 1
                aborted = True
                break
            continue

        runner = _pick_runner(options.runner_preference, config)
        logger.info("Running scenario", context={"scenario": name, "runner": runner})
        try:
            out_dir = _execute_runner(runner, sc, results_root)
            summary.append(
                ScenarioResult(
                    scenario=name,
                    status="success",
                    runner=runner,
                    output=str(out_dir),
                )
            )
            logger.info("Scenario completed", context={"scenario": name, "output": out_dir})
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Scenario execution failed", context={"scenario": name})
            summary.append(
                ScenarioResult(
                    scenario=name,
                    status="error",
                    runner=runner,
                    error=str(exc),
                )
            )
            errors_list.append(
                {
                    "code": "scenario_execution_failed",
                    "message": "Scenario execution failed",
                    "details": {"scenario": name, "error": str(exc)},
                }
            )
            if options.stop_on_error:
                exit_code = 1
                aborted = True
                break

    if not aborted:
        successes = [row for row in summary if row.status == "success"]
        if successes:
            try:
                aggregate_path = aggregate(str(results_root))
                logger.info(
                    "Aggregated comparison complete", context={"aggregate": aggregate_path}
                )
            except Exception as exc:  # pragma: no cover
                logger.exception(
                    "Aggregation failed", context={"results_root": results_root}
                )
                errors_list.append(
                    {
                        "code": "aggregation_failed",
                        "message": "Aggregation failed",
                        "details": {
                            "results_root": str(results_root),
                            "error": str(exc),
                        },
                    }
                )
                if options.stop_on_error:
                    exit_code = 1
                    aborted = True

    return ExecutionResult(
        summary=summary,
        errors=errors_list,
        aggregate=aggregate_path,
        exit_code=exit_code,
        aborted=aborted,
    )


__all__ = [
    "DriverSummary",
    "ExecutionOptions",
    "ExecutionResult",
    "collect_status",
    "discover_scenarios",
    "execute_scenarios",
    "filter_scenarios",
    "has_successful_run",
    "matches_filters",
    "RunStatusEntry",
    "ScenarioResult",
    "ScenarioSelection",
    "resolve_scenario_reference",
    "resolve_scenario_references",
]
