"""Runner registry and optional dependency management for driver execution."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Dict


RunnerCallable = Callable[[Path, Path], Path]


class RunnerNotRegisteredError(LookupError):
    """Raised when a runner key is not known to the registry."""

    def __init__(self, runner: str) -> None:
        super().__init__(f"Unknown runner '{runner}'")
        self.runner = runner


class RunnerNotAvailableError(RuntimeError):
    """Raised when an optional runner cannot be constructed."""

    def __init__(
        self,
        runner: str,
        message: str,
        *,
        missing_dependency: str | None = None,
    ) -> None:
        super().__init__(message)
        self.runner = runner
        self.missing_dependency = missing_dependency


_RUNNER_FACTORIES: Dict[str, Callable[[], RunnerCallable]] = {}
_RUNNER_CACHE: Dict[str, RunnerCallable] = {}


def register_runner(key: str, factory: Callable[[], RunnerCallable]) -> None:
    """Register *factory* for *key*, replacing any existing runner."""

    _RUNNER_FACTORIES[key] = factory
    _RUNNER_CACHE.pop(key, None)


def get_runner(key: str) -> RunnerCallable:
    """Return the callable registered for *key*, constructing it if needed."""

    if key in _RUNNER_CACHE:
        return _RUNNER_CACHE[key]

    factory = _RUNNER_FACTORIES.get(key)
    if factory is None:
        raise RunnerNotRegisteredError(key)

    try:
        runner = factory()
    except RunnerNotAvailableError:
        raise
    _RUNNER_CACHE[key] = runner
    return runner


def clear_runner_cache() -> None:
    """Reset the cached runner callables."""

    _RUNNER_CACHE.clear()


def _build_scenario_runner() -> RunnerCallable:
    from leadlag.training.run_scenario import run_scenario

    def _runner(scenario_path: Path, results_root: Path) -> Path:
        return run_scenario(str(scenario_path), str(results_root))

    return _runner


def _build_dynamic_runner() -> RunnerCallable:
    try:
        from leadlag.training.run_dynamic_baselines import run_dynamic
    except ImportError as exc:  # pragma: no cover - optional dependency path
        missing = getattr(exc, "name", None) or str(exc)
        raise RunnerNotAvailableError(
            "dynamic",
            "Dynamic baseline runner unavailable. Install optional dependencies for dynamic "
            f"baselines (missing module: {missing}).",
            missing_dependency=missing,
        ) from exc

    def _runner(scenario_path: Path, results_root: Path) -> Path:
        return run_dynamic(str(scenario_path), str(results_root))

    return _runner


def _build_rl_runner() -> RunnerCallable:
    try:
        from leadlag.training.run_rl import run_rl
    except ImportError as exc:  # pragma: no cover - optional dependency path
        missing = getattr(exc, "name", None) or str(exc)
        raise RunnerNotAvailableError(
            "rl",
            "RL runner unavailable. Install the RL extras (pip install -r requirements-rl.txt) "
            f"(missing module: {missing}).",
            missing_dependency=missing,
        ) from exc

    def _runner(scenario_path: Path, results_root: Path) -> Path:
        return run_rl(str(scenario_path), str(results_root))

    return _runner


register_runner("scenario", _build_scenario_runner)
register_runner("dynamic", _build_dynamic_runner)
register_runner("rl", _build_rl_runner)


__all__ = [
    "RunnerCallable",
    "RunnerNotAvailableError",
    "RunnerNotRegisteredError",
    "clear_runner_cache",
    "get_runner",
    "register_runner",
]
