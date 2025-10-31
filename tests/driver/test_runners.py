from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.driver import runners


def test_runner_registry_unknown_key() -> None:
    with pytest.raises(runners.RunnerNotRegisteredError):
        runners.get_runner("missing")


def test_runner_registry_dispatch_and_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runners, "_RUNNER_FACTORIES", {})
    monkeypatch.setattr(runners, "_RUNNER_CACHE", {})

    def factory() -> runners.RunnerCallable:
        return lambda scenario_path, results_root: results_root / "custom"

    runners.register_runner("custom", factory)
    runner = runners.get_runner("custom")
    out_dir = runner(Path("scenario.yaml"), tmp_path)

    assert out_dir == tmp_path / "custom"
    assert runners.get_runner("custom") is runner


def test_runner_registry_optional_dependency_error(monkeypatch) -> None:
    monkeypatch.setattr(runners, "_RUNNER_FACTORIES", {})
    monkeypatch.setattr(runners, "_RUNNER_CACHE", {})

    def factory() -> runners.RunnerCallable:
        raise runners.RunnerNotAvailableError("custom", "missing dep", missing_dependency="dep")

    runners.register_runner("custom", factory)
    with pytest.raises(runners.RunnerNotAvailableError) as excinfo:
        runners.get_runner("custom")

    assert excinfo.value.runner == "custom"
    assert excinfo.value.missing_dependency == "dep"
