from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.driver import scenario_registry


def test_discover_scenarios_prefers_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "configs" / "scenarios"
    config_dir.mkdir(parents=True)
    (config_dir / "b.yaml").write_text("run:\n  run_name: b\n", encoding="utf-8")
    (config_dir / "a.yaml").write_text("run:\n  run_name: a\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    scenarios = scenario_registry.discover_scenarios()

    assert [path.name for path in scenarios] == ["a.yaml", "b.yaml"]
    assert all(path.is_absolute() for path in scenarios)


def test_resolve_scenario_reference_resource() -> None:
    path = scenario_registry.resolve_scenario_reference("fixed_30")
    assert path.name == "fixed_30.yaml"
    assert path.exists()


def test_resolve_scenario_references_collects_errors(tmp_path: Path) -> None:
    local = tmp_path / "alpha.yaml"
    local.write_text("run:\n  run_name: alpha\n", encoding="utf-8")

    resolved, errors = scenario_registry.resolve_scenario_references(
        [str(local), "missing_scenario"]
    )

    assert resolved == [local.resolve()]
    assert len(errors) == 1
    assert "missing_scenario" in errors[0]
