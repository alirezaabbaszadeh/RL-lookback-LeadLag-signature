from __future__ import annotations

import json
from pathlib import Path

from leadlag.driver import selection


def test_matches_filters_supports_include_and_exclude() -> None:
    assert selection.matches_filters("alpha", include=["alp"], exclude=None)
    assert not selection.matches_filters("beta", include=["alp"], exclude=None)
    assert not selection.matches_filters("alpha", include=None, exclude=["alp"])
    assert selection.matches_filters("alpha", include=None, exclude=["zzz"])


def test_filter_scenarios_returns_matching_paths(tmp_path: Path) -> None:
    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml", tmp_path / "gamma.yaml"]
    filtered = selection.filter_scenarios(scenarios, include=["alp"], exclude=["g"])
    assert filtered == [tmp_path / "alpha.yaml"]


def test_has_successful_run_detects_existing_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    existing = results_root / "alpha_20240101_000000"
    existing.mkdir(parents=True)
    (existing / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    assert selection.has_successful_run("alpha", results_root) is True
    assert selection.has_successful_run("beta", results_root) is False


def test_collect_status_reports_runs(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    aggregate_dir = results_root / "aggregate"
    aggregate_dir.mkdir(parents=True)
    (aggregate_dir / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    success = results_root / "alpha_20240101_000000"
    success.mkdir()
    (success / "run_metadata.json").write_text(
        json.dumps({"config_path": "leadlag/configs/scenarios/alpha.yaml"}), encoding="utf-8"
    )
    (success / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    empty = results_root / "beta_20240101_000010"
    empty.mkdir()

    runs = selection.collect_status(results_root)

    statuses = {entry.run_dir: entry.status for entry in runs}
    assert str(aggregate_dir) in statuses and statuses[str(aggregate_dir)] == "aggregate"
    assert statuses[str(success)] == "success"
    assert statuses[str(empty)] == "empty"

    payloads = [entry.to_payload() for entry in runs]
    assert all("run_dir" in payload and "status" in payload for payload in payloads)
