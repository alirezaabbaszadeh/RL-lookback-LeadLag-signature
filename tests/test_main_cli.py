from __future__ import annotations

import json
from pathlib import Path

import pytest

try:  # pragma: no cover - dependency guard for optional pandas wheels
    from leadlag import main
except ValueError as exc:
    pytest.skip(f"Dependency import failed: {exc}", allow_module_level=True)


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.records.append(("info", message, kwargs.get("context")))

    def warning(
        self, message: str, *args, **kwargs
    ) -> None:  # pragma: no cover - exercised in tests
        self.records.append(("warning", message, kwargs.get("context")))

    def exception(
        self, message: str, *args, **kwargs
    ) -> None:  # pragma: no cover - exercised in tests
        self.records.append(("exception", message, kwargs.get("context")))


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs" / "scenarios"
    config_dir.mkdir(parents=True)
    dummy_logger = DummyLogger()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "get_logger", lambda *args, **kwargs: dummy_logger)

    return tmp_path, config_dir, dummy_logger


def test_match_filters_basic():
    assert main._match_filters("alpha", include=["alp"], exclude=None)
    assert not main._match_filters("beta", include=["alp"], exclude=None)
    assert not main._match_filters("alpha", include=None, exclude=["alp"])
    assert main._match_filters("alpha", include=None, exclude=["zzz"])


def test_discover_scenarios_sorted(cli_env):
    _, config_dir, _ = cli_env
    (config_dir / "b.yaml").write_text("run: {}\n", encoding="utf-8")
    (config_dir / "a.yaml").write_text("run: {}\n", encoding="utf-8")
    (config_dir / "ignored.txt").write_text("ignore", encoding="utf-8")

    scenarios = main.discover_scenarios()
    assert [path.name for path in scenarios] == ["a.yaml", "b.yaml"]


def test_main_dry_run_filters(cli_env, monkeypatch):
    _, config_dir, logger = cli_env
    (config_dir / "alpha.yaml").write_text("run: {}\n", encoding="utf-8")
    (config_dir / "beta.yaml").write_text("run: {}\n", encoding="utf-8")

    def fail_aggregate(_root: str) -> Path:  # pragma: no cover - should never run
        raise AssertionError("aggregate must not run during dry-run")

    monkeypatch.setattr(main, "aggregate", fail_aggregate)

    exit_code = main.main(["--dry-run", "--include", "alpha"])
    assert exit_code == 0

    dry_run_messages = [
        msg for level, msg, _ in logger.records if level == "info" and "[dry-run]" in msg
    ]
    assert dry_run_messages == [f"[dry-run] {Path('configs/scenarios/alpha.yaml')}"]


def test_main_list_outputs_names(cli_env, capsys):
    _, config_dir, _ = cli_env
    (config_dir / "beta.yaml").write_text("run: {}\n", encoding="utf-8")
    (config_dir / "alpha.yaml").write_text("run: {}\n", encoding="utf-8")

    exit_code = main.main(["--list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == ["alpha", "beta"]


def test_main_list_json(cli_env, capsys):
    _, config_dir, _ = cli_env
    (config_dir / "beta.yaml").write_text("run: {}\n", encoding="utf-8")
    (config_dir / "alpha.yaml").write_text("run: {}\n", encoding="utf-8")

    exit_code = main.main(["--list", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    data = json.loads(captured.out)
    assert data == {"scenarios": ["alpha", "beta"]}


def test_main_runs_scenario_and_aggregates(cli_env, monkeypatch):
    root, config_dir, _ = cli_env
    scenario_path = config_dir / "dynamic.yaml"
    scenario_path.write_text("run: {}\n", encoding="utf-8")

    merge_calls: list[Path] = []
    runner_calls: list[tuple[str, Path, Path]] = []
    aggregate_calls: list[Path] = []

    def fake_merge(path: Path) -> dict:
        merge_calls.append(Path(path))
        return {"dynamic": {}}

    def fake_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        runner_calls.append((runner, Path(sc_path), Path(results_root)))
        return Path(results_root) / f"{Path(sc_path).stem}_output"

    def fake_aggregate(root_str: str) -> Path:
        root_path = Path(root_str)
        aggregate_calls.append(root_path)
        return root_path / "aggregate"

    monkeypatch.setattr(main, "_merge_extends", fake_merge)
    monkeypatch.setattr(main, "_execute_runner", fake_execute)
    monkeypatch.setattr(main, "aggregate", fake_aggregate)

    results_root = root / "outputs"
    exit_code = main.main(["--results-root", str(results_root)])

    assert exit_code == 0
    assert merge_calls == [scenario_path]
    assert runner_calls == [("dynamic", scenario_path, results_root.resolve())]
    assert aggregate_calls == [results_root.resolve()]


def test_main_json_summary(cli_env, monkeypatch, capsys):
    root, config_dir, _ = cli_env
    scenario_path = config_dir / "alpha.yaml"
    scenario_path.write_text("run: {}\n", encoding="utf-8")

    def fake_merge(path: Path) -> dict:
        assert path == scenario_path
        return {}

    def fake_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        return results_root / f"{Path(sc_path).stem}_output"

    def fake_aggregate(root_str: str) -> Path:
        return Path(root_str) / "aggregate"

    monkeypatch.setattr(main, "_merge_extends", fake_merge)
    monkeypatch.setattr(main, "_execute_runner", fake_execute)
    monkeypatch.setattr(main, "aggregate", fake_aggregate)

    exit_code = main.main(["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["selected"] == ["alpha"]
    assert payload["aggregate"].endswith("aggregate")
    assert Path(payload["results_root"]).is_absolute()
    assert payload["summary"] == [
        {
            "scenario": "alpha",
            "status": "success",
            "output": str(Path(payload["results_root"]) / "alpha_output"),
            "runner": "scenario",
        }
    ]


def test_main_runner_error_without_stop(cli_env, monkeypatch):
    root, config_dir, logger = cli_env
    scenario_path = config_dir / "failure.yaml"
    scenario_path.write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(main, "_merge_extends", lambda path: {})

    def failing_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        raise RuntimeError("boom")

    aggregate_calls: list[Path] = []

    def fake_aggregate(_root: str) -> Path:
        aggregate_calls.append(Path(_root))
        return Path(_root) / "aggregate"

    monkeypatch.setattr(main, "_execute_runner", failing_execute)
    monkeypatch.setattr(main, "aggregate", fake_aggregate)

    results_root = root / "results"
    exit_code = main.main(["--results-root", str(results_root)])

    assert exit_code == 0  # default is continue-on-error
    assert aggregate_calls == []  # no successes, so no aggregation
    warning_messages = [msg for level, msg, _ in logger.records if level == "warning"]
    assert any("did not complete" in msg for msg in warning_messages)


def test_main_runner_error_with_stop(cli_env, monkeypatch):
    root, config_dir, _ = cli_env
    (config_dir / "failure.yaml").write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(main, "_merge_extends", lambda path: {})

    def raising_execute(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "_execute_runner", raising_execute)
    monkeypatch.setattr(main, "aggregate", lambda root: Path(root) / "aggregate")

    results_root = root / "results"
    exit_code = main.main(["--results-root", str(results_root), "--stop-on-error"])

    assert exit_code == 1


def test_main_merge_failure_respects_stop(cli_env, monkeypatch):
    root, config_dir, logger = cli_env
    (config_dir / "bad.yaml").write_text("run: {}\n", encoding="utf-8")

    def failing_merge(_path: Path) -> dict:
        raise ValueError("broken config")

    monkeypatch.setattr(main, "_merge_extends", failing_merge)
    monkeypatch.setattr(main, "aggregate", lambda root: Path(root) / "aggregate")

    exit_continue = main.main([])
    assert exit_continue == 0

    exit_stop = main.main(["--stop-on-error"])
    assert exit_stop == 1

    exception_messages = [msg for level, msg, _ in logger.records if level == "exception"]
    assert any("Failed to load scenario" in msg for msg in exception_messages)


def test_main_no_matching_filters(cli_env, monkeypatch):
    _, config_dir, _ = cli_env
    (config_dir / "alpha.yaml").write_text("run: {}\n", encoding="utf-8")

    monkeypatch.setattr(main, "aggregate", lambda root: Path(root) / "aggregate")

    exit_code = main.main(["--include", "missing"])
    assert exit_code == 1
