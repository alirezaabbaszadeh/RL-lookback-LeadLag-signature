from __future__ import annotations

import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from leadlag.driver import execution

scipy_stub = types.ModuleType("scipy")
stats_stub = types.ModuleType("scipy.stats")

class _DummyTTestResult:
    def __init__(self, statistic: float = 0.0, pvalue: float = 1.0):
        self.statistic = statistic
        self.pvalue = pvalue

def _ttest_ind(*args, **kwargs):
    return _DummyTTestResult()

stats_stub.ttest_ind = _ttest_ind
scipy_stub.stats = stats_stub
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.stats", stats_stub)

sklearn_stub = types.ModuleType("sklearn")
linear_model_stub = types.ModuleType("sklearn.linear_model")
class _DummyLogisticRegression:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return [0] * (len(X) if hasattr(X, '__len__') else 0)

linear_model_stub.LogisticRegression = _DummyLogisticRegression
metrics_stub = types.ModuleType("sklearn.metrics")
def _accuracy_score(*args, **kwargs):
    return 1.0

metrics_stub.accuracy_score = _accuracy_score
model_selection_stub = types.ModuleType("sklearn.model_selection")
def _train_test_split(X, y, *args, **kwargs):
    return X, X, y, y

model_selection_stub.train_test_split = _train_test_split
decomposition_stub = types.ModuleType("sklearn.decomposition")
class _DummyPCA:
    def __init__(self, *args, **kwargs):
        pass

    def fit_transform(self, X):
        return X

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

decomposition_stub.PCA = _DummyPCA
preprocessing_stub = types.ModuleType("sklearn.preprocessing")
class _DummyStandardScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

preprocessing_stub.StandardScaler = _DummyStandardScaler
sklearn_stub.linear_model = linear_model_stub
sklearn_stub.metrics = metrics_stub
sklearn_stub.model_selection = model_selection_stub
sklearn_stub.decomposition = decomposition_stub
sklearn_stub.preprocessing = preprocessing_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.linear_model", linear_model_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)
sys.modules.setdefault("sklearn.decomposition", decomposition_stub)
sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)

try:  # pragma: no cover - dependency guard for optional pandas wheels
    from leadlag import main
except ValueError as exc:
    pytest.skip(f"Dependency import failed: {exc}", allow_module_level=True)

from leadlag.cli.dependencies import build_driver_service


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

    base_service = build_driver_service()

    def fake_prepare_execution(args):
        results_root = Path(args.results_root).expanduser().resolve()
        results_root.mkdir(parents=True, exist_ok=True)
        options = base_service.ExecutionOptions(
            results_root=results_root,
            runner_preference=args.runner,
            skip_existing=args.skip_existing,
            stop_on_error=args.stop_on_error,
            dry_run=args.dry_run,
        )
        command = getattr(args, "_leadlag_command", "leadlag")
        return base_service.ExecutionSetup(
            results_root=results_root,
            logger=dummy_logger,
            options=options,
            command=command,
        )

    service = replace(base_service, prepare_execution=fake_prepare_execution)

    return tmp_path, config_dir, dummy_logger, service


def _write_basic_scenario(path: Path, *, run_name: str | None = None) -> None:
    run_name = run_name or path.stem
    path.write_text(
        "\n".join(
            [
                "run:",
                f"  run_name: {run_name}",
                "data:",
                "  price_csv: data.csv",
                "analysis:",
                "  method: signature",
                "  lookback: 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_match_filters_basic():
    service = build_driver_service()
    assert service.matches_filters("alpha", include=["alp"], exclude=None)
    assert not service.matches_filters("beta", include=["alp"], exclude=None)
    assert not service.matches_filters("alpha", include=None, exclude=["alp"])
    assert service.matches_filters("alpha", include=None, exclude=["zzz"])


def test_discover_scenarios_sorted(cli_env):
    _, config_dir, _, service = cli_env
    _write_basic_scenario(config_dir / "b.yaml", run_name="b")
    _write_basic_scenario(config_dir / "a.yaml", run_name="a")
    (config_dir / "ignored.txt").write_text("ignore", encoding="utf-8")

    scenarios = service.discover_scenarios()
    assert [path.name for path in scenarios] == ["a.yaml", "b.yaml"]


def test_main_dry_run_filters(cli_env):
    _, config_dir, logger, service = cli_env
    _write_basic_scenario(config_dir / "alpha.yaml", run_name="alpha")
    _write_basic_scenario(config_dir / "beta.yaml", run_name="beta")

    def fail_aggregate(_root: str) -> Path:  # pragma: no cover - should never run
        raise AssertionError("aggregate must not run during dry-run")

    service = replace(service, aggregate=fail_aggregate)

    exit_code = main.main(
        ["--dry-run", "--include", "alpha"], build_driver_service=lambda: service
    )
    assert exit_code == 0

    dry_run_messages = [
        msg for level, msg, _ in logger.records if level == "info" and "[dry-run]" in msg
    ]
    assert dry_run_messages == [f"[dry-run] {Path('configs/scenarios/alpha.yaml')}"]


def test_main_list_outputs_names(cli_env, capsys):
    _, config_dir, _, _service = cli_env
    _write_basic_scenario(config_dir / "beta.yaml", run_name="beta")
    _write_basic_scenario(config_dir / "alpha.yaml", run_name="alpha")

    exit_code = main.main(["--list"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.strip().splitlines() == ["alpha", "beta"]


def test_main_list_json(cli_env, capsys):
    _, config_dir, _, service = cli_env
    _write_basic_scenario(config_dir / "beta.yaml", run_name="beta")
    _write_basic_scenario(config_dir / "alpha.yaml", run_name="alpha")

    exit_code = main.main(["--list", "--json"], build_driver_service=lambda: service)
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    assert payload["data"]["scenarios"] == ["alpha", "beta"]


def test_main_list_format_json(cli_env, capsys):
    _, config_dir, _, service = cli_env
    _write_basic_scenario(config_dir / "beta.yaml", run_name="beta")
    _write_basic_scenario(config_dir / "alpha.yaml", run_name="alpha")

    exit_code = main.main(
        ["--list", "--format", "json"], build_driver_service=lambda: service
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    assert payload["data"]["scenarios"] == ["alpha", "beta"]


def test_main_runs_scenario_and_aggregates(cli_env, monkeypatch):
    root, config_dir, _, service = cli_env
    scenario_path = config_dir / "dynamic.yaml"
    _write_basic_scenario(scenario_path, run_name="dynamic")

    merge_calls: list[Path] = []
    runner_calls: list[tuple[str, Path, Path]] = []
    aggregate_calls: list[Path] = []

    def fake_merge(path: Path) -> dict:
        merge_calls.append(Path(path))
        return {
            "run": {"run_name": "dynamic"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
            "dynamic": {},
        }

    def fake_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        runner_calls.append((runner, Path(sc_path), Path(results_root)))
        return Path(results_root) / f"{Path(sc_path).stem}_output"

    def fake_aggregate(root_str: str) -> Path:
        root_path = Path(root_str)
        aggregate_calls.append(root_path)
        return root_path / "aggregate"

    monkeypatch.setattr(main, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_execute_runner", fake_execute)
    monkeypatch.setattr(execution, "aggregate", fake_aggregate)

    results_root = root / "outputs"
    exit_code = main.main(
        ["--results-root", str(results_root)],
        build_driver_service=lambda: service,
    )

    assert exit_code == 0
    assert merge_calls == [scenario_path]
    assert runner_calls == [("dynamic", scenario_path, results_root.resolve())]
    assert aggregate_calls == [results_root.resolve()]


def test_main_json_summary(cli_env, monkeypatch, capsys):
    root, config_dir, _, service = cli_env
    scenario_path = config_dir / "alpha.yaml"
    _write_basic_scenario(scenario_path, run_name="alpha")

    def fake_merge(path: Path) -> dict:
        assert path == scenario_path
        return {
            "run": {"run_name": "alpha"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
        }

    def fake_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        return results_root / f"{Path(sc_path).stem}_output"

    def fake_aggregate(root_str: str) -> Path:
        return Path(root_str) / "aggregate"

    monkeypatch.setattr(main, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_merge_extends", fake_merge)
    monkeypatch.setattr(execution, "_execute_runner", fake_execute)
    monkeypatch.setattr(execution, "aggregate", fake_aggregate)

    exit_code = main.main(["--json"], build_driver_service=lambda: service)
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["selected"] == ["alpha"]
    assert data["aggregate"].endswith("aggregate")
    assert Path(data["results_root"]).is_absolute()
    assert data["summary"] == [
        {
            "scenario": "alpha",
            "status": "success",
            "output": str(Path(data["results_root"]) / "alpha_output"),
            "runner": "scenario",
        }
    ]


def test_main_runner_error_without_stop(cli_env, monkeypatch):
    root, config_dir, logger, service = cli_env
    scenario_path = config_dir / "failure.yaml"
    _write_basic_scenario(scenario_path, run_name="failure")

    def failing_execute(runner: str, sc_path: Path, results_root: Path) -> Path:
        raise RuntimeError("boom")

    aggregate_calls: list[Path] = []

    def fake_aggregate(_root: str) -> Path:
        aggregate_calls.append(Path(_root))
        return Path(_root) / "aggregate"

    monkeypatch.setattr(
        main,
        "_merge_extends",
        lambda path: {
            "run": {"run_name": "failure"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
        },
    )
    monkeypatch.setattr(
        execution,
        "_merge_extends",
        lambda path: {
            "run": {"run_name": "failure"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
        },
    )
    monkeypatch.setattr(execution, "_execute_runner", failing_execute)
    monkeypatch.setattr(execution, "aggregate", fake_aggregate)

    results_root = root / "results"
    exit_code = main.main(
        ["--results-root", str(results_root)],
        build_driver_service=lambda: service,
    )

    assert exit_code == 0  # default is continue-on-error
    assert aggregate_calls == []  # no successes, so no aggregation
    warning_messages = [msg for level, msg, _ in logger.records if level == "warning"]
    assert any("did not complete" in msg for msg in warning_messages)


def test_main_runner_error_with_stop(cli_env, monkeypatch):
    root, config_dir, _, service = cli_env
    _write_basic_scenario(config_dir / "failure.yaml", run_name="failure")

    def raising_execute(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        main,
        "_merge_extends",
        lambda path: {
            "run": {"run_name": "failure"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
        },
    )
    monkeypatch.setattr(
        execution,
        "_merge_extends",
        lambda path: {
            "run": {"run_name": "failure"},
            "data": {"price_csv": "data.csv"},
            "analysis": {"method": "signature", "lookback": 30},
        },
    )
    monkeypatch.setattr(execution, "_execute_runner", raising_execute)
    monkeypatch.setattr(execution, "aggregate", lambda root: Path(root) / "aggregate")

    results_root = root / "results"
    exit_code = main.main(
        ["--results-root", str(results_root), "--stop-on-error"],
        build_driver_service=lambda: service,
    )

    assert exit_code == 1


def test_main_merge_failure_respects_stop(cli_env, monkeypatch):
    root, config_dir, logger, service = cli_env
    _write_basic_scenario(config_dir / "bad.yaml", run_name="bad")

    def failing_merge(_path: Path) -> dict:
        raise ValueError("broken config")

    monkeypatch.setattr(main, "_merge_extends", failing_merge)
    monkeypatch.setattr(execution, "_merge_extends", failing_merge)
    monkeypatch.setattr(execution, "aggregate", lambda root: Path(root) / "aggregate")

    exit_continue = main.main([], build_driver_service=lambda: service)
    assert exit_continue == 0

    exit_stop = main.main(["--stop-on-error"], build_driver_service=lambda: service)
    assert exit_stop == 1

    exception_messages = [msg for level, msg, _ in logger.records if level == "exception"]
    assert any("Failed to load scenario" in msg for msg in exception_messages)


def test_main_no_matching_filters(cli_env):
    _, config_dir, _, service = cli_env
    _write_basic_scenario(config_dir / "alpha.yaml", run_name="alpha")

    service = replace(service, aggregate=lambda root: Path(root) / "aggregate")

    exit_code = main.main(["--include", "missing"], build_driver_service=lambda: service)
    assert exit_code == 1


def test_main_explicit_scenarios(cli_env, monkeypatch, capsys):
    root, config_dir, _, service = cli_env
    alpha = config_dir / "alpha.yaml"
    beta = config_dir / "beta.yaml"
    _write_basic_scenario(alpha, run_name="alpha")
    _write_basic_scenario(beta, run_name="beta")

    monkeypatch.setattr(execution, "_execute_runner", lambda *args, **kwargs: Path("unused"))
    monkeypatch.setattr(execution, "aggregate", lambda root: Path(root) / "aggregate")

    exit_code = main.main(
        [
            "--scenarios",
            str(alpha),
            str(beta),
            "--max-scenarios",
            "1",
            "--dry-run",
            "--json",
        ],
        build_driver_service=lambda: service,
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["data"]["selected"] == ["alpha"]


def test_main_validate_packaged(capsys):
    exit_code = main.main(["--validate", "fixed_30", "--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["valid"] is True
    assert data["scenario"] == "fixed_30"


def test_main_validate_path_failure(tmp_path, capsys):
    scenario = tmp_path / "broken.yaml"
    scenario.write_text(
        "\n".join(
            [
                "run:",
                "  run_name: broken",
                "analysis:",
                "  method: signature",
                "  lookback: 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = main.main(["--validate", str(scenario), "--format", "json"])
    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["success"] is False
    data = payload.get("data", {})
    assert data.get("valid") is False
    assert payload["errors"][0]["code"] == "scenario_validation_failed"
    assert "missing" in (data.get("error") or "")


def test_main_status_json(tmp_path, capsys):
    results_root = tmp_path / "results"
    success = results_root / "alpha_20240101_000000"
    success.mkdir(parents=True)
    metadata = {
        "config_path": "configs/scenarios/alpha.yaml",
        "scenario": "alpha",
    }
    (success / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (success / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    incomplete = results_root / "beta_20240101_000010"
    incomplete.mkdir()
    (incomplete / "run_metadata.json").write_text(
        json.dumps({"config_path": "configs/scenarios/beta.yaml"}), encoding="utf-8"
    )

    exit_code = main.main(["--status", "--results-root", str(results_root), "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["results_root"] == str(results_root.resolve())
    statuses = {entry["scenario"]: entry["status"] for entry in data["runs"] if "scenario" in entry}
    assert statuses.get("alpha") == "success"
    assert statuses.get("beta") == "incomplete"


def test_main_skip_existing(cli_env, monkeypatch, capsys):
    root, config_dir, _, service = cli_env
    scenario_path = config_dir / "alpha.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "run:",
                "  run_name: alpha",
                "data:",
                "  price_csv: data.csv",
                "analysis:",
                "  method: signature",
                "  lookback: 30",
            ]
        ),
        encoding="utf-8",
    )

    results_root = root / "results"
    existing = results_root / "alpha_20240101_000000"
    existing.mkdir(parents=True)
    (existing / "summary.csv").write_text("metric,mean\n", encoding="utf-8")

    executed: list[str] = []

    def fail_execute(*_args, **_kwargs):  # pragma: no cover - should not run
        executed.append("called")
        raise AssertionError("runner should be skipped")

    monkeypatch.setattr(execution, "_execute_runner", fail_execute)
    exit_code = main.main(
        [
            "--results-root",
            str(results_root),
            "--include",
            "alpha",
            "--skip-existing",
            "--json",
        ],
        build_driver_service=lambda: service,
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert not executed
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    statuses = {entry["scenario"]: entry["status"] for entry in data["summary"]}
    assert statuses.get("alpha") == "skipped"
