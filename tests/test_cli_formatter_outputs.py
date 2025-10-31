from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest


class _DummyPlotModule:
    def figure(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def xlabel(self, *args, **kwargs):
        return None

    def ylabel(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def tight_layout(self, *args, **kwargs):
        return None

    def savefig(self, *args, **kwargs):
        return None

    def close(self, *args, **kwargs):
        return None


matplotlib_stub = types.ModuleType("matplotlib")
matplotlib_stub.pyplot = _DummyPlotModule()
sys.modules.setdefault("matplotlib", matplotlib_stub)
sys.modules.setdefault("matplotlib.pyplot", matplotlib_stub.pyplot)

gym_stub = types.ModuleType("gymnasium")
class _DummyEnv:  # pragma: no cover - lightweight shim
    pass

class _DummyDiscrete:  # pragma: no cover - lightweight shim
    def __init__(self, *args, **kwargs):
        self.n = args[0] if args else None

class _DummyBox:  # pragma: no cover - lightweight shim
    def __init__(self, *args, **kwargs):
        if "shape" in kwargs:
            self.shape = kwargs["shape"]
        elif args:
            first = args[0]
            self.shape = getattr(first, "shape", None)
        else:
            self.shape = None

class _DummyDict:  # pragma: no cover - lightweight shim
    def __init__(self, *args, **kwargs):
        self.spaces = kwargs

gym_stub.Env = _DummyEnv
gym_stub.spaces = types.SimpleNamespace(
    Box=_DummyBox,
    Discrete=_DummyDiscrete,
    MultiDiscrete=lambda *a, **k: None,
    Dict=_DummyDict,
)
sys.modules.setdefault("gymnasium", gym_stub)

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
sklearn_stub.linear_model = linear_model_stub
sklearn_stub.metrics = metrics_stub
sklearn_stub.model_selection = model_selection_stub
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.linear_model", linear_model_stub)
sys.modules.setdefault("sklearn.metrics", metrics_stub)
sys.modules.setdefault("sklearn.model_selection", model_selection_stub)

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
sklearn_stub.decomposition = decomposition_stub
sys.modules.setdefault("sklearn.decomposition", decomposition_stub)

preprocessing_stub = types.ModuleType("sklearn.preprocessing")
class _DummyStandardScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X

preprocessing_stub.StandardScaler = _DummyStandardScaler
sklearn_stub.preprocessing = preprocessing_stub
sys.modules.setdefault("sklearn.preprocessing", preprocessing_stub)

from leadlag.cli import formatters as cli_formatters
from leadlag.pipelines import run_ablation
from leadlag.reporting import (
    compare_scenarios,
    generate_report,
    plot_balance_history,
    status_summary,
)
from leadlag.research.offline_rl import log_trajectories, train_offline


class DummyLogger:
    def info(self, *args, **kwargs):  # pragma: no cover - shim used for tests
        pass

    def warning(self, *args, **kwargs):  # pragma: no cover - shim used for tests
        pass


@pytest.fixture
def dummy_logger():
    return DummyLogger()


def test_emit_formatted_output_envelope_keys(capsys):
    args = argparse.Namespace(format="json", json=True)
    cli_formatters.emit_formatted_output(
        args,
        success=True,
        data={"alpha": 1},
        command="leadlag --status",
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    assert payload["command"] == "leadlag --status"
    assert payload["format"] == "json"
    assert payload["data"] == {"alpha": 1}
    assert "args" in payload and isinstance(payload["args"], dict)
    assert payload["errors"] == []


def test_status_summary_json(tmp_path, monkeypatch, capsys, dummy_logger):
    roadmap = tmp_path / "roadmap.pseudo"
    roadmap.write_text(
        "OPEN_ITEMS [ITEM {MODULE: \"alpha\", ISSUE: \"missing docs\", NEXT_STEP: \"write\"}]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(status_summary, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(status_summary, "get_logger", lambda *args, **kwargs: dummy_logger)

    exit_code = status_summary.main([
        "--roadmap",
        str(roadmap),
        "--format",
        "json",
        "--log-path",
        str(tmp_path / "status.log"),
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data == [{"module": "alpha", "issue": "missing docs", "next_step": "write"}]


def test_run_ablation_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    monkeypatch.setattr(run_ablation, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_ablation, "get_logger", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setattr(
        run_ablation,
        "load_scenario",
        lambda name: run_ablation.ScenarioInfo(
            name=name,
            runner="scenario",
            requires_sb3=False,
            requires_sb3_contrib=False,
            requires_signature=False,
        ),
    )
    monkeypatch.setattr(
        run_ablation,
        "ensure_dependencies",
        lambda info, skip, logger: (True, None),
    )

    exit_code = run_ablation.main([
        "--output-root",
        str(tmp_path),
        "--scenarios",
        "alpha",
        "beta",
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert data["executed"] == ["alpha", "beta"]
    assert data["skipped"] == []
    assert data["output_root"] == str(tmp_path.resolve())
    assert "comparison_output" not in data


def test_compare_scenarios_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    results_root = tmp_path / "results"
    results_root.mkdir()
    (results_root / "alpha_aggregate").mkdir()
    (results_root / "beta_aggregate").mkdir()

    monkeypatch.setattr(compare_scenarios, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(compare_scenarios, "get_logger", lambda *args, **kwargs: dummy_logger)

    exit_code = compare_scenarios.main([
        "--results_root",
        str(results_root),
        "--out",
        str(tmp_path / "evaluation"),
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert set(data["aggregates"]) == {
        str((results_root / "alpha_aggregate").resolve()),
        str((results_root / "beta_aggregate").resolve()),
    }
    assert payload.get("artifacts") in ({}, None)


def test_plot_balance_history_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    class DummySeries:
        index = range(3)
        values = [1.0, 1.2, 1.1]

    run_info = plot_balance_history.RunInfo(
        run_dir=tmp_path / "results" / "alpha",
        scenario="alpha",
        method="signature",
        lookback_label="L=30",
        seed_label="seed=42",
        label="alpha | L=30 | seed=42",
        equity=DummySeries(),
    )
    monkeypatch.setattr(plot_balance_history, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(plot_balance_history, "get_logger", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setattr(plot_balance_history, "collect_run_infos", lambda *a, **k: [run_info])

    exit_code = plot_balance_history.main([
        "--results-root",
        str(tmp_path / "results"),
        "--out",
        str(tmp_path / "plots"),
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert len(data["runs"]) == 1
    assert data["runs"][0]["label"] == "alpha | L=30 | seed=42"


def test_generate_report_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    results_root = tmp_path / "results"
    (results_root / "alpha_aggregate").mkdir(parents=True)
    monkeypatch.setattr(generate_report, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(generate_report, "get_logger", lambda *args, **kwargs: dummy_logger)

    exit_code = generate_report.main([
        "--results-root",
        str(results_root),
        "--output-dir",
        str(tmp_path / "reports"),
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert data["aggregates"] == [str((results_root / "alpha_aggregate").resolve())]


def test_log_trajectories_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    monkeypatch.setattr(log_trajectories, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(log_trajectories, "get_logger", lambda *args, **kwargs: dummy_logger)
    monkeypatch.setattr(log_trajectories, "_merge_extends", lambda scenario: {"run": {}})
    monkeypatch.setattr(log_trajectories, "_set_seed", lambda seed: None)
    monkeypatch.setattr(
        log_trajectories,
        "_read_prices",
        lambda cfg: (object(), Path("prices.csv")),
    )
    monkeypatch.setattr(log_trajectories, "build_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(log_trajectories, "run_quality_checks", lambda prices: {})
    monkeypatch.setattr(log_trajectories, "record_manifest", lambda manifest, parent, name: parent / name)

    exit_code = log_trajectories.main([
        "--scenario",
        str(tmp_path / "scenario.yaml"),
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert data["output"] == "results/offline/offline_dataset.csv"


def test_train_offline_dry_run_json(tmp_path, monkeypatch, capsys, dummy_logger):
    monkeypatch.setattr(train_offline, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(train_offline, "get_logger", lambda *args, **kwargs: dummy_logger)

    exit_code = train_offline.main([
        "--dataset",
        str(tmp_path / "dataset.csv"),
        "--scenario",
        str(tmp_path / "scenario.yaml"),
        "--output-root",
        str(tmp_path / "offline"),
        "--dry-run",
        "--format",
        "json",
    ])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["dry_run"] is True
    assert data["dataset"] == str(tmp_path / "dataset.csv")
