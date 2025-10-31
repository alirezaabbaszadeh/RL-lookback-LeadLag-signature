import json
import sys
from dataclasses import replace

import pytest

from leadlag.pipelines import run_full_suite


class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, *args, **kwargs):
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs):
        self.messages.append(("warning", args, kwargs))


@pytest.fixture
def base_config(tmp_path):
    return run_full_suite.parse_args(
        [
            "--output-root",
            str(tmp_path / "suite"),
            "--format",
            "json",
            "--dry-run",
        ]
    )


def _make_context(config, tmp_path, dependency_status=None):
    output_root = run_full_suite.ensure_path(config.output_root)
    logs_dir = run_full_suite.ensure_path(output_root / "logs")
    paths = run_full_suite.FullSuitePaths(
        output_root=output_root,
        logs_dir=logs_dir,
        baseline_root=run_full_suite.ensure_path(output_root / "core"),
        robustness_root=run_full_suite.ensure_path(output_root / "robustness"),
        ablation_root=output_root / "ablations",
        meta_root=output_root / "meta_rl",
        offline_root=output_root / "offline",
    )
    return run_full_suite.FullSuiteContext(
        config=config,
        logger=DummyLogger(),
        dependency_status=dependency_status or {},
        paths=paths,
        python_executable=sys.executable,
        run_log={},
        start_time=0.0,
    )


@pytest.fixture
def stage_mocks(monkeypatch):
    monkeypatch.setattr(
        run_full_suite,
        "inspect_scenario",
        lambda name: run_full_suite.ScenarioRequirements(
            name=name,
            requires_signature=False,
            requires_sb3=False,
            requires_sb3_contrib=False,
        ),
    )
    monkeypatch.setattr(run_full_suite, "check_optional_dependencies", lambda *args, **kwargs: True)
    monkeypatch.setattr(run_full_suite.hydra_main, "_load_scenario_cfg", lambda name: {})
    monkeypatch.setattr(run_full_suite.hydra_main, "validate_scenario_cfg", lambda cfg: None)


def test_parse_args_returns_dataclass(base_config):
    assert isinstance(base_config, run_full_suite.FullSuiteCLIOptions)
    assert base_config.output_root.name == "suite"
    assert base_config.output_format == "json"
    assert base_config.dry_run is True


def test_dataset_audit_stage_runs_command(tmp_path, base_config, monkeypatch):
    config = replace(
        base_config,
        data_path=tmp_path / "data.csv",
        skip_audit=False,
        fail_on_quality=True,
    )
    context = _make_context(config, tmp_path)
    commands = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(run_full_suite, "run_command", fake_run)
    run_full_suite._run_dataset_audit(context)
    assert len(commands) == 1
    assert commands[0][0] == sys.executable
    assert "--exit-on-fail" in commands[0]


def test_dataset_audit_stage_skips_when_disabled(tmp_path, base_config, monkeypatch):
    config = replace(base_config, skip_audit=True)
    context = _make_context(config, tmp_path)
    called = False

    def fake_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(run_full_suite, "run_command", fake_run)
    run_full_suite._run_dataset_audit(context)
    assert called is False


def test_prepare_baselines_updates_run_log(tmp_path, base_config, stage_mocks):
    config = replace(base_config, baseline_scenarios=("a", "b"))
    context = _make_context(config, tmp_path)
    validated = run_full_suite._prepare_baselines(context)
    assert validated == ["a", "b"]
    assert context.run_log["baseline_scenarios_requested"] == ["a", "b"]
    assert context.run_log["validated_baselines"] == ["a", "b"]


def test_run_baselines_invokes_commands(tmp_path, base_config, stage_mocks, monkeypatch):
    config = replace(base_config, baseline_scenarios=("alpha",), baseline_single_seed=True)
    context = _make_context(config, tmp_path)
    commands = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(run_full_suite, "run_command", fake_run)
    validated = run_full_suite._prepare_baselines(context)
    run_full_suite._run_baselines(context, validated)
    assert commands
    assert "--multi_seed_enabled" not in commands[0]


def test_ablation_stage_passes_scenarios(tmp_path, base_config, stage_mocks, monkeypatch):
    config = replace(base_config, ablation_scenarios=("x", "y"), skip_ablation=False)
    context = _make_context(config, tmp_path)
    commands = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(run_full_suite, "run_command", fake_run)
    validated = run_full_suite._run_ablation(context)
    assert validated == ["x", "y"]
    assert any("--scenarios" in cmd for cmd in commands)


def test_reporting_stage_respects_skip_flag(tmp_path, base_config, monkeypatch):
    config = replace(base_config, skip_report=True)
    context = _make_context(config, tmp_path)
    commands = []

    def fake_run(cmd, *_args, **_kwargs):
        commands.append(cmd)

    monkeypatch.setattr(run_full_suite, "run_command", fake_run)
    run_full_suite._run_reporting(context)
    assert len(commands) == 1  # compare_scenarios only


def test_full_suite_json_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_full_suite, "run_command", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_suite, "dependency_preflight", lambda skip, logger: {})
    monkeypatch.setattr(run_full_suite, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_full_suite, "get_logger", lambda *args, **kwargs: DummyLogger())
    monkeypatch.setattr(run_full_suite.hydra_main, "_load_scenario_cfg", lambda name: {})
    monkeypatch.setattr(run_full_suite.hydra_main, "validate_scenario_cfg", lambda cfg: None)
    monkeypatch.setattr(
        run_full_suite,
        "inspect_scenario",
        lambda name: run_full_suite.ScenarioRequirements(
            name=name,
            requires_signature=False,
            requires_sb3=False,
            requires_sb3_contrib=False,
        ),
    )

    exit_code = run_full_suite.main(
        [
            "--output-root",
            str(tmp_path / "suite"),
            "--skip-baseline",
            "--skip-ablation",
            "--skip-report",
            "--skip-meta-offline",
            "--skip-audit",
            "--skip-schema-check",
            "--format",
            "json",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["output_root"].endswith("suite")
    assert data["dry_run"] is True
