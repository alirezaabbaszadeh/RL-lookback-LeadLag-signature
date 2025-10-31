from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from leadlag import main as cli_main


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, tuple]] = []
        self.warnings: list[tuple[str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.infos.append((message, args))

    def warning(self, message: str, *args, **kwargs) -> None:  # pragma: no cover - exercised in tests
        self.warnings.append((message, kwargs.get("context")))


def _base_args(tmp_path: Path, **overrides) -> Namespace:
    base = dict(
        include=None,
        exclude=None,
        max_scenarios=None,
        runner="auto",
        skip_existing=False,
        stop_on_error=False,
        dry_run=False,
        list=False,
        status=False,
        scenarios=None,
        validate=None,
        log_level="INFO",
        log_path=None,
        results_root=str(tmp_path),
        format="text",
        json=False,
    )
    base.update(overrides)
    args = Namespace(**base)
    setattr(args, "_leadlag_command", "leadlag")
    return args


def test_validate_success(monkeypatch, tmp_path):
    args = _base_args(tmp_path, validate="demo")
    cli = cli_main.LeadLagCLI(args)
    scenario_path = tmp_path / "demo.yaml"

    monkeypatch.setattr(
        cli_main.driver_service,
        "resolve_scenario_reference",
        lambda value: scenario_path,
    )
    monkeypatch.setattr(cli_main, "_merge_extends", lambda path: {"config": True})

    validated: dict[str, object] = {}

    def _validate(config, scenario):
        validated["scenario"] = scenario

    monkeypatch.setattr(cli_main, "_validate_scenario_schema", _validate)

    result = cli.validate()

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.payload["data"]["valid"] is True
    assert result.payload["command"] == "leadlag"
    assert validated["scenario"] == "demo"


def test_validate_failure(monkeypatch, tmp_path):
    args = _base_args(tmp_path, validate="broken")
    cli = cli_main.LeadLagCLI(args)

    def _raise(_value):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        cli_main.driver_service,
        "resolve_scenario_reference",
        _raise,
    )

    result = cli.validate()

    assert result.exit_code == 1
    assert result.emitter == "error"
    assert result.payload["code"] == "scenario_validation_failed"


def test_status_summary(monkeypatch, tmp_path):
    args = _base_args(tmp_path)
    cli = cli_main.LeadLagCLI(args)

    monkeypatch.setattr(
        cli_main.driver_service,
        "collect_status",
        lambda root: ["status"],
    )

    class DummyStatus:
        def __init__(self) -> None:
            self.data = {"runs": 1}
            self.text = "ok"
            self.errors = None
            self.success = True

    monkeypatch.setattr(
        cli_main,
        "render_status_summary",
        lambda root, runs: DummyStatus(),
    )

    result = cli.status()

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.payload["data"] == {"runs": 1}
    assert result.payload["command"] == "leadlag"


def test_list_discovers_scenarios(monkeypatch, tmp_path):
    args = _base_args(tmp_path)
    cli = cli_main.LeadLagCLI(args)

    discovered = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    monkeypatch.setattr(
        cli_main.driver_service,
        "discover_scenarios",
        lambda: discovered,
    )

    result = cli.list()

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.payload["data"] == {"scenarios": ["alpha", "beta"]}
    assert "alpha\nbeta" == result.payload["text"]


def test_execute_dry_run(monkeypatch, tmp_path):
    args = _base_args(tmp_path, dry_run=True)
    cli = cli_main.LeadLagCLI(args)

    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    monkeypatch.setattr(
        cli_main.driver_service,
        "discover_scenarios",
        lambda: scenarios,
    )
    monkeypatch.setattr(
        cli_main.driver_service,
        "filter_scenarios",
        lambda discovered, include, exclude: list(discovered),
    )

    def _prepare(_args):
        prepared_root = tmp_path / "results"
        options = cli_main.driver_service.ExecutionOptions(
            results_root=prepared_root,
            runner_preference=_args.runner,
            skip_existing=_args.skip_existing,
            stop_on_error=_args.stop_on_error,
            dry_run=_args.dry_run,
        )
        return cli_main.driver_service.ExecutionSetup(
            results_root=prepared_root,
            logger=DummyLogger(),
            options=options,
            command="leadlag --dry-run",
        )

    monkeypatch.setattr(cli_main.driver_service, "prepare_execution", _prepare)

    class DummyExecution:
        dry_run = True
        dry_run_entries = ["alpha"]
        summary = []
        errors: list[dict] = []
        aggregate = None
        exit_code = 0
        aborted = False

    monkeypatch.setattr(
        cli_main.driver_service,
        "execute_scenarios",
        lambda selected, options, logger=None: DummyExecution(),
    )

    monkeypatch.setattr(
        cli_main,
        "render_dry_run_summary",
        lambda payload: Namespace(data={"selected": payload.selected}, text="dry"),
    )

    result = cli.execute()

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.payload["message"] == "Dry-run completed."
    assert result.payload["data"] == {"selected": ["alpha", "beta"]}
    assert result.payload["command"] == "leadlag --dry-run"


def test_execute_full_run(monkeypatch, tmp_path):
    args = _base_args(tmp_path)
    cli = cli_main.LeadLagCLI(args)

    scenarios = [tmp_path / "alpha.yaml", tmp_path / "beta.yaml"]
    monkeypatch.setattr(
        cli_main.driver_service,
        "discover_scenarios",
        lambda: scenarios,
    )
    monkeypatch.setattr(
        cli_main.driver_service,
        "filter_scenarios",
        lambda discovered, include, exclude: [discovered[0]],
    )

    def _prepare(_args):
        prepared_root = tmp_path / "results"
        options = cli_main.driver_service.ExecutionOptions(
            results_root=prepared_root,
            runner_preference=_args.runner,
            skip_existing=_args.skip_existing,
            stop_on_error=_args.stop_on_error,
            dry_run=_args.dry_run,
        )
        return cli_main.driver_service.ExecutionSetup(
            results_root=prepared_root,
            logger=DummyLogger(),
            options=options,
            command="leadlag",
        )

    monkeypatch.setattr(cli_main.driver_service, "prepare_execution", _prepare)

    class DummyExecution:
        dry_run = False
        dry_run_entries: list = []
        summary = [
            cli_main.driver_service.ScenarioResult(
                scenario="alpha",
                status="success",
                runner="auto",
                output="done",
            )
        ]
        errors: list[dict] = []
        aggregate = tmp_path / "agg.json"
        exit_code = 0
        aborted = False

    monkeypatch.setattr(
        cli_main.driver_service,
        "execute_scenarios",
        lambda selected, options, logger=None: DummyExecution(),
    )

    class DummyRender:
        def __init__(self) -> None:
            self.data = {
                "selected": ["alpha"],
                "results_root": str(tmp_path / "results"),
                "summary": [
                    {
                        "scenario": "alpha",
                        "status": "success",
                        "runner": "auto",
                        "output": "done",
                    }
                ],
                "aggregate": str(tmp_path / "agg.json"),
                "dry_run": False,
            }
            self.text = "ok"
            self.message = "LeadLag scenarios completed."
            self.artifacts = None
            self.errors = None
            self.success = True

    monkeypatch.setattr(
        cli_main,
        "render_execution_summary",
        lambda root, **kwargs: DummyRender(),
    )

    result = cli.execute()

    assert result.exit_code == 0
    assert result.emitter == "output"
    assert result.payload["message"] == "LeadLag scenarios completed."
    assert result.payload["data"]["selected"] == ["alpha"]
    assert result.payload["success"] is True
    assert result.payload["command"] == "leadlag"
