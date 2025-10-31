from __future__ import annotations

from argparse import Namespace

from leadlag import main as cli_main


class DummyLogger:
    def __init__(self) -> None:
        self.infos: list[tuple[str, tuple]] = []
        self.warnings: list[tuple[str, dict | None]] = []

    def info(self, message: str, *args, **kwargs) -> None:
        self.infos.append((message, args))

    def warning(self, message: str, *args, **kwargs) -> None:
        self.warnings.append((message, kwargs.get("context")))


def _base_args(**overrides) -> Namespace:
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
    )
    base.update(overrides)
    return Namespace(**base)


def test_handle_validate_success(monkeypatch, tmp_path):
    args = _base_args(validate="demo")
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)
    scenario_path = tmp_path / "demo.yaml"

    monkeypatch.setattr(
        cli_main.driver_service,
        "resolve_scenario_reference",
        lambda value: scenario_path,
    )
    monkeypatch.setattr(cli_main, "_merge_extends", lambda path: {"config": True})

    validated = {}

    def _validate(config, scenario):
        validated["scenario"] = scenario

    monkeypatch.setattr(cli_main, "_validate_scenario_schema", _validate)

    captured: dict[str, object] = {}

    def _emit(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_formatted_output", _emit)

    exit_code = cli_main._handle_validate(args, context)

    assert exit_code == 0
    assert validated["scenario"] == "demo"
    assert captured["data"]["valid"] is True
    assert captured["command"] == "leadlag"


def test_handle_validate_failure(monkeypatch, tmp_path):
    args = _base_args(validate="broken")
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)

    def _raise(_value):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        cli_main.driver_service,
        "resolve_scenario_reference",
        _raise,
    )

    captured: dict[str, object] = {}

    def _emit_error(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_error", _emit_error)

    exit_code = cli_main._handle_validate(args, context)

    assert exit_code == 1
    assert captured["code"] == "scenario_validation_failed"


def test_handle_status(monkeypatch, tmp_path):
    args = _base_args(status=True)
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)

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

    captured: dict[str, object] = {}

    def _emit(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_formatted_output", _emit)

    exit_code = cli_main._handle_status(args, context)

    assert exit_code == 0
    assert captured["data"] == {"runs": 1}
    assert captured["command"] == "leadlag"


def test_handle_list(monkeypatch, tmp_path):
    args = _base_args(list=True)
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)
    context.discovered_scenarios = [
        tmp_path / "alpha.yaml",
        tmp_path / "beta.yaml",
    ]

    captured: dict[str, object] = {}

    def _emit(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_formatted_output", _emit)

    exit_code = cli_main._handle_list(args, context)

    assert exit_code == 0
    assert captured["data"] == {"scenarios": ["alpha", "beta"]}
    assert captured["text"] == "alpha\nbeta"


def test_handle_execute_dry_run(monkeypatch, tmp_path):
    args = _base_args(dry_run=True)
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)
    context.discovered_scenarios = [
        tmp_path / "alpha.yaml",
        tmp_path / "beta.yaml",
    ]

    monkeypatch.setattr(
        cli_main.driver_service,
        "filter_scenarios",
        lambda scenarios, include, exclude: list(scenarios),
    )

    class DummyExecution:
        dry_run = True
        dry_run_entries = ["alpha"]
        summary = []
        errors = []
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
        "configure_driver_logger",
        lambda *args, **kwargs: DummyLogger(),
    )

    monkeypatch.setattr(
        cli_main,
        "render_dry_run_summary",
        lambda payload: Namespace(data={"selected": payload.selected}, text="dry"),
    )

    captured: dict[str, object] = {}

    def _emit(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_formatted_output", _emit)

    exit_code = cli_main._handle_execute(args, context)

    assert exit_code == 0
    assert captured["data"] == {"selected": ["alpha", "beta"]}
    assert captured["message"] == "Dry-run completed."


def test_handle_execute_full_run(monkeypatch, tmp_path):
    args = _base_args()
    context = cli_main._CLIContext(command="leadlag", results_root=tmp_path)
    context.discovered_scenarios = [
        tmp_path / "alpha.yaml",
        tmp_path / "beta.yaml",
    ]

    monkeypatch.setattr(
        cli_main.driver_service,
        "filter_scenarios",
        lambda scenarios, include, exclude: list(scenarios)[:1],
    )

    class DummyExecution:
        dry_run = False
        dry_run_entries = []
        summary = [
            cli_main.driver_service.ScenarioResult(
                scenario="alpha",
                status="success",
                runner="auto",
                output="done",
            )
        ]
        errors = []
        aggregate = tmp_path / "agg.json"
        exit_code = 0
        aborted = False

    logger = DummyLogger()

    monkeypatch.setattr(
        cli_main.driver_service,
        "execute_scenarios",
        lambda selected, options, logger=None: DummyExecution(),
    )
    monkeypatch.setattr(
        cli_main,
        "configure_driver_logger",
        lambda *args, **kwargs: logger,
    )

    captured: dict[str, object] = {}

    def _emit(*_args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "emit_formatted_output", _emit)

    exit_code = cli_main._handle_execute(args, context)

    assert exit_code == 0
    assert captured["message"] == "LeadLag scenarios completed."
    assert captured["success"] is True
    assert logger.infos

