from __future__ import annotations

from argparse import Namespace

from leadlag.cli.commands import CommandContext, ListCommand, ScenarioManager


def test_list_command_returns_shared_response_when_no_scenarios(tmp_path):
    list_command = ListCommand(scenarios=ScenarioManager(lambda: []))
    context = CommandContext(
        args=Namespace(),
        results_root=tmp_path,
        command="leadlag list",
    )

    response = list_command(context)

    assert response.exit_code == 1
    assert response.code == "no_scenarios_available"
    assert response.message.startswith("No scenarios found")
    assert response.details == {"results_root": str(tmp_path)}
    assert response.command == "leadlag list"
    assert response.results_root == tmp_path
