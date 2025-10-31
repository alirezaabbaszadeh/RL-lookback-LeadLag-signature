from __future__ import annotations

from pathlib import Path

import pytest

from leadlag.cli import responses


@pytest.mark.parametrize(
    "command",
    [None, "leadlag execute"],
)
def test_no_scenarios_available_structure(command):
    results_root = Path("/tmp/results")

    response = responses.no_scenarios_available(
        command=command,
        results_root=results_root,
    )

    assert response.exit_code == 1
    assert response.code == "no_scenarios_available"
    assert response.command == command
    assert response.results_root == results_root
    assert response.details == {"results_root": str(results_root)}
    assert response.message.startswith("No scenarios found")


def test_invalid_scenarios_structure():
    command = "leadlag execute"
    results_root = Path("/tmp/invalid")

    response = responses.invalid_scenarios(
        errors=("missing", "typo"),
        requested=["foo"],
        command=command,
        results_root=results_root,
    )

    assert response.exit_code == 1
    assert response.code == "invalid_scenarios"
    assert response.command == command
    assert response.results_root == results_root
    assert response.details == {
        "errors": ["missing", "typo"],
        "requested": ["foo"],
        "results_root": str(results_root),
    }
    assert response.message == "One or more scenarios not found"


@pytest.mark.parametrize(
    "include, exclude",
    [
        (None, None),
        (["pattern"], ("skip",)),
    ],
)
def test_no_matches_structure(include, exclude):
    command = "leadlag execute"
    results_root = Path("/tmp/no-match")

    response = responses.no_matches(
        include=include,
        exclude=exclude,
        command=command,
        results_root=results_root,
    )

    assert response.exit_code == 1
    assert response.code == "no_scenarios_matched"
    assert response.command == command
    assert response.results_root == results_root
    assert response.details == {
        "include": list(include or []),
        "exclude": list(exclude or []),
        "results_root": str(results_root),
    }
    assert response.message == "No scenarios match the provided filters."
