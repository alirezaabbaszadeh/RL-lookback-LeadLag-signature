from __future__ import annotations

from pathlib import Path

import pytest

try:  # pragma: no cover - environment guard for optional pandas wheels
    import pandas  # noqa: F401
except Exception as exc:  # pragma: no cover
    pytest.skip(f"pandas import failed: {exc}", allow_module_level=True)

from leadlag.research.offline_rl.log_trajectories import build_arg_parser as log_parser
from leadlag.research.offline_rl.train_offline import build_arg_parser as train_parser


def test_train_offline_default_scenario_resolves():
    parser = train_parser()
    args = parser.parse_args([])
    assert isinstance(args.scenario, Path)
    assert args.scenario.exists()


def test_log_trajectories_default_scenario_resolves():
    parser = log_parser()
    args = parser.parse_args([])
    assert isinstance(args.scenario, Path)
    assert args.scenario.exists()
