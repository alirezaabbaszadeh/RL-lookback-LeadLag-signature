from __future__ import annotations

import importlib
import importlib.machinery
import json
import sys
import types
from argparse import Namespace

from leadlag import main as cli_main


def test_main_reports_missing_iisignature(monkeypatch, capsys):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):  # pragma: no cover - shimmed for test determinism
        if name == "iisignature":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(sys, "argv", ["leadlag", "--format", "json", "--status"])

    exit_code = cli_main.main(["--format", "json", "--status"])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False
    assert payload["errors"]
    error = payload["errors"][0]
    assert error["code"] == "missing_dependency"
    assert error["message"] == "iisignature is not installed"
    assert error["details"]["package"] == "iisignature"
    assert "install_commands" in error["details"]


def test_ensure_iisignature_records_version(monkeypatch):
    dummy_module = types.ModuleType("iisignature")
    dummy_module.__version__ = "9.9.9"
    monkeypatch.setitem(sys.modules, "iisignature", dummy_module)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):  # pragma: no cover - shimmed for test determinism
        if name == "iisignature":
            return importlib.machinery.ModuleSpec(name, loader=None)
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    args = Namespace(format="json", json=True)
    assert cli_main.ensure_iisignature(args) is True
    assert getattr(args, "iisignature_version") == "9.9.9"
