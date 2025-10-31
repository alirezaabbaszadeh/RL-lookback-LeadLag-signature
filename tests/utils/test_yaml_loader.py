from __future__ import annotations

import pytest

from leadlag.utils.yaml import load_yaml


def test_load_yaml_required_success(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("foo: bar\nvalue: 3", encoding="utf-8")

    data = load_yaml(path)

    assert data == {"foo": "bar", "value": 3}


def test_load_yaml_missing_optional_returns_default(tmp_path):
    path = tmp_path / "missing.yaml"
    default = {"fallback": True}

    data = load_yaml(path, required=False, default=default)

    assert data == default
    assert data is not default


def test_load_yaml_missing_required_raises(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml(path)


def test_load_yaml_invalid_optional_returns_default(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(": bad: yaml", encoding="utf-8")

    data = load_yaml(path, required=False, default={})

    assert data == {}


def test_load_yaml_empty_returns_default(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    data = load_yaml(path, required=False, default={})

    assert data == {}
