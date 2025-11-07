from __future__ import annotations

import json
from pathlib import Path

from leadlag.utils import print_manifest


def _write_manifest(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "unit-s00-w00",
                "seed": 0,
                "window_index": 0,
                "env_steps_reported": 10,
                "env_steps_actual": 10,
                "feature_time": {
                    "column": "t_feat",
                    "rows": 8,
                    "checked_rows": 8,
                    "min_lag_ns": 1_000,
                    "max_lag_ns": 2_000,
                    "tz": "UTC",
                    "freq_hint": "D",
                },
                "config_sources": ["hydra:file"],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_print_manifest_text_format(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_manifest(run_dir)

    exit_code = print_manifest.main(["--root", str(run_dir)])
    assert exit_code == 0

    stdout = capsys.readouterr().out.strip()
    assert stdout.startswith("run_id=unit-s00-w00")
    assert "min_lag_ns=1000" in stdout
    assert "config_sources=hydra:file" in stdout


def test_print_manifest_json_format(tmp_path, capsys):
    run_dir = tmp_path / "run"
    manifest_path = _write_manifest(run_dir)

    exit_code = print_manifest.main([
        "--root",
        str(manifest_path),
        "--format",
        "json",
    ])
    assert exit_code == 0

    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["success"] is True
    assert payload["data"]["manifest_path"] == str(manifest_path)
    assert payload["data"]["feature_min_lag_ns"] == 1000
    assert payload["data"]["config_sources"] == ["hydra:file"]
