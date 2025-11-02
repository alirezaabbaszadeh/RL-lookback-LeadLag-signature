from __future__ import annotations

import json

import pandas as pd
import numpy as np

from leadlag.training.run_support import prepare_run_environment


def test_prepare_run_environment_creates_expected_artifacts(tmp_path):
    cfg_path = tmp_path / "scenario.yaml"
    cfg_path.write_text("run: {}\n", encoding="utf-8")

    out_root = tmp_path / "outputs"
    cfg = {
        "run": {
            "run_name": "unit",
            "seed": 123,
            "output_root": str(out_root),
        },
        "data": {"price_csv": ""},
    }

    frame = pd.DataFrame(
        {
            "AssetA": [1.0, 1.5, 2.0],
            "AssetB": [2.0, 1.0, 0.5],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )

    def loader(_: dict):
        return frame.copy(), None

    prep = prepare_run_environment(
        cfg,
        cfg_path=cfg_path,
        module="test",
        logger_name="unit-test",
        read_prices_fn=loader,
        run_name="unit",
        profile_label="load",
        extra_metadata={"extra_field": "value"},
    )

    prep.logger.info("helper smoke test log")

    assert prep.out_dir.is_dir()
    assert prep.out_dir.parent == out_root
    assert prep.manifest_path.exists()
    assert (prep.out_dir / "config_merged.yaml").is_file()
    assert (prep.out_dir / "profiles" / "load.json").is_file()
    assert prep.prices.equals(frame)
    assert prep.resolved_price_path is None

    metadata = json.loads((prep.out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["extra_field"] == "value"
    assert metadata["data_manifest"] == str(prep.manifest_path)
    assert metadata["run_manifest"] == str(prep.run_manifest_path)
    assert "environment" in metadata

    # Seeded RNG should be deterministic
    assert prep.seed == 123
    assert prep.run_name == "unit"
    assert prep.timestamp
    assert prep.logger is not None
    assert prep.out_dir.name.startswith("unit_")

    run_manifest = json.loads((prep.run_manifest_path).read_text(encoding="utf-8"))
    assert run_manifest["run"]["seed"] == 123
    assert run_manifest["determinism"]["seed"] == 123
    assert "environment" in run_manifest

    # The NumPy RNG seeded by the helper should yield a deterministic value
    assert np.random.randint(0, 1000) == 510
