from __future__ import annotations

import pandas as pd

from leadlag.governance.dataset import build_manifest


def test_build_manifest_preserves_timezone_information():
    naive_index = pd.date_range("2023-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"price": [1.0, 1.5, 1.75]}, index=naive_index)

    manifest = build_manifest(frame)

    assert manifest["index_timezone"] is None

    tz_index = pd.date_range("2023-01-01", periods=3, freq="D", tz="UTC")
    frame_tz = pd.DataFrame({"price": [1.0, 1.5, 1.75]}, index=tz_index)

    manifest_tz = build_manifest(frame_tz)

    assert manifest_tz["index_timezone"] == "UTC"
