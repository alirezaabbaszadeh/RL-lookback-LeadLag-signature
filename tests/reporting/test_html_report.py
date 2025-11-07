from __future__ import annotations

from pathlib import Path

import pandas as pd

from leadlag.reporting.html_report import build


CSV_NAMES = [
    "main_results.csv",
    "ablations.csv",
    "hac_sharpe_ci.csv",
    "psr_dsr_pvalues.csv",
    "spa_table.csv",
    "mcs_table.csv",
]


def test_build_creates_html_report(tmp_path: Path) -> None:
    for name in CSV_NAMES:
        frame = pd.DataFrame([{"column": name, "value": 1}])
        frame.to_csv(tmp_path / name, index=False)

    for image in ("forest.png", "heatmap.png", "pnl.png"):
        (tmp_path / image).write_bytes(b"PNG")

    report_path = build(tmp_path, title="Sample Report")
    assert report_path.exists()
    content = report_path.read_text()
    assert "Sample Report" in content
    assert "<h2>main_results.csv</h2>" in content
    assert "forest.png" in content
