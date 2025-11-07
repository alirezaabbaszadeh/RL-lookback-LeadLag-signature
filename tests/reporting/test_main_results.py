import json
from pathlib import Path

import pandas as pd
import pytest

from leadlag.reporting.main_results import aggregate_main_results
from leadlag.reporting.main_results import build_parser
from leadlag.reporting.main_results import main as main_cli


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def test_aggregate_main_results_produces_confidence_intervals(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    shared = {
        "agent": "ppo",
        "policy": "gaussian",
        "universe": "sp500",
        "timeframe": "1h",
        "split_scheme": "walk_forward",
        "reward": "sharpe",
        "features_signature": True,
        "signature_depth": 2,
        "features_leadlag": True,
        "time_channel": True,
        "cost_fee_bps": 0.1,
        "slippage_bps": 0.0,
        "experiment_id": "run-0",
        "seed": 1,
        "window_index": 0,
    }

    run1 = results_root / "run_1"
    run1.mkdir()
    _write_metrics(
        run1 / "metrics.csv",
        [
            {
                **shared,
                "Sharpe": 1.0,
                "Sortino": 1.2,
                "MaxDD": -0.2,
                "PnL": 100.0,
                "Turnover": 0.5,
                "Exposure": 0.8,
                "EnvSteps": 200,
            }
        ],
    )

    run2 = results_root / "run_2"
    run2.mkdir()
    _write_metrics(
        run2 / "metrics.csv",
        [
            {
                **shared,
                "experiment_id": "run-1",
                "seed": 2,
                "Sharpe": 2.0,
                "Sortino": 1.5,
                "MaxDD": -0.1,
                "PnL": 130.0,
                "Turnover": 0.6,
                "Exposure": 0.9,
                "EnvSteps": 220,
            }
        ],
    )

    result = aggregate_main_results(results_root, out_dir)

    main_df = result.main_results
    assert not main_df.empty
    row = main_df.iloc[0]
    assert row["n_runs"] == 2
    assert row["n_seeds"] == 2
    assert row["n_windows"] == 1
    assert row["winsor_alpha"] == pytest.approx(0.0)
    assert row["Sharpe"] == pytest.approx(1.5)
    assert row["Sharpe_std"] == pytest.approx(0.7071067, rel=1e-5)
    assert row["Sharpe_lo"] == pytest.approx(0.52, rel=1e-6)
    assert row["Sharpe_hi"] == pytest.approx(2.48, rel=1e-6)
    assert row["PnL"] == pytest.approx(115.0)

    ablations_df = result.ablations
    assert not ablations_df.empty
    main_csv = out_dir / "main_results.csv"
    ablations_csv = out_dir / "ablations.csv"
    main_tex = out_dir / "main_results.tex"
    ablations_tex = out_dir / "ablations.tex"

    assert main_csv.exists()
    assert ablations_csv.exists()
    assert main_tex.exists()
    assert ablations_tex.exists()

    fixtures = Path(__file__).parent / "fixtures"
    expected_main = (fixtures / "main_results.tex").read_text(encoding="utf-8").strip()
    expected_ablations = (fixtures / "ablations.tex").read_text(encoding="utf-8").strip()

    assert main_tex.read_text(encoding="utf-8").strip() == expected_main
    assert ablations_tex.read_text(encoding="utf-8").strip() == expected_ablations
    assert (out_dir / "all_metrics_raw.csv").exists()
    assert len(result.all_metrics) == 2


def test_aggregate_main_results_without_metrics_raises(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    results_root.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(RuntimeError):
        aggregate_main_results(results_root, out_dir)


def test_main_results_cli_json_envelope(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_root = tmp_path / "results"
    out_dir = tmp_path / "out"
    results_root.mkdir()
    out_dir.mkdir()

    shared = {
        "agent": "ppo",
        "policy": "gaussian",
        "universe": "sp500",
        "timeframe": "1h",
        "split_scheme": "walk_forward",
        "reward": "sharpe",
        "features_signature": True,
        "signature_depth": 2,
        "features_leadlag": True,
        "time_channel": True,
        "cost_fee_bps": 0.1,
        "slippage_bps": 0.0,
        "experiment_id": "run-0",
        "seed": 1,
        "window_index": 0,
    }

    run_dir = results_root / "run_1"
    run_dir.mkdir()
    _write_metrics(
        run_dir / "metrics.csv",
        [
            {
                **shared,
                "Sharpe": 1.0,
                "PnL": 10.0,
                "EnvSteps": 200,
            }
        ],
    )

    exit_code = main_cli(
        [
            "--results",
            str(results_root),
            "--out",
            str(out_dir),
            "--format",
            "json",
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["success"] is True
    data = payload["data"]
    assert data["winsor_alpha"] == pytest.approx(0.0)
    tables = data["tables"]
    assert tables["main_results"]["rows"] == 1
    assert tables["main_results"]["latex_path"].endswith("main_results.tex")
    assert payload["artifacts"]["main_results"].endswith("main_results.csv")
    assert payload["artifacts"]["main_results_tex"].endswith("main_results.tex")


def test_cli_help_mentions_latex() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "LaTeX" in help_text
