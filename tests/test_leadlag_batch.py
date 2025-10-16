import numpy as np
import pytest

try:
    import pandas as pd
except ValueError:
    pytest.skip("Pandas binary incompatible with current numpy", allow_module_level=True)

from models.LeadLag_main import LeadLagAnalyzer, LeadLagConfig
from models.leadlag.matrix_builder import build_matrix


def _make_price_df() -> "pd.DataFrame":
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    data = {
        "A": np.linspace(100, 108, num=8),
        "B": np.linspace(200, 210, num=8),
        "C": np.linspace(50, 55, num=8),
    }
    return pd.DataFrame(data, index=dates)


def _make_config() -> LeadLagConfig:
    return LeadLagConfig(
        method="ccf_at_lag",
        correlation_method="pearson",
        lookback=3,
        update_freq=1,
        use_parallel=False,
        num_cpus=1,
        quantiles=4,
        show_progress=False,
        Scaling_Method="mean-centering",
        lag=1,
    )


def test_compute_matrices_batch_matches_single_window():
    price_df = _make_price_df()
    analyzer = LeadLagAnalyzer(_make_config())
    windows = [
        (price_df.index[3], price_df.index[5]),
        (price_df.index[4], price_df.index[7]),
    ]

    result = analyzer.compute_matrices_batch(price_df, windows)

    expected_keys = {
        (pd.Timestamp(price_df.index[3]), pd.Timestamp(price_df.index[5])),
        (pd.Timestamp(price_df.index[4]), pd.Timestamp(price_df.index[7])),
    }
    assert set(result.keys()) == expected_keys

    for key, matrix in result.items():
        log_returns = analyzer.window_processor.get_log_returns(price_df, key[0], key[1])
        baseline = build_matrix(log_returns, analyzer._compute_lead_lag_measure_optimized)
        pd.testing.assert_frame_equal(matrix, baseline)


def test_compute_matrices_batch_empty_windows_returns_empty_dict():
    price_df = _make_price_df()
    analyzer = LeadLagAnalyzer(_make_config())

    assert analyzer.compute_matrices_batch(price_df, []) == {}
