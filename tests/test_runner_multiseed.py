import pytest

try:
    import numpy as np
    import pandas as pd
except ValueError:
    pytest.skip("NumPy/Pandas binary mismatch", allow_module_level=True)

from training.runner_multiseed import _bootstrap_ci, _aggregate_summaries


def test_bootstrap_ci_returns_bounds():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    low, high = _bootstrap_ci(values, resamples=200)
    assert low <= values.mean() <= high


def test_aggregate_summaries_with_single_metric():
    df = pd.DataFrame(
        {
            'scenario': ['s1', 's1', 's1'],
            'metric': ['m1', 'm1', 'm1'],
            'value_mean': [1.0, 2.0, 3.0],
        }
    )
    result = _aggregate_summaries([df])
    assert not result.empty
    row = result.iloc[0]
    assert row['scenario'] == 's1'
    assert row['metric'] == 'm1'
    assert 'value_mean' in row
