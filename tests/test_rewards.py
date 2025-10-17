import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.rewards import RewardContext, compute_components, load_reward_template


def _metrics():
    return {
        'mean_abs': 0.6,
        'row_range': 0.4,
        'stability_matrix': 0.2,
        'stability_rows': 0.1,
        'delta_mean': 0.05,
    }


def _extras():
    return {
        'regime': {'returns_mean': 0.02, 'returns_vol': 0.01, 'cross_sectional_vol': 0.015},
        'signature': {},
    }


def _context(lookback=30, prev=25):
    return RewardContext(lookback=lookback, prev_lookback=prev, min_lookback=10, max_lookback=50)


def test_compute_components_contains_expected_keys():
    components = compute_components(_metrics(), _extras(), _context(), {})
    assert set(components.keys()) == {'S', 'C', 'E', 'Sharpe', 'LookbackPenalty'}
    assert components['S'] > 0
    assert math.isfinite(components['Sharpe'])


def test_reward_template_from_dict_applies_weights():
    template = load_reward_template({
        'components': {'S': 1.0, 'Sharpe': 0.5},
        'params': {'lookback': {'width': 1.0}},
    })
    reward, comps = template.compute(_metrics(), _extras(), _context())
    expected = comps['S'] + 0.5 * comps['Sharpe']
    assert pytest.approx(reward) == expected


def test_reward_template_loads_from_file(tmp_path: Path):
    data = {
        'components': {'S': 1.0},
        'params': {},
    }
    path = tmp_path / 'custom.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    template = load_reward_template(str(path))
    reward, comps = template.compute(_metrics(), _extras(), _context())
    assert reward == pytest.approx(comps['S'])
