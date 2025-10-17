import sys
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from training.policy_factory import SB3_CONTRIB_AVAILABLE, make_algorithm_spec
from training.policies.attention_policy import AttentionPolicy


def test_make_algorithm_spec_mlp_defaults_to_ppo():
    spec = make_algorithm_spec({'policy': 'mlp'})
    assert spec.algo_cls.__name__ == 'PPO'
    assert spec.policy == 'MlpPolicy'


@pytest.mark.skipif(not SB3_CONTRIB_AVAILABLE, reason="sb3-contrib not installed")
def test_make_algorithm_spec_recurrent_uses_contrib():
    spec = make_algorithm_spec({'policy': 'ppo_lstm'})
    assert spec.algo_cls.__name__ == 'RecurrentPPO'


def test_make_algorithm_spec_attention_policy():
    spec = make_algorithm_spec({'policy': 'attention', 'policy_kwargs': {'features_extractor_kwargs': {'features_dim': 64}}})
    assert spec.algo_cls.__name__ == 'PPO'
    assert spec.policy is AttentionPolicy
    assert spec.policy_kwargs['features_extractor_kwargs']['features_dim'] == 64
