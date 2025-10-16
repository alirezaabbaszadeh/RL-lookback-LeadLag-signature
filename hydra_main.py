from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import hydra
    from omegaconf import OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    OmegaConf = None  # type: ignore

from training.runner_multiseed import run_multiseed
from training.run_scenario import run_scenario
from training.run_dynamic_baselines import run_dynamic


SCENARIO_DIR = Path("configs/scenario")

BASE_CONFIG = {
    'run': {
        'output_root': 'results',
        'run_name': 'auto',
        'seed': 42,
    },
    'data': {
        'price_csv': 'raw_data/daily_price.csv',
        'universe_csv': None,
    },
    'analysis': {
        'method': 'signature',
        'lookback': 30,
        'update_freq': 1,
        'use_parallel': False,
        'num_cpus': 4,
        'show_progress': False,
        'Scaling_Method': 'mean-centering',
        'ccf_at_lag': {
            'lag': 1,
            'correlation_method': 'pearson',
            'quantiles': 4,
        },
        'signature': {
            'sig_method': 'levy',
            'correlation_method': 'pearson',
            'quantiles': 4,
        },
    },
    'metrics': {
        'compute': [
            'mean_abs_matrix',
            'max_abs_matrix',
            'row_sum_range',
            'row_sum_std',
            'stability_matrix_corr',
            'stability_rowsum_corr',
        ],
        'plots': ['signal_strength', 'stability'],
    },
}


def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


SCENARIO_PRESETS = {
    'fixed_30': {
        'name': 'fixed_30',
        'path': 'configs/scenarios/fixed_30.yaml',
        'runner': 'scenario',
        'multi_seed': {'seeds': [42, 52, 62]},
        'raw_config': _merge(BASE_CONFIG, {'analysis': {'lookback': 30}, 'run': {'run_name': 'fixed_30'}}),
    },
    'fixed_90': {
        'name': 'fixed_90',
        'path': 'configs/scenarios/fixed_90.yaml',
        'runner': 'scenario',
        'raw_config': _merge(BASE_CONFIG, {'analysis': {'lookback': 90}, 'run': {'run_name': 'fixed_90'}}),
    },
    'dynamic_adaptive': {
        'name': 'dynamic_adaptive',
        'path': 'configs/scenarios/dynamic_adaptive.yaml',
        'runner': 'dynamic',
        'raw_config': _merge(
            BASE_CONFIG,
            {
                'analysis': {'lookback': 30},
                'run': {'run_name': 'dynamic_adaptive'},
                'dynamic': {
                    'min_lookback': 10,
                    'max_lookback': 120,
                    'step': 5,
                },
            },
        ),
    },
    'rl_ppo': {
        'name': 'rl_ppo',
        'path': 'configs/scenarios/rl_ppo.yaml',
        'runner': 'rl',
        'raw_config': _merge(
            BASE_CONFIG,
            {
                'analysis': {
                    'lookback': 60,
                    'method': 'signature',
                    'signature': {'sig_method': 'levy'},
                },
                'run': {'run_name': 'rl_ppo'},
                'rl': {
                    'min_lookback': 10,
                    'max_lookback': 120,
                    'discrete_actions': True,
                    'action_mode': 'hybrid',
                    'relative_step': 5,
                    'episode_length': 252,
                    'random_start': True,
                    'ema_alpha': 0.2,
                    'policy': 'attention',
                    'policy_kwargs': {
                        'features_extractor_kwargs': {'features_dim': 96, 'n_heads': 4}
                    },
                    'total_timesteps': 2000,
                    'n_steps': 256,
                    'batch_size': 128,
                    'learning_rate': 3e-4,
                    'gamma': 0.99,
                    'ent_coef': 0.0,
                    'eval_freq': 0,
                    'verbose': False,
                    'reward_weights': {'alpha': 1.0, 'beta': 0.3, 'gamma': 0.08},
                    'penalty_same': 0.05,
                    'penalty_step': 10,
                },
            },
        ),
    },
    'fast_smoke': {
        'name': 'fast_smoke',
        'path': 'configs/scenarios/fixed_30.yaml',
        'runner': 'scenario',
        'multi_seed': {'enabled': False, 'seeds': [1]},
        'raw_config': _merge(
            BASE_CONFIG,
            {
                'analysis': {'lookback': 10, 'update_freq': 9999},
                'run': {'run_name': 'fast_smoke', 'seed': 1},
                'metrics': {
                    'compute': ['mean_abs_matrix'],
                    'plots': [],
                },
            },
        ),
    },
}


def _load_scenario_cfg(entry):
    if isinstance(entry, dict):
        return entry
    if isinstance(entry, str):
        if HYDRA_AVAILABLE:
            path = SCENARIO_DIR / f"{entry}.yaml"
            cfg = OmegaConf.load(path)
            return OmegaConf.to_container(cfg, resolve=True)
        preset = SCENARIO_PRESETS.get(entry)
        if preset:
            return json.loads(json.dumps(preset))
        raise ValueError(f"Unknown scenario preset '{entry}'. Install hydra-core for YAML support.")
    raise TypeError("Scenario entry must be string or dict")


def _run_single(scenario_cfg, output_root):
    runner = scenario_cfg.get('runner', 'scenario')
    path = scenario_cfg['path']
    overrides = {}
    if 'raw_config' in scenario_cfg:
        overrides['_raw_config'] = scenario_cfg['raw_config']
    if runner == 'dynamic':
        return run_dynamic(path, output_root, overrides)
    if runner == 'rl':
        return run_rl(path, output_root, overrides)
    return run_scenario(path, output_root, overrides)


def _run_workflow(cfg_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    output_root = cfg_dict.get('output_root', 'results')
    multi_seed_cfg = cfg_dict.get('multi_seed', {})
    enabled = multi_seed_cfg.get('enabled', False)
    seeds = multi_seed_cfg.get('seeds', [])

    scenarios = cfg_dict.get('scenarios') or [cfg_dict['scenario']]
    results: List[Dict[str, Any]] = []

    for entry in scenarios:
        scenario_cfg = _load_scenario_cfg(entry)
        scenario_seeds = scenario_cfg.get('multi_seed', {}).get('seeds', seeds)
        scenario_enabled = scenario_cfg.get('multi_seed', {}).get('enabled', enabled)

        if scenario_enabled and scenario_seeds:
            agg_dir = run_multiseed(scenario_cfg, scenario_seeds, output_root)
            results.append({'scenario': scenario_cfg.get('name'), 'path': str(agg_dir)})
            print(f"Aggregated results saved to: {agg_dir}")
        else:
            out_dir = _run_single(scenario_cfg, output_root)
            results.append({'scenario': scenario_cfg.get('name'), 'path': str(out_dir)})
            print(f"Single run saved to: {out_dir}")

    summary_path = Path(output_root) / 'hydra_runs.json'
    summary_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if HYDRA_AVAILABLE:
    @hydra.main(config_path="configs", config_name="config", version_base=None)
    def hydra_entry(cfg):  # pragma: no cover
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        _run_workflow(cfg_dict)


def _parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run lead-lag scenarios")
    parser.add_argument('--scenario', type=str, default='fixed_30', help='Scenario name in configs/scenario/')
    parser.add_argument('--scenarios', type=str, nargs='*', help='Multiple scenario names')
    parser.add_argument('--output_root', type=str, default='results', help='Output directory root')
    parser.add_argument('--multi_seed_enabled', action='store_true', help='Enable multi-seed runs')
    parser.add_argument('--seeds', type=int, nargs='*', default=[], help='Seeds for multi-seed runs')
    return parser.parse_args()


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        hydra_entry()  # type: ignore
    else:
        args = _parse_args()
        scenario_entries = args.scenarios or [args.scenario]
        cfg_dict = {
            'scenario': scenario_entries[0],
            'scenarios': scenario_entries,
            'output_root': args.output_root,
            'multi_seed': {
                'enabled': args.multi_seed_enabled,
                'seeds': args.seeds,
            },
        }
        _run_workflow(cfg_dict)
