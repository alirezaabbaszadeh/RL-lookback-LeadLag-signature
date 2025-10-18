from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.run_scenario import run_scenario
import hydra_main

def _preset_raw_config_ccf(preset: dict) -> dict:
    import copy
    raw = copy.deepcopy(preset['raw_config'])
    # Force method to ccf_at_lag to avoid optional signature dependency during probes
    raw.setdefault('analysis', {})
    raw['analysis']['method'] = 'ccf_at_lag'
    return raw


def run_placebo_probe(preset: dict, output_root: str = "results", seed: int = 7, limit_days: Optional[int] = None) -> Path:
    overrides = {
        '_raw_config': _preset_raw_config_ccf(preset),
        'run': {'seed': seed, 'run_name': 'leakage_placebo'},
        'data': {'placebo_shuffle': True},
    }
    if limit_days is not None:
        overrides['data']['limit_days'] = int(limit_days)
    return run_scenario(preset['path'], output_root, overrides)


def run_control(preset: dict, output_root: str = "results", seed: int = 7, limit_days: Optional[int] = None) -> Path:
    overrides = {
        '_raw_config': _preset_raw_config_ccf(preset),
        'run': {'seed': seed, 'run_name': 'leakage_control'},
    }
    if limit_days is not None:
        overrides['data'] = {'limit_days': int(limit_days)}
    return run_scenario(preset['path'], output_root, overrides)


def compare_stability(control_dir: Path, placebo_dir: Path) -> pd.DataFrame:
    ctrl = pd.read_csv(control_dir / 'metrics_timeseries.csv', parse_dates=['date']).set_index('date')
    plc = pd.read_csv(placebo_dir / 'metrics_timeseries.csv', parse_dates=['date']).set_index('date')
    def summarize(df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'stability_matrix_corr_mean': float(df['stability_matrix_corr'].mean()),
            'stability_rowsum_corr_mean': float(df['stability_rowsum_corr'].mean()),
            'mean_abs_matrix_mean': float(df['mean_abs_matrix'].mean()),
            'row_sum_std_mean': float(df['row_sum_std'].mean()),
        })
    a = summarize(ctrl)
    b = summarize(plc)
    out = pd.DataFrame({'control': a, 'placebo': b})
    out['delta'] = out['placebo'] - out['control']
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Leakage probes: control vs placebo shuffle')
    ap.add_argument('--scenario', type=str, default='fixed_30', help='Scenario preset name (see hydra_main presets)')
    ap.add_argument('--out', type=str, default='results')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--limit_days', type=int, default=180)
    args = ap.parse_args()

    preset = hydra_main.SCENARIO_PRESETS.get(args.scenario)
    if not preset:
        raise SystemExit(f"Unknown scenario preset: {args.scenario}")
    ctrl_dir = run_control(preset, args.out, args.seed, args.limit_days)
    plc_dir = run_placebo_probe(preset, args.out, args.seed, args.limit_days)
    df = compare_stability(ctrl_dir, plc_dir)
    print(df)

    report_dir = Path('docs/audit/phase-2')
    report_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / 'leakage_probe_summary.csv')
    print(f'Saved leakage probe summary to {report_dir / "leakage_probe_summary.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
