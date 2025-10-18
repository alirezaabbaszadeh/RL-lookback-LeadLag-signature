from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from training.run_scenario import run_scenario
import hydra_main


def _preset_raw_config_ccf(preset: dict) -> dict:
    import copy
    raw = copy.deepcopy(preset['raw_config'])
    raw.setdefault('analysis', {})
    raw['analysis']['method'] = 'ccf_at_lag'
    return raw


def run_with_limit(
    preset: dict,
    run_name: str,
    seed: int,
    limit_days: Optional[int],
    output_root: Path,
) -> Path:
    overrides = {
        '_raw_config': _preset_raw_config_ccf(preset),
        'run': {'seed': seed, 'run_name': run_name},
    }
    if limit_days is not None:
        overrides['data'] = {'limit_days': int(limit_days)}
    return run_scenario(preset['path'], str(output_root), overrides)


def compare_at_pivot(full_dir: Path, trunc_dir: Path) -> pd.DataFrame:
    full = pd.read_csv(full_dir / 'metrics_timeseries.csv', parse_dates=['date']).set_index('date')
    trunc = pd.read_csv(trunc_dir / 'metrics_timeseries.csv', parse_dates=['date']).set_index('date')
    pivot = trunc.index.max()
    if pivot not in full.index:
        raise RuntimeError('Pivot date not found in full run index')
    cols = [c for c in trunc.columns if c in full.columns]
    a = trunc.loc[pivot, cols]
    b = full.loc[pivot, cols]
    out = pd.DataFrame({'truncated': a, 'full': b})
    out['delta'] = out['full'] - out['truncated']
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description='Walk-forward verification: compare metrics at pivot date')
    ap.add_argument('--scenario', type=str, default='fixed_30')
    ap.add_argument('--seed', type=int, default=13)
    ap.add_argument('--limit_days', type=int, default=120)
    ap.add_argument('--output-root', type=Path, default=Path('results'))
    args = ap.parse_args()

    preset = hydra_main.SCENARIO_PRESETS.get(args.scenario)
    if not preset:
        raise SystemExit(f'Unknown scenario preset: {args.scenario}')

    args.output_root.mkdir(parents=True, exist_ok=True)
    full_dir = run_with_limit(preset, 'wf_full', args.seed, None, args.output_root)
    trunc_dir = run_with_limit(preset, 'wf_trunc', args.seed, args.limit_days, args.output_root)
    df = compare_at_pivot(full_dir, trunc_dir)
    out_dir = Path('docs/audit/phase-2')
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'walk_forward_check.csv'
    df.to_csv(path)
    print(f'Saved walk-forward check to {path}')
    # Print max absolute delta as quick signal
    max_abs = df['delta'].abs().max()
    print('max|delta| =', max_abs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
