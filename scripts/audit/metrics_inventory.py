from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def count_logged_metrics(run_dir: Path) -> Dict[str, int]:
    summary_p = run_dir / 'summary.csv'
    metrics = 0
    by_metric: Dict[str, int] = {}
    if summary_p.exists():
        try:
            df = pd.read_csv(summary_p)
            # Count numeric columns per metric (excluding the 'metric' label column)
            num_cols = [c for c in df.columns if c != 'metric']
            for _, row in df.iterrows():
                m = str(row.get('metric', 'metric'))
                by_metric[m] = len(num_cols)
                metrics += len(num_cols)
        except Exception:
            pass
    return {'total': metrics, **{f'm_{k}': v for k, v in by_metric.items()}}


def main() -> int:
    root = Path('results')
    rows: List[Dict[str, int]] = []
    for run in root.glob('*_*'):
        if not run.is_dir() or run.name.endswith('_aggregate'):
            continue
        rec = {'run_dir': run.name}
        rec.update(count_logged_metrics(run))
        rows.append(rec)
    if rows:
        df = pd.DataFrame(rows).sort_values('run_dir')
        out_dir = Path('docs/audit/phase-6')
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / 'metrics_inventory.csv', index=False)
        print(f'Wrote inventory for {len(rows)} runs to {out_dir / "metrics_inventory.csv"}')
    else:
        print('No runs found under results/.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

