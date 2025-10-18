from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


def collect_pairs(agg_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in agg_dir.glob('significance_*_pairs.csv'):
        try:
            df = pd.read_csv(p)
            df['source'] = p.name
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    agg_dir = Path('results/aggregate')
    out_dir = Path('docs/audit/phase-3')
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(agg_dir)
    if pairs.empty:
        print('No pairs CSVs found in results/aggregate')
        return 0

    # Keep the most interesting results: sorted by q-value then absolute effect size
    if 'q_value' in pairs.columns:
        pairs['abs_d'] = pairs.get('cohens_d', pd.Series([float('nan')] * len(pairs))).abs()
        pairs = pairs.sort_values(['q_value', 'abs_d'], ascending=[True, False])
    sig = pairs
    if 'q_value' in pairs.columns:
        sig = pairs[pairs['q_value'] < 0.05]
    sig.to_csv(out_dir / 'significant_pairs.csv', index=False)
    pairs.to_csv(out_dir / 'all_pairs.csv', index=False)
    print(f'Wrote {len(sig)} significant pairs to {out_dir / "significant_pairs.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

