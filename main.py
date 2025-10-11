from __future__ import annotations

import sys
from pathlib import Path

from training.run_scenario import run_scenario, _merge_extends
from evaluation.aggregate import aggregate
try:
    import yaml
except Exception:
    yaml = None
from training.run_dynamic_baselines import run_dynamic
from training.run_rl import run_rl


def discover_scenarios() -> list[Path]:
    base = Path('configs/scenarios')
    return sorted(base.glob('*.yaml'))


def main():
    scenarios = discover_scenarios()
    if not scenarios:
        print("No scenarios found in configs/scenarios/*.yaml")
        sys.exit(1)

    results_root = 'results'
    for sc in scenarios:
        print(f"Running scenario: {sc}")
        # dispatch: if scenario defines a 'dynamic' block -> dynamic baseline runner
        try:
            cfg = _merge_extends(sc)
        except Exception:
            cfg = {}
        if 'dynamic' in cfg:
            out_dir = run_dynamic(str(sc), results_root)
        elif 'rl' in cfg:
            out_dir = run_rl(str(sc), results_root)
        else:
            out_dir = run_scenario(str(sc), results_root)
        print(f" -> saved: {out_dir}")

    aggr = aggregate(results_root)
    print(f"Aggregated comparison at: {aggr}")


if __name__ == '__main__':
    main()
