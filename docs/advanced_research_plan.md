# Advanced Research Execution Plan

This plan tracks the two remaining backlog items—AR-01 (Meta-RL) and AR-02 (Offline RL)—and documents the concrete artefacts newly added to the repository. Each section lists objectives, implementation hooks, validation metrics, and follow-up work.

## AR-01 Meta-RL – Context-Aware Adaptation
- **Objective**: learn a regime-aware policy that generalises across synthetic market regimes and demonstrates positive transfer.
- **Implementation**:
  - Synthetic regimes generated via `research/meta_rl/datasets.py` (bull, bear, range configurations).
  - Lightweight meta agent in `research/meta_rl/agent.py` mapping regime context to lookback recommendations.
  - Orchestration script `research/meta_rl/run_meta_rl.py` produces datasets under `meta_rl/` and writes transfer metrics to both `meta_rl/meta_analysis.csv` and `results/meta_analysis.csv`.
- **Usage**:
  ```
  python research/meta_rl/run_meta_rl.py --output-root meta_rl --samples 750 --seed 123
  ```
  Outputs include `train_regimes.csv`, `test_regimes.csv`, the trained regime lookup (`meta_agent.json`), and MAE-based transfer analysis.
- **Validation**:
  - Positive transfer: MAE on held-out dataset lower than naive global average baseline (tracked in `meta_analysis.csv`).
  - Dataset diversity: at least three regimes with distinct volatility/trend profiles.
  - Artefact completeness: datasets + agent weights + summary CSV versioned.
- **Next steps**: integrate regime embeddings into `LeadLagEnv` observations and benchmark the meta agent against PPO baselines on historical data.

## AR-02 Offline RL – Behaviour Cloning Baseline
- **Objective**: build an offline training path that approaches online PPO performance within 10% reward gap.
- **Implementation**:
  - Trajectory capture via `research/offline_rl/log_trajectories.py`, writing `results/offline/offline_dataset.h5` plus manifest/metadata.
  - Behaviour cloning trainer `research/offline_rl/train_offline.py` fits a logistic regression policy, evaluates it inside the environment, and exports `offline_results.{json,csv}` as well as optional comparison against an online PPO `summary.csv`.
  - Dataset governance shared with online runs through `governance/dataset.py` utilities.
- **Usage**:
  ```
  python research/offline_rl/log_trajectories.py --episodes 10 --output results/offline/offline_dataset.h5
  python research/offline_rl/train_offline.py --dataset results/offline/offline_dataset.h5 --online-summary <path-to-online-summary.csv>
  ```
- **Validation**:
  - Offline classification accuracy reported in `offline_results.csv` (target ≥ 0.85 for parity).
  - Reward gap computed in `offline_vs_online.csv` when an online summary is supplied; success criterion ≤ 10%.
  - Dataset manifest + metadata persisted for reproducibility.
- **Next steps**: expand trainer to support alternative algorithms (e.g., CQL), schedule automatic logging from live PPO runs, and integrate evaluation outputs into the reporting dashboards.

## Shared Considerations
- **Governance**: Both pipelines rely on dataset manifests (`data_manifest.json`) so provenance persists with artefacts.
- **Automation**: Add CI smoke targets to run `run_meta_rl.py` and `train_offline.py` with reduced sample counts to guard against regressions.
- **Documentation**: Update the research note once transfer MAE and offline reward-gap metrics meet thresholds; include plots derived from the generated CSVs.
