# Advanced Research Execution Plan

This plan tracks the current research backlog and documents the concrete artefacts already implemented in the repository. The previous Meta-RL prototype was retired during the 2025 refactor, so only active workstreams appear below.

## AR-01 Offline RL – Behaviour Cloning Baseline
- **Objective**: build an offline training path that approaches online PPO performance within a 10% reward gap.
- **Implementation**:
  - Trajectory capture via `research/offline_rl/log_trajectories.py`, writing `results/offline/offline_dataset.csv` together with manifests/metadata.
  - Behaviour cloning trainer `research/offline_rl/train_offline.py` fits a logistic regression policy, evaluates it inside the environment, and exports `offline_results.{json,csv}` plus optional comparisons against an online PPO `summary.csv`.
  - Dataset governance is shared with online runs through `governance/dataset.py` utilities.
- **Usage**:
  ```
  python research/offline_rl/log_trajectories.py --episodes 10 --output results/offline/offline_dataset.csv
  python research/offline_rl/train_offline.py --dataset results/offline/offline_dataset.csv --online-summary <path-to-online-summary.csv>
  ```
- **Validation**:
  - Offline classification accuracy reported in `offline_results.csv` (target ≥ 0.85 for parity).
  - Reward gap computed in `offline_vs_online.csv` when an online summary is supplied; success criterion ≤ 10%.
  - Dataset manifest + metadata persisted for reproducibility.
- **Next steps**: expand the trainer to support alternative algorithms (e.g., CQL), schedule automatic logging from live PPO runs, and integrate evaluation outputs into the reporting dashboards.

## Deferred Work – Meta-RL Concepts
- **Status**: the earlier Meta-RL experiment suite was removed; no source files remain under `research/meta_rl/`.
- **Open questions**:
  - Revisit the synthetic regime generator once a concrete use-case resurfaces.
  - Decide whether regime embeddings should be produced offline or derived from live PPO trajectories.
- **Exit criteria**: Meta-RL only returns to the roadmap once we have documented requirements, resource estimates, and a hosting module in `research/` to keep artefacts reproducible.

## Shared Considerations
- **Governance**: All research pipelines rely on dataset manifests (`data_manifest.json`) so provenance persists with artefacts.
- **Automation**: Add CI smoke targets to run `train_offline.py` with reduced sample counts to guard against regressions; revisit Meta-RL tasks once code is re-introduced.
- **Documentation**: Update this note whenever offline RL metrics cross parity thresholds or when Meta-RL prototypes land back in the tree; include plots derived from the generated CSVs.
