# Advanced Research Execution Plan

This document outlines concrete steps for the remaining research backlog items
AR-01 (Meta-RL) and AR-02 (Offline RL) so that scope, resourcing, and success
criteria are explicit ahead of implementation.

## AR-01 Meta-RL

- **Objective**: deliver a context-aware policy that adapts to market regime
  changes and demonstrates positive transfer across at least two synthetic
  regimes.
- **Data requirements**:
  1. Generate two synthetic datasets with controlled regime switches
     (persistent bull/bear plus mixed drift/vol regime).
  2. Augment with one historical slice using regime labels inferred from the
     signature features.
- **Implementation steps**:
  1. Extend `training/policy_factory.py` with a context encoder interface
     (GRU or attention) that ingests regime embeddings.
  2. Create `training/policies/meta_context_policy.py` implementing the
     encoder and adaptation logic.
  3. Add environment hooks to inject regime context into observations
     (toggle via config `rl.meta_context.enabled`).
  4. Update Hydra configs: `configs/scenarios/meta_rl.yaml` with policy,
     reward, and logging defaults.
  5. Instrument multi-seed runs through `training/runner_multiseed.py` with
     at least three seeds per regime dataset.
- **Validation metrics**:
  - Transfer gain = target regime reward mean - baseline reward mean > 0.
  - Stability of adaptation measured via rolling Sharpe over regime windows.
  - Episode completion rate >= 98% after adaptation.
- **Artifacts**:
  - `meta_rl/` package with policy implementation and helper modules.
  - Result bundle `results/manual/meta_rl_aggregate/`.
  - Visuals: regime conditioned reward curves, attention/encoder diagnostics.
- **Risks and mitigations**:
  - _Instability_: apply gradient clipping and curriculum warm-up.
  - _Data scarcity_: fall back to bootstrapped synthetic blends if historical
    regime labelling is noisy.
  - _Compute budget_: run smoke preset (short horizon) before full backtests.
- **Milestones**:
  1. Prototype policy with synthetic regimes (ETA 3 days).
  2. Integrate with historical slice and rerun benchmarks (ETA +2 days).
  3. Draft research note summarising meta transfer outcome (ETA +1 day).

## AR-02 Offline RL

- **Objective**: build a reproducible offline training pipeline that matches
  online PPO performance within 10% on the evaluation reward.
- **Data requirements**:
  1. Export trajectory buffers from existing EO-03 logging (ensure policy,
     reward, observation metadata present).
  2. Store consolidated dataset at `results/offline/offline_dataset.h5` with
     schema: observations, actions, rewards, dones, info.
- **Implementation steps**:
  1. Add `scripts/export_offline_dataset.py` to collate logged trajectories
     into the consolidated HDF5 format.
  2. Implement `training/offline_runner.py` supporting CQL and BC algorithms
     (leveraging stable-baselines3 or custom losses).
  3. Create Hydra config `configs/scenarios/offline_rl.yaml` with dataset
     pointers and algorithm knobs.
  4. Extend evaluation tooling to compare offline vs online runs
     (`reporting/compare_scenarios.py` to accept offline tags).
  5. Schedule nightly smoke run to validate dataset freshness (add CI target).
- **Validation metrics**:
  - Reward gap <= 10% vs online PPO baseline.
  - Behavior cloning pre-training accuracy >= 85% on held-out dataset split.
  - Dataset coverage: at least 500 episodes spanning three action regimes.
- **Artifacts**:
  - `offline_dataset.h5` with metadata manifest (`dataset_meta.json`).
  - Offline training logs in `results/manual/offline_rl_*`.
  - Comparison tables in `reports/offline_vs_online.md`.
- **Risks and mitigations**:
  - _Dataset bias_: include scenario diversity and randomised start seeds.
  - _Overfitting_: enforce validation split and early stopping hooks.
  - _Tooling drift_: pin dataset schema version in `.gitattributes`.
- **Milestones**:
  1. Complete dataset export and schema validation (ETA 2 days).
  2. Train baseline BC and CQL agents (ETA +3 days).
  3. Publish offline vs online comparison report (ETA +1 day).

