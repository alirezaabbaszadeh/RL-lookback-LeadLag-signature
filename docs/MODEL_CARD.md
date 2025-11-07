# Model Card

## Overview

The lead-lag signature agent estimates directional signals between paired assets by combining recurrent encoders with
log-signature features. Policy optimisation primarily relies on proximal policy optimisation (PPO) variants with
recurrent latent states.

## Intended Use

- Support research into robust trading signals under anonymised settings.
- Benchmark reinforcement learning against supervised lead-lag baselines.
- Provide reproducible experiments for peer review and anonymised evaluation.

This model card is not an endorsement for live trading or production deployment.

## Training Data and Features

- Inputs: normalised price differentials, rolling volume indicators, and derived log-signature terms up to order three.
- Rewards: shaped to favour stable lead/lag predictions, penalising drawdowns and excessive turnover.
- Training data follows the splits described in `docs/DATA_CARD.md`.

## Model Assumptions

- Market microstructure can be approximated by the synthetic regimes and anonymised historical windows supplied.
- Lead-lag relationships remain locally stationary across the training and validation windows.
- Replay buffers capture sufficient exploration diversity to seed offline training without major distribution shifts.

## Failure Modes and Limitations

- **Regime shifts:** Sudden macro events or structural breaks unseen in the training regimes can degrade performance.
- **Sparse liquidity:** Asset pairs with thin trading volumes may produce noisy signatures that the model overfits.
- **Latency sensitivity:** The approach assumes consistent sampling intervals; irregular timestamps can misalign the
  log-signature features.
- **Anonymisation impacts:** Hashing identifiers prevents domain experts from applying instrument-specific heuristics,
  which may limit remediation when anomalies are detected.

Users should inspect evaluation diagnostics under `results/` before extending to new datasets.

## Compute Profile

- **Hardware:** Reference runs use a single NVIDIA A100 40GB GPU (or Kaggle T4 fallback) with 16 vCPUs and 64 GB RAM.
- **Training duration:** Full suite runs complete within 6 GPU-hours for the anonymised corpus; synthetic-only sweeps
  finish in 2 GPU-hours.
- **Frameworks:** PyTorch 2.x with CUDA 12.x, Hydra for configuration, and Weights & Biases disabled in anonymous mode.
- **Determinism:** Seeds are fixed via Hydra configs; stochasticity persists due to GPU kernels but falls within
  reproducibility tolerances logged in run metadata.

## Evaluation

Validation follows the metrics documented in `docs/metrics_dictionary.md`, emphasising Sharpe ratio, maximum drawdown,
turnover, and calibration error. Test results are reserved for final reporting and remain frozen once published.

## Ethical Considerations

- The anonymisation pipeline removes identifiers and aggregates sensitive fields, but downstream users must still ensure
  compliance with local regulations when introducing external data.
- Reinforcement learning policies that interact with live markets could amplify risk if deployed without safeguards;
  this repository focuses solely on offline and replay-based experimentation.

## Maintenance

Model hyperparameters and scenario definitions live under `hydra_main.py` and `configs/`. Update the model card when
new architectures, reward formulations, or evaluation protocols are introduced.
