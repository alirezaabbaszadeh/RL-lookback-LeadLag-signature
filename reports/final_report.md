# Lead-Lag Signature RL Research Report

_Generated on 2025-10-17 13:36:37Z_

This report compiles the latest reinforcement learning lookback experiments and baseline
comparisons.
Scenarios covered: fixed_30, research_full.

## Methodology

Experiments were run via the ExperimentOrchestrator multi-seed pipeline with Hydra-driven
configs.
Aggregates include mean/median statistics, bootstrap confidence intervals, and optional Welch
tests.

- **fixed_30**: aggregate directory `results\manual\fixed_30_aggregate`; seeds: 42, 52, 62
- **research_full**: aggregate directory `results\manual\research_full_aggregate`; seeds: 101, 202, 303


## Experiments

### Scenario: fixed_30

| Metric | Mean | Std Dev |
| --- | --- | --- |
| mean_abs_matrix | 0.1686 | 0.0000 |
| row_sum_std | 9.3700 | 0.0000 |
| row_sum_range | 48.9851 | 0.0000 |
| stability_matrix_corr | 0.9575 | 0.0000 |

Bootstrap confidence intervals:
- mean_abs_matrix: bootstrap 95% CI [0.1686, 0.1686]
- row_sum_std: bootstrap 95% CI [9.3700, 9.3700]
- row_sum_range: bootstrap 95% CI [48.9851, 48.9851]
- stability_matrix_corr: bootstrap 95% CI [0.9575, 0.9575]

Per-run highlights:
- Seed 42, created 20251011_211857, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`
- Seed 52, created 20251011_212658, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`
- Seed 62, created 20251011_213510, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`

### Scenario: research_full

| Metric | Mean | Std Dev |
| --- | --- | --- |
| mean_abs_matrix | 0.1203 | 0.0000 |
| row_sum_std | 6.7064 | 0.0000 |
| row_sum_range | 35.4104 | 0.0000 |
| stability_matrix_corr | 0.9796 | 0.0000 |

Bootstrap confidence intervals:
- mean_abs_matrix: bootstrap 95% CI [0.1203, 0.1203]
- row_sum_std: bootstrap 95% CI [6.7064, 6.7064]
- row_sum_range: bootstrap 95% CI [35.4104, 35.4104]
- stability_matrix_corr: bootstrap 95% CI [0.9796, 0.9796]

Per-run highlights:
- Seed 101, created 20251016_181558, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`
- Seed 202, created 20251016_183505, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`
- Seed 303, created 20251016_185410, config `D:\uni\Dr.AI.Ali\FinTech\LeadLag-signature - RL\RL-lookback-LeadLag-signature\configs\scenarios\fixed_30.yaml`


## Conclusion

fixed_30 achieved the strongest mean_abs_matrix signal (0.1686), indicating the highest overall
signal strength among evaluated runs.
Next actions: finalise the research narrative, attach visual artefacts from ER-01, and schedule
peer review.