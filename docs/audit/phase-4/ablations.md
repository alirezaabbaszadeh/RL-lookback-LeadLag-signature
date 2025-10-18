# Phase 4 — Ablation Study Plan

## Factor Grid

| Factor                | Values / Scenarios                                                  | Coverage status |
|-----------------------|---------------------------------------------------------------------|-----------------|
| Method                | `signature` (fixed_30, fixed_90, abl_*), `dynamic` (dynamic_adaptive) | ✅ signature, ⚠️ dynamic (needs rerun) |
| Lookback window       | 10 (abl_smoke), 30 (fixed_30, abl_lite_gpu), 60 (abl_server), adaptive (dynamic_adaptive) | ✅ |
| Action mode (RL)      | Hybrid attention policy (abl_lite_gpu, abl_server), baseline static (fixed_*), random control (abl_random) | ⚠️ random requires optional gym/sb3 deps |
| Reward weights        | Default only                                                        | ⚠️ introduce variants (e.g., emphasize Sharpe vs drawdown) |
| Seeds                 | Single (abl_smoke, fast smoke), multi-seed [42,52,62], [101,202,303] | ✅ for key scenarios |

Existing scenarios (Hydra presets / YAML):
- `fixed_30`, `fixed_90`, `dynamic_adaptive`, `rl_ppo` (core baselines)
- Ablation presets: `abl_smoke`, `abl_lite_gpu`, `abl_server`

## Coverage Summary
- Signature baselines (fixed lookback) covered at 30/90.
- RL variants covered at lite (reduced dimensions) and server-grade (multi-seed).
- Dynamic adaptive baseline available but requires fresh runs to capture metrics in latest format.
- Random policy preset `abl_random` added (requires optional gym/sb3 deps for LeadLagEnv); execute when dependencies available.

## Negative Control Plan
- Add random/frozen policy scenario by extending RL runner to support `policy: random` (placeholder item).
- Include static signal-only baseline (no RL, no adaptive mechanism) to contrast with RL policies.

## Next Steps
1. Run `dynamic_adaptive` with current pipeline to populate metrics/finance KPIs.
2. Implement random/frozen policy toggle in RL runner and create `configs/scenarios/abl_random.yaml`.
3. Introduce reward-weight variants (e.g., Sharpe emphasis) and log effect on financial KPIs.
4. Update coverage table with run evidence (link to results directories & plots).
