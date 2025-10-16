# Hydra Configuration Reference

```
configs/
├── config.yaml            # base entry point for hydra_main.py
└── scenario/              # reusable scenario descriptors (Hydra/YAML mode)
    ├── fixed_30.yaml
    ├── fixed_90.yaml
    ├── dynamic_adaptive.yaml
    └── rl_ppo.yaml
```

## config.yaml

| Key | Type | Description |
|-----|------|-------------|
| `defaults.scenario` | str | Hydra default pointing to a file in `configs/scenario/`. Override via CLI: `python hydra_main.py scenario=fixed_90`. |
| `output_root` | str | Root directory for run artifacts. |
| `multi_seed.enabled` | bool | Global flag to execute multi-seed runs. |
| `multi_seed.seeds` | list[int] | Default seeds when multi-seed is enabled. |
| `scenarios` | list[str or dict] | Optional list to run multiple scenarios sequentially. Each entry can be a scenario name or an inline dict matching the columns below. |

## Scenario descriptor (`configs/scenario/*.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | Scenario identifier shown in logs and aggregate outputs. |
| `path` | str | Path to YAML file describing analysis parameters. |
| `runner` | str | One of `scenario` (baseline), `dynamic`, `rl`. Determines which execution function is used. |
| `multi_seed.seeds` | list[int] | Optional seeds specific to this scenario. |
| `multi_seed.enabled` | bool | Optional override of the multi-seed flag. |

## Built-in presets (available even without Hydra/YAML)

Hydra fallback mode exposes several named presets via the `--scenario` flag:

| Preset | Purpose | Key settings |
|--------|---------|--------------|
| `fast_smoke` | Ultra-fast smoke test | Synthetic data if CSV missing, lookback=10, update_freq=9999 (یک پنجره)، فقط متریک mean_abs_matrix، بدون plot، تک seed. |
| `fixed_30` | Baseline تحلیل ثابت | lookback=30، می‌تواند multi-seed `[42,52,62]` داشته باشد. |
| `dynamic_adaptive` | قانون‌محور پویا | تنظیمات min/max/step برای انتخاب داینامیک lookback. |
| `rl_ppo` | اجرای سبک RL (نیازمند stable-baselines3) | total_timesteps=2000 و سایر هایپرپارامترهای سبک. |

## CLI Examples

```bash
# run single scenario (default fixed_30)
python hydra_main.py

# run ultra-fast smoke test
python hydra_main.py --scenario fast_smoke --multi_seed_enabled=false

# research-grade baseline with multi-seed aggregation
python hydra_main.py --scenario research_full

# run RL preset (requires stable-baselines3)
python hydra_main.py --scenario rl_ppo

# run multiple scenarios sequentially (Hydra/YAML mode)
python hydra_main.py scenarios='[fixed_30, rl_ppo]' output_root=experiments
```

هر اجرا خروجی‌هایی مانند `metrics_timeseries.csv`, `summary.csv`, و در حالت multi-seed فایل‌های `stats.csv`, `significance.csv`, `runs.json` را تولید می‌کند. در preset‌های سبک (fast_smoke) نمودارها غیرفعال شده‌اند تا زمان اجرا حداقلی باشد؛ در preset‌های پژوهشی خروجی کامل برای مقاله فراهم می‌شود.
