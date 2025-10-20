# نقشهٔ خروجی‌های Kaggle

این سند مسیرها و فایل‌هایی را که پس از اجرای دستور `!python kaggle/run_all.py` روی Kaggle ساخته می‌شوند، فهرست می‌کند. برای هر آیتم یک برچسب داخل براکت آمده تا نقش آن سریع مشخص شود.

## جمع‌بندی سریع
- `[bundle] /kaggle/working/multi_stage_artifacts.zip` تنها بستهٔ قابل دانلود؛ حاوی کل دایرکتوری `multi_stage_artifacts/`.
- `[summary] multi_stage_artifacts/summary.json` گزارش وضعیت هر استیج (موفق/ناموفق، مدت، مسیر لاگ).
- `[stage] multi_stage_artifacts/full_suite`, `[stage] …/sb3_leadlag`, `[stage] …/dopamine` ریشهٔ خروجی سه مرحله.

## ساختار مرحلهٔ full_suite
- `[log] multi_stage_artifacts/full_suite/stdout.log` و `…/stderr.log` لاگ‌های خام استیج (در اجراهای موفق معمولاً خالی می‌مانند).
- `[req] multi_stage_artifacts/full_suite/requirements.txt` نسخهٔ دقیق وابستگی‌های نصب‌شده هنگام اجرا.
- `[metadata] multi_stage_artifacts/full_suite/run_metadata.json` زمان اجرا، کد خروج و مسیر ریشهٔ خروجی را ثبت می‌کند.
- `[runs-root] multi_stage_artifacts/full_suite/runs` همهٔ فرآیندهای درون full_suite را با زیرشاخه‌های جدا نگه می‌دارد:
  - `[baselines] runs/core` خروجی سناریوهای LeadLag (ثابت، دینامیک، RL، …).
    - `[log] */run.log` لاگ ساختارمند هر اجرا.
    - `[config] */config_merged.yaml` پیکربندی نهایی پس از merge.
    - `[manifest] */data_manifest.json` مشخصات دادهٔ ورودی و چک‌های کیفیت.
    - `[metadata] */run_metadata.json` محیط اجرا (git، نسخهٔ پایتون، مسیر داده).
    - `[results] */metrics_timeseries.csv`, `[summary] */summary.csv`.
    - `[plots] */fig_signal_strength.png`, `*/fig_stability.png`.
    - `[profiles] */profiles/*.pstats`, `*.json` نتیجهٔ پروفایلینگ بخش‌های اصلی.
    - `[matrix] */matrix_<YYYY-MM-DD>.csv` ماتریس‌های نمونهٔ ابتدا/انتها برای بررسی سریع.
    - `[aggregate] *_aggregate/` (وقتی multi-seed فعال است) شامل `stats.csv`, `significance.csv`, `welch.csv`, `aggregate.log`, `runs.json`.
  - `[meta-rl] runs/meta_rl` داده‌های مصنوعی (`train_regimes.csv`, `test_regimes.csv`)، نتایج (`meta_analysis.csv`)، مدل (`meta_agent.json`).
  - `[offline] runs/offline` شامل `offline_dataset.csv`, `data_manifest.json`, `offline_metadata.json`, `offline_results.json|csv` و در صورت مقایسه با آنلاین `offline_vs_online.csv`.
  - `[robustness] runs/robustness` خروجی پروب نشت (`leakage_*`) و Walk-Forward (`wf_*`)، هر کدام با ساختار مشابه baseline.
  - `[evaluation] runs/evaluation/finance_kpis.csv` به‌علاوه‌ی `plots/` با نمودارهای KPI.
  - `[comparison] runs/aggregate_comparison/aggregate_comparison.csv` و نمودارهای میله‌ای متریک‌ها.
  - `[reports] runs/reports/final_report.md`, `appendix.md`, `final_report.pdf` مستند نهایی.
  - `[audit-report] runs/audit/scan_report.json`, `scan_report.md` خروجی اعتبارسنجی آرتیفکت‌ها.
- `[logs] multi_stage_artifacts/full_suite/logs/run_summary_<timestamp>.json` خلاصهٔ کلی اجرای پایپ‌لاین (دستور، آرگومان‌ها، طول زمان، موفقیت).

## ساختار مرحلهٔ sb3_leadlag
- `[metadata] multi_stage_artifacts/sb3_leadlag/run_metadata.json` فهرست سناریوهای RL و پارامترهای تزریق‌شده از محیط.
- `[scenario-run] multi_stage_artifacts/sb3_leadlag/<scenario-name>_*` برای هر سناریو پوشه‌ای شامل:
  - `[model] model.zip` مدل آموزش‌دیدهٔ SB3 (اگر سیاست تصادفی نباشد).
  - `[results] metrics_timeseries.csv`, `[summary] summary.csv`, `[plots] fig_signal_strength.png`, `fig_stability.png`.
  - `[log] run.log`, `[manifest] data_manifest.json`, `[metadata] run_metadata.json`.
  - `[eval] eval/` و `eval_logs/` وقتی `EvalCallback` فعال باشد.

## ساختار مرحلهٔ dopamine
- `[metadata] multi_stage_artifacts/dopamine/run_metadata.json` نسخهٔ Gymnasium/Dopamine، بازده اپیزودها و تنظیمات اجرا.
- `[stats] multi_stage_artifacts/dopamine/iteration_stats.json` آمار داخلی Dopamine به شکل JSON.
- `[log] stdout.log`, `[log] stderr.log` خروجی‌های خام استیج.

## اثرات جانبی خارج از دایرکتوری استیج‌ها
- `[cache] /kaggle/working/wheelhouse/` بسته‌های whl دانلود‌شده برای نصب سریع.
- `[cache] /kaggle/working/.cache/pip/` کش Pip اختصاصی اجرای Kaggle.
- `[side-effect] results/meta_analysis.csv` و `[side-effect] results/offline_results.csv` کپی جهانی نتایج متا/آفلاین برای استفادهٔ سایر ابزارها.
- `[side-effect] docs/audit/phase-2/leakage_probe_summary.csv` و `walk_forward_check.csv` خروجی‌های تحلیلی پروب‌ها.

## نکات دانلود
- برای دریافت همهٔ خروجی‌ها، `multi_stage_artifacts.zip` را از پنل Kaggle دانلود و سپس اکسترکت کنید؛ ساختار فوق دقیقاً همان محتویات داخل زیپ است.
- در صورت نیاز به بخشی خاص (مثلاً فقط نتایج RL)، می‌توان پس از دانلود زیپ، پوشهٔ مربوطه را جداگانه آرشیو کرد یا در Kaggle با `shutil.make_archive` زیپ جدید ساخت.
