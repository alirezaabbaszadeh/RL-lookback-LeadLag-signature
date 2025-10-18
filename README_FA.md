# پلتفرم LeadLag Signature (راهنمای جامع فارسی)

این مخزن یک محیط پژوهشی کامل برای تحلیل ماتریس‌های لید–لگ، آزمایش سیاست‌های نگاه‌به‌عقب (Lookback) تطبیقی و ارزیابی عامل‌های یادگیری تقویتی است. هدف ما این است که همه مراحل (داده → آزمایش → گزارش) با **یک فرمان** روی محیط‌هایی مثل Kaggle تکرارپذیر اجرا شوند.

---

## پیش‌نیازهای اصلی

| دسته | بسته‌ها | توضیح |
|------|---------|--------|
| ضروری | `numpy`, `pandas`, `scipy`, `hydra-core`, `iisignature`, `dcor`, `gym`, `matplotlib`, `scikit-learn`, `tqdm`, `pyyaml` | در فایل `requirements-kaggle.txt` جمع شده است؛ برای اجرای سناریوهای Signature/CCF کافی هستند. |
| اختیاری (برای RL) | `stable-baselines3`, `torch`, `sb3-contrib` | برای سناریوهای یادگیری تقویتی و نسخه‌ی LSTM لازم‌اند. اگر نصب نشوند، می‌توان با فلگ `--skip-optional-deps` سناریوهای RL را رد کرد. |
| ابزار پژوهشی | `mlflow` (اختیاری) | در صورت نیاز به لاگ‌گیری خارجی. |

> **نصب در Kaggle (بدون اینترنت):** ابتدا `requirements-kaggle.txt` را نصب کنید. اگر قصد اجرای سناریوهای RL دارید، wheelهای `stable-baselines3`، `torch` و `sb3-contrib` باید در دیتاست ورودی شما قرار داشته باشند یا قبل از قطع اینترنت نصب شوند. در غیر این صورت، از فلگ `--skip-optional-deps` استفاده کنید.

---

## اجرای همه آزمایش‌ها با یک فرمان

1. **نصب پیش‌نیازها**

   ```bash
   !pip install -r /kaggle/input/<dataset>/requirements-kaggle.txt
   # در صورت نیاز به سناریوهای RL
   !pip install -q stable-baselines3 torch sb3-contrib
   ```

2. **اجرای پایپلاین کامل**

   ```bash
   python pipelines/run_full_suite.py \
       --output-root /kaggle/working/full_suite \
       --baseline-seeds 42 52 62 \
       --max-missing-ratio 0.01 \
       --max-zero-variance 0
   ```

   این فرمان، تمام مراحل زیر را پشت سر هم انجام می‌دهد:
   - ممیزی کیفیت داده (`dataset_quality.py`)
   - اجرای سناریوهای پایه (پیش‌فرض: `fixed_30`) با چند seed
   - اجرای Meta-RL و RL آفلاین (دیتاست آفلاین به صورت CSV ذخیره می‌شود)
   - استخراج KPIهای مالی (بازده سالانه، شارپ، دراودان)
   - اجرای آزمایش‌های حذفی کامل (signature، CCF، دینامیک، RL به همراه واریانت‌های وزن‌دهی و کنترل تصادفی)
   - آزمون‌های نشتی و واک‌فوروارد
   - مقایسه آماری، گزارش نهایی، و اسکن نهایی خروجی‌ها

3. **خروجی‌ها**

   پس از اجرا تمام نتایج در پوشه‌های زیر ذخیره می‌شوند:
   ```
   /core                  # سناریوهای پایه + KPI مالی
   /meta_rl               # نتایج Meta-RL
   /offline               # دیتاست آفلاین و نتایج Behavior Cloning
   /ablations             # خروجی سناریوهای حذفی + تجمیع
   /robustness            # آزمون‌های نشتی و واک‌فوروارد
   /aggregate_comparison  # مقایسه‌های آماری و نمودارها
   /reports               # گزارش نهایی و پیوست‌ها
   /audit                 # گزارش معتبر بودن خروجی‌ها (scan_report)
   ```

---

## فلگ‌ها و تنظیمات مهم

| فلگ | کاربرد |
|-----|--------|
| `--baseline-seeds` / `--baseline-single-seed` | کنترل تعداد seed برای سناریوهای پایه |
| `--baseline-scenarios` | اجرای چند سناریوی پایه (مثلاً `fixed_30 ccf_fixed rl_ppo`) |
| `--ablation-scenarios` و `--ablation-single-seed` | انتخاب دقیق سناریوهای حذفی و سبک‌سازی اجرا |
| `--skip-ablation`, `--skip-meta-offline`, `--skip-audit`, `--skip-report`, `--skip-baseline` | حذف بخش‌های پرهزینه یا غیرضروری |
| `--skip-optional-deps` | در نبود SB3/Torch/sb3-contrib، سناریوهای RL به‌طور خودکار رد می‌شوند |
| `--max-missing-ratio`, `--max-zero-variance`, `--fail-on-quality` | کنترل آستانه‌های ممیزی داده |
| `--skip-schema-check` | حذف مرحله نهایی اسکن خروجی‌ها (در صورت سفید بودن نتایج قبلی) |

---

## سناریوهای مهم

- Signature (پایه): `fixed_30`, `fixed_90`
- CCF-at-lag: `ccf_fixed`
- دینامیک حریصانه: `dynamic_adaptive`
- RL استاندارد و واریانت‌ها: `rl_ppo`, `rl_ppo_sharpe`, `rl_ppo_drawdown`, `rl_ppo_lstm`
- کنترل تصادفی: `abl_random`
- نسخه‌های سبک آزمایشی: `abl_smoke`, `abl_lite_gpu`, `abl_server`

> اگر `iisignature` نصب نباشد، سناریوهای Signature خودکار رد می‌شوند. سناریوهای CCF و دینامیک همچنان اجرا خواهند شد.

---

## ابزارهای کمکی

| فایل | توضیح |
|------|-------|
| `pipelines/run_ablation.py` | فقط آزمایش‌های حذفی (signature/ccf/dynamic/RL/random) را اجرا می‌کند؛ با فلگ `--skip-missing-deps` وابستگی‌های RL را نادیده می‌گیرد. |
| `kaggle/starter.py` | برای نوت‌بوک‌های ساده: اجرای یک سناریو + (اختیاری) Meta-RL و RL آفلاین. |
| `scripts/smoke_kaggle.py` | اسکریپت دودتست سریع (fast_smoke + گزینه‌های Meta/Offline). |
| `evaluation/finance_kpis.py` | استخراج KPI‌های مالی برای هر پوشه خروجی (قابل فراخوانی مجزا). |
| `scripts/audit/dataset_quality.py` | ممیزی داده با آستانه‌های سفارشی و گزینه `--exit-on-fail`. |
| `scripts/audit/validate_artifacts.py` | اسکن یک پوشه خروجی و تولید گزارش وجود فایل‌های استاندارد. |

---

## ساختار مخزن

| مسیر | کاربرد |
|------|--------|
| `configs/` | کانفیگ‌های Hydra (سناریوهای پایه و حذفی). |
| `pipelines/` | پایپلاین‌های یک‌فرمانی (`run_full_suite.py`, `run_ablation.py`). |
| `research/meta_rl/` و `research/offline_rl/` | ابزارهای پژوهشی برای رژیم‌های مصنوعی و RL آفلاین. |
| `scripts/audit/` | اسکریپت‌های ممیزی (کیفیت داده، اسکن خروجی، نشتی و واک‌فوروارد). |
| `evaluation/` | محاسبه KPI مالی و گزارش‌های مقایسه‌ای. |
| `docs/` | راهنماها و گزارش‌های ممیزی؛ بخش Deployment به ویژه برای Kaggle مفصل است. |
| `tests/` | مجموعه PyTest برای صحت‌سنجی اجزای کلیدی. |

---

## توصیه‌های پایانی برای Kaggle

1. قبل از قطع اینترنت، نسخه CPU از Torch و پکیج‌های RL را نصب کنید یا از فلگ `--skip-optional-deps` بهره ببرید.
2. برای صرفه‌جویی در زمان می‌توانید سناریوها را با `--baseline-single-seed` و `--ablation-single-seed` سبک کنید.
3. خروجی‌ها (خصوصاً `/reports` و `/aggregate_comparison`) را بررسی کنید؛ اگر حجم زیاد است، فایل‌های میانی را حذف یا فشرده کنید.
4. گزارش نهایی PDF/Markdown در `/reports` تولید می‌شود؛ می‌توان آن را به همراه `audit/scan_report.md` ضمیمه گزارش یا مقاله کرد.

با این چارچوب، تنها یک فرمان کافی است تا کل آزمایش‌ها، ممیزی‌ها و گزارش‌ها در محیط Kaggle ساخته شوند و برای تحلیل نهایی آماده باشند. موفق باشید! 🎯
