# Ù¾Ù„ØªÙØ±Ù… LeadLag Signature (Ø±Ø§Ù‡Ù†Ù…Ø§ÛŒ Ø¬Ø§Ù…Ø¹ ÙØ§Ø±Ø³ÛŒ)

Ø§ÛŒÙ† Ù…Ø®Ø²Ù† ÛŒÚ© Ù…Ø­ÛŒØ· Ù¾Ú˜ÙˆÙ‡Ø´ÛŒ Ú©Ø§Ù…Ù„ Ø¨Ø±Ø§ÛŒ ØªØ­Ù„ÛŒÙ„ Ù…Ø§ØªØ±ÛŒØ³â€ŒÙ‡Ø§ÛŒ Ù„ÛŒØ¯â€“Ù„Ú¯ØŒ Ø¢Ø²Ù…Ø§ÛŒØ´ Ø³ÛŒØ§Ø³Øªâ€ŒÙ‡Ø§ÛŒ Ù†Ú¯Ø§Ù‡â€ŒØ¨Ù‡â€ŒØ¹Ù‚Ø¨ (Lookback) ØªØ·Ø¨ÛŒÙ‚ÛŒ Ùˆ Ø§Ø±Ø²ÛŒØ§Ø¨ÛŒ Ø¹Ø§Ù…Ù„â€ŒÙ‡Ø§ÛŒ ÛŒØ§Ø¯Ú¯ÛŒØ±ÛŒ ØªÙ‚ÙˆÛŒØªÛŒ Ø§Ø³Øª. Ù‡Ø¯Ù Ù…Ø§ Ø§ÛŒÙ† Ø§Ø³Øª Ú©Ù‡ Ù‡Ù…Ù‡ Ù…Ø±Ø§Ø­Ù„ (Ø¯Ø§Ø¯Ù‡ â†’ Ø¢Ø²Ù…Ø§ÛŒØ´ â†’ Ú¯Ø²Ø§Ø±Ø´) Ø¨Ø§ **ÛŒÚ© ÙØ±Ù…Ø§Ù†** Ø±ÙˆÛŒ Ù…Ø­ÛŒØ·â€ŒÙ‡Ø§ÛŒÛŒ Ù…Ø«Ù„ Kaggle ØªÚ©Ø±Ø§Ø±Ù¾Ø°ÛŒØ± Ø§Ø¬Ø±Ø§ Ø´ÙˆÙ†Ø¯.

---

## Ù¾ÛŒØ´â€ŒÙ†ÛŒØ§Ø²Ù‡Ø§ÛŒ Ø§ØµÙ„ÛŒ

| Ø¯Ø³ØªÙ‡ | Ø¨Ø³ØªÙ‡â€ŒÙ‡Ø§ | ØªÙˆØ¶ÛŒØ­ |
|------|---------|--------|
| Ø¶Ø±ÙˆØ±ÛŒ | `numpy`, `pandas`, `scipy`, `hydra-core`, `iisignature`, `dcor`, `gym`, `matplotlib`, `scikit-learn`, `tqdm`, `pyyaml` | Ø¯Ø± ÙØ§ÛŒÙ„ `requirements-kaggle.txt` Ø¬Ù…Ø¹ Ø´Ø¯Ù‡ Ø§Ø³ØªØ› Ø¨Ø±Ø§ÛŒ Ø§Ø¬Ø±Ø§ÛŒ Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ Signature/CCF Ú©Ø§ÙÛŒ Ù‡Ø³ØªÙ†Ø¯. |
| Ø§Ø®ØªÛŒØ§Ø±ÛŒ (Ø¨Ø±Ø§ÛŒ RL) | `stable-baselines3`, `torch`, `sb3-contrib` | Ø¨Ø±Ø§ÛŒ Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ ÛŒØ§Ø¯Ú¯ÛŒØ±ÛŒ ØªÙ‚ÙˆÛŒØªÛŒ Ùˆ Ù†Ø³Ø®Ù‡â€ŒÛŒ LSTM Ù„Ø§Ø²Ù…â€ŒØ§Ù†Ø¯. Ø§Ú¯Ø± Ù†ØµØ¨ Ù†Ø´ÙˆÙ†Ø¯ØŒ Ù…ÛŒâ€ŒØªÙˆØ§Ù† Ø¨Ø§ ÙÙ„Ú¯ `--skip-optional-deps` Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ RL Ø±Ø§ Ø±Ø¯ Ú©Ø±Ø¯. |
| Ø§Ø¨Ø²Ø§Ø± Ù¾Ú˜ÙˆÙ‡Ø´ÛŒ | `mlflow` (Ø§Ø®ØªÛŒØ§Ø±ÛŒ) | Ø¯Ø± ØµÙˆØ±Øª Ù†ÛŒØ§Ø² Ø¨Ù‡ Ù„Ø§Ú¯â€ŒÚ¯ÛŒØ±ÛŒ Ø®Ø§Ø±Ø¬ÛŒ. |

> **Ù†ØµØ¨ Ø¯Ø± Kaggle (Ø¨Ø¯ÙˆÙ† Ø§ÛŒÙ†ØªØ±Ù†Øª):** Ø§Ø¨ØªØ¯Ø§ `requirements-kaggle.txt` Ø±Ø§ Ù†ØµØ¨ Ú©Ù†ÛŒØ¯. Ø§Ú¯Ø± Ù‚ØµØ¯ Ø§Ø¬Ø±Ø§ÛŒ Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ RL Ø¯Ø§Ø±ÛŒØ¯ØŒ wheelÙ‡Ø§ÛŒ `stable-baselines3`ØŒ `torch` Ùˆ `sb3-contrib` Ø¨Ø§ÛŒØ¯ Ø¯Ø± Ø¯ÛŒØªØ§Ø³Øª ÙˆØ±ÙˆØ¯ÛŒ Ø´Ù…Ø§ Ù‚Ø±Ø§Ø± Ø¯Ø§Ø´ØªÙ‡ Ø¨Ø§Ø´Ù†Ø¯ ÛŒØ§ Ù‚Ø¨Ù„ Ø§Ø² Ù‚Ø·Ø¹ Ø§ÛŒÙ†ØªØ±Ù†Øª Ù†ØµØ¨ Ø´ÙˆÙ†Ø¯. Ø¯Ø± ØºÛŒØ± Ø§ÛŒÙ† ØµÙˆØ±ØªØŒ Ø§Ø² ÙÙ„Ú¯ `--skip-optional-deps` Ø§Ø³ØªÙØ§Ø¯Ù‡ Ú©Ù†ÛŒØ¯.

---

## مسیر نوت‌بوک Kaggle (ویژه ارسال مجله)

تنظیمات نوت‌بوک: اینترنت **روشن** و شتاب‌دهنده **GPU** (T4 به بالا). تمام مسیرها در `/kaggle/working` ساخته می‌شوند.

1. **بررسی GPU و محیط**

   ```python
   import torch

   print("CUDA available:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print(torch.cuda.get_device_name(0))
   !nvidia-smi -L
   ```

2. **آماده‌سازی محیط و پیش‌دریافت wheel** – اگر مخزن را با نام دیگری متصل کرده‌اید، مسیر را مطابق نیاز عوض کنید.

   ```bash
   %%bash
   set -e
   WORK=/kaggle/working
   cd "$WORK"

   if [ ! -f kaggle/run_all.py ]; then
     cp -r /kaggle/input/leadlag-signature/* "$WORK"/
   fi

   python -m pip install --upgrade pip
   mkdir -p wheelhouse .cache/pip
   python -m pip download -d wheelhouse -r requirements-kaggle.txt
   python -m pip download -d wheelhouse "gymnasium==0.29.1" "stable-baselines3==2.1.0" "sb3-contrib==2.1.0" "torch>=2.1,<2.7"
   python -m pip download -d wheelhouse "dopamine-rl==4.1.2" "gymnasium==1.0.0"

   export PIP_CACHE_DIR=/kaggle/working/.cache/pip
   export PIP_FIND_LINKS=/kaggle/working/wheelhouse
   export PIP_NO_INDEX=1
   ```

3. **اجرای ارکستریتور چندمرحله‌ای** – هر مرحله در یک virtualenv جدا اجرا شده و خروجی نهایی فشرده می‌شود.

   ```python
   !python kaggle/run_all.py
   ```

   فلگ‌های مهم: `--no-prefetch` برای رد کردن مرحلهٔ دانلود wheel و `--artifacts-root` برای تغییر مسیر خروجی.

4. **نمایش خروجی مقاله** – این قطعه خروجی‌های `paper_outputs/` را به سطح بالای نوت‌بوک کپی می‌کند تا سریع بررسی شوند.

   ```python
   import shutil, pathlib

   src = pathlib.Path("/kaggle/working/multi_stage_artifacts/full_suite/paper_outputs")
   dst = pathlib.Path("/kaggle/working/paper_outputs")
   if src.exists():
       shutil.copytree(src, dst, dirs_exist_ok=True)
       print("Paper outputs copied to", dst)
   else:
       print("paper_outputs missing – inspect stage logs")
   ```

5. **دانلود بستهٔ نهایی برای داوران** – فایل `multi_stage_artifacts.zip` آمادهٔ بارگیری از پانل سمت راست Kaggle است. در صورت نیاز، پوشهٔ `paper_outputs/` کپی‌شده را نیز جداگانه بارگیری کنید.

تمام موارد مورد نیاز داوران (لاگ‌ها، مانفیست‌ها، جداول مقاله و آمار) در همین زیپ وجود دارد.

---

## فلگ‌ها و تنظیمات مهم

| فلگ | کاربرد |
|-----|--------|
| `--list` | فهرست کردن سناریوهای بسته‌بندی‌شده و خروج پس از نمایش. |
| `--scenarios <نام ...>` | اجرای سناریوهای مشخص‌شده (نام YAML یا مسیر دلخواه). |
| `--include / --exclude` | فیلتر کردن سناریوها بر اساس زیررشتهٔ نام فایل. |
| `--dry-run` | فقط گزارش انتخاب سناریو و مسیر خروجی بدون اجرای محاسبات. |
| `--status` | خلاصهٔ وضعیت اجراهای موجود زیر `results_root` را چاپ می‌کند. |
| `--format text|json` | انتخاب قالب خروجی استاندارد؛ `json` برای اتوماسیون توصیه می‌شود. |
| `--results-root` | مسیر ذخیرهٔ خروجی‌ها (پیش‌فرض: متغیر محیطی `LEADLAG_RESULTS_ROOT` یا `results`). |
| `--runner auto|scenario|dynamic|rl` | اجبار رانر خاص یا اجازهٔ انتخاب خودکار بر اساس پیکربندی سناریو. |
| `--skip-existing` | عبور از سناریوهایی که قبلاً اجرا و موفق شده‌اند. |
| `--stop-on-error` | در صورت بروز خطا اجرا را فوراً متوقف می‌کند. |
| `--validate <سناریو>` | اعتبارسنجی یک سناریو بدون اجرا (نام یا مسیر). |

---

## Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ Ù…Ù‡Ù…

- Signature (Ù¾Ø§ÛŒÙ‡): `fixed_30`, `fixed_90`
- CCF-at-lag: `ccf_fixed`
- Ø¯ÛŒÙ†Ø§Ù…ÛŒÚ© Ø­Ø±ÛŒØµØ§Ù†Ù‡: `dynamic_adaptive`
- RL Ø§Ø³ØªØ§Ù†Ø¯Ø§Ø±Ø¯ Ùˆ ÙˆØ§Ø±ÛŒØ§Ù†Øªâ€ŒÙ‡Ø§: `rl_ppo`, `rl_ppo_sharpe`, `rl_ppo_drawdown`, `rl_ppo_lstm`
- Ú©Ù†ØªØ±Ù„ ØªØµØ§Ø¯ÙÛŒ: `abl_random`
- Ù†Ø³Ø®Ù‡â€ŒÙ‡Ø§ÛŒ Ø³Ø¨Ú© Ø¢Ø²Ù…Ø§ÛŒØ´ÛŒ: `abl_smoke`, `abl_lite_gpu`, `abl_server`

> Ø§Ú¯Ø± `iisignature` Ù†ØµØ¨ Ù†Ø¨Ø§Ø´Ø¯ØŒ Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ Signature Ø®ÙˆØ¯Ú©Ø§Ø± Ø±Ø¯ Ù…ÛŒâ€ŒØ´ÙˆÙ†Ø¯. Ø³Ù†Ø§Ø±ÛŒÙˆÙ‡Ø§ÛŒ CCF Ùˆ Ø¯ÛŒÙ†Ø§Ù…ÛŒÚ© Ù‡Ù…Ú†Ù†Ø§Ù† Ø§Ø¬Ø±Ø§ Ø®ÙˆØ§Ù‡Ù†Ø¯ Ø´Ø¯.

---

## ابزارهای کمکی

| فایل/فرمان | توضیح |
|-------------|-------|
| `leadlag` | CLI اصلی برای فهرست، Dry-run، اجرا و گزارش وضعیت سناریوها با خروجی متنی یا JSON. |
| `leadlag-full-suite` | مسیر سازگار با Hydra برای زمانی که به overrideهای مستقیم نیاز دارید (`-- env.allow_short=false`). |
| `pipelines/run_ablation.py` | فقط آزمایش‌های حذفی (signature/ccf/dynamic/RL/random) را اجرا می‌کند؛ با فلگ `--skip-missing-deps` وابستگی‌های RL را نادیده می‌گیرد. |
| `kaggle/starter.py` | برای نوت‌بوک‌های ساده: اجرای یک سناریو + (اختیاری) Meta-RL و RL آفلاین. |
| `scripts/smoke_kaggle.py` | اسکریپت دودتست سریع (fast_smoke + گزینه‌های Meta/Offline). |
| `evaluation/finance_kpis.py` | استخراج KPIهای مالی برای هر پوشهٔ خروجی (قابل فراخوانی مجزا). |
| `scripts/audit/dataset_quality.py` | ممیزی داده با آستانه‌های سفارشی و گزینهٔ `--exit-on-fail`. |
| `scripts/audit/validate_artifacts.py` | اسکن یک پوشهٔ خروجی و تولید گزارش وجود فایل‌های استاندارد. |

---

## ساختار بستهٔ ارسال

`multi_stage_artifacts/` همان دایرکتوری است که برای داوران زیپ می‌شود:

- `summary.json` – وضعیت هر مرحله، مدت اجرا و مسیر لاگ.
- `full_suite/` – خروجی پایپلاین Hydra شامل `results/`، پوشهٔ `paper_outputs/`، ممیزی‌ها و گزارش‌های نهایی.
- `sb3_leadlag/` – نتایج آموزش Stable-Baselines3 روی محیط LeadLag (فایل‌های `metrics_timeseries.csv`، `summary.csv`، `model.zip` و مانفیست‌ها).
- `dopamine/` – آزمون سلامت Gymnasium 1.x به همراه لاگ‌ها و آمار تکرار.

در `full_suite/paper_outputs/` آمار مقاله قرار دارد:

- `all_metrics_raw.csv`، `psr_dsr_pvalues.csv`، `hac_sharpe_confidence_intervals.csv`، خروجی‌های SPA و نمودارها.
- `paper_results.md` و `paper_status.txt` برای گزارش مستقیم به هیئت داوران.

اگر `scripts/reproduce_all.sh` را محلی اجرا کنید همان ساختار زیر مسیرهای `RES` و `OUT` تولید می‌شود.

## ساختار مخزن

| مسیر | کاربرد |
|------|--------|
| `src/leadlag/main.py` | دیسپچر CLI اصلی که به عنوان `leadlag` اکسپورت شده است. |
| `src/leadlag/cli/` | منطق مشترک CLI (فرمت JSON/متن، مدیریت خطا، رجیستری فرمان‌ها). |
| `src/leadlag/driver/` | کشف سناریو، انتخاب، اجرا و تجمیع خروجی‌ها. |
| `src/leadlag/pipelines/` | لانچرهای Hydra قدیمی (`run_full_suite.py`, `run_ablation.py`). |
| `src/leadlag/configs/scenarios/` | سناریوهای بسته‌بندی‌شده که CLI آن‌ها را کشف می‌کند. |
| `research/meta_rl/` و `research/offline_rl/` | ابزارهای پژوهشی برای رژیم‌های مصنوعی و RL آفلاین. |
| `scripts/audit/` | اسکریپت‌های ممیزی (کیفیت داده، اسکن خروجی، نشت و واک‌آوروارْد). |
| `evaluation/` | محاسبهٔ KPI مالی و گزارش‌های مقایسه‌ای. |
| `docs/` | راهنماها و مستندات ممیزی؛ بخش استقرار مخصوص Kaggle. |
| `tests/` | مجموعهٔ PyTest برای صحت‌سنجی اجزای کلیدی. |

---

## توصیه‌های پایانی برای Kaggle

1. قبل از قطع اینترنت، نسخهٔ CPU از Torch و بسته‌های RL را نصب کنید یا با فلگ `--skip-optional-deps` سناریوهای RL را کنار بگذارید.
2. برای کاهش زمان اجرا، از `--include` یا `--max-scenarios` استفاده کنید و فقط سناریوهای دلخواه را روی GPU اجرا کنید.
3. خروجی‌ها (به‌ویژه `/reports` و `aggregate.json`) را بررسی کنید؛ در صورت حجیم بودن، فایل‌های میانی را فشرده یا حذف کنید.
4. گزارش نهایی PDF/Markdown در `/reports` تولید می‌شود؛ آن را همراه با گزارش ممیزی `audit/scan_report.md` آرشیو کنید.

با این چارچوب، تنها یک فرمان CLI کافی است تا کل آزمایش‌ها، ممیزی‌ها و گزارش‌ها در محیط Kaggle ساخته شوند و برای تحلیل نهایی آماده باشند. موفق باشید! 🎯

### لاگ اجرای پایپلاین
- پس از هر اجرای کامل، فایل خلاصه‌ای در مسیر `/logs/run_summary_<timestamp>.json` ذخیره می‌شود که شامل فرمان اجرا، پارامترها، وضعیت وابستگی‌ها، زمان شروع/پایان و مدت اجرا است.
- این اطلاعات بدون نیاز به سلول اضافی در نوت‌بوک نگهداری می‌شود و برای مقایسهٔ اجراها یا مستندسازی مفید است.

### نمودارهای مسیر موجودی
- پایپلاین کامل به صورت خودکار اسکریپت `reporting/plot_balance_history.py` را اجرا می‌کند و نمودارهای Portfolio Balance را برای تمام آزمایش‌ها می‌سازد.
- خروجی‌ها در `evaluation/plots/balance` قرار می‌گیرند؛ شامل:
  - `balance_all_runs.png`: همهٔ منحنی‌ها در یک نمودار.
  - `scenario/balance_<scenario>.png`: تفکیک بر اساس هر سناریو.
  - `method/balance_method_<name>.png`: تفکیک بر اساس روش (`signature`, `ccf_at_lag`, `dynamic`, `rl`, ...).
  - `lookback/balance_lookback_<value>.png`: تفکیک بر اساس طول پنجره.
- برای اجرای مستقل یا سفارشی‌سازی، دستور زیر قابل استفاده است:
  ```bash
  python reporting/plot_balance_history.py \
      --results-root /kaggle/working/full_suite \
      --out /kaggle/working/full_suite/evaluation/plots/balance \
      --start-balance 100000
  ```
- در صورت نیاز می‌توانید با `--max-lines` تعداد خطوط نمودار کلی را محدود کنید و سپس نمودارهای سفارشی (مثلاً فقط سناریوهای منتخب) بسازید.
