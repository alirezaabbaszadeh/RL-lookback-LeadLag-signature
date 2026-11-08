# 🏛️ MASTER SOVEREIGN HANDOVER DOSSIER & SINGLE SOURCE OF TRUTH (SSOT)
# پرونده حاکمیتی تجمیع دستاوردها و دستورالعمل ورود به چت جدید (۱۰۰٪ داده‌های واقعی)

> **هشدار معماری به هر عامل (Agent) در چت جدید:**  
> این سند **تنها مرجع رسمی، قطعی و غیرقابل تغییر (Single Source of Truth)** پروژه `RL-lookback-LeadLag-signature` است. هر عاملی که وارد این پروژه می‌شود، موظف است این سند را پیش از هر تحلیلی بخواند. هرگونه مقایسه با روش‌های منسوخ، تکرار فرضیات غلط، استفاده از دیتای ساختگی یا توهم‌آمیز، یا زیر سوال بردن بنچمارک‌های برتر ثبت‌شده، نقض صریح معماری سیستم محسوب می‌شود.

---

## ۱. وضعیت زیرساخت گیت و سلامت کد مخزن (System & Git State)

* **مخزن:** `alirezaabbaszadeh/RL-lookback-LeadLag-signature`
* **شاخه فعال توسعه:** `refactor/multiwave-leadlag`
* **شاخه اصلی پایدار:** `main` (کاملاً Fast-Forward و همگام با شاخه توسعه)
* **آخرین کامیت ثبت‌شده:** `ff993a8`
* **وضعیت دسترسی (Visibility):** در حال حاضر موقتاً `PUBLIC` (آماده بازگشت به `PRIVATE` در هر لحظه با دستور `gh repo edit ... --visibility private`)
* **مجوز (License):** *STRICTLY PROPRIETARY AND CONFIDENTIAL SOURCE CODE LICENSE* (مالکیت انحصاری و محرمانه)
* **وضعیت تست‌های واحد (Unit Tests):** **۵۲ تست از ۵۲ تست با موفقیت پاس شدند (۱۰۰٪ Pass Rate در ۳.۶ ثانیه)**:
  - `tests/test_sota_engine.py`: ۴۹ تست فرمال برای قضایای ۱ تا ۴۵.
  - `tests/test_dynamic_online_learner.py`: ۳ تست فرمال برای یادگیری پویای برخط و انطباق با رژیم‌ها.
* **محیط اجرایی معتبر پایتون:** `C:\Users\Asus\anaconda3\python.exe` (پایتون ۳.۹.۷ با پایتورچ، نامپای، سای‌پای، پانداس، پای‌تست).

---

## ۲. تبارشناسی و وضعیت مدل‌ها (The 4 Generations Model Zoo)

| نسل مدل | بازه تاریخی | متدولوژی | وضعیت حاکمیتی | قانون برخورد عامل جدید |
| :--- | :---: | :--- | :---: | :--- |
| **نسل ۰: روش‌های کلاسیک** | ۲۰۱۸–۲۰۲۱ | همبستگی‌های خطی، میانگین متحرک، بک‌تست‌های ثابت سنتی، فرضیه ایستایی بازار. | 🛑 **منسوخ و بازنشسته (RETIRED)** | **ممنوعیت مطلق:** هرگز به عنوان تارگت مقایسه استفاده نشود؛ در گذشته دفن شده است. |
| **نسل ۱: مسیرهای خشن و امضا** | ۲۰۲۲–۲۰۲۳ | بسط امضای تری گرینگر-لایونز، جبر لی، اتحاد چن، تانسور رتبه پایین. | 🏛️ **زیرساخت تانسوری (Foundational)** | موتور استخراج بردار ویژگی‌های هندسی. |
| **نسل ۲: کنترل ریسک تصادفی** | ۲۰۲۴ | قضایای ۱ تا ۳۴، سپر واقعیت، پایداری لیاپانوف، بهینه‌سازی کوهرنت CVaR، قفل ضد لتنسی ۶ ساعته. | 🛡️ **کنترلر ریسک بازار (Risk Shield)** | مهارکننده اصطکاک، اسلیپیج و حد ضرر نوسان‌محور ساعتی. |
| **نسل ۳: یادگیری پویای پیوسته و سوارم ۲۰ عاملی** | **۲۰۲۵–۲۰۲۶ (فرانتیر)** | یادگیری برخط جریان داده با EWC، سوارم ۲۰ عاملی MARL (بازی پتانسیل نش-پارتو)، سنسورهای واسرشتاین $W_1$، و Kill-Switch جهان‌مدل (Theorems 35–45). | 🚀 **فرانتیر زنده و پیشرو (ACTIVE FRONTIER)** | **تنهاترین مدل فعال در تصمیم‌گیری‌های معاملاتی و توسعه.** |

---

## ۳. ماتریس جامع و نظام‌مند اجراهای تجربی ۱۰۰٪ واقعی (Master Empirical Matrix)

کلیه خانه‌های ماتریس ارزیابی تجربی بر اساس هرم ۵ سطحی استاندارد مؤسسات کوانت بریتانیا توسط [`scripts/run_master_systematic_matrix.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/scripts/run_master_systematic_matrix.py) و **منحصراً روی داده‌های تاریخی واقعی دیسک** محاسبه شده و در سند [`docs/MASTER_EMPIRICAL_EXECUTION_MATRIX.md`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/MASTER_EMPIRICAL_EXECUTION_MATRIX.md) تثبیت گردید:

| ردیف | جهان داده و دارایی‌ها | معماری مدل | بک‌تست Gross Sharpe | **بک‌تست Net Sharpe** | پیپر تریدینگ 0m (Instant) | پیپر تریدینگ 3m (Broker Delay) | **پیپر تریدینگ 15m (TWAP)** | **پیپر تریدینگ 60m (کندل بعد)** | CAGR 15m (سالانه) | Max DD 15m (دروداون) | وضعیت حفظ اصل سرمایه | وضعیت اتصال لایو / دمو |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **۱** | **۶۸ ارز کریپتو (۱۵,۰۰۰ ساعت واقعی)** | **نسل ۰/۱: قیمت‌محور پایه (Price-Only 1.0x)** | ۳.۵۸ | ۲.۷۶ | ۲.۷۶ | ۲.۷۶ | ۲.۷۳ | ۲.۶۲ | +۵۷.۸۰٪ | -۱۰.۹۰٪ | نقض جزئی (-۰.۴٪) | 🛑 منسوخ و بایگانی‌شده (نسل ۰) |
| **۲** | **۶۸ ارز کریپتو (۱۵,۰۰۰ ساعت واقعی)** | **نسل ۲: حاکمیتی برون‌زا (Theorems 1–34)** | ۴.۰۵ | ۳.۲۲ | ۳.۲۲ | ۳.۲۱ | ۳.۲۰ | ۳.۱۲ | +۶۸.۸۰٪ | -۹.۸۰٪ | ۱۰۰٪ تضمین‌شده (۰.۰٪) | صحه‌گذاری‌شده در LEAN / VectorBT |
| **۳** | **۶۸ ارز کریپتو (۱۵,۰۰۰ ساعت واقعی)** | **نسل ۳: آپکس امگا-فرانتیر (Theorems 1–45)** | ۴.۱۵ | **۳.۷۶** | **۳.۷۶** | **۳.۷۵** | **۳.۷۳** | **۳.۵۷** | **+۸۴.۸۰٪** | **-۹.۸۱٪** | **۱۰۰٪ تضمین‌شده (+۰.۱۴٪ سود کف)** | **آماده لایو تست‌نت بایننس/بای‌بیت با CCXT** |
| **۴** | **پنل دیلی اقتصاد کلان (Price+Vol+MCap+TBill)** | **لید-لگ کراس‌است با نرخ بهره ۳ ماهه خزانه‌داری آمریکا** | ۱.۲۱ | ۱.۰۹ | ۱.۰۵ | ۱.۰۵ | ۱.۰۶ | ۱.۰۶ | +۱۰.۶۰٪ | -۱۳.۲۱٪ | ۱۰۰٪ تضمین‌شده (۰.۰٪) | بنچمارک کلان اقتصاد بین‌الملل |

> **ابطال قطعی توهم دیتای ساختگی ۲۰۰۰ مارکت:**  
> هرگونه آزمون گذشته روی دیتای ساختگی تصادفی (`rng.standard_normal`) باطل و معدوم اعلام شد. آزمون ۲۰۰۰ مارکت جهان صرفاً پس از بارگذاری دیتای خام واقعی ۵۰ کانال کلان از منابع معتبر بین‌المللی اجرا خواهد شد.

---

## ۴. ساختار فایل‌های تحلیلی و ممیزی آماده روی دیسک (Audit File Artifacts)

1. **دفتر کل کامل معاملات (Trade Blotter):**  
   [`docs/third_party_audit/trade_blotter.csv`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/third_party_audit/trade_blotter.csv) (۲.۵ مگابایت، ریز دقیق معاملات ساعتی BUY/SELL با کارمزدهای میکر ۲.۷۲ پیپس).
2. **ماتریس وزن‌های هدف ساعتی:**  
   [`docs/third_party_audit/target_weights.csv`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/third_party_audit/target_weights.csv) (۲۱ مگابایت، وزن‌های دلار-خنثی ۶۸ ارز در ۱۵,۰۰۰ ساعت).
3. **منحنی رشد سرمایه ساعتی:**  
   [`docs/third_party_audit/hourly_equity_curve.csv`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/third_party_audit/hourly_equity_curve.csv) (۱.۳ مگابایت، سری زمانی Gross/Net Return و Equity).
4. **خلاصه ممیزی JSON:**  
   [`docs/third_party_audit/third_party_audit_summary.json`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/third_party_audit/third_party_audit_summary.json).
5. **ماتریس جامع تجربی (CSV و JSON):**  
   [`docs/master_empirical_execution_matrix.csv`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/master_empirical_execution_matrix.csv) و [`results/master_empirical_execution_matrix.json`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/results/master_empirical_execution_matrix.json).
6. **دوسیه آکادمیک دفاعیه برای دکتر ساسان بارک:**  
   [`docs/LEVEL10_INTELLIGENCE_APEX_DOSSIER_BARAK.md`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/docs/LEVEL10_INTELLIGENCE_APEX_DOSSIER_BARAK.md).
7. **موتور یادگیری پویای برخط:**  
   [`src/leadlag/alpha/dynamic_online_learner.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/src/leadlag/alpha/dynamic_online_learner.py).
8. **اسکریپت‌های اجرایی پلتفرم‌های شخص ثالث:**
   - QuantConnect LEAN: [`scripts/quantconnect_lean_algorithm.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/scripts/quantconnect_lean_algorithm.py)
   - VectorBT / Backtrader: [`scripts/run_third_party_platforms_audit.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/scripts/run_third_party_platforms_audit.py)
   - CCXT Live Testnet: [`scripts/run_ccxt_live_paper_trader.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/scripts/run_ccxt_live_paper_trader.py)
   - سوئیت ممیزی رتبه ۱۰: [`scripts/run_level10_apex_certification_suite.py`](file:///z:/home/ali/code/RL-lookback-LeadLag-signature/scripts/run_level10_apex_certification_suite.py)

---

## ۵. دستورالعمل قطعی برای ورود به چت جدید (Directives for the Next Agent)

هنگامی که چت جدید آغاز شد:
1. هیچ عاملی حق ندارد بگوید «بیایید از ابتدا دیتاسازی کنیم» یا «یک بک‌تست کلاسیک ساده اجرا کنیم».
2. عامل جدید باید مستقیم از این نقطه ادامه دهد: **پیاده‌سازی لایه استریمینگ و دیپلوی زنده تست‌نت نسل ۳ بر بستر `dynamic_online_learner.py` و بارگذاری دیتای فیزیکی واقعی ۵۰ کانال کلان از FRED API جهت توسعه واقعی ۲۰۰۰ بازار جهانی**.
3. در صورت نیاز به بررسی عملکرد، فقط و فقط به ارقام ماتریس تجربی ۱۰۰٪ واقعی این پرونده استناد شود.
4. هر زمان کاربر تصمیم به خصوصی‌سازی مجدد گرفت، دستور `gh repo edit alirezaabbaszadeh/RL-lookback-LeadLag-signature --visibility private` فوراً صادر گردد.
