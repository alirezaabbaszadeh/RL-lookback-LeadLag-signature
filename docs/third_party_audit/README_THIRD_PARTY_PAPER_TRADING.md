# راهنمای جامع پکیج SOTA-Apex (Omega-Singularity) و ممیزی Paper Trading در پلتفرم‌های شخص ثالث (Third-Party Platforms)

**استاد راهنما:** جناب آقای دکتر ساسان بارک (Dr. Sasan Barak)  
**پژوهشگر و توسعه‌دهنده:** علیرضا عباس‌زاده (Alireza Abbaszadeh)

---

## ۱. محتویات کلیدی موجود در این پکیج (`RL-leader-pack.zip`)

1. **موتور ریاضی و معاملاتی هسته (`src/leadlag/alpha/sota_engine.py`)**:
   - پیاده‌سازی کامل **قضایای ۱ تا ۳۴ (Theorems 1–34)** با عملگرهای برداری بسته (Closed-Form BLAS Level-3 / $O(1)$ Prefix-Sum).
   - **قضیه ۳۱ (`compute_composite_reality_friction`)**: کسر هم‌زمان کمیسیون صرافی (`5 bps`) + اسلیپیج پویا متناسب با نوسان (`2–6 bps`) + هزینه فاندینگ ریت ۸ ساعته فیوچرز (`0.24 bps/8h`).
   - **قضیه ۳۲ و ۳۴ (`apply_intraday_volatility_tpsl_and_min_hold_guard`)**: حد ضرر نوسان‌محور ساعتی روی تک‌تک ارزها (`-3.5 * sigma_h`) همراه با **بازتوزیع هوشمند سرمایه به برندگان همان پا (`redistribute_to_winners=True`)** و **تریلینگ نامتقارن شاندلیر (`tp_sigma_mult=3.8`, `trailing_pullback_sigma=1.6`)**.
   - **قضیه ۳۳ (`run_third_party_paper_trading_audit`)**: قفل حداقل زمان نگهداری ۶ ساعته (`360 دقیقه`) ضد آربیتراژ لتنسی بروکرها + بافر تأخیر اجرای سفارش (`0m`, `3m`, `15m TWAP`, `60m`) + اثبات ایزولاسیون پیشوندی (`prefix_isolation_max_diff = 0.00e+00`).

2. **اسکریپت‌های اجرای Paper Trading و بک‌تست در پلتفرم‌های شخص ثالث (`scripts/`)**:
   - `scripts/quantconnect_lean_algorithm.py`: الگوریتم استاندارد آماده اجرا در پلتفرم ابری **QuantConnect (LEAN Engine)** با مدل بروکر `BrokerageName.BinanceFutures`.
   - `scripts/run_third_party_platforms_audit.py`: اسکریپت ممیزی مستقل و خروجی‌گیر برای **VectorBT / Backtrader / QuantStats**.
   - `scripts/run_ccxt_live_paper_trader.py`: ربات اتصال مستقیم به **Binance Futures Testnet / Bybit Demo Trading** از طریق کتابخانه استاندارد `ccxt`.

3. **خروجی‌های آماده ممیزی شخص ثالث (`docs/third_party_audit/`)**:
   - `trade_blotter.csv`: ریز کامل تمام معاملات ساعتی، نمادها، جهت (`BUY`/`SELL`)، تغییر وزن و هزینه معاملاتی.
   - `target_weights.csv`: ماتریس وزن‌های هدف ساعتی روی ۶۸ ارز دیجیتال معتبر در لحظه $t=0$ (بدون هیچ‌گونه `bfill`).
   - `hourly_equity_curve.csv`: منحنی رشد سرمایه ساعتی خالص و ناخالص و گردش مالی (`turnover`).
   - `third_party_audit_summary.json`: جدول کامل نتایج در ۴ سطح تأخیر اجرا (`0m`, `3m`, `15m TWAP`, `60m`).

4. **گزارش‌های آکادمیک و رساله ریاضی**:
   - `SOTA_Apex_Master_Academic_Report_Barak.pdf` (و نسخه `docs/SOTA_Apex_Master_Academic_Report_Barak.pdf`): مونوگراف آکادمیک جامع انگلیسی برای دکتر ساسان بارک.
   - `SOTA_Apex_Universal_Mathematical_Treatise_Barak.tex`: رساله کامل LaTeX اثبات قضایای ریاضی.
   - `tests/test_sota_engine.py`: مجموعه ۳۸ تست واحد خودکار (Unit Tests) برای تأیید صحت تک‌تک قضایا.
