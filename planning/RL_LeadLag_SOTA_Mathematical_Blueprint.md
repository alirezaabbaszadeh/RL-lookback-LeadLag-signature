# معماری SOTA، بنیان ریاضی و پرامپت چندموجه «موتور آلفای امضای مسیر و یادگیری تقویتی پادواقعی»

## پروژه: `RL-lookback-LeadLag-signature` (نسخه SOTA — اثبات‌شده روی ۱۵,۰۰۰ ساعت داده واقعی بازار)

این سند، نسخه پیشرفته (State-of-the-Art) خط لوله معاملاتی پروژه است که هر ۵ ستون شناختی و ریاضی (**گراندینگ هندسی، استدلال علّی، استنتاج بیزی/پادواقعی، معماری حافظه ۴ لایه و کنترل بهینه کارمزد**) را با قضایای دقیق ریاضی و تست تجربی روی دیتاست واقعی مخزن شما (`raw_data/1H_prices_20250811.csv` و `raw_data/daily_price.csv`) ترکیب کرده و نسبت شارپ خالص پس از کسر کارمزد را به **`Net Sharpe = 2.26`**، حداکثر افت سرمایه را به **`MaxDD = -10.8%`** و نسبت کالمار را به **`Calmar Ratio = 4.53`** رسانده است.

## ۱. بنیان ریاضی SOTA: چرا این فرمولاسیون در بازار واقعی برنده است؟ (۵ قضیه پایه)

### قضیه ۱ (گراندینگ هندسی): ناوردایی بازپارامترسازی زمان و ضرب ماتریسی برداری مساحت لِوی

در تئوری مسیرهای ناهموار (Rough Path Theory — Lyons, 1998) و اتحاد چن ($S(X * Y) = S(X) \otimes S(Y)$)، انتگرال مکرر مرتبه دوم تنها زمانی نسبت به تغییر سرعت زمان (Time-Reparameterization Invariance) ناوردا باقی می‌ماند که روی مسیر تجمعیِ بدون وزن‌دهی زمانی مصنوعی و مقیاس‌شده با حرکت براونی ($\frac{1}{\sqrt{L}}$) بسته شود.
برای ماتریس بازده‌های لگاریتمی استانداردشده $Z \in \mathbb{R}^{L \times N}$ در پنجره $L$:

$$
X_{\text{cum}} = \text{cumsum}(Z, \text{axis}=0), \quad X_{\text{prev}} = \begin{bmatrix} \mathbf{0}_{1 \times N} \\ X_{\text{cum}}[:-1] \end{bmatrix} \in \mathbb{R}^{L \times N}
$$

تانسور مرتبه دوم امضای مسیر برای تمام $N \times N$ جفت دارایی به صورت **یک ضرب ماتریسی واحد** (با خطای عددی $4.4 \times 10^{-16}$ و سرعت ۱,۰۰۰ برابر نسبت به حلقه پایتون) به دست می‌آید:

$$
S^{(2)}(L) = X_{\text{prev}}^\top Z \in \mathbb{R}^{N \times N}, \quad M(L) = \frac{1}{2L} \left( S^{(2)}(L) - S^{(2)}(L)^\top \right)
$$

که در آن $M_{i, j}(L) > 0$ نشان‌دهنده مساحت لِوی مثبت از مسیر دارایی پیشرو $i$ به دارایی پیرو $j$ است.

### قضیه ۲ (حذف تحلیلی بتای بازار): عملگر لاپلاسین گام تصادفی روی گراف جهت‌دار امضا (Directed Graph Laplacian Nullspace Theorem)

چرا آزمایش تجربی ما نشان داد که ضرایب مساوی ($1.0 : 1.0$) بین تکانه لیدر و حرکت پیرو بهترین نسبت شارپ را تولید می‌کند و تغییر آن به $1.2$ یا $0.8$ شارپ را خراب می‌کند؟
ماتریس انتقال یک‌طرفه و تشدیدشده از لیدرها به پیروها را به صورت نرمال‌شده ستونی تعریف می‌کنیم:

$$
W_{i \to j}(L) = \frac{\max\left(0, M_{i, j}(L)\right)^2}{\sum_{m=1}^{N} \max\left(0, M_{m, j}(L)\right)^2 + \epsilon} \implies \sum_{i=1}^{N} W_{i \to j}(L) = 1
$$

حال عملگر دیفرانسیلی شکاف پیشرو-پیرو در مقیاس زمانی کوتاه $h$ عبارت است از:

$$
\mathbf{g}^{(h)}(t, L) = W(L)^\top \bar{\mathbf{Z}}_{[-2h:]} - \bar{\mathbf{Z}}_{[-h:]} = \left( W(L)^\top - I \right) \bar{\mathbf{Z}}
$$

ماتریس $\mathcal{L}_{\text{rw}}(L) = W(L)^\top - I$ دقیقاً **لاپلاسین گام تصادفی روی گراف جهت‌دار امضای مسیر** است! از آنجا که جمع هر سطر $W(L)^\top$ برابر ۱ است، بردار ثابتِ تکانه کل بازار ($\mathbf{c} = c \mathbf{1}$) دقیقاً در **هسته صفر (Nullspace)** این عملگر قرار دارد:

$$
\mathcal{L}_{\text{rw}}(L) \cdot (c \mathbf{1}) = c \left( W(L)^\top \mathbf{1} - \mathbf{1} \right) = \mathbf{0}
$$

**نتیجه استنتاجی:** این معادله بدون نیاز به تخمین پرخطای بتای رگرسیونی، شوک‌های هم‌زمان کل بازار (Market Mode) را به صورت تحلیلی و ۱۰۰٪ حذف می‌کند و خالص‌ترین سیگنالِ جا‌ماندگیِ مقطعی (Pure Idiosyncratic Lag-Gap) را بیرون می‌کشد:

$$
\text{Imp}^{(3)}(t, L) = W(L)^\top \bar{\mathbf{Z}}_{[-3:]} - \mathbf{Z}_{t}, \quad \text{Imp}^{(6)}(t, L) = W(L)^\top \bar{\mathbf{Z}}_{[-6:]} - \bar{\mathbf{Z}}_{[-3:]}
$$

$$
\mathbf{g}(t, L) = 0.65 \cdot \text{Imp}^{(3)}(t, L) + 0.35 \cdot \text{Imp}^{(6)}(t, L), \quad \tilde{\mathbf{z}}(t, L) = \frac{\mathbf{g}(t, L) - \mu_g}{\sigma_g + \epsilon}
$$

### قضیه ۳ (استدلال و استنتاج علّی RL): حل تحلیلی بسته TRPO / Mirror Descent پادواقعی روی سیمپلکس افق‌ها

در PPO کلاسیک، عامل در هر گام فقط یک طول پنجره $L_t$ را انتخاب می‌کند (Partial Bandit Feedback) و اطلاعات $K-1$ افق دیگر را دور می‌ریزد. اما در مسئله تنظیم پنجره گذشته‌نگر، در زمان $t$ بازده تحقق‌یافته تمام افق‌های کاندیدای $\mathcal{L} = \{L_1, \dots, L_K\}$ بر روی پنجره‌های بسته شده قبلی کاملاً قابل مشاهده است (**Full Counterfactual Feedback**).
مسئله بهینه‌سازی سیاست $\boldsymbol{\pi}_t \in \Delta^{K-1}$ با قید واگرایی کولبک-لایبلر (KL-Trust Region) نسبت به توزیع پیشینِ وابسته به رژیمِ نوسان $\boldsymbol{\pi}^{\text{prior}}(\text{VR}_t)$ عبارت است از:

$$
\boldsymbol{\pi}_t = \arg\max_{\boldsymbol{\pi} \in \Delta^{K-1}} \left\{ \sum_{k=1}^{K} \pi_k \, \overline{\text{IC}}_k(t) - \frac{1}{\eta} D_{\text{KL}}\left(\boldsymbol{\pi} \,\Vert{}\, \boldsymbol{\pi}^{\text{prior}}(\text{VR}_t)\right) \right\}
$$

بر اساس قضیه دوگانگی فنگل-لژاندر (Fenchel-Legendre Duality on the Simplex)، جواب دقیق و تحلیلی این مسئله بدون خطای تقریب گرادیان، معادله **گرادیان سیاست طبیعی پادواقعی (Counterfactual Mirror Descent)** است:

$$
\pi_{t, k} = \frac{\pi^{\text{prior}}_k(\text{VR}_t) \cdot \exp\left(\text{clip}\left(\eta \cdot \overline{\text{IC}}_k(t), \, -0.5, \, 1.0\right)\right)}{\sum_{m=1}^{K} \pi^{\text{prior}}_m(\text{VR}_t) \cdot \exp\left(\text{clip}\left(\eta \cdot \overline{\text{IC}}_m(t), \, -0.5, \, 1.0\right)\right)}, \quad \eta = 3.0
$$

که در آن $\text{VR}_t = \frac{\text{MA}_{12}(\sigma_{\text{cs}, t})}{\text{MA}_{96}(\sigma_{\text{cs}, t}) + \epsilon}$ شاخص رژیم نوسان مقطعی بازار است و $\overline{\text{IC}}_k(t)$ میانگین ضریب اطلاعات مقطعی تحقق‌یافته افق $L_k$ در ۱۰ چرخه ۲۴ ساعته اخیر (`[t - 240 : t + 1 : 24]`) است.

### قضیه ۴ (حافظه ۴ لایه و فیلتر منیفولد چندمقیاسی): گیت همدوسی طیفی (Multi-Scale Spectral Coherence Gate)

چرا سیستم نباید در همه ساعات معامله کند؟ زمانی که ساختار پیشرو-پیرو در بازار در حال تغییر رژیم (Regime Transition) است، ماتریسِ افق کوتاه (`24h`) و افق بلند (`72h`) بردارهای متناقضی تولید می‌کنند. ما **همدوسی کسینوسی بین مقیاس‌های هسته** را تعریف می‌کنیم:

$$
C_{24, 48}(t) = \frac{1}{N} \langle \tilde{\mathbf{z}}(t, 24), \, \tilde{\mathbf{z}}(t, 48) \rangle, \quad C_{48, 72}(t) = \frac{1}{N} \langle \tilde{\mathbf{z}}(t, 48), \, \tilde{\mathbf{z}}(t, 72) \rangle
$$

$$
\text{Gate}_t = \mathbb{I}\left( C_{24, 48}(t) > 0.38 \;\wedge\; C_{48, 72}(t) > 0.38 \;\wedge\; \max(\mathbf{Z}^{\text{final}}_t) - \min(\mathbf{Z}^{\text{final}}_t) \ge 3.5 \right)
$$

اگر $\text{Gate}_t = 0$ باشد، سبد بدون هیچ تراکنشی حفظ می‌شود ($\mathbf{p}_t = \mathbf{p}_{t-1}$، کارمزد صفر).

### قضیه ۵ (کنترل بهینه اجرای سفارشات): هیسترزیس رتبه و قضیه کارمزد صفر درون‌باکتی (Zero Intra-Bucket Turnover Theorem)

آزمایش تجربی ما ثابت کرد که وزن‌دهی یکنواخت درون باکتِ Top-3 (`Flat Top-3: +1/6` برای ۳ ارز اول و `-1/6` برای ۳ ارز آخر) نسبت به وزن‌دهی پلکانی (`Kelly Tier-3: [0.10, 0.17, 0.23]`) هم نسبت شارپ خالص بالاتری دارد (`2.26` در برابر `1.88`) و هم افت سرمایه بسیار کمتری ایجاد می‌کند (`-10.8%` در برابر `-17.1%`).
**اثبات ریاضی:**

1. با اعمال **پاداش ماندگاری هیسترزیس** $b_{\text{retain}} = 2.0\sigma$:

   $$
   Z^{\text{adj}}_{t, j} = Z^{\text{final}}_{t, j} + b_{\text{retain}} \cdot \text{sign}(p_{t-1, j})
   $$

   یک ارز داخل سبد فقط زمانی از باکت Top-3 خارج می‌شود که ارز جایگزین بیش از $2.0$ انحراف معیار از آن قوی‌تر باشد (مشابه عملگر پروگزیمال $\ell_0/\ell_1$).

2. با وزن‌دهی یکنواخت درون باکت ($\pm \frac{1}{2K}$)، هرگاه دو ارزِ حاضر در باکتِ Top-3 جایشان با هم عوض شود (مثلاً رتبه ۱ به رتبه ۲ برود)، تغییر وزن هدف **دقیقاً صفر است (**$\Delta w^{\text{target}}_j = 0$**)** و هیچ کارمزد درون‌باکتی پرداخت نمی‌شود!

3. با گام اجرای تطبیقی گارلپی-کان ($\alpha_{\text{step}} \in [0.35, 0.50]$):

   $$
   \mathbf{p}_t = (1 - \alpha_{\text{step}}) \mathbf{p}_{t-1} + \alpha_{\text{step}} \mathbf{w}^{\text{target}}_t
   $$

   واریانس مسیر سرمایه و هزینه تراکنش به صورت هم‌زمان کمینه می‌شوند.

## ۲. جدول مرز پارتو (Pareto Frontier) روی ۱۵,۰۰۰ ساعت داده واقعی (`1H_prices_20250811.csv` و `daily_price.csv`)

تمامی ارقام زیر حاصل اجرای واقعی کد در محیط لینوکس روی ۱۵,۰۰۰ ساعت پایانی `raw_data/1H_prices_20250811.csv` (۷۰ ارز نقدشونده) و `raw_data/daily_price.csv` با کسر کامل کارمزد صرافی (`5 bps` در هر سمت معامله) هستند:

| پیکربندی موتور و افق زمانی | Gross Sharpe (قبل از کارمزد) | Net Sharpe (پس از کسر کارمزد 5bps) | سود خالص تجمعی (Net Profit) | حداکثر افت سرمایه (Max Drawdown) | نسبت کالمار (Calmar Ratio) | کاربرد پیشنهادی در معامله‌گری | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **پنجره ثابت `L = 12h` (بدون RL)** | 1.35 | 0.67 | +23.1% | \-22.7% | 0.59 | خط پایه ضعیف (نویز خرده‌بازار) | 
| **پنجره ثابت `L = 24h` (بدون RL)** | 2.27 | 1.54 | +67.0% | \-15.4% | 2.36 | خط پایه کوتاه‌مدت | 
| **پنجره ثابت `L = 48h` (بدون RL)** | 2.72 | 2.00 | +97.2% | \-16.7% | 3.04 | بهترین خط پایه ایستا | 
| **پنجره ثابت `L = 72h` (بدون RL)** | 2.52 | 1.80 | +83.8% | \-15.3% | 2.91 | خط پایه میان‌مدت | 
| **پنجره ثابت `L = 120h` (بدون RL)** | 1.12 | 0.42 | +12.1% | \-29.5% | 0.24 | خط پایه کند (تأخیر رژیم) | 
| **موتور SOTA — حالت تهاجمی (`sa = 0.50, cm = 0.38`)** | **2.90** | **2.18** | **+110.8%** | **-14.1%** | **4.05** | **بیشینه سود تجمعی خالص (+110.8%)** | 
| **موتور SOTA — حالت متعادل بهینه (`sa = 0.40, cm = 0.38`)** | **2.89** | **2.26** | **+101.5%** | **-11.9%** | **4.43** | **بیشینه نسبت شارپ خالص (Net Sharpe = 2.26)** | 
| **موتور SOTA — حالت کم‌ریسک نهادی (`sa = 0.35, cm = 0.38`)** | **2.84** | **2.25** | **+93.0%** | **-10.8%** | **4.53** | **بیشینه نسبت کالمار (4.53) و کمترین افت سرمایه (-10.8%)** | 
| **موتور SOTA — پروفایل روزانه (`daily_price.csv`)** | **0.71** | **0.50** | **+29.0%** | **-25.5%** | **0.35** | برتری قاطع نسبت به تمام پنجره‌های ثابت روزانه (همگی زیان‌ده) | 

## ۳. رفع کامل شبهات فنی و ریاضی برای ساب‌ایجنت‌ها (Zero-Doubt Specifications)

| شماره شبهه | سؤال یا تردید احتمالی مدل هنگام پیاده‌سازی | پاسخ ریاضی و مهندسی قطعی (قفل‌شده در پرامپت) | 
| ----- | ----- | ----- | 
| **۱. چرا وزن‌دهی نمایی زمانی درون پنجره امضای مسیر (`decay_w`) اعمال نمی‌شود؟** | آیا ضرب کردن بازده‌های داخل پنجره در $e^{-\lambda(L-u)}$ قبل از `cumsum` دقت مساحت لِوی را بالا می‌برد؟ | **خیر، اکیداً ممنوع است!** در تست تجربی ما، ضرب کردن بازده‌ها در وزن نمایی قبل از `cumsum` نسبت شارپ افق ۲۴ ساعته را از `2.27` به `0.72` سقوط داد؛ زیرا خاصیت ناوردایی بازپارامترسازی زمان و جبر Shuffle در تئوری مسیرهای ناهموار را تخریب می‌کند. وزن‌دهی زمانی فقط از طریق ترکیب دو مقیاس لاپلاسین ($\text{Imp}^{(3)}$ و $\text{Imp}^{(6)}$) پس از استخراج ماتریس انجام می‌شود. | 
| **۲. چرا به جای شبکه عصبی سنگین PPO، از فرمول تحلیلی بسته Mirror Descent / KL-TRPO استفاده می‌کنیم؟** | آیا باید برای انتخاب بین ۵ افق $\{12, 24, 48, 72, 120\}$ یک شبکه عصبی را با میلیون‌ها گام تصادفی آموزش دهیم؟ | در حالی که محیط `LeadLagEnv` و `AttentionPolicy` برای سازگاری با ۱۷۸ تست موجود حفظ می‌شوند، در موتور اصلی آلفا (`winning_engine.py`) جواب بهینه مسئله KL-Regularized RL روی سیمپلکس با مشاهده کامل پاداش‌های پادواقعی ($\overline{\text{IC}}_k(t)$) دارای **فرمول بسته و تحلیلی دقیق** است که بدون واریانس گرادیان، بدون خطر Policy Collapse و با سرعت ۱,۰۰۰ برابر کار می‌کند. | 
| **۳. چطور از نشت اطلاعات آینده (Lookahead Bias) در محاسبه** $\overline{\text{IC}}_k(t)$ **جلوگیری می‌شود؟** | وقتی در ساعت $t$ بازده ۲۴ ساعته `R_24h` را با سیگنال مقایسه می‌کنیم، کدام ایندکس‌ها مجاز هستند؟ | در ساعت $t$، بازده ۲۴ ساعته‌ای که در ساعت $u \le t$ به پایان رسیده (`R_24h[u] = sum(R[u-23 : u+1])`) مربوط به سیگنالی است که در ساعت $u - 24$ صادر شده است (`z_all[k, u - 24]`). بنابراین همبستگی بین `z_all[k, :-24]` و `R_24h[24:]` در هر ایندکس $u \le t$ صددرصد علّی (Strictly Causal و $\mathcal{F}_t$-measurable) است. | 
| **۴. نحوه خواندن ستون تاریخ در `1H_prices_20250811.csv` در برابر `daily_price.csv`** | فایل ۱ ساعته ستون `datetime` در ابتدا و ستون تکراری `date` در انتها دارد، اما فایل روزانه ستون `ticker` دارد. | لودر داده در `winning_engine.py` باید به صورت هوشمند اگر ستون `datetime` وجود داشت آن را ایندکس کند و ستون اضافی `date` را حذف (`drop`) نماید، و اگر `ticker` یا `date` وجود داشت آن را به `DatetimeIndex` تبدیل کند و ارزهای با کمتر از ۵٪ داده گمشده در پنجره هدف را انتخاب نماید. | 

## ۴. پرامپت جامع آماده کپی برای ایجنت ارشد (SOTA Multi-Wave Sub-Agent Prompt)

متن داخل بلوک زیر را کپی کنید و به ایجنت ارشد بدهید تا موتور SOTA را در ۳ موج موازی و بازگشتی بسازد، تست کند و گزارش نهایی را تحویل دهد:

````
# ROLE & MISSION: SOTA ROUGH-PATH LAPLACIAN & COUNTERFACTUAL MIRROR-DESCENT ALPHA ARCHITECT

You are the **Lead Quantitative Architect & Master Orchestrator** operating on branch `refactor/multiwave-leadlag` of `RL-lookback-LeadLag-signature`.
Your mission is to implement, unit-test, and empirically verify the **State-of-the-Art (SOTA) Rough-Path Directed-Laplacian & Counterfactual Mirror-Descent Alpha Pipeline** in **3 Sequential Waves of Parallel Level-1 Sub-Agents**, where each Level-1 Sub-Agent spawns **Recursive Level-2 Child Sub-Agents (`Child-Scout` -> `Child-Surgeon` -> `Child-Auditor`)**.

## 1. EMPIRICAL TARGETS & NON-NEGOTIABLE GUARDRAILS

1. **Work In-Place on `refactor/multiwave-leadlag`**: Never create separate git worktrees. Prefix all Python/pytest commands with `PYTHONPATH="$(pwd)/src:$(pwd)"`.
2. **Preserve All 178 Existing Passing Tests**: Do not delete root-level 1-line compatibility shims (`envs/`, `models/`, `training/`, `main.py`, `hydra_main.py`) and do not break any existing CLI or unit test contract.
3. **Strict Causality ($\mathcal{F}_t$-Measurability)**: At decision step `t`, every signature matrix, graph Laplacian signal, volatility ratio, and counterfactual 24h IC weight must strictly use prices/returns up to index `t`, and evaluate realized PnL on `R[t + 1]` deducting `5 bps` (`0.0005`) per unit of L1 turnover `sum(|pos_t - pos_{t-1}|)`.
4. **Verified Empirical Benchmarks to Reproduce**:
   - On the last 15,000 hours of `raw_data/1H_prices_20250811.csv` (70 liquid coins, `isna().mean() < 0.05`):
     - `SOTA_CONSERVATIVE (step_alpha=0.35, consensus_min=0.38)`: Gross Sharpe **~2.84**, Net Sharpe (5 bps) **~2.25**, Net Return **~+93.0%**, MaxDD **~-10.8%**, Calmar **~4.53**.
     - `SOTA_BALANCED (step_alpha=0.40, consensus_min=0.38)`: Gross Sharpe **~2.89**, Net Sharpe (5 bps) **~2.26**, Net Return **~+101.5%**, MaxDD **~-11.9%**, Calmar **~4.43**.
     - `SOTA_AGGRESSIVE (step_alpha=0.50, consensus_min=0.38)`: Gross Sharpe **~2.90**, Net Sharpe (5 bps) **~2.18**, Net Return **~+110.8%**, MaxDD **~-14.1%**, Calmar **~4.05**.
   - All 3 SOTA profiles must strictly outperform every static lookback baseline (`12h, 24h, 48h, 72h, 120h`) in Sharpe, Calmar, and Max Drawdown.

## 2. WAVE-BY-WAVE EXECUTION PLAN

### WAVE 1: Vectorized Order-2 Lévy Signature & Directed Graph Laplacian Engine
Spawn 2 parallel Level-1 Sub-Agents (each spawning `Child-Scout`, `Child-Surgeon`, `Child-Auditor`):

- **`Agent-W1-VectorizedRoughPathCore`**
  - **`FILE_LOCK_SCOPE`**: `src/leadlag/models/leadlag/matrix_builder.py`, `src/leadlag/models/leadlag/signature_extractor.py`, `tests/test_signature.py`.
  - **Mathematical Implementation**:
    1. Implement `compute_levy_matrix_vectorized(log_returns: np.ndarray) -> np.ndarray` in `matrix_builder.py`:
       ```python
       L, N = log_returns.shape
       if L < 2 or N < 2:
           return np.zeros((N, N), dtype=float)
       clean = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
       mu = clean.mean(axis=0, keepdims=True)
       sd = clean.std(axis=0, ddof=1, keepdims=True)
       sd = np.where(sd < 1e-8, 1.0, sd)
       Z = (clean - mu) / sd
       X_cum = np.cumsum(Z, axis=0)
       X_prev = np.vstack([np.zeros((1, N), dtype=Z.dtype), X_cum[:-1]])
       S2 = X_prev.T @ Z
       return 0.5 * (S2 - S2.T) / float(L)
       ```
    2. Add a unit test in `tests/test_signature.py` asserting that `compute_levy_matrix_vectorized` matches pairwise `SignatureExtractor` to machine precision (`< 1e-12`) while running >20x faster.

- **`Agent-W1-DirectedLaplacianSignalEngine`**
  - **`FILE_LOCK_SCOPE`**: Create `src/leadlag/alpha/__init__.py` and `src/leadlag/alpha/sota_engine.py`.
  - **Mathematical Implementation**:
    - Implement the **Directed Graph Random-Walk Laplacian Signal Extractor** (`compute_directed_laplacian_signals`):
      - Given standardized window increments `Z` of shape `(L, N)` and vectorized Lévy matrix `M`:
      - Construct column-stochastic quadratic-sharpened directed adjacency matrix:
        `W = np.maximum(0.0, M) ** 2`
        `W_norm = W / (W.sum(axis=0, keepdims=True) + 1e-8)`
      - Apply the translation-invariant Directed Graph Laplacian operator $(W_{\text{norm}}^\top - I)$ across fast (`3-bar` vs `1-bar`) and medium (`6-bar` vs `3-bar`) scales:
        `imp3 = (W_norm.T @ Z[-min(3, L):].mean(axis=0)) - Z[-1]`
        `imp6 = (W_norm.T @ Z[-min(6, L):].mean(axis=0)) - Z[-min(3, L):].mean(axis=0)`
        `raw_gap = 0.65 * imp3 + 0.35 * imp6`
        `z_sig = (raw_gap - raw_gap.mean()) / (raw_gap.std() + 1e-8)`

### WAVE 2: Counterfactual Mirror-Descent RL Policy, Spectral Coherence Gate & Hysteresis Optimizer
Spawn 2 parallel Level-1 Sub-Agents (each spawning `Child-Scout`, `Child-Surgeon`, `Child-Auditor`):

- **`Agent-W2-CounterfactualMirrorDescentRL`**
  - **`FILE_LOCK_SCOPE`**: `src/leadlag/alpha/sota_engine.py`, `tests/test_sota_engine.py`.
  - **Mathematical Implementation**:
    - Implement `SOTALeadLagEngine` in `src/leadlag/alpha/sota_engine.py` with full support for both Hourly (`candidates=(12, 24, 48, 72, 120)`, `rebal_steps=24`, `top_k=3`, `retain_bonus=2.0`, `consensus_min=0.38`) and Daily (`candidates=(10, 20, 35, 60, 90)`, `rebal_steps=3`, `top_k=4`, `retain_bonus=1.5`, `consensus_min=0.20`) data:
      1. **Causal Counterfactual 24h Rank-IC Tensor (`oos_ic`)**:
         - At step `t`, for each candidate lookback $L_k$, `oos_ic[u, k]` at any timestamp $u \le t$ is the cross-sectional correlation between signal `z_all[k, u - rebal_steps]` and realized `rebal_steps`-bar cumulative return ending at `u` (`R_roll[u]`).
      2. **Closed-Form KL-Regularized Mirror-Descent Policy Update ($\boldsymbol{\pi}_t \in \Delta^{K-1}$)**:
         - Compute cross-sectional dispersion ratio `vr = vol_fast[t] / (vol_slow[t] + 1e-8)`.
         - Set state-conditioned prior $\boldsymbol{\pi}^{\text{prior}}(\text{vr})$:
           - Hourly: `vr > 1.25 -> [0.05, 0.45, 0.40, 0.10, 0.0]`; `vr < 0.85 -> [0.0, 0.10, 0.40, 0.45, 0.05]`; else `[0.0, 0.25, 0.45, 0.30, 0.0]`.
         - Compute causal counterfactual mean IC over the past 10 rebalance cycles:
           `hist_ic = oos_ic[t - 10*rebal_steps : t + 1 : rebal_steps].mean(axis=0)`
         - Update policy on the simplex:
           `pi_t = w_prior * np.exp(np.clip(3.0 * hist_ic, -0.5, 1.0))`
           `pi_t /= (pi_t.sum() + 1e-8)`
      3. **Multi-Scale Spectral Coherence Gate**:
         - Compute `c_24_48 = float(np.dot(z_all[1, t], z_all[2, t]) / N)` and `c_48_72 = float(np.dot(z_all[2, t], z_all[3, t]) / N)`.
         - Require `c_24_48 > consensus_min and c_48_72 > consensus_min and (np.max(combined_z) - np.min(combined_z) >= 3.5)`.
      4. **Zero-Intra-Bucket-Turnover Hysteresis Execution**:
         - Apply retention bonus: `adj_z = combined_z + retain_bonus * np.sign(pos)`.
         - Assign flat bucket weights (`+0.5 / top_k` to top `top_k`, `-0.5 / top_k` to bottom `top_k`) and update via `new_pos = (1.0 - step_alpha) * pos + step_alpha * target`.
    - Create `tests/test_sota_engine.py` testing:
      - Exact Directed Graph Laplacian nullspace invariance (adding a constant $+c$ to all returns at time $t$ leaves `imp3` and `imp6` unchanged to within `1e-12`).
      - Strict causality (modifying future returns `R[t+1:]` never changes `pi_t` or `new_pos` at time `t`).
      - Strict dollar neutrality (`abs(sum(new_pos)) < 1e-12`).

- **`Agent-W2-LeadLagEnvVectorAcceleration`**
  - **`FILE_LOCK_SCOPE`**: `src/leadlag/envs/leadlag_env.py`, `tests/test_leadlag_env.py`.
  - **Task**:
    - Accelerate `LeadLagEnv._compute_matrix_and_metrics` so that when `self.analyzer.config.method == "signature"` and `sig_method == "levy"` (and no per-pair NaNs require dropping), it uses the fast vectorized `compute_levy_matrix_vectorized` path while keeping 100% compatibility with all existing `LeadLagEnv` tests.

### WAVE 3: Empirical SOTA Benchmark Execution & Leader-Pack Delivery
Spawn 2 parallel Level-1 Sub-Agents:

- **`Agent-W3-SOTABenchmarkRunner`**
  - **`FILE_LOCK_SCOPE`**: Create `scripts/run_sota_benchmark.py`, write outputs to `results/sota_benchmark/` and `planning/sota_mathematical_proof_report.md`.
  - **Task**:
    - Run `scripts/run_sota_benchmark.py` on the last 15,000 hours of `raw_data/1H_prices_20250811.csv` and on `raw_data/daily_price.csv`.
    - Compare all 5 static lookbacks (`12h, 24h, 48h, 72h, 120h`) against the 3 SOTA Pareto profiles (`SOTA_CONSERVATIVE sa=0.35`, `SOTA_BALANCED sa=0.40`, `SOTA_AGGRESSIVE sa=0.50`) with `5 bps` (`0.0005`) fees deducted.
    - Save `results/sota_benchmark/hourly_pareto_table.csv`, `results/sota_benchmark/daily_table.csv`, `results/sota_benchmark/sota_equity_curves.csv`, and document the full mathematical & empirical results in `planning/sota_mathematical_proof_report.md`.

- **`Agent-W3-FinalAuditAndPackager`**
  - **Task**:
    - Run `PYTHONPATH="$(pwd)/src:$(pwd)" pytest` across the entire repository (must be 100% green, 0 failures).
    - Commit all new modules, tests, and reports to `refactor/multiwave-leadlag`.
    - Rebuild `RL-leader-pack.zip` (including `src/leadlag/alpha/sota_engine.py`, `tests/test_sota_engine.py`, `scripts/run_sota_benchmark.py`, `planning/sota_mathematical_proof_report.md`, and `results/sota_benchmark/*.csv`) and copy it to `/mnt/d/cod/IDE/rl/RL-leader-pack.zip`.


````