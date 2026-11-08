# UK University Institutional Data Request & Global 3,000-Market Architecture (Theorems 1–55)

## 1. Formal Academic Data Request List (For UK University Professor / Library Licence)

Most top UK research universities (via Wharton Research Data Services — **WRDS**, **LSEG/Refinitiv Workspace & Tick History**, and **Bloomberg Academic**) hold institutional subscriptions to the following gold-standard, survivorship-bias-free datasets that cost tens of thousands of dollars commercially:

| Priority | Database / Provider (Available via UK University / WRDS) | Exact Tables / Feeds to Request | Role in 3,000-Market SOTA-Apex Engine |
| :--- | :--- | :--- | :--- |
| **1 (Highest)** | **WRDS — CRSP & Compustat Global** | `crsp.dsf` (US Daily Stock File including delisted securities), `comp.g_secd` (Global Equities Daily), `crsp.msf` | Point-in-time, survivorship-bias-free prices, split/dividend adjustments, and shares outstanding for 1,800 Global Equities (Theorem 49 & 52). |
| **2 (Highest)** | **NYSE Daily TAQ / LSEG Tick History (Refinitiv)** | Consolidated 1-Minute (`1m`) OHLCV + Bid-Ask Spread & Depth bars for Global Equities, Futures (CME/ICE/Eurex), and FX | Feeds Theorem 55 (`compute_1min_subbar_rough_signature_aggregator`) to extract intra-hour 1-minute Order-2 Lévy Lead-Lag signatures and Bipower Jump Variation. |
| **3 (High)** | **Markit Securities Finance (via WRDS) / Ortex** | Daily Equity Short-Loan Fee (`IndicativeFee`), Utilisation %, and Lendable Quantity | Feeds Theorem 52 (`compute_borrow_aware_short_leg_optimizer`) to decide dynamically between Single-Stock Shorting vs. Index Futures Hedging. |
| **4 (High)** | **OptionMetrics IvyDB (US & Europe via WRDS)** | Implied Volatility Surface (`StdOptionPrice`), 25-Delta Put-Call Skew, Total Option Volume & Open Interest | Institutional Smart-Money Options & Gamma Exposure (`Y_exo` channels) leading underlying equities. |
| **5 (Medium)** | **Databento Academic / CME DataMine** | Continuous & Contract-Level 1-Minute / 1-Hour Futures OHLCV + Open Interest (ES, NQ, ZN, ZB, CL, GC, NG, HG, 6E, 6J) | Feeds Theorem 53 (`compute_oi_volume_proportional_futures_roll`) for arbitrage-free Open-Interest-weighted futures rolling. |
| **6 (Medium)** | **Thomson Reuters / LSEG 13F & Insider Filings (WRDS)** | Institutional Holdings (`s34`) and Corporate Insider Transactions (`insiders`) | Smart-Money Institutional Accumulation / Distribution exogenous channels (`Y_exo`). |

### Ready-to-Send Email / Message Template for Your UK Professor

```text
Dear Professor,

For the next empirical phase of our Rough-Path Lead-Lag & Multi-Agent Reinforcement Learning benchmark across the Global 3,000-Market Multi-Asset Universe (Theorems 1–55), we are expanding our strictly causal point-in-time panel to include 1-minute (1m) and 1-hour (1H) cross-asset bars alongside institutional smart-money conditioning channels.

If available through the university's WRDS, LSEG/Refinitiv Tick History, or Databento academic licences, could we obtain an export (Parquet or CSV, 2019–present) for the following modules?
1. Survivorship-bias-free Global & US Equities (CRSP / Compustat Global + 1-minute or hourly bars via LSEG Tick History / NYSE TAQ) for the top 1,800 liquid equities by rolling dollar turnover.
2. Markit Securities Finance (via WRDS) daily indicative stock-borrow fees and utilisation rates (for our Theorem 52 borrow-aware short-leg optimizer).
3. CME / ICE / Eurex front- and second-month Futures 1-minute/hourly bars with daily Open Interest (for our Theorem 53 OI-proportional roll across Indices, Rates, Energy, Metals, and FX).
4. OptionMetrics IvyDB daily implied volatility skew and put/call volume ratios (for our Bipartite Smart-Money exogenous tensor).

All free public macro and derivatives channels (FRED, CFTC Disaggregated COT, DefiLlama, and Binance Vision archives) are already integrated in our pipeline.

Best regards,
```
