# ==============================================================================
# QuantConnect Research Notebook: Spearman Rank-IC Diagnostic (research.ipynb)
# Purpose: Measure Spearman Rank-IC across forward horizons (1d, 5d, 10d, 20d)
#          to validate signal polarity, information ratio, and lookback horizon.
# ==============================================================================

from AlgorithmImports import *
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# 1. Initialize QuantBook
qb = QuantBook()

# 2. Add Equities at Daily Resolution (18 Homogeneous Assets)
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    "JPM", "XOM", "AMD", "JNJ", "V", "BAC",
    "SPY", "QQQ", "IWM", "EFA", "EEM"
]
symbols = [qb.AddEquity(t, Resolution.Daily).Symbol for t in tickers]

# 3. Fetch Historical Data (2010 to 2026)
start = datetime(2010, 1, 1)
end = datetime(2026, 8, 1)
history = qb.History(symbols, start, end, Resolution.Daily)

# 4. Extract Close Prices and Resample
closes = history['close'].unstack(level=0).ffill().dropna(how='all')
print(f"Data loaded successfully: shape = {closes.shape}, columns = {len(closes.columns)}")

# 5. Compute Log Returns
P = closes.values
T, N = P.shape
dates = closes.index
LR = np.zeros_like(P)
LR[1:] = np.clip(np.log(np.maximum(P[1:], 1e-8) / np.maximum(P[:-1], 1e-8)), -0.30, 0.30)

# 6. Directed Laplacian Signal Function (Theorems 1 & 2)
def compute_directed_laplacian_signal(win_log_returns):
    L, N = win_log_returns.shape
    if L < 4 or N < 2:
        return np.zeros(N, dtype=float)
    clean = np.nan_to_num(win_log_returns, nan=0.0, posinf=0.0, neginf=0.0)
    mu = clean.mean(axis=0, keepdims=True)
    sd = clean.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (clean - mu) / sd

    X_cum = np.cumsum(Z, axis=0)
    X_prev = np.vstack([np.zeros((1, N), dtype=Z.dtype), X_cum[:-1]])
    S2 = X_prev.T @ Z
    M = 0.5 * (S2 - S2.T) / float(L)

    W = np.maximum(0.0, M)
    W *= W
    col_sums = W.sum(axis=0)
    inv_col = 1.0 / np.maximum(col_sums, 1e-12)

    h3 = min(3, L)
    h6 = min(6, L)
    z_bar3 = Z[-h3:].mean(axis=0)
    z_bar6 = Z[-h6:].mean(axis=0)

    wt_3 = z_bar3 @ W
    wt_6 = z_bar6 @ W
    imp3 = np.where(col_sums < 1e-12, z_bar3, wt_3 * inv_col) - Z[-1]
    imp6 = np.where(col_sums < 1e-12, z_bar6, wt_6 * inv_col) - z_bar3
    raw_gap = 0.65 * imp3 + 0.35 * imp6

    gap_std = float(raw_gap.std())
    if gap_std < 1e-12:
        return np.zeros(N, dtype=float)
    return (raw_gap - float(raw_gap.mean())) / gap_std

# 7. Evaluate Rank-IC across Horizons & Forward Returns
horizons = [5, 10, 21, 42, 63]
forward_horizons = [1, 5, 10, 20]
rebal_stride = 5  # Weekly evaluation grid

print("\n" + "="*70)
print("SPEARMAN RANK-IC EVALUATION TABLE (2010 - 2026)")
print("="*70)

summary_rows = []
for fwd in forward_horizons:
    for H in horizons:
        ics = []
        for t in range(max(horizons) + 1, T - fwd, rebal_stride):
            win = LR[t - H + 1 : t + 1]
            z_s = compute_directed_laplacian_signal(win)
            fwd_ret = (P[t + fwd] - P[t]) / np.maximum(P[t], 1e-8)
            if np.std(z_s) > 1e-6 and np.std(fwd_ret) > 1e-6:
                corr, _ = spearmanr(z_s, fwd_ret)
                if not np.isnan(corr):
                    ics.append(corr)
        arr = np.array(ics)
        mean_ic = float(np.mean(arr)) if len(arr) > 0 else 0.0
        std_ic = float(np.std(arr)) if len(arr) > 0 else 1.0
        ir = mean_ic / (std_ic + 1e-8)
        t_stat = ir * np.sqrt(len(arr)) if len(arr) > 0 else 0.0
        win_rate = float(np.mean(arr > 0)) if len(arr) > 0 else 0.0
        summary_rows.append({
            "Forward Return": f"{fwd}d",
            "Lookback Horizon": f"{H}d",
            "Mean IC": f"{mean_ic:+.4f}",
            "IR": f"{ir:+.3f}",
            "t-stat": f"{t_stat:+.2f}",
            "Win Rate": f"{win_rate*100:.1f}%",
            "Observations": len(arr)
        })

df_ic = pd.DataFrame(summary_rows)
print(df_ic.to_string(index=False))

# 8. Cumulative IC Plot for Horizon 21 and 63 (Forward 1d)
plt.figure(figsize=(12, 5))
for H in [21, 63]:
    ic_series = []
    dt_series = []
    for t in range(max(horizons) + 1, T - 1, rebal_stride):
        win = LR[t - H + 1 : t + 1]
        z_s = compute_directed_laplacian_signal(win)
        fwd_ret = (P[t + 1] - P[t]) / np.maximum(P[t], 1e-8)
        if np.std(z_s) > 1e-6 and np.std(fwd_ret) > 1e-6:
            corr, _ = spearmanr(z_s, fwd_ret)
            if not np.isnan(corr):
                ic_series.append(corr)
                dt_series.append(dates[t])
    plt.plot(dt_series, np.cumsum(ic_series), label=f"Lookback {H}d (Mean IC={np.mean(ic_series):+.4f})")

plt.title("Cumulative Spearman Rank-IC on 18 US Equities (2010 - 2026)")
plt.xlabel("Date")
plt.ylabel("Cumulative IC")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
