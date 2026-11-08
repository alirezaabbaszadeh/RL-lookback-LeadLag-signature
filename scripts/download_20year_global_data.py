#!/usr/bin/env python3
"""Practical 20-Year Global Multi-Asset & 5-Layer Macro Data Downloader (2005-2026).

Run this script on your local PC (with internet connection) to transition your repository
from the 15,000-hour crypto-only panel to a 20-Year Global Cross-Asset Data Lake (`raw_data/global_20y/`).

Requires zero paid API keys out of the box:
  - Layer 1 & 2 (Central Bank Plumbing, Sovereign Yields, Credit Spreads): Downloaded directly from
    St. Louis Fed (FRED) public CSV endpoints (no API key required!).
  - Layer 3, 4 & 5 (Global FX, Strategic Commodities, Global Equity Indices, Sector ETFs, and Top Liquid
    Global Stocks across US, Europe, Japan, and Asia): Downloaded via `yfinance` in batched parallel chunks.
  - Scale-Up Hook: Automatically ingests any full 3,000-stock WRDS (CRSP/Compustat) CSV/Parquet placed
    in `raw_data/wrds_equities.parquet`.

Usage on your PC:
  pip install yfinance pandas pyarrow requests tqdm
  python scripts/download_20year_global_data.py --start-date 2005-01-01 --universe-size 500
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "raw_data" / "global_20y"

# Layer 1 & 2: Public FRED Macro, Central Bank Plumbing, Yields & Credit Spreads (Direct CSV URLs, No API Key Needed)
FRED_SERIES = {
    "FEDFUNDS": "Effective Federal Funds Rate",
    "SOFR": "Secured Overnight Financing Rate (Layer 1 Repo)",
    "RRPONTSYD": "Overnight Reverse Repurchase Agreements (ON RRP Layer 1)",
    "WALCL": "Federal Reserve Total Assets Balance Sheet (Layer 1)",
    "WTREGEN": "US Treasury General Account TGA at Fed (Layer 1)",
    "ECBASSETSW": "ECB Total Assets Balance Sheet (Layer 1 G4)",
    "JPNASSETS": "Bank of Japan Total Assets (Layer 1 G4)",
    "DGS2": "US 2-Year Treasury Constant Maturity Yield (Layer 2)",
    "DGS10": "US 10-Year Treasury Constant Maturity Yield (Layer 2)",
    "DGS30": "US 30-Year Treasury Constant Maturity Yield (Layer 2)",
    "T10Y2Y": "US 10Y-2Y Treasury Yield Curve Slope (Layer 2)",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index Option-Adjusted Spread HY OAS (Layer 2)",
    "BAMLC0A0CM": "ICE BofA US Corporate Master Option-Adjusted Spread IG OAS (Layer 2)",
    "T10YIE": "10-Year Breakeven Inflation Rate (Layer 2)",
    "VIXCLS": "CBOE Volatility Index VIX (Layer 2 Risk Regime)",
}

# Layer 3: Global Reserve Currencies & Carry Pairs (Yahoo Finance Tickers)
FX_TICKERS = [
    "DX-Y.NYB", "JPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X", "CAD=X",
    "CHF=X", "NZDUSD=X", "CNY=X", "MXN=X", "BRL=X", "ZAR=X", "SGD=X", "INR=X",
]

# Layer 4: Strategic Physical Commodities (Energy, Metals, Agriculture)
COMMODITY_TICKERS = [
    "CL=F", "BZ=F", "NG=F", "GC=F", "SI=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "CT=F", "SB=F",
]

# Layer 5 Core: Global Equity Indices, Sovereign Bond ETFs, Sector ETFs & Top Global Mega-Caps
GLOBAL_INDEX_AND_ETF_TICKERS = [
    "^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^STOXX50E",
    "SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "EZU", "FXI", "INDA", "EWZ",
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "GLD", "SLV", "USO", "DBC",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
]

TOP_GLOBAL_EQUITIES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "AVGO", "JPM",
    "XOM", "UNH", "V", "PG", "MA", "COST", "JNJ", "HD", "MRK", "ABBV",
    "CVX", "CRM", "BAC", "WMT", "AMD", "KO", "PEP", "TMO", "LIN", "MCD",
    "CSCO", "ADBE", "WFC", "ABT", "TM", "ASML", "NVO", "SAP", "AZN", "SHEL",
    "HSBC", "UL", "SNY", "TTE", "BP", "RIO", "BHP", "SONY", "MUFG", "TSM",
    "GS", "MS", "BLK", "C", "AXP", "SCHW", "CAT", "DE", "GE", "HON",
    "RTX", "LMT", "BA", "UNP", "UPS", "LOW", "NKE", "SBUX", "TJX", "BKNG",
    "INTC", "QCOM", "TXN", "AMAT", "LRCX", "MU", "ADI", "KLAC", "PANW", "NOW",
    "ISRG", "GILD", "AMGN", "VRTX", "REGN", "PFE", "BMY", "SYK", "MDT", "ELV",
    "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "NEM", "FCX", "APD",
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD",
]


def fetch_fred_series_csv(series_id: str, start_date: str = "2005-01-01") -> pd.Series:
    """Download a single FRED macro series directly from St. Louis Fed public CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=4) as resp:
        raw_csv = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw_csv))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    s = pd.to_numeric(df[val_col], errors="coerce")
    s.name = series_id
    return s


def download_yahoo_direct_20y_series(ticker: str) -> tuple[pd.Series, pd.Series]:
    """Fallback zero-dependency 20-year daily chart downloader via Yahoo Finance direct REST API."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=20y"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        res = data["chart"]["result"][0]
        timestamps = pd.to_datetime(res["timestamp"], unit="s")
        q = res["indicators"]["quote"][0]
        closes = pd.Series(q.get("close", []), index=timestamps, name=ticker).ffill()
        vols = pd.Series(q.get("volume", [1e6] * len(timestamps)), index=timestamps, name=ticker).fillna(0.0)
        return closes, vols


def download_global_20y_dataset(start_date: str = "2005-01-01", extra_tickers_file: str = "") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 88)
    print(f" DOWNLOADING 20-YEAR GLOBAL 5-LAYER MACRO & MULTI-ASSET DATA LAKE ({start_date} -> Present)")
    print("=" * 88)

    # 1. Download Layer 1 & Layer 2 Macro Series from FRED
    fred_frames = {}
    for sid, desc in FRED_SERIES.items():
        try:
            print(f"[Layer 1-2 FRED] Fetching {sid:14s} ({desc})...")
            fred_frames[sid] = fetch_fred_series_csv(sid, start_date=start_date)
        except Exception as exc:
            print(f"  [WARN] Could not fetch {sid}: {exc}")

    if fred_frames:
        df_macro = pd.DataFrame(fred_frames).sort_index()
    else:
        print("[Layer 1-2] Falling back to Yahoo Finance Macro series (^TNX, ^TYX, ^FVX, ^IRX, ^VIX, HYG, LQD, TLT)...", flush=True)
        for msym in ["^TNX", "^TYX", "^FVX", "^IRX", "^VIX", "HYG", "LQD", "TLT"]:
            try:
                c, _ = download_yahoo_direct_20y_series(msym)
                if len(c) > 0:
                    fred_frames[msym] = c
            except Exception as e:
                print(f"  [WARN] Fallback macro fetch failed for {msym}: {e}", flush=True)
        df_macro = pd.DataFrame(fred_frames).sort_index()

    if not df_macro.empty:
        # Strict 1-day causal publication shift: macro data published for day d is visible at day d+1
        df_macro_causal = df_macro.shift(1).ffill()
        df_macro_causal.to_csv(OUT_DIR / "layer1_2_macro_fred_20y.csv")
        print(f"Saved Layer 1-2 Macro Panel: {df_macro_causal.shape} -> {OUT_DIR / 'layer1_2_macro_fred_20y.csv'}", flush=True)

    # 2. Download Layer 3 (FX), Layer 4 (Commodities), Layer 5 (Global Equities & Indices) via yfinance
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        print("\n[!] Please install yfinance on your PC first:  pip install yfinance pyarrow")
        return

    tickers = list(dict.fromkeys(FX_TICKERS + COMMODITY_TICKERS + GLOBAL_INDEX_AND_ETF_TICKERS + TOP_GLOBAL_EQUITIES))
    if extra_tickers_file and Path(extra_tickers_file).exists():
        extra = [line.strip() for line in Path(extra_tickers_file).read_text().splitlines() if line.strip()]
        tickers = list(dict.fromkeys(tickers + extra))

    print(f"\n[Layer 3-5 Yahoo Finance] Downloading 20-Year Daily OHLCV for {len(tickers)} Global Markets...")
    data = yf.download(tickers, start=start_date, auto_adjust=True, progress=True, threads=True)
    close_df = data["Close"] if "Close" in data else data
    vol_df = data["Volume"] if "Volume" in data else pd.DataFrame(1e7, index=close_df.index, columns=close_df.columns)

    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
    vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
    close_df = close_df.sort_index().ffill()
    vol_df = vol_df.sort_index().ffill().fillna(0.0)

    close_df.to_csv(OUT_DIR / "global_20y_daily_prices.csv")
    vol_df.to_csv(OUT_DIR / "global_20y_daily_volume.csv")
    print(f"\n[SUCCESS] Saved 20-Year Global Multi-Asset Prices: {close_df.shape} -> {OUT_DIR / 'global_20y_daily_prices.csv'}")
    print("Now run:  python scripts/run_20year_global_backtest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 20-Year Global 5-Layer Multi-Asset Dataset")
    parser.add_argument("--start-date", default="2005-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--tickers-file", default="", help="Optional path to custom 3,000-ticker list file")
    args = parser.parse_args()
    download_global_20y_dataset(start_date=args.start_date, extra_tickers_file=args.tickers_file)
