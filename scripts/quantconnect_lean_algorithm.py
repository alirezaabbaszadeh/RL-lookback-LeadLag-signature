# QuantConnect LEAN Python Algorithm — SOTA-Apex (Omega-Singularity) Paper Trading & Backtest
# Platform: QuantConnect Cloud (https://www.quantconnect.com) or LEAN CLI (`lean backtest` / `lean live`)
from AlgorithmImports import *
import numpy as np


class SOTAApexOmegaSingularityAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 11, 18)
        self.SetEndDate(2025, 8, 11)
        self.SetAccountCurrency("USDT")
        self.SetCash(100000)
        self.SetBrokerageModel(BrokerageName.BinanceFutures, AccountType.Margin)

        tickers = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
            "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT", "TRXUSDT", "LTCUSDT",
            "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "FILUSDT", "NEARUSDT",
            "APTUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "RNDRUSDT", "STXUSDT",
        ]
        self.symbols = []
        for t in tickers:
            sym = self.AddCryptoFuture(t, Resolution.Hour, Market.Binance).Symbol
            self.symbols.append(sym)

        self.horizons = [24, 72, 240, 720]
        self.rebal_hours = 24
        self.min_hold_hours = 6
        self.sl_sigma = 3.5
        self.tp_sigma = 3.8
        self.trail_sigma = 1.6
        self.hour_counter = 0
        self.hold_hours = {s: 0 for s in self.symbols}
        self.entry_prices = {s: 0.0 for s in self.symbols}
        self.peak_pnl = {s: 0.0 for s in self.symbols}
        self.current_weights = {s: 0.0 for s in self.symbols}
        self.SetWarmUp(750, Resolution.Hour)

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return
        self.hour_counter += 1
        hist = self.History(self.symbols, 725, Resolution.Hour)
        if hist.empty:
            return

        closes = hist["close"].unstack(level=0).ffill().dropna(axis=1)
        valid_syms = [s for s in self.symbols if s in closes.columns]
        if len(valid_syms) < 8:
            return

        P = closes[valid_syms].values
        LR = np.diff(np.log(np.maximum(P, 1e-12)), axis=0)
        LR = np.clip(LR, -0.30, 0.30)
        vol_1h = np.maximum(LR[-24:].std(axis=0), 0.005)

        # 1. Intraday Theorem 34 Check (Stop-Loss + Chandelier Trailing + Same-Leg Winner Redistribution)
        stopped_or_tp = False
        for idx, s in enumerate(valid_syms):
            w = self.current_weights.get(s, 0.0)
            if abs(w) > 1e-6 and self.entry_prices[s] > 0:
                self.hold_hours[s] += 1
                ret_since = (P[-1, idx] / self.entry_prices[s] - 1.0) * np.sign(w)
                self.peak_pnl[s] = max(self.peak_pnl[s], ret_since)
                sig_h = vol_1h[idx] * np.sqrt(max(self.hold_hours[s], 1))
                is_sl = ret_since <= -self.sl_sigma * sig_h
                is_tp = (
                    self.hold_hours[s] >= self.min_hold_hours
                    and self.peak_pnl[s] >= self.tp_sigma * sig_h
                    and (self.peak_pnl[s] - ret_since) >= self.trail_sigma * vol_1h[idx]
                )
                if is_sl or is_tp:
                    self.current_weights[s] = 0.0
                    self.hold_hours[s] = 0
                    stopped_or_tp = True

        if stopped_or_tp:
            pos_s = [s for s in valid_syms if self.current_weights.get(s, 0.0) > 1e-6]
            neg_s = [s for s in valid_syms if self.current_weights.get(s, 0.0) < -1e-6]
            if pos_s and neg_s:
                pos_sum = sum(self.current_weights[s] for s in pos_s)
                neg_sum = sum(-self.current_weights[s] for s in neg_s)
                for s in pos_s:
                    self.current_weights[s] *= 0.50 / max(pos_sum, 1e-8)
                for s in neg_s:
                    self.current_weights[s] *= 0.50 / max(neg_sum, 1e-8)
                self._execute_targets(valid_syms)

        # 2. Scheduled 24-Hour Rebalance with Anti-Latency Min-Hold Guard
        if self.hour_counter % self.rebal_hours != 0:
            return

        N = len(valid_syms)
        score = np.zeros(N, dtype=float)
        for H in self.horizons:
            W = LR[-H:]
            mu = np.median(W, axis=0, keepdims=True)
            mad = np.median(np.abs(W - mu), axis=0, keepdims=True) + 1e-8
            Z = np.tanh((W - mu) / (1.4826 * mad * 3.0))
            Pref = np.vstack([np.zeros((1, N)), np.cumsum(Z, axis=0)])
            S = (2.0 * Pref[:-1] - Pref[-1:].repeat(H, axis=0)).T @ Z / float(H)
            A = 0.5 * (S - S.T)
            r_h = W.sum(axis=0)
            sig_h = A @ np.tanh(r_h / (r_h.std() + 1e-8))
            score += (sig_h - sig_h.mean()) / (sig_h.std() + 1e-8)

        k_sel = max(2, N // 4)
        order = np.argsort(score)
        target = np.zeros(N, dtype=float)
        target[order[-k_sel:]] = 0.50 / k_sel
        target[order[:k_sel]] = -0.50 / k_sel

        for idx, s in enumerate(valid_syms):
            old_w = self.current_weights.get(s, 0.0)
            new_w = float(target[idx])
            if abs(old_w) > 1e-6 and self.hold_hours[s] < self.min_hold_hours and np.sign(new_w) != np.sign(old_w):
                new_w = old_w
            if abs(new_w - old_w) > 1e-4:
                self.hold_hours[s] = 0
                self.entry_prices[s] = float(P[-1, idx])
                self.peak_pnl[s] = 0.0
            self.current_weights[s] = new_w

        self._execute_targets(valid_syms)

    def _execute_targets(self, valid_syms):
        targets = [PortfolioTarget(s, self.current_weights.get(s, 0.0)) for s in valid_syms]
        self.SetHoldings(targets)
