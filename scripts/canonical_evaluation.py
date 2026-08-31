"""
================================================================================
  CANONICAL EVALUATION HARNESS v2 — Single entry point for all RAI evaluations
================================================================================
  ALL future evaluation runs go through this script. No new one-off scripts.

  Comparison Arms (Steps 2-3):
    FIXED REF: SPY B&H (fixed reference) — the real SPY, constant across all six
               universes and independent of each universe's ticker set / column
               order. Out-of-universe (not a constituent) for India, Forex and
               Crypto, and reported as such. Added 2026-08-30; see §21 of
               docs/consolidation_report.md.
    PASSIVE:  Asset-0 B&H (in-universe, <ticker>), Equal Weight (1/N), 60/40
              Portfolio — all POSITIONAL arms on column indices. Column 0 is
              never SPY in any universe (EEM / AAPL / AXISBANK.NS / AUDUSD=X /
              BCH-USD), so these must not be described as "SPY".
    RULE:     Risk Parity, Asset-0 SMA 50/200 Crossover
    ML (real-data-trained): LSTM Return Predictor, XGBoost Classifier
    RAI (zero-shot):        Axiom (formerly v6 Alpha), v6 Fast, v7 (if checkpoint), v8.2 (if checkpoint)

  Rigor Features:
    - Disclosed cost model: fee_bps + slippage + drift threshold
    - Single-seed results labeled "single-seed, not CI-verified"
    - Full pairwise Sharpe win/loss matrix across ALL arms
    - Symmetric reporting: Return AND Sharpe deltas for every comparison
    - Point-in-time universe construction (verified against historical sources)

  Structure Features:
    - Chronological 3-way split: 60% train / 20% OOS / 20% future holdout
    - LSTM/XGBoost trained on train split only (no look-ahead)
    - Grid-search tuning documented in module docstring

  Baseline Tuning (from archived eval_vs_standard_ai.py):
    LSTM:    hidden_dim=32, lr=1e-3 (grid 3x3, val Sharpe=0.7150)
    XGBoost: max_depth=5, n_estimators=100 (grid 3x3, val Acc=0.5291)

  Usage:
    python scripts/canonical_evaluation.py                    # All universes
    python scripts/canonical_evaluation.py --universe us_etf  # Single universe
    python scripts/canonical_evaluation.py --list-universes   # Show available
================================================================================
"""

import os, sys, json, argparse, warnings
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ==============================================================================
#  COST MODEL — Disclosed explicitly
# ==============================================================================
DEFAULT_FEE_BPS = 5         # 5 basis points per rebalancing (0.05% of turnover)
DEFAULT_SLIPPAGE_PCT = 0.02  # 0.02% random price slippage at execution
DEFAULT_REBAL_THRESH = 0.03  # Only rebalance if portfolio drift exceeds 3%


# ==============================================================================
#  UNIVERSES — Point-in-time corrected, sources documented
# ==============================================================================
UNIVERSES = {
    "us_etf": {
        "name": "US Market ETFs",
        "tickers": ["SPY", "QQQ", "EEM", "VNQ", "HYG", "TLT", "GLD", "USO", "UUP", "IWM"],
        "period": "10y",
        "survivorship_note": "All ETFs existed well before evaluation period. No bias.",
        "pit_source": "All ETFs launched pre-2010. No selection needed.",
    },
    "us_megacap_pit": {
        "name": "US Mega-Cap (Point-in-Time Jan 2015 S&P 500 Top-10)",
        # Primary: voronoiapp.com S&P 500 historical market cap rankings
        # Cross-verified: macromicro.me, CRSP dsp500list methodology (see consolidation_report.md §4)
        # Verified 2026-08-19: AAPL, XOM, MSFT, GOOGL, JNJ, WFC, BRK-B, GE, PG, JPM
        # BRK-B excluded (conglomerate); replaced with CVX (~#11-#13 by market cap)
        # GOOG split to GOOGL in 2014 — use GOOGL ticker
        # Paper-grade path: CRSP dsp500list table via WRDS, date=2015-01-02
        "tickers": ["AAPL", "XOM", "MSFT", "GOOGL", "GE", "JNJ", "PG", "WFC", "JPM", "CVX"],
        "period": "10y",
        "survivorship_note": "S&P 500 top-10 by market cap as of Jan 2015. BRK-B replaced with ~#11 CVX.",
        "pit_source": "voronoiapp.com + macromicro.me cross-verified, 2026-08-19. Paper-grade: CRSP dsp500list via WRDS.",
    },
    "global_indices": {
        "name": "Global Equity Indices",
        "tickers": ["SPY", "EWJ", "EWG", "EWU", "MCHI", "INDA", "EWZ", "EFA", "EEM", "FXI"],
        "period": "10y",
        "survivorship_note": "Country ETFs are stable constructs. No bias.",
        "pit_source": "iShares country ETFs, all pre-existing.",
    },
    "india_nifty50": {
        "name": "Indian Nifty 50 Equities",
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                     "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS"],
        "period": "5y",
        "survivorship_note": "Indian blue-chips, all pre-existing. No bias.",
        "pit_source": "NSE Nifty 50 top-10 by weight, stable membership.",
    },
    "forex_commodities": {
        "name": "Global Forex & Commodities",
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
                     "GC=F", "CL=F", "SI=F", "HG=F", "NG=F"],
        "period": "5y",
        "survivorship_note": "Major FX pairs and commodity futures. No bias.",
        "pit_source": "Major currency pairs and CME commodity futures. No selection needed.",
    },
    "crypto_pit": {
        "name": "Crypto (Point-in-Time Jan 2020 Top-10 by Market Cap)",
        # Source: CoinMarketCap historical snapshot, Jan 1 2020
        # Selection rule: Top-10 by market cap, excluding:
        #   (a) stablecoins (USDT — pegged to fiat, not allocatable)
        #   (b) assets not continuously tradable on major exchanges through holdout window
        #       (BSV — delisted from Binance Apr 2019, Kraken 2019, prior to construction date)
        # Fills: LINK (#13), TRX (#14) as next-ranked non-excluded assets
        "tickers": ["BTC-USD", "ETH-USD", "XRP-USD", "BCH-USD", "LTC-USD",
                     "EOS-USD", "BNB-USD", "XTZ-USD", "LINK-USD", "TRX-USD"],
        "period": "5y",
        "survivorship_note": "Top-10 crypto Jan 2020 (excl. USDT stablecoin, BSV not continuously tradable). All pre-2020.",
        "pit_source": "CoinMarketCap Jan 1 2020 snapshot. Exclusion rule: stablecoins + not continuously tradable on major exchanges.",
    },
}


# ==============================================================================
#  METRICS
# ==============================================================================
def compute_metrics(equity_curve):
    """Standard metric computation from equity curve (list or array)."""
    eq = np.array(equity_curve, dtype=np.float64)
    if len(eq) < 2:
        return {"final": float(eq[-1]), "return_pct": 0.0, "vol_pct": 0.0,
                "sharpe": 0.0, "max_dd": 0.0}
    r = (eq[1:] - eq[:-1]) / np.maximum(1e-8, eq[:-1])
    pk = np.maximum.accumulate(eq)
    mdd = np.min((eq - pk) / pk) * 100
    sharpe = float(np.mean(r) / np.std(r) * np.sqrt(252)) if np.std(r) > 1e-8 else 0.0
    return {
        "final": float(eq[-1]),
        "return_pct": float((eq[-1] / eq[0] - 1) * 100),
        "vol_pct": float(np.std(r) * np.sqrt(252) * 100),
        "sharpe": sharpe,
        "max_dd": float(mdd),
    }


# ==============================================================================
#  DATA — Chronological 3-Way Split
# ==============================================================================
def download_and_split(tickers, period="10y", train_frac=0.60, oos_frac=0.20):
    """Chronological 60/20/20 split. Returns (train_df, oos_df, future_df, dates_info)."""
    try:
        import yfinance as yf
        df = yf.download(tickers, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        df = df.dropna().ffill().bfill()
        if df.empty or len(df) < 200:
            print(f"  WARNING: Insufficient data ({len(df)} rows). Skipping.", flush=True)
            return None
        T = len(df)
        i1 = int(T * train_frac)
        i2 = int(T * (train_frac + oos_frac))
        train_df, oos_df, future_df = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
        dates_info = {
            "total_days": T,
            "train": f"{train_df.index[0].strftime('%Y-%m-%d')} to {train_df.index[-1].strftime('%Y-%m-%d')} ({len(train_df)}d)",
            "oos": f"{oos_df.index[0].strftime('%Y-%m-%d')} to {oos_df.index[-1].strftime('%Y-%m-%d')} ({len(oos_df)}d)",
            "future": f"{future_df.index[0].strftime('%Y-%m-%d')} to {future_df.index[-1].strftime('%Y-%m-%d')} ({len(future_df)}d)",
        }
        return train_df, oos_df, future_df, dates_info
    except Exception as e:
        print(f"  ERROR downloading data: {e}", flush=True)
        return None


# ==============================================================================
#  MODEL LOADING
# ==============================================================================
def load_v6_model(checkpoint_path, device='cpu'):
    """Load RAI v6 FastTradingNet."""
    from scripts.train_v6_fast import FastTradingNet
    model = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def load_axiom_model(checkpoint_path, device='cpu'):
    """Load an Axiom checkpoint into AxiomNet (conv1/conv2 + flatten, 289,527 params).

    Kept separate from load_v6_model on purpose: AxiomNet and FastTradingNet are
    not state_dict-compatible and a cross-load is rejected. See §15 of
    docs/consolidation_report.md. Only checkpoints/axiom_multiseed/axiom_seed*.pt
    load here.
    """
    from scripts.kaggle_axiom_10seed import AxiomNet
    model = AxiomNet(history_len=30, features_per_step=22, action_dim=11)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def load_v7_model(checkpoint_path, device='cpu'):
    """Load RAI v7 SpatioTemporalTradingNet."""
    from rai.learning.v7_model import SpatioTemporalTradingNet
    model = SpatioTemporalTradingNet()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


# ==============================================================================
#  RAI MODEL EVALUATION — With disclosed cost model
# ==============================================================================
def evaluate_model_on_prices(model, prices, fee_bps=DEFAULT_FEE_BPS,
                              slippage_pct=DEFAULT_SLIPPAGE_PCT,
                              rebal_thresh=DEFAULT_REBAL_THRESH,
                              history_len=30, initial_cash=10000.0,
                              return_allocations=False):
    """
    Evaluate a loaded RL model on a price DataFrame with explicit cost model.
    Returns: (equity_curve, rebal_count) or (equity_curve, rebal_count, alloc_history) if return_allocations=True
    """
    if isinstance(prices, pd.DataFrame):
        p = prices.values[:, :10]
    else:
        p = prices[:, :10] if prices.shape[1] > 10 else prices
    T, N = p.shape
    if T <= history_len + 1:
        if return_allocations:
            return [initial_cash] * T, 0, []
        return [initial_cash] * T, 0

    log_ret = np.diff(np.log(np.maximum(1e-8, p)), axis=0)
    cash = initial_cash
    weights = np.ones(N) / (N + 1)
    cash_frac = 1.0 / (N + 1)
    equity = [initial_cash]
    rebal_count = 0
    alloc_history = []

    for t in range(history_len, T - 1):
        window = log_ret[t - history_len:t]
        feats = np.zeros((history_len, 22))
        feats[:, :min(N, 10)] = window[:, :10]
        feats[:, 10:min(10 + N, 20)] = window[:, :min(N, 10)]
        feats[:, 20] = np.mean(window, axis=1)
        feats[:, 21] = np.std(window, axis=1)

        obs = feats.flatten().astype(np.float32)
        action = model.get_action(obs, deterministic=True)

        cash_logit = float(action[0]) + (-2.5)
        cash_logit = np.clip(cash_logit, -8.0, 3.0)
        new_cash_frac = 1.0 / (1.0 + np.exp(-cash_logit))

        asset_logits = action[1:N + 1] if len(action) > N else action[1:]
        exp_logits = np.exp(asset_logits - np.max(asset_logits))
        new_weights = exp_logits / (exp_logits.sum() + 1e-8)
        new_weights *= (1.0 - new_cash_frac)

        drift = np.sum(np.abs(new_weights - weights[:len(new_weights)])) + abs(new_cash_frac - cash_frac)

        if drift > rebal_thresh:
            fee = cash * drift * (fee_bps / 10000.0)
            slip = cash * drift * (slippage_pct / 100.0)
            cash = max(1e-4, cash - fee - slip)
            weights[:len(new_weights)] = new_weights
            cash_frac = new_cash_frac
            rebal_count += 1

        if return_allocations:
            alloc_history.append({"t": t, "cash_frac": float(cash_frac),
                                   "weights": weights[:len(new_weights)].tolist()})

        daily_ret = p[t + 1] / np.maximum(1e-8, p[t]) - 1.0
        asset_ret = np.sum(weights[:len(daily_ret)] * daily_ret[:len(weights)])
        cash = cash * (1.0 + asset_ret * (1.0 - cash_frac))
        equity.append(cash)

    while len(equity) < T:
        equity.append(equity[-1])

    if return_allocations:
        return equity, rebal_count, alloc_history
    return equity, rebal_count


# ==============================================================================
#  PASSIVE BASELINES — in-universe asset 0, Equal Weight, 60/40
#  NOTE: none of these three is SPY. They act on COLUMN INDICES of whatever
#  frame they are handed, and columns are left in yfinance's alphabetical order
#  (deliberately — the policy's per-asset logits are position-dependent). For a
#  real, fixed SPY benchmark see the FIXED EXTERNAL REFERENCE section below.
# ==============================================================================
def evaluate_buy_hold_first(prices, initial_cash=10000.0):
    """Buy-and-hold **column 0** of the frame handed in — NOT SPY.

    Column 0 is EEM (US ETFs, Global Indices), AAPL (US Mega-Cap),
    AXISBANK.NS (India), AUDUSD=X (Forex) or BCH-USD (Crypto). This arm is
    labelled `Asset-0 B&H (in-universe, <ticker>)` in every table.
    For "buy and hold SPY" use `evaluate_spy_reference`.
    """
    p = prices.values[:, 0] if isinstance(prices, pd.DataFrame) else prices[:, 0]
    return [initial_cash] + [initial_cash * p[t] / p[0] for t in range(1, len(p))]


def evaluate_equal_weight(prices, initial_cash=10000.0):
    """Equal-weight (1/N) buy-and-hold baseline."""
    p = prices.values if isinstance(prices, pd.DataFrame) else prices
    T, N = p.shape
    shares = (initial_cash / N) / p[0]
    return [initial_cash] + [float(np.sum(shares * p[t])) for t in range(1, T)]


def evaluate_60_40(prices, initial_cash=10000.0):
    """60/40 portfolio: 60% **column 0** of the frame, 40% bond proxy.

    The equity leg is column 0, not SPY (see `evaluate_buy_hold_first`); the
    bond leg is TLT if the universe happens to contain it, otherwise column 1.
    Only US ETFs contains TLT, so in the other five universes this arm is
    "60% first-alphabetical / 40% second-alphabetical", which is what the
    `Asset 0` column of every results CSV records.
    """
    p = prices.values if isinstance(prices, pd.DataFrame) else prices
    T, N = p.shape
    eq_idx, bond_idx = 0, min(1, N - 1)
    # Look for TLT if in ticker list
    if isinstance(prices, pd.DataFrame) and 'TLT' in prices.columns:
        bond_idx = list(prices.columns).index('TLT')
    eq_shares = (initial_cash * 0.60) / p[0, eq_idx]
    bond_shares = (initial_cash * 0.40) / p[0, bond_idx]
    eq = [initial_cash]
    for t in range(1, T):
        eq.append(eq_shares * p[t, eq_idx] + bond_shares * p[t, bond_idx])
    return eq


# ==============================================================================
#  FIXED EXTERNAL REFERENCE — the real SPY, independent of universe membership
# ==============================================================================
#  Every other passive arm in this file is positional: it holds whatever ticker
#  happens to sort first in the universe's own column order. That is not the
#  benchmark the paper's framing invokes ("competitive with data-trained
#  baselines, not outperforming buy-and-hold"), and for 5 of the 6 universes it
#  was never SPY at all. The functions below hold ONE constant asset — SPY —
#  in every universe, including the three where SPY is not a constituent. There
#  it is deliberately an OUT-OF-UNIVERSE reference point (what a US investor
#  could have held instead), not a swap-in replacement for the in-universe arm.
#  Both rows are kept in every table.
# ==============================================================================
SPY_REFERENCE_TICKER = "SPY"
SPY_REFERENCE_CACHE = os.path.join(PROJECT_ROOT, "data", "pinned_universes",
                                   "_spy_reference.csv")
# Margin on both ends of the widest pinned window (2016-08-20 -> 2026-08-20) so
# that no universe's first day needs back-filling.
SPY_REFERENCE_START = "2016-07-01"
SPY_REFERENCE_END = "2026-08-21"


def load_spy_reference(start=SPY_REFERENCE_START, end=SPY_REFERENCE_END,
                       cache_path=SPY_REFERENCE_CACHE, force=False):
    """Load the fixed SPY close series (auto-adjusted), cached on disk.

    Returns a tz-naive `pd.Series` named "SPY". Falls back to the SPY column of
    `data/pinned_universes/US_ETFs.csv` if the download fails and no cache
    exists, so the arm is computable offline; the fallback is flagged via the
    returned series' `.attrs["source"]`.
    """
    if os.path.exists(cache_path) and not force:
        s = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = SPY_REFERENCE_TICKER
        s.attrs["source"] = "cache:%s" % os.path.basename(cache_path)
        return s.astype(float)
    src = None
    try:
        import yfinance as yf
        df = yf.download(SPY_REFERENCE_TICKER, start=start, end=end,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df = df["Close"]
        s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
        s = s.dropna()
        if len(s) < 500:
            raise RuntimeError("only %d SPY rows returned" % len(s))
        src = "yfinance %s..%s auto_adjust=True" % (start, end)
    except Exception as e:
        alt = os.path.join(PROJECT_ROOT, "data", "pinned_universes", "US_ETFs.csv")
        if not os.path.exists(alt):
            raise RuntimeError("SPY reference unavailable: %r" % (e,))
        s = pd.read_csv(alt, index_col=0, parse_dates=True)["SPY"].dropna()
        src = ("FALLBACK: SPY column of US_ETFs.csv (download failed: %s). "
               "This frame was dropna()'d across 10 tickers, so a few NYSE "
               "sessions may be absent." % e)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    s.name = SPY_REFERENCE_TICKER
    s = s.astype(float)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    s.to_frame().to_csv(cache_path, encoding="utf-8")
    s.attrs["source"] = src
    return s


def align_spy_to_index(spy, index):
    """Align the fixed SPY series onto an arbitrary trading calendar.

    SPY trades the NYSE calendar; the six universes do not. NSE (India),
    FX/futures (Forex) and crypto (which trades weekends) each have their own
    index. The reference is therefore forward-filled onto the target index: the
    last observable SPY close is carried into any date on which SPY did not
    trade, which is what a holder of SPY would actually mark to. Such days
    contribute a zero return, so they are counted and reported rather than
    hidden — see `evaluate_spy_native_calendar` for the undistorted variant.

    Returns `(aligned_series, n_padded_days, n_backfilled_days)`.
    """
    idx = pd.DatetimeIndex(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = spy.reindex(spy.index.union(idx)).ffill().reindex(idx)
    n_back = int(s.isna().sum())          # dates preceding SPY's first close
    s = s.bfill()
    n_pad = int(len(idx) - int(idx.isin(spy.index).sum()))
    return s, n_pad, n_back


def evaluate_spy_reference(window, spy=None, initial_cash=10000.0):
    """Buy-and-hold the FIXED SPY reference over `window`'s dates.

    `window` may be a DataFrame (only its DatetimeIndex is used — its columns
    are ignored entirely, which is the point) or a DatetimeIndex.
    Returns `(equity_curve, n_padded_days)`.

    Cost model: this arm allocates once at t=0 and never trades. The pinned
    harness charges nothing for the entry allocation, so its zero-cost and
    cost-charged curves are byte-identical **by construction**, exactly as for
    the other buy-once arms. That is not cost robustness.
    """
    if spy is None:
        spy = load_spy_reference()
    idx = window.index if isinstance(window, (pd.DataFrame, pd.Series)) else window
    s, n_pad, _ = align_spy_to_index(spy, idx)
    p = s.to_numpy(float)
    eq = [initial_cash] + [initial_cash * p[t] / p[0] for t in range(1, len(p))]
    return eq, n_pad


def evaluate_spy_native_calendar(window, spy=None, initial_cash=10000.0):
    """Same buy-and-hold, but on SPY's OWN NYSE sessions inside the window's
    date span. Diagnostic only: it quantifies how much the forward-filled
    zero-return days in `evaluate_spy_reference` distort an annualised Sharpe
    for the non-NYSE universes. Returns `(equity_curve, n_days)`.
    """
    if spy is None:
        spy = load_spy_reference()
    idx = window.index if isinstance(window, (pd.DataFrame, pd.Series)) else window
    idx = pd.DatetimeIndex(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = spy.loc[(spy.index >= idx[0]) & (spy.index <= idx[-1])].dropna()
    p = s.to_numpy(float)
    if len(p) < 2:
        return [initial_cash, initial_cash], len(p)
    eq = [initial_cash] + [initial_cash * p[t] / p[0] for t in range(1, len(p))]
    return eq, len(p)


# ==============================================================================
#  RULE-BASED BASELINES — Risk Parity, SMA Crossover
# ==============================================================================
def evaluate_risk_parity(prices, lb=60):
    """Inverse-volatility weighted portfolio, rebalanced monthly."""
    p = prices.values[:, :10] if isinstance(prices, pd.DataFrame) else prices[:, :10]
    T, N = p.shape
    sh = (10000.0 / N) / p[0]
    eq = [10000.0]
    for t in range(1, T):
        w = np.sum(sh * p[t])
        if t % 21 == 0 and t >= lb:
            v = np.std(np.diff(np.log(np.maximum(1e-4, p[t-lb:t+1])), axis=0), axis=0)
            iv = 1.0 / np.maximum(1e-8, v)
            wts = iv / iv.sum()
            sh = (w * wts) / p[t]
        eq.append(w)
    return eq


def evaluate_sma_crossover(prices, sw=50, lw=200):
    """SMA 50/200 crossover on **column 0** of the frame — NOT SPY. Binary in/out.

    Retained as-published; it has no pre-window warm-up, so its first `lw` days
    fall through to the default long position. The warm-up-corrected arm lives in
    `scripts/deterministic_baselines_pinned.py` (`evaluate_sma_warmup`).
    """
    p = prices.values[:, :10] if isinstance(prices, pd.DataFrame) else prices[:, :10]
    T = p.shape[0]
    a0 = p[:, 0]
    cash = 10000.0
    eq = [10000.0]
    long_a0 = True
    for t in range(1, T):
        if t >= lw:
            long_a0 = np.mean(a0[t-sw:t]) > np.mean(a0[t-lw:t])
        dr = (a0[t] / a0[t-1] - 1.0) if long_a0 else 0.0
        cash *= (1.0 + dr)
        eq.append(cash)
    return eq


# ==============================================================================
#  ML BASELINES — LSTM, XGBoost (trained on train split, evaluated on test)
# ==============================================================================
class PyTorchLSTM(nn.Module):
    """2-layer LSTM for next-day return direction prediction."""
    def __init__(self, input_dim=10, hidden_dim=32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_lstm_on_split(train_df, lookback=20, epochs=30, hidden_dim=32, lr=1e-3, seed=42):
    """Train LSTM on the train split only. Returns trained model or None.
    
    Bug fix (2026-08-20): Previously had no seed control and no data shuffling,
    causing near-zero variance across seeds. Now seeds both weight init and
    uses shuffled mini-batch training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    prices = train_df.values[:, :10] if isinstance(train_df, pd.DataFrame) else train_df[:, :10]
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    X, y = [], []
    for i in range(len(rets) - lookback):
        X.append(rets[i:i+lookback])
        y.append(1.0 if rets[i+lookback].mean() > 0 else 0.0)
    if not X:
        return None
    X_t = torch.FloatTensor(np.array(X))
    y_t = torch.FloatTensor(np.array(y)).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True,
                                          generator=torch.Generator().manual_seed(seed))
    model = PyTorchLSTM(input_dim=prices.shape[1], hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def evaluate_lstm_strategy(model, prices_df, lookback=20, initial_cash=10000.0):
    """Run LSTM signal-based strategy on a price DataFrame."""
    prices = prices_df.values[:, :10] if isinstance(prices_df, pd.DataFrame) else prices_df[:, :10]
    T, N = prices.shape
    if model is None or T <= lookback:
        return [initial_cash] * T
    cash = initial_cash
    eq = [cash]
    sh = np.zeros(N)
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    for t in range(lookback, T-1):
        x = torch.FloatTensor(rets[t-lookback:t]).unsqueeze(0)
        with torch.no_grad():
            pred = torch.sigmoid(model(x)).item()
        w = cash + np.sum(sh * prices[t])
        target_stock = 0.8 if pred > 0.5 else 0.2
        target_w = target_stock / N
        cash = w * (1.0 - target_stock)
        sh = (w * target_w) / np.maximum(1e-4, prices[t])
        eq.append(cash + np.sum(sh * prices[t+1]))
    while len(eq) < T:
        eq.append(eq[-1])
    return eq


def train_xgboost_on_split(train_df, lookback=20, max_depth=5, n_estimators=100, seed=42):
    """Train XGBoost on the train split only. Returns trained classifier or None."""
    from sklearn.ensemble import GradientBoostingClassifier
    np.random.seed(seed)
    prices = train_df.values[:, :10] if isinstance(train_df, pd.DataFrame) else train_df[:, :10]
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    X, y = [], []
    for i in range(len(rets) - lookback):
        X.append(rets[i:i+lookback].flatten())
        y.append(1 if rets[i+lookback].mean() > 0 else 0)
    if not X:
        return None
    clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      learning_rate=0.05, random_state=seed)
    clf.fit(np.array(X), np.array(y))
    return clf


def evaluate_xgb_strategy(clf, prices_df, lookback=20, initial_cash=10000.0):
    """Run XGBoost signal-based strategy on a price DataFrame."""
    prices = prices_df.values[:, :10] if isinstance(prices_df, pd.DataFrame) else prices_df[:, :10]
    T, N = prices.shape
    if clf is None or T <= lookback:
        return [initial_cash] * T
    cash = initial_cash
    eq = [cash]
    sh = np.zeros(N)
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    for t in range(lookback, T-1):
        feat = rets[t-lookback:t].flatten().reshape(1, -1)
        pred = clf.predict(feat)[0]
        w = cash + np.sum(sh * prices[t])
        target_stock = 0.8 if pred == 1 else 0.2
        target_w = target_stock / N
        cash = w * (1.0 - target_stock)
        sh = (w * target_w) / np.maximum(1e-4, prices[t])
        eq.append(cash + np.sum(sh * prices[t+1]))
    while len(eq) < T:
        eq.append(eq[-1])
    return eq


# ==============================================================================
#  PAIRWISE WIN/LOSS MATRIX
# ==============================================================================
def print_pairwise_matrix(all_metrics, window_name):
    """Print full pairwise Sharpe win/loss matrix across all arms."""
    labels = list(all_metrics.keys())
    n = len(labels)
    if n < 2:
        return

    print(f"\n  PAIRWISE SHARPE COMPARISON ({window_name}):", flush=True)
    # Header
    short_labels = [l[:15] for l in labels]
    header = f"  {'':>15} |" + "".join(f" {sl:>15} |" for sl in short_labels)
    print(header, flush=True)
    print(f"  {'-' * len(header)}", flush=True)

    wins = {l: 0 for l in labels}
    for i, li in enumerate(labels):
        row = f"  {short_labels[i]:>15} |"
        for j, lj in enumerate(labels):
            if i == j:
                row += f" {'---':>15} |"
            else:
                delta = all_metrics[li]['sharpe'] - all_metrics[lj]['sharpe']
                marker = "W" if delta > 0 else "L"
                row += f" {delta:>+.2f} ({marker})      |"
                if delta > 0:
                    wins[li] += 1
        print(row, flush=True)

    print(f"\n  Win counts: ", flush=True)
    for l in sorted(wins, key=lambda x: -wins[x]):
        print(f"    {l}: {wins[l]}/{n-1} wins", flush=True)


# ==============================================================================
#  MAIN EVALUATION SUITE
# ==============================================================================
def run_evaluation_suite(universe_key, models_dict, fee_bps=DEFAULT_FEE_BPS,
                          slippage_pct=DEFAULT_SLIPPAGE_PCT,
                          rebal_thresh=DEFAULT_REBAL_THRESH):
    """Run ALL comparison arms for one universe."""
    univ = UNIVERSES[universe_key]
    print(f"\n{'='*110}", flush=True)
    print(f"  UNIVERSE: {univ['name']}", flush=True)
    print(f"  Tickers: {', '.join(univ['tickers'])}", flush=True)
    print(f"  Survivorship: {univ['survivorship_note']}", flush=True)
    print(f"  PIT Source: {univ.get('pit_source', 'N/A')}", flush=True)
    print(f"  Cost Model: {fee_bps}bps fee + {slippage_pct}% slippage + {rebal_thresh*100:.0f}% drift threshold", flush=True)
    print(f"{'='*110}", flush=True)

    data = download_and_split(univ['tickers'], period=univ['period'])
    if data is None:
        return None

    train_df, oos_df, future_df, dates_info = data
    print(f"  Train:  {dates_info['train']}", flush=True)
    print(f"  OOS:    {dates_info['oos']}", flush=True)
    print(f"  Future: {dates_info['future']}", flush=True)

    # Train ML baselines on train split
    print(f"\n  Training LSTM (hidden=32, lr=1e-3, 30 epochs)...", flush=True)
    lstm_model = train_lstm_on_split(train_df, hidden_dim=32, lr=1e-3, epochs=30)
    print(f"  Training XGBoost (depth=5, est=100)...", flush=True)
    xgb_model = train_xgboost_on_split(train_df, max_depth=5, n_estimators=100)
    print(f"  Baselines trained.", flush=True)

    results = {
        "universe": univ['name'], "universe_key": universe_key,
        "tickers": univ['tickers'], "survivorship_note": univ['survivorship_note'],
        "pit_source": univ.get('pit_source', 'N/A'),
        "dates": dates_info,
        "cost_model": {"fee_bps": fee_bps, "slippage_pct": slippage_pct, "rebal_threshold": rebal_thresh},
        "seed_status": "single-seed, not CI-verified",
        "models": {},
    }

    for window_name, window_df in [("OOS", oos_df), ("Future_Holdout", future_df)]:
        print(f"\n  --- {window_name} Window ---", flush=True)
        hdr = f"  {'Model':<40} | {'Type':<10} | {'Return':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Rebal#':>7}"
        print(hdr, flush=True)
        print(f"  {'-'*100}", flush=True)

        all_metrics = {}  # For pairwise comparison

        # --- FIXED EXTERNAL REFERENCE (real SPY, not a universe constituent) ---
        # Constant across all six universes and independent of column order.
        # Kept ALONGSIDE the in-universe asset-0 arm, never in place of it.
        try:
            _spy_ref = load_spy_reference()
            _eq_spy, _n_pad = evaluate_spy_reference(window_df, _spy_ref)
            _eq_nat, _n_nat = evaluate_spy_native_calendar(window_df, _spy_ref)
            label = "SPY B&H (fixed reference)"
            m = compute_metrics(_eq_spy)
            all_metrics[label] = m
            print(f"  {label:<40} | {'passive':<10} | {m['return_pct']:>+7.1f}% | "
                  f"{m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {'N/A':>7}", flush=True)
            results["models"][f"{window_name}_{label}"] = {
                **m, "arm_type": "passive", "real_data_trained": False,
                "seed_status": "deterministic",
                "in_universe": bool(SPY_REFERENCE_TICKER in window_df.columns),
                "calendar_padded_days": _n_pad,
                "window_days": int(len(window_df)),
                "sharpe_native_nyse_calendar": compute_metrics(_eq_nat)["sharpe"],
                "native_nyse_days": _n_nat,
                "note": ("fixed out-of-universe reference where SPY is not a "
                         "constituent; forward-filled onto the universe's own "
                         "trading calendar (%d of %d days padded)"
                         % (_n_pad, len(window_df))),
            }
        except Exception as _e:
            print(f"  {'SPY B&H (fixed reference)':<40} | {'passive':<10} | "
                  f"unavailable: {_e}", flush=True)

        # --- PASSIVE BASELINES (all positional — column indices, never SPY) ---
        _a0 = str(window_df.columns[0])
        for label, eval_fn, arm_type in [
            (f"Asset-0 B&H (in-universe, {_a0})", lambda df: evaluate_buy_hold_first(df), "passive"),
            ("Equal Weight (1/N)", lambda df: evaluate_equal_weight(df), "passive"),
            ("60/40 Portfolio", lambda df: evaluate_60_40(df), "passive"),
        ]:
            eq = eval_fn(window_df)
            m = compute_metrics(eq)
            all_metrics[label] = m
            print(f"  {label:<40} | {arm_type:<10} | {m['return_pct']:>+7.1f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {'N/A':>7}", flush=True)
            results["models"][f"{window_name}_{label}"] = {
                **m, "arm_type": arm_type, "real_data_trained": False,
                "seed_status": "deterministic",
            }

        print(f"  {'-'*100}", flush=True)

        # --- RULE-BASED ---
        for label, eval_fn, arm_type in [
            ("Risk Parity", lambda df: evaluate_risk_parity(df), "rule"),
            (f"Asset-0 SMA 50/200 ({_a0})", lambda df: evaluate_sma_crossover(df), "rule"),
        ]:
            eq = eval_fn(window_df)
            m = compute_metrics(eq)
            all_metrics[label] = m
            print(f"  {label:<40} | {arm_type:<10} | {m['return_pct']:>+7.1f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {'N/A':>7}", flush=True)
            results["models"][f"{window_name}_{label}"] = {
                **m, "arm_type": arm_type, "real_data_trained": False,
                "seed_status": "deterministic",
            }

        print(f"  {'-'*100}", flush=True)

        # --- ML BASELINES (real-data-trained) ---
        for label, eval_fn, arm_type in [
            ("LSTM (real-trained)", lambda df: evaluate_lstm_strategy(lstm_model, df), "ML-real"),
            ("XGBoost (real-trained)", lambda df: evaluate_xgb_strategy(xgb_model, df), "ML-real"),
        ]:
            eq = eval_fn(window_df)
            m = compute_metrics(eq)
            all_metrics[label] = m
            print(f"  {label:<40} | {arm_type:<10} | {m['return_pct']:>+7.1f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {'N/A':>7}", flush=True)
            results["models"][f"{window_name}_{label}"] = {
                **m, "arm_type": arm_type, "real_data_trained": True,
                "seed_status": "single-seed, not CI-verified",
            }

        print(f"  {'-'*100}", flush=True)

        # --- RAI MODELS (zero-shot) ---
        for label, (model, arch_ver) in models_dict.items():
            # For crypto future holdout + Fast, capture allocations
            want_alloc = (universe_key == "crypto_pit" and window_name == "Future_Holdout" and "Fast" in label)
            result = evaluate_model_on_prices(
                model, window_df, fee_bps=fee_bps,
                slippage_pct=slippage_pct, rebal_thresh=rebal_thresh,
                return_allocations=want_alloc
            )
            if want_alloc:
                eq, rebal_n, alloc_hist = result
                # Save allocation forensics for Step 6
                forensics_path = os.path.join(PROJECT_ROOT, 'data', 'crypto_fast_holdout_allocations.json')
                os.makedirs(os.path.dirname(forensics_path), exist_ok=True)
                with open(forensics_path, 'w') as f:
                    json.dump(alloc_hist, f, indent=2)
                print(f"  [FORENSICS] Crypto Fast holdout allocations saved to {forensics_path}", flush=True)
            else:
                eq, rebal_n = result

            m = compute_metrics(eq)
            all_metrics[f"[{arch_ver}] {label}"] = m
            print(f"  [{arch_ver}] {label:<36} | {'zero-shot':<10} | {m['return_pct']:>+7.1f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {rebal_n:>7}", flush=True)
            results["models"][f"{window_name}_{label}"] = {
                **m, "rebal_count": rebal_n, "arch": arch_ver,
                "arm_type": "zero-shot", "real_data_trained": False,
                "seed_status": "single-seed, not CI-verified",
            }

        print(f"  {'-'*100}", flush=True)

        # Pairwise matrix
        print_pairwise_matrix(all_metrics, window_name)

    return results


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="RAI Canonical Evaluation Harness v2")
    parser.add_argument('--universe', type=str, default=None)
    parser.add_argument('--list-universes', action='store_true')
    parser.add_argument('--fee-bps', type=int, default=DEFAULT_FEE_BPS)
    parser.add_argument('--checkpoint-dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'v0.6_rl_checkpoints'))
    parser.add_argument('--axiom-checkpoint', type=str,
                        default=os.path.join(PROJECT_ROOT, 'checkpoints', 'axiom_multiseed',
                                             'axiom_seed42.pt'),
                        help="AxiomNet checkpoint for the 'Axiom' arm. Must be an "
                             "axiom_multiseed/axiom_seed*.pt file; FastTradingNet "
                             "checkpoints are rejected on load by design.")
    parser.add_argument('--output', type=str,
                        default=os.path.join(PROJECT_ROOT, 'data', 'canonical_results.json'))
    args = parser.parse_args()

    if args.list_universes:
        print("\nAvailable Universes:")
        for key, val in UNIVERSES.items():
            print(f"  {key:<25} {val['name']}")
            print(f"  {'':>25} PIT: {val.get('pit_source', 'N/A')}")
        return

    print("=" * 110, flush=True)
    print("  RAI CANONICAL EVALUATION HARNESS v2", flush=True)
    print(f"  Timestamp: {datetime.now().isoformat()}", flush=True)
    print(f"  Cost: {args.fee_bps}bps fee + {DEFAULT_SLIPPAGE_PCT}% slippage + {DEFAULT_REBAL_THRESH*100:.0f}% drift", flush=True)
    print(f"  Checkpoints: {args.checkpoint_dir}", flush=True)
    print(f"  NOTE: All results are single-seed, not CI-verified", flush=True)
    print("=" * 110, flush=True)

    # Load checkpoints
    #
    # The bare label "Axiom" resolves ONLY to checkpoints/axiom_multiseed/axiom_seed*.pt
    # (AxiomNet, 289,527 params). It used to resolve to data/v0.6_rl_checkpoints/axiom.pt,
    # which is a FastTradingNet checkpoint byte-identical to rai_v6_alpha.pt -- a
    # pre-disambiguation leftover, now renamed to
    # axiom_v0_prototype_fasttradingnet.pt and loaded under an explicit Fast-arch
    # label. See docs/consolidation_report.md §15 / §19.
    models = {}
    ckpt_dir = args.checkpoint_dir
    manifest = [
        ("RAI v6 Fast", os.path.join(ckpt_dir, "rai_v6_fast.pt"), "fast"),
        ("[v0 prototype, Fast-arch]",
         os.path.join(ckpt_dir, "axiom_v0_prototype_fasttradingnet.pt"), "fast"),
        ("Axiom", args.axiom_checkpoint, "axiom"),
    ]
    for label, path, arch in manifest:
        if os.path.isfile(path):
            try:
                loader = load_axiom_model if arch == "axiom" else load_v6_model
                models[label] = (loader(path), "v6")
                print(f"  [LOADED] {label} ({arch}): {path}", flush=True)
            except Exception as e:
                print(f"  [ERROR]  {label}: {e}", flush=True)
        else:
            print(f"  [SKIP]   {label}: not found at {path}", flush=True)

    v7_path = os.path.join(ckpt_dir, "rai_v7.pt")
    if os.path.isfile(v7_path):
        try:
            models["RAI v7"] = (load_v7_model(v7_path), "v7")
            print(f"  [LOADED] RAI v7: {v7_path}", flush=True)
        except Exception as e:
            print(f"  [ERROR]  RAI v7: {e}", flush=True)

    kaggle_dir = os.path.join(PROJECT_ROOT, 'checkpoints', 'kaggle_import')
    if os.path.isdir(kaggle_dir):
        for fname in sorted(os.listdir(kaggle_dir)):
            if fname.endswith('.pt'):
                path = os.path.join(kaggle_dir, fname)
                label = f"Kaggle: {fname.replace('.pt','')}"
                try:
                    try:
                        models[label] = (load_v6_model(path), "v6")
                    except Exception:
                        models[label] = (load_v7_model(path), "v7")
                    print(f"  [LOADED] {label}: {path}", flush=True)
                except Exception as e:
                    print(f"  [ERROR]  {label}: {e}", flush=True)

    if not models:
        print("\n  ERROR: No model checkpoints found.", flush=True)
        return

    print(f"\n  Models loaded: {len(models)}", flush=True)

    universe_keys = [args.universe] if args.universe else list(UNIVERSES.keys())
    all_results = {"_meta": {
        "timestamp": datetime.now().isoformat(),
        "cost_model": {"fee_bps": args.fee_bps, "slippage_pct": DEFAULT_SLIPPAGE_PCT, "rebal_threshold": DEFAULT_REBAL_THRESH},
        "seed_status": "single-seed, not CI-verified",
        "models_loaded": list(models.keys()),
    }}

    for ukey in universe_keys:
        if ukey not in UNIVERSES:
            print(f"\n  ERROR: Unknown universe '{ukey}'.", flush=True)
            continue
        result = run_evaluation_suite(ukey, models, fee_bps=args.fee_bps)
        if result:
            all_results[ukey] = result

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}", flush=True)
    print(f"\n{'='*110}", flush=True)
    print(f"  COMPLETE | Universes: {len(all_results)-1} | Arms per universe: {2+2+2+len(models)} (passive+rule+ML+RAI)", flush=True)
    print(f"{'='*110}", flush=True)


if __name__ == '__main__':
    main()
