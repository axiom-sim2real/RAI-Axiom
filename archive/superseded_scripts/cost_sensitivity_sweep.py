"""
================================================================================
  P1 TASK 5: Transaction Cost Sensitivity Sweep
  
  Sweeps cost tiers [0bps, 5bps, 10bps, 25bps] across all three v6 variants
  on the 2020-2024 out-of-sample period. Reports Sharpe / Return / MaxDD at
  each tier. Discloses current cost assumptions from env files.
================================================================================
"""
import os, sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.download_data import ensure_real_market_checkpoints
from scripts.train_v6_fast import FastTradingNet
from scripts.eval_vs_standard_ai import metrics

COST_TIERS_BPS = [0, 5, 10, 25]  # basis points: 0bps=frictionless, 25bps=high-cost broker
SLIPPAGE_PCT   = 0.02             # 0.02% random price slippage at execution (fixed)
REBAL_THRESH   = 0.03             # 3% drift threshold to trigger rebalancing

V6_VARIANTS = {
    "RAI v6 Fast":       "rai_v6_fast.pt",
    "RAI v6 Alpha":      "rai_v6_alpha.pt",
    "RAI v6 Pro-Growth": "rai_v6_pro_growth.pt",
}


def eval_v6_with_costs(model, df, fee_bps, slippage=SLIPPAGE_PCT, thresh=REBAL_THRESH):
    prices_raw = df.values[:, :10]
    T, N = prices_raw.shape
    if T < 35:
        return [10000.0]

    cash = 5000.0
    shares = (5000.0 / N) / prices_raw[30]
    peak = 10000.0
    wealth_hist = [10000.0]
    rng = np.random.RandomState(42)
    rebal_count = 0

    obs_history = []
    for t in range(30):
        p = prices_raw[t]; p_prev = prices_raw[max(0, t - 1)]
        obs_history.append(np.concatenate([
            p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [0.5, 0.0]
        ]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)
        cl = np.clip(act[0] - 2.5, -8.0, 3.0)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash
        ea = np.exp(act[1:] - np.max(act[1:]))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_raw[t]
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w; ccf = cash / w
        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))

        if drift > thresh:
            rebal_count += 1
            p_ex = p * (1.0 + rng.uniform(-slippage / 100., slippage / 100., N))
            tv = abs(cash - w * target_cash) + np.sum(np.abs(shares * p - w * target_aw))
            fee = fee_bps / 10000.0
            net = max(1e-4, w - tv * fee)
            cash = net * target_cash
            shares = (net * target_aw) / np.maximum(1e-4, p_ex)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        wealth_hist.append(nw)

        p_prev = prices_raw[t - 1]
        obs_history.pop(0)
        obs_history.append(np.concatenate([
            p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)),
            [cash / nw, np.clip((nw - peak) / peak, -1, 0)]
        ]).astype(np.float32))

    return wealth_hist, rebal_count


def main():
    ensure_real_market_checkpoints()
    test_csv = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    test_df  = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    ckpt_dir = os.path.join(PROJECT_ROOT, "data", "v0.6_rl_checkpoints")

    # ── Audit environment cost assumptions ────────────────────────────────────
    print("=" * 90, flush=True)
    print("  COST ASSUMPTION AUDIT — Current Hardcoded Values", flush=True)
    print("=" * 90, flush=True)
    env_files = ["rai/world/real_ai_env.py", "rai/world/real_env.py", "rai/world/env.py"]
    for ef in env_files:
        fpath = os.path.join(PROJECT_ROOT, ef)
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            cost_lines = [l.strip() for l in content.splitlines()
                          if any(k in l.lower() for k in ['fee', 'cost', 'slip', 'commission', 'bps', 'transaction'])]
            if cost_lines:
                print(f"\n  [{ef}]")
                for cl in cost_lines[:8]:
                    print(f"    {cl}")
            else:
                print(f"\n  [{ef}] — No explicit cost/slippage terms found (may be 0-cost)")
        else:
            print(f"\n  [{ef}] — File not found")

    print(f"\n  eval scripts use: fee_bps=5, slippage=0.02%, rebal_thresh=3%", flush=True)

    # ── Load variants ─────────────────────────────────────────────────────────
    loaded = {}
    for label, fname in V6_VARIANTS.items():
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            m = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            loaded[label] = m

    if not loaded:
        print("\n  No checkpoints found. Train models first.", flush=True)
        return

    # ── SPY benchmark (frictionless — buy and hold) ────────────────────────
    spy = test_df['SPY'].values if 'SPY' in test_df.columns else test_df.values[:, 0]
    eq_spy_wh = (10000.0 * spy / spy[0]).tolist()
    m_spy = metrics(eq_spy_wh)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*90}", flush=True)
    print(f"  COST SENSITIVITY SWEEP — 2020-2024 Out-of-Sample", flush=True)
    print(f"  Slippage: {SLIPPAGE_PCT}% fixed | Rebalance threshold: {REBAL_THRESH*100:.0f}% drift", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"  {'Model':<25} | {'Cost':>6} | {'Return':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Rebal#':>7} | vs SPY Sharpe"
    print(f"\n{header}")
    print(f"  {'-'*90}")

    # SPY row (0-cost benchmark)
    print(f"  {'SPY Buy & Hold':<25} | {'0bps':>6} | {m_spy['return_pct']:>+7.1f}% | {m_spy['sharpe']:>7.2f} | {m_spy['max_dd']:>+7.1f}% | {'N/A':>7} | baseline")
    print(f"  {'-'*90}")


    for label, model in loaded.items():
        for fee_bps in COST_TIERS_BPS:
            result = eval_v6_with_costs(model, test_df, fee_bps=fee_bps)
            if isinstance(result, tuple):
                wh, rebal_n = result
            else:
                wh, rebal_n = result, 0
            m = metrics(wh)
            vs_spy = "BEATS" if m['sharpe'] > m_spy['sharpe'] else "below"
            print(f"  {label:<25} | {fee_bps:>4}bps | {m['return_pct']:>+7.1f}% | {m['sharpe']:>7.2f} | {m['max_dd']:>+7.1f}% | {rebal_n:>7} | {vs_spy} SPY ({m_spy['sharpe']:.2f})")

        print(f"  {'-'*90}")

    print(f"\n  Key: 0bps=frictionless, 5bps=low-cost ETF, 10bps=typical retail, 25bps=high-cost")
    print(f"  Rebal# = number of rebalancing events triggered (3% drift threshold)", flush=True)


if __name__ == "__main__":
    main()
