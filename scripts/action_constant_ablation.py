"""
================================================================================
  P2 TASK 7: Action-Decoding Constant Ablation
  
  Sweeps three hand-coded constants in the action-decoding layer:
    1. cash_logit_offset: the -2.5 subtracted from act[0] before sigmoid
    2. clip_range: the [-8.0, 3.0] applied to the logit
    3. rebal_threshold: the 3% drift threshold before rebalancing triggers
  
  Reports sensitivity of Sharpe / Return / MaxDD to each sweep.
  Justifies (or flags) the "end-to-end learned" framing.
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
# eval_vs_standard_ai was moved to archive/superseded_scripts/ during
# consolidation; its metrics() is byte-identical to the canonical harness's
# compute_metrics(), so import from the canonical harness instead.
from scripts.canonical_evaluation import compute_metrics as metrics


# ── Sweep ranges ──────────────────────────────────────────────────────────────
CASH_LOGIT_OFFSETS = [-3.5, -2.5, -1.5, -0.5, 0.0]   # default = -2.5
CLIP_RANGES        = [(-6, 2), (-8, 3), (-10, 4), (-12, 5)]   # default = (-8, 3)
REBAL_THRESHOLDS   = [0.01, 0.02, 0.03, 0.05, 0.10]    # default = 0.03

DEFAULT_OFFSET    = -2.5
DEFAULT_CLIP      = (-8.0, 3.0)
DEFAULT_REBAL     = 0.03
FEE_BPS           = 5
SLIPPAGE          = 0.02

V6_VARIANTS = {
    "RAI v6 Fast":  "rai_v6_fast.pt",
    # NOT Axiom. This file (formerly axiom.pt, byte-identical to rai_v6_alpha.pt)
    # is a FastTradingNet checkpoint that was mislabelled "axiom" before the two
    # architectures were disambiguated on 2026-08-29. The real Axiom arm is
    # AxiomNet and lives in checkpoints/axiom_multiseed/axiom_seed*.pt -- it is
    # swept by scripts/action_constant_ablation_multiuniverse.py, not here.
    # See docs/consolidation_report.md §15 / §19.
    "[v0 prototype, Fast-arch]": "axiom_v0_prototype_fasttradingnet.pt",
}



def eval_v6_custom(model, df, logit_offset=DEFAULT_OFFSET,
                   clip_lo=DEFAULT_CLIP[0], clip_hi=DEFAULT_CLIP[1],
                   rebal_thresh=DEFAULT_REBAL, fee_bps=FEE_BPS):
    prices_raw = df.values[:, :10]
    T, N = prices_raw.shape
    if T < 35:
        return [10000.0]

    cash = 5000.0
    shares = (5000.0 / N) / prices_raw[30]
    peak = 10000.0
    wealth_hist = [10000.0]
    rng = np.random.RandomState(42)

    obs_history = []
    for t in range(30):
        p = prices_raw[t]; p_prev = prices_raw[max(0, t - 1)]
        obs_history.append(np.concatenate([
            p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)), [0.5, 0.0]
        ]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)
        # Apply swept constants:
        cl = np.clip(act[0] + logit_offset, clip_lo, clip_hi)
        target_cash = 1.0 / (1.0 + np.exp(-cl))
        target_stock = 1.0 - target_cash
        ea = np.exp(act[1:] - np.max(act[1:]))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_raw[t]
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w; ccf = cash / w
        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))

        if drift > rebal_thresh:
            tv = abs(cash - w * target_cash) + np.sum(np.abs(shares * p - w * target_aw))
            net = max(1e-4, w - tv * fee_bps / 10000.0)
            cash = net * target_cash
            shares = (net * target_aw) / np.maximum(1e-4, p * (1 + rng.uniform(-SLIPPAGE/100, SLIPPAGE/100, N)))

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        wealth_hist.append(nw)

        p_prev = prices_raw[t - 1]
        obs_history.pop(0)
        obs_history.append(np.concatenate([
            p / prices_raw[30], np.log(p / np.maximum(1e-4, p_prev)),
            [cash / nw, np.clip((nw - peak) / peak, -1, 0)]
        ]).astype(np.float32))

    return wealth_hist


def print_sweep_table(label, param_name, param_values, default_val, results, spy_sharpe):
    W = 80
    print(f"\n  {'─'*W}")
    print(f"  MODEL: {label} | SWEEP: {param_name} (default = {default_val})")
    print(f"  {'─'*W}")
    print(f"  {param_name:<20} | {'Return':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Fragile?':>10}")
    print(f"  {'-'*60}")
    base_ret, base_sh = None, None
    for idx, (val, (ret, sh, dd)) in enumerate(zip(param_values, results)):
        is_default = str(val) == str(default_val) or idx == 0 and base_sh is None and str(val) == str(default_val)
        marker = " <-- DEFAULT" if is_default else ""
        if is_default:
            base_ret, base_sh = ret, sh
        fragile = ""
        if base_sh is not None and abs(sh - base_sh) > 0.15:
            fragile = "FRAGILE"
        print(f"  {str(val):<20} | {ret:>+7.1f}% | {sh:>7.2f} | {dd:>+7.1f}% | {fragile:<10}{marker}")
    print(f"  {'-'*60}")



def main():
    ensure_real_market_checkpoints()
    test_csv = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    test_df  = pd.read_csv(test_csv, index_col=0, parse_dates=True)
    ckpt_dir = os.path.join(PROJECT_ROOT, "data", "v0.6_rl_checkpoints")

    spy = test_df['SPY'].values if 'SPY' in test_df.columns else test_df.values[:, 0]
    spy_wh = (10000.0 * spy / spy[0]).tolist()
    spy_sharpe = metrics(spy_wh)['sharpe']

    loaded = {}
    for label, fname in V6_VARIANTS.items():
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            m = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            loaded[label] = m

    if not loaded:
        print("No checkpoints found. Train models first.")
        return

    print("=" * 90, flush=True)
    print("  P2 TASK 7: ACTION-DECODING CONSTANT ABLATION", flush=True)
    print("  Tests whether Sharpe/Return/MaxDD are fragile to manually-set constants", flush=True)
    print("  A FRAGILE flag means Sharpe changes by >0.15 from the default", flush=True)
    print("=" * 90, flush=True)
    print(f"\n  Constants being tested:", flush=True)
    print(f"    cash_logit_offset: act[0] + offset → clipped → sigmoid → cash fraction")
    print(f"    clip_range:        (lo, hi) applied to the logit before sigmoid")
    print(f"    rebal_threshold:   minimum portfolio drift % before rebalancing fires")
    print(f"    SPY Sharpe (reference): {spy_sharpe:.2f}", flush=True)

    for label, model in loaded.items():
        print(f"\n\n  {'#'*70}")
        print(f"  MODEL: {label}")
        print(f"  {'#'*70}", flush=True)

        # Sweep 1: cash_logit_offset
        res1 = []
        for offset in CASH_LOGIT_OFFSETS:
            wh = eval_v6_custom(model, test_df, logit_offset=offset,
                                clip_lo=DEFAULT_CLIP[0], clip_hi=DEFAULT_CLIP[1],
                                rebal_thresh=DEFAULT_REBAL)
            m = metrics(wh)
            res1.append((m['return_pct'], m['sharpe'], m['max_dd']))
        print_sweep_table(label, "cash_logit_offset", CASH_LOGIT_OFFSETS, DEFAULT_OFFSET, res1, spy_sharpe)

        # Sweep 2: clip_range
        res2 = []
        for cr in CLIP_RANGES:
            wh = eval_v6_custom(model, test_df, logit_offset=DEFAULT_OFFSET,
                                clip_lo=cr[0], clip_hi=cr[1],
                                rebal_thresh=DEFAULT_REBAL)
            m = metrics(wh)
            res2.append((m['return_pct'], m['sharpe'], m['max_dd']))
        print_sweep_table(label, "clip_range", [str(c) for c in CLIP_RANGES],
                          str(DEFAULT_CLIP), res2, spy_sharpe)

        # Sweep 3: rebal_threshold
        res3 = []
        for rt in REBAL_THRESHOLDS:
            wh = eval_v6_custom(model, test_df, logit_offset=DEFAULT_OFFSET,
                                clip_lo=DEFAULT_CLIP[0], clip_hi=DEFAULT_CLIP[1],
                                rebal_thresh=rt)
            m = metrics(wh)
            res3.append((m['return_pct'], m['sharpe'], m['max_dd']))
        print_sweep_table(label, "rebal_threshold", [f"{x*100:.0f}%" for x in REBAL_THRESHOLDS],
                          f"{DEFAULT_REBAL*100:.0f}%", res3, spy_sharpe)

    print(f"\n\n  INTERPRETATION GUIDE:")
    print(f"  - If Sharpe stays stable across all sweeps: policy is robust to constant choices.")
    print(f"  - If FRAGILE flags appear: disclose in paper that these constants materially affect results.")
    print(f"  - 'end-to-end' framing is defensible only if cash_logit_offset and clip_range sweeps are stable.")


if __name__ == "__main__":
    main()
