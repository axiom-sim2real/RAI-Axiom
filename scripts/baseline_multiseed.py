"""
================================================================================
  MULTI-SEED BASELINE RE-RUN  (TASK 5)
================================================================================
  Re-runs the LSTM and XGBoost(GBT) real-data-trained baselines across the SAME
  10 seeds used for Axiom v0.9 -- [42, 101, 202, 303, 404, 505, 606, 707, 808,
  909] -- on the same 6 universes, and reports mean +/- SD with a bootstrap 95%
  CI, replacing the single-seed numbers.

  Why this script exists rather than canonical_evaluation.py:
    * canonical_evaluation.py trains ONE seed per universe.
    * Axiom's CI numbers were produced by scripts/kaggle_axiom_10seed.py on
      Kaggle, using period="10y"/"5y" relative downloads. To make the baselines
      comparable, the price windows are PINNED to explicit start/end dates that
      reproduce the Kaggle run's windows, and the script verifies this by
      re-evaluating the 10 saved Axiom checkpoints locally and comparing against
      data/axiom_per_seed_results.csv.

  Seed-propagation fix being exercised (consolidation_report.md 12):
    train_lstm_on_split()    -> torch.manual_seed(seed) + shuffled DataLoader
    train_xgboost_on_split() -> random_state=seed
  The script asserts both are present before running.

  Cost-model asymmetry:
    The existing baseline evaluators (evaluate_lstm_strategy /
    evaluate_xgb_strategy in canonical_evaluation.py) rebalance daily and charge
    NO transaction cost, while Axiom pays 5bps + 0.02% slippage on turnover
    above a 3% drift threshold. This script reports BOTH the as-published
    zero-cost variant (for like-for-like replacement of the old numbers) and a
    cost-charged variant using Axiom's cost model.

  Usage:
    python scripts/baseline_multiseed.py                 # full run
    python scripts/baseline_multiseed.py --repro-only    # only Axiom repro check
    python scripts/baseline_multiseed.py --universe US_ETFs
  Outputs:
    data/baseline_per_seed_results.csv
    data/baseline_multiseed_summary.json
    data/pinned_universes/<name>.csv   (price cache)
================================================================================
"""

import os
import sys
import json
import glob
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.canonical_evaluation import (  # noqa: E402
    train_lstm_on_split, evaluate_lstm_strategy,
    train_xgboost_on_split, evaluate_xgb_strategy,
    compute_metrics,
    DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_PCT, DEFAULT_REBAL_THRESH,
)
from scripts.kaggle_axiom_10seed import (  # noqa: E402
    AxiomNet, evaluate_on_real_data, UNIVERSES as KAGGLE_UNIVERSES,
)

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

# Kaggle Axiom run date (from consolidation_report.md 11 / backup timestamps).
# period="10y" / "5y" relative to that date -> pinned absolute windows.
PIN_END = "2026-08-20"
PIN_START = {"10y": "2016-08-20", "5y": "2021-08-20"}

CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "pinned_universes")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "axiom_multiseed")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "baseline_per_seed_results.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", "baseline_multiseed_summary.json")
AXIOM_CSV = os.path.join(PROJECT_ROOT, "data", "axiom_per_seed_results.csv")

N_BOOT = 10000
BOOT_SEED = 20260829


# ==============================================================================
#  SEED-FIX VERIFICATION
# ==============================================================================
def verify_seed_fix():
    """Confirm the seed-propagation fix is actually in the code paths we call."""
    src = os.path.join(PROJECT_ROOT, "scripts", "canonical_evaluation.py")
    with open(src, "r", encoding="utf-8") as f:
        code = f.read()
    checks = {
        "LSTM torch.manual_seed(seed)": "torch.manual_seed(seed)" in code,
        "LSTM DataLoader shuffle=True": "shuffle=True" in code,
        "LSTM generator seeded": "torch.Generator().manual_seed(seed)" in code,
        "XGB random_state=seed": "random_state=seed" in code,
        "no hardcoded random_state=42": "random_state=42" not in code,
    }
    import inspect
    checks["train_lstm_on_split takes seed"] = "seed" in inspect.signature(train_lstm_on_split).parameters
    checks["train_xgboost_on_split takes seed"] = "seed" in inspect.signature(train_xgboost_on_split).parameters

    print("  SEED-PROPAGATION FIX VERIFICATION")
    ok = True
    for k, v in checks.items():
        print("    [%s] %s" % ("OK " if v else "FAIL", k))
        ok = ok and v
    if not ok:
        raise SystemExit("  Seed fix is NOT fully applied. Aborting.")
    print()
    return checks


# ==============================================================================
#  DATA
# ==============================================================================
def load_universe(name, info, force=False):
    """Download (or load cached) pinned price window for a universe."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "%s.csv" % name)
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    import yfinance as yf
    start = PIN_START[info["period"]]
    df = yf.download(info["tickers"], start=start, end=PIN_END,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    # NOTE: columns are deliberately LEFT in yfinance's natural (alphabetical)
    # order. The Kaggle Axiom run never reordered them, and the policy's
    # per-asset logits are position-dependent, so reordering to the ticker-list
    # order changes every result. Verified: with alphabetical order and
    # end=2026-08-20 the 10 saved checkpoints reproduce
    # data/axiom_per_seed_results.csv exactly (max |err| = 0.000).
    df = df.dropna()
    df.to_csv(path, encoding="utf-8")
    return df



def split_60_20_20(df):
    T = len(df)
    i1, i2 = int(T * 0.6), int(T * 0.8)
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


# ==============================================================================
#  COST-CHARGED VARIANT OF THE BASELINE STRATEGIES
# ==============================================================================
def evaluate_signal_strategy_with_costs(predict_fn, prices_df, lookback=20,
                                        initial_cash=10000.0,
                                        fee_bps=DEFAULT_FEE_BPS,
                                        slippage_pct=DEFAULT_SLIPPAGE_PCT,
                                        rebal_thresh=DEFAULT_REBAL_THRESH):
    """
    Same 0.8/0.2 equal-weight-across-N signal strategy as the published LSTM /
    XGBoost evaluators, but charging Axiom's cost model on turnover and only
    trading when portfolio drift exceeds the 3% threshold.
    predict_fn(window_rets) -> 1 (risk-on) or 0 (risk-off).
    """
    prices = prices_df.values[:, :10] if isinstance(prices_df, pd.DataFrame) else prices_df[:, :10]
    T, N = prices.shape
    if T <= lookback:
        return [initial_cash] * T, 0
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0

    cash = initial_cash
    shares = np.zeros(N)
    eq = [initial_cash]
    rebal = 0
    for t in range(lookback, T - 1):
        sig = predict_fn(rets[t - lookback:t])
        target_stock = 0.8 if sig == 1 else 0.2
        tw = np.full(N, target_stock / N)
        wealth = max(1e-4, cash + float(np.sum(shares * prices[t])))
        cur_w = (shares * prices[t]) / wealth
        drift = abs(cash / wealth - (1.0 - target_stock)) + float(np.sum(np.abs(cur_w - tw)))
        if drift > rebal_thresh:
            turnover = abs(cash - wealth * (1.0 - target_stock)) + \
                float(np.sum(np.abs(shares * prices[t] - wealth * tw)))
            net = max(1e-4, wealth - turnover * fee_rate)
            cash = net * (1.0 - target_stock)
            shares = (net * tw) / np.maximum(1e-4, prices[t])
            rebal += 1
        eq.append(float(cash + np.sum(shares * prices[t + 1])))
    while len(eq) < T:
        eq.append(eq[-1])
    return eq, rebal


# ==============================================================================
#  AXIOM REPRODUCTION CHECK — confirms the pinned window matches the Kaggle run
# ==============================================================================
def load_axiom_checkpoint(path):
    model = AxiomNet(history_len=30, features_per_step=22, action_dim=11)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def axiom_repro_check(universes, verbose=True):
    """Re-evaluate the 10 saved Axiom checkpoints on the pinned windows and
    compare with data/axiom_per_seed_results.csv."""
    ref = pd.read_csv(AXIOM_CSV)
    ckpts = {}
    for seed in SEEDS:
        p = os.path.join(CKPT_DIR, "axiom_seed%d.pt" % seed)
        if os.path.exists(p):
            ckpts[seed] = load_axiom_checkpoint(p)
    if not ckpts:
        print("  WARNING: no Axiom checkpoints in %s — skipping repro check" % CKPT_DIR)
        return None

    rows = []
    for name, info in universes.items():
        df = load_universe(name, info)
        _, oos, fut = split_60_20_20(df)
        for seed, model in ckpts.items():
            oos_m = compute_metrics(evaluate_on_real_data(model, oos.values))
            fut_m = compute_metrics(evaluate_on_real_data(model, fut.values))
            r = ref[(ref["Universe"] == name) & (ref["Seed"] == seed)]
            rows.append({
                "Universe": name, "Seed": seed,
                "oos_repro": oos_m["sharpe"],
                "oos_ref": float(r["OOS Sharpe"].iloc[0]) if len(r) else np.nan,
                "fut_repro": fut_m["sharpe"],
                "fut_ref": float(r["Future Sharpe"].iloc[0]) if len(r) else np.nan,
            })
        if verbose:
            sub = pd.DataFrame([x for x in rows if x["Universe"] == name])
            print("    %-18s repro OOS %+.3f vs ref %+.3f | repro HOLD %+.3f vs ref %+.3f"
                  % (name, sub["oos_repro"].mean(), sub["oos_ref"].mean(),
                     sub["fut_repro"].mean(), sub["fut_ref"].mean()))
    rep = pd.DataFrame(rows)
    rep["oos_abs_err"] = (rep["oos_repro"] - rep["oos_ref"]).abs()
    rep["fut_abs_err"] = (rep["fut_repro"] - rep["fut_ref"]).abs()
    return rep


# ==============================================================================
#  BOOTSTRAP HELPERS (same methodology as scripts/aggregate_ci.py)
# ==============================================================================
def boot_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    v = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    draws = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def strat_boot_ci(mat, n_boot=N_BOOT, seed=BOOT_SEED):
    """mat[k universes x n seeds]; resample seeds within universe."""
    rng = np.random.default_rng(seed)
    k, n = mat.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=(k, n))
        draws[b] = np.take_along_axis(mat, idx, axis=1).mean(axis=1).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cluster_boot_ci(mat, n_boot=N_BOOT, seed=BOOT_SEED + 1):
    rng = np.random.default_rng(seed)
    k, n = mat.shape
    draws = np.empty(n_boot)
    for b in range(n_boot):
        u = rng.integers(0, k, size=k)
        s = rng.integers(0, n, size=(k, n))
        draws[b] = np.take_along_axis(mat[u], s, axis=1).mean(axis=1).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ==============================================================================
#  MAIN
# ==============================================================================
def run_baselines(universes):
    rows = []
    for name, info in universes.items():
        df = load_universe(name, info)
        train, oos, fut = split_60_20_20(df)
        print("\n  --- %s --- %d days (train %d / oos %d / holdout %d)  %s..%s"
              % (name, len(df), len(train), len(oos), len(fut),
                 df.index[0].date(), df.index[-1].date()), flush=True)

        for seed in SEEDS:
            # ---- LSTM ----
            lstm = train_lstm_on_split(train, seed=seed)
            l_oos = compute_metrics(evaluate_lstm_strategy(lstm, oos))
            l_fut = compute_metrics(evaluate_lstm_strategy(lstm, fut))

            def lstm_pred(w, _m=lstm):
                x = torch.FloatTensor(w).unsqueeze(0)
                with torch.no_grad():
                    return 1 if torch.sigmoid(_m(x)).item() > 0.5 else 0

            lc_oos = compute_metrics(evaluate_signal_strategy_with_costs(lstm_pred, oos)[0])
            lc_fut = compute_metrics(evaluate_signal_strategy_with_costs(lstm_pred, fut)[0])

            # ---- XGBoost (sklearn GradientBoostingClassifier) ----
            xgb = train_xgboost_on_split(train, seed=seed)
            x_oos = compute_metrics(evaluate_xgb_strategy(xgb, oos))
            x_fut = compute_metrics(evaluate_xgb_strategy(xgb, fut))

            def xgb_pred(w, _c=xgb):
                return int(_c.predict(w.flatten().reshape(1, -1))[0])

            xc_oos = compute_metrics(evaluate_signal_strategy_with_costs(xgb_pred, oos)[0])
            xc_fut = compute_metrics(evaluate_signal_strategy_with_costs(xgb_pred, fut)[0])

            for model, o, f, oc, fc in (
                    ("LSTM (real)", l_oos, l_fut, lc_oos, lc_fut),
                    ("XGBoost (real)", x_oos, x_fut, xc_oos, xc_fut)):
                rows.append({
                    "Universe": name, "Model": model, "Seed": seed,
                    "OOS Sharpe": o["sharpe"], "OOS Return (%)": o["return_pct"],
                    "OOS Max DD (%)": o["max_dd"],
                    "Future Sharpe": f["sharpe"], "Future Return (%)": f["return_pct"],
                    "Future Max DD (%)": f["max_dd"],
                    "OOS Sharpe (costed)": oc["sharpe"],
                    "Future Sharpe (costed)": fc["sharpe"],
                })
            print("    seed %-4d LSTM oos %+.3f hold %+.3f | XGB oos %+.3f hold %+.3f"
                  % (seed, l_oos["sharpe"], l_fut["sharpe"], x_oos["sharpe"], x_fut["sharpe"]),
                  flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    out = {}
    for model in sorted(df["Model"].unique()):
        sub = df[df["Model"] == model]
        universes = list(dict.fromkeys(sub["Universe"]))
        entry = {"per_universe": {}, "n_seeds": int(sub.groupby("Universe").size().min())}
        for col, key in (("OOS Sharpe", "oos"), ("Future Sharpe", "holdout"),
                         ("OOS Sharpe (costed)", "oos_costed"),
                         ("Future Sharpe (costed)", "holdout_costed")):
            mat = np.vstack([sub.loc[sub["Universe"] == u, col].to_numpy(float) for u in universes])
            per_u = {}
            for u, r in zip(universes, mat):
                lo, hi = boot_ci(r)
                per_u[u] = {"mean": float(r.mean()), "sd": float(r.std(ddof=1)),
                            "ci95": [lo, hi]}
            s_lo, s_hi = strat_boot_ci(mat)
            c_lo, c_hi = cluster_boot_ci(mat)
            u_means = mat.mean(axis=1)
            entry[key] = {
                "per_universe": per_u,
                "overall_mean": float(u_means.mean()),
                "seed_level_ci95_stratified_boot": [s_lo, s_hi],
                "market_level_ci95_cluster_boot": [c_lo, c_hi],
                "cross_universe_dispersion_sd": float(u_means.std(ddof=1)),
            }
        entry.pop("per_universe")
        out[model] = entry
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=None, help="run a single universe")
    ap.add_argument("--repro-only", action="store_true")
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()

    universes = dict(KAGGLE_UNIVERSES)
    if args.universe:
        universes = {args.universe: universes[args.universe]}

    print("=" * 100)
    print("  MULTI-SEED BASELINE RE-RUN — LSTM + XGBoost, seeds %s" % SEEDS)
    print("  pinned window: start %s / %s  end %s" % (PIN_START["10y"], PIN_START["5y"], PIN_END))
    print("=" * 100)
    checks = verify_seed_fix()

    if not args.skip_repro:
        print("  AXIOM REPRODUCTION CHECK (pinned window vs Kaggle per-seed CSV)")
        rep = axiom_repro_check(universes)
        if rep is not None:
            print("    max |err| OOS = %.3f   holdout = %.3f   mean |err| = %.3f / %.3f"
                  % (rep["oos_abs_err"].max(), rep["fut_abs_err"].max(),
                     rep["oos_abs_err"].mean(), rep["fut_abs_err"].mean()))
            rep.to_csv(os.path.join(PROJECT_ROOT, "data", "axiom_repro_check.csv"), index=False)
    if args.repro_only:
        return

    df = run_baselines(universes)
    df.to_csv(OUT_CSV, index=False)
    print("\n  Wrote %s (%d rows)" % (os.path.relpath(OUT_CSV, PROJECT_ROOT), len(df)))

    summary = summarize(df)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated": "2026-08-29", "seeds": SEEDS,
                   "pinned_start": PIN_START, "pinned_end": PIN_END,
                   "seed_fix_checks": checks, "summary": summary}, f, indent=2)
    print("  Wrote %s" % os.path.relpath(OUT_JSON, PROJECT_ROOT))

    for model, e in summary.items():
        for key, label in (("oos", "OOS"), ("holdout", "HOLDOUT")):
            k = e[key]
            print("\n  %s — %s (10 seeds, zero-cost harness as published)" % (model, label))
            for u, v in k["per_universe"].items():
                print("    %-20s %+.3f +/- %.3f  CI [%+.2f, %+.2f]"
                      % (u, v["mean"], v["sd"], v["ci95"][0], v["ci95"][1]))
            print("    %-20s %+.3f  seed-level CI [%+.2f, %+.2f]  market-level CI [%+.2f, %+.2f]  dispersion SD %.2f"
                  % ("OVERALL", k["overall_mean"],
                     k["seed_level_ci95_stratified_boot"][0], k["seed_level_ci95_stratified_boot"][1],
                     k["market_level_ci95_cluster_boot"][0], k["market_level_ci95_cluster_boot"][1],
                     k["cross_universe_dispersion_sd"]))
            ck = e[key + "_costed"]
            print("    %-20s %+.3f  (cost-charged variant, 5bps + 0.02%% slip, 3%% drift)"
                  % ("OVERALL costed", ck["overall_mean"]))


if __name__ == "__main__":
    main()



