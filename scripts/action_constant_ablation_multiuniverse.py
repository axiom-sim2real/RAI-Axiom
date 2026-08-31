"""
================================================================================
  ACTION-CONSTANT ABLATION OUTSIDE EQUITIES  (TASK 6)
================================================================================
  The cash-logit offset (-2.5) was validated as "not fragile" ONLY on the US
  equities / sector-ETF 2020-2024 window (scripts/action_constant_ablation.py).
  This script re-runs the SAME offset sweep -- [-3.5, -2.5, -1.5, -0.5, 0.0] --
  on the universes where the claim was never tested:

      India_Nifty_50, Forex_Commodities, Crypto_PIT
      + US_ETFs as an in-scope equities control

  Differences from the original ablation, deliberately:
    * Uses the SAME evaluation harness that produced the published Axiom CI
      numbers (evaluate_on_real_data from kaggle_axiom_10seed.py), with the
      hardcoded -2.5 lifted into a parameter. Verified to reproduce
      data/axiom_per_seed_results.csv at offset=-2.5.
    * Runs all 10 Axiom seeds, so "fragile" is judged against seed noise rather
      than against a single checkpoint. Also runs the single RAI v6 Fast
      checkpoint, because the documented crypto failure (98.9% BCH-USD) is a
      Fast failure.
    * Evaluates both the OOS and the future-holdout window.

  Fragility rule (same threshold as the original script): an offset is FRAGILE
  if the mean Sharpe moves more than 0.15 from the -2.5 default. For the 10-seed
  Axiom arm the paired-by-seed delta and its SD are also reported, so the shift
  can be compared against seed-level noise.

  Usage:  python scripts/action_constant_ablation_multiuniverse.py
  Output: data/action_constant_ablation_multiuniverse.csv
          data/action_constant_ablation_multiuniverse.json
================================================================================
"""

import os
import sys
import json
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

from scripts.canonical_evaluation import compute_metrics  # noqa: E402
from scripts.kaggle_axiom_10seed import (  # noqa: E402
    AxiomNet, UNIVERSES as KAGGLE_UNIVERSES,
)
# NOTE: AxiomNet and FastTradingNet are two DIFFERENT architectures, not two
# checkpoints of one. AxiomNet (Kaggle Axiom seeds) is conv1/conv2 ->
# flatten(64*30) -> fc(128)+LayerNorm -> actor/critic, 289,527 params.
# FastTradingNet (local Fast / alpha checkpoints) is a conv1d Sequential ->
# mean-pool -> fc(128) -> actor_head/critic_head, 51,703 params. They are not
# state_dict-compatible, so each family is loaded with its own class. Both
# expose the same get_action(flat_obs, deterministic=True) -> 11-dim contract,
# which is why the offset sweep below can treat them as interchangeable
# *policies* even though they are not interchangeable *networks*.
from scripts.train_v6_fast import FastTradingNet  # noqa: E402

from scripts.baseline_multiseed import (  # noqa: E402
    load_universe, split_60_20_20, SEEDS, CKPT_DIR,
)

OFFSETS = [-3.5, -2.5, -1.5, -0.5, 0.0]
DEFAULT_OFFSET = -2.5
FRAGILE_THRESH = 0.15

TARGET_UNIVERSES = ["US_ETFs", "India_Nifty_50", "Forex_Commodities", "Crypto_PIT"]
EQUITY_CONTROL = "US_ETFs"

OUT_CSV = os.path.join(PROJECT_ROOT, "data", "action_constant_ablation_multiuniverse.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", "action_constant_ablation_multiuniverse.json")
FAST_CKPT = os.path.join(PROJECT_ROOT, "data", "v0.6_rl_checkpoints", "rai_v6_fast.pt")


def evaluate_with_offset(model, prices, cash_logit_offset=DEFAULT_OFFSET,
                         clip=(-8.0, 3.0), fee_bps=5, slippage_pct=0.02,
                         rebal_thresh=0.03):
    """
    Exact mirror of kaggle_axiom_10seed.evaluate_on_real_data with the
    cash-logit offset, clip range and rebalance threshold exposed as arguments.
    Returns (equity_curve, rebalance_count, mean_cash_fraction, mean_top_weight).
    """
    T, N = prices.shape
    history_len = 30
    if T <= history_len + 1:
        return np.ones(T) * 10000.0, 0, np.nan, np.nan

    norm_prices = prices / prices[0]
    cash = 10000.0 * 0.05
    shares = (10000.0 * 0.95 / N) / prices[history_len]
    peak = 10000.0
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0

    obs_history = []
    for i in range(history_len):
        p = norm_prices[i]
        p_prev = norm_prices[max(0, i - 1)]
        w = max(1e-4, cash + np.sum(shares * prices[i]))
        obs_history.append(np.concatenate([
            p, np.log(p / np.maximum(1e-4, p_prev)),
            [cash / w, np.clip((w - peak) / max(1e-4, peak), -1.0, 0.0)]
        ]).astype(np.float32))

    equity = [10000.0] * history_len
    rebal = 0
    cash_fracs = []
    top_ws = []

    for t in range(history_len, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        action = model.get_action(flat_obs, deterministic=True)

        cash_logit = np.clip(action[0] + cash_logit_offset, clip[0], clip[1])
        target_cash = 1.0 / (1.0 + np.exp(-cash_logit))
        stock_portion = 1.0 - target_cash
        asset_logits = action[1:N + 1] if len(action) > N else action[1:]
        exp_a = np.exp(asset_logits - np.max(asset_logits))
        target_w = (exp_a / np.sum(exp_a)) * stock_portion

        wealth = max(1e-4, cash + np.sum(shares * prices[t]))
        cur_w = (shares * prices[t]) / wealth
        cur_cash = cash / wealth
        drift = abs(cur_cash - target_cash) + np.sum(np.abs(cur_w - target_w))

        if drift > rebal_thresh:
            t_vol = abs(cash - wealth * target_cash) + \
                np.sum(np.abs(shares * prices[t] - wealth * target_w))
            net = max(1e-4, wealth - t_vol * fee_rate)
            cash = net * target_cash
            shares = (net * target_w) / np.maximum(1e-4, prices[t])
            rebal += 1

        wealth = cash + np.sum(shares * prices[t])
        peak = max(peak, wealth)
        equity.append(wealth)
        cash_fracs.append(cash / max(1e-4, wealth))
        top_ws.append(float(np.max((shares * prices[t]) / max(1e-4, wealth))))

        p = norm_prices[t]
        p_prev = norm_prices[max(0, t - 1)]
        cash_r = cash / max(1e-4, wealth)
        dd = np.clip((wealth - peak) / max(1e-4, peak), -1.0, 0.0)
        obs_history.pop(0)
        obs_history.append(np.concatenate([
            p, np.log(p / np.maximum(1e-4, p_prev)), [cash_r, dd]
        ]).astype(np.float32))

    return (np.array(equity), rebal,
            float(np.mean(cash_fracs)) if cash_fracs else np.nan,
            float(np.mean(top_ws)) if top_ws else np.nan)


def load_ckpt(path, cls=AxiomNet):
    model = cls(history_len=30, features_per_step=22, action_dim=11)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def collect(universes):
    rows = []
    axiom = {}
    for s in SEEDS:
        p = os.path.join(CKPT_DIR, "axiom_seed%d.pt" % s)
        if os.path.exists(p):
            axiom[s] = load_ckpt(p)
    fast = load_ckpt(FAST_CKPT, cls=FastTradingNet) if os.path.exists(FAST_CKPT) else None
    print("  loaded %d Axiom seed checkpoints; Fast checkpoint: %s"
          % (len(axiom), "yes" if fast else "MISSING"))

    for uname in universes:
        df = load_universe(uname, KAGGLE_UNIVERSES[uname])
        _, oos, fut = split_60_20_20(df)
        print("\n  --- %s --- oos %dd / holdout %dd" % (uname, len(oos), len(fut)), flush=True)
        for window, arr in (("OOS", oos.values), ("Holdout", fut.values)):
            for offset in OFFSETS:
                for seed, model in axiom.items():
                    eq, rb, cf, tw = evaluate_with_offset(model, arr, cash_logit_offset=offset)
                    m = compute_metrics(eq)
                    rows.append({"Universe": uname, "Window": window, "Model": "Axiom v0.9",
                                 "Seed": seed, "Offset": offset, "Sharpe": m["sharpe"],
                                 "Return (%)": m["return_pct"], "Max DD (%)": m["max_dd"],
                                 "Rebalances": rb, "Mean Cash Frac": cf, "Mean Top Weight": tw})
                if fast is not None:
                    eq, rb, cf, tw = evaluate_with_offset(fast, arr, cash_logit_offset=offset)
                    m = compute_metrics(eq)
                    rows.append({"Universe": uname, "Window": window, "Model": "RAI v6 Fast",
                                 "Seed": -1, "Offset": offset, "Sharpe": m["sharpe"],
                                 "Return (%)": m["return_pct"], "Max DD (%)": m["max_dd"],
                                 "Rebalances": rb, "Mean Cash Frac": cf, "Mean Top Weight": tw})
                print("    %-8s offset %+.1f done" % (window, offset), flush=True)
    return pd.DataFrame(rows)


def analyse(df):
    """Per (universe, window, model): mean Sharpe by offset, delta vs -2.5,
    paired-by-seed delta SD, and a FRAGILE verdict."""
    out = []
    for (u, w, mdl), sub in df.groupby(["Universe", "Window", "Model"], sort=False):
        piv = sub.pivot_table(index="Seed", columns="Offset", values="Sharpe")
        base = piv[DEFAULT_OFFSET]
        entry = {"Universe": u, "Window": w, "Model": mdl,
                 "n_seeds": int(len(piv)), "offsets": {}}
        fragile_offsets = []
        for off in OFFSETS:
            col = piv[off]
            delta = col - base
            mean_sharpe = float(col.mean())
            mean_delta = float(delta.mean())
            frag = (off != DEFAULT_OFFSET) and abs(mean_delta) > FRAGILE_THRESH
            if frag:
                fragile_offsets.append(off)
            entry["offsets"][str(off)] = {
                "mean_sharpe": mean_sharpe,
                "sd_sharpe": float(col.std(ddof=1)) if len(col) > 1 else 0.0,
                "mean_delta_vs_default": mean_delta,
                "sd_paired_delta": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
                "mean_return_pct": float(sub.loc[sub["Offset"] == off, "Return (%)"].mean()),
                "mean_max_dd_pct": float(sub.loc[sub["Offset"] == off, "Max DD (%)"].mean()),
                "mean_cash_frac": float(sub.loc[sub["Offset"] == off, "Mean Cash Frac"].mean()),
                "mean_top_weight": float(sub.loc[sub["Offset"] == off, "Mean Top Weight"].mean()),
                "mean_rebalances": float(sub.loc[sub["Offset"] == off, "Rebalances"].mean()),
                "fragile": bool(frag),
            }
        entry["fragile_offsets"] = fragile_offsets
        entry["verdict"] = "FRAGILE" if fragile_offsets else "not fragile"
        entry["max_abs_delta"] = max(abs(v["mean_delta_vs_default"])
                                     for v in entry["offsets"].values())
        # Sharpe is a ratio and is largely invariant to scaling exposure up or
        # down, so a "not fragile" Sharpe verdict does NOT mean the constant is
        # inconsequential. Track the swing in level metrics separately.
        rets = [v["mean_return_pct"] for v in entry["offsets"].values()]
        dds = [v["mean_max_dd_pct"] for v in entry["offsets"].values()]
        cash = [v["mean_cash_frac"] for v in entry["offsets"].values()]
        entry["return_pct_range"] = float(max(rets) - min(rets))
        entry["max_dd_pct_range"] = float(max(dds) - min(dds))
        entry["cash_frac_range"] = float(max(cash) - min(cash))
        out.append(entry)
    return out


def print_report(analysis):
    print("\n" + "=" * 104)
    print("  CASH-LOGIT OFFSET SWEEP — mean Sharpe by offset (default = -2.5)")
    print("  FRAGILE = mean Sharpe moves more than %.2f from the default" % FRAGILE_THRESH)
    print("=" * 104)
    hdr = "  %-18s %-8s %-13s %-6s" % ("Universe", "Window", "Model", "n") + \
          "".join("%9s" % ("%+.1f" % o) for o in OFFSETS) + "   %-9s %s" % ("max|d|", "verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for e in analysis:
        line = "  %-18s %-8s %-13s %-6d" % (e["Universe"], e["Window"], e["Model"], e["n_seeds"])
        for o in OFFSETS:
            v = e["offsets"][str(o)]
            mark = "*" if v["fragile"] else " "
            line += "%8.2f%s" % (v["mean_sharpe"], mark)

        line += "   %-9.2f %s" % (e["max_abs_delta"], e["verdict"])
        print(line)
    print("  (* = FRAGILE at that offset)")

    print("\n  SHARPE IS A RATIO — the same sweep expressed in LEVEL metrics")
    print("  (swing from the best to the worst offset within each arm)")
    print("  %-18s %-8s %-13s %10s %10s %10s"
          % ("Universe", "Window", "Model", "ret swing", "DD swing", "cash swing"))
    for e in analysis:
        print("  %-18s %-8s %-13s %9.1fpp %9.1fpp %9.1fpp"
              % (e["Universe"], e["Window"], e["Model"], e["return_pct_range"],
                 e["max_dd_pct_range"], 100 * e["cash_frac_range"]))

    print("\n  CASH FRACTION / CONCENTRATION AT EACH OFFSET (holdout window)")
    print("  %-18s %-13s %-8s %10s %10s %10s" %
          ("Universe", "Model", "offset", "cash_frac", "top_weight", "rebalances"))
    for e in analysis:
        if e["Window"] != "Holdout":
            continue
        for o in OFFSETS:
            v = e["offsets"][str(o)]
            print("  %-18s %-13s %+8.1f %9.1f%% %9.1f%% %10.1f"
                  % (e["Universe"], e["Model"], o, 100 * v["mean_cash_frac"],
                     100 * v["mean_top_weight"], v["mean_rebalances"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universes", nargs="*", default=TARGET_UNIVERSES)
    args = ap.parse_args()

    print("=" * 104)
    print("  ACTION-CONSTANT ABLATION OUTSIDE EQUITIES — cash-logit offset sweep")
    print("  offsets: %s   universes: %s" % (OFFSETS, args.universes))
    print("=" * 104)

    df = collect(args.universes)
    df.to_csv(OUT_CSV, index=False)
    print("\n  Wrote %s (%d rows)" % (os.path.relpath(OUT_CSV, PROJECT_ROOT), len(df)))

    analysis = analyse(df)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated": "2026-08-29", "offsets": OFFSETS,
                   "default_offset": DEFAULT_OFFSET,
                   "fragile_threshold": FRAGILE_THRESH,
                   "equity_control": EQUITY_CONTROL,
                   "analysis": analysis}, f, indent=2)
    print("  Wrote %s" % os.path.relpath(OUT_JSON, PROJECT_ROOT))
    print_report(analysis)

    eq = [e for e in analysis if e["Universe"] == EQUITY_CONTROL]
    non_eq = [e for e in analysis if e["Universe"] != EQUITY_CONTROL]
    print("\n  SUMMARY")
    print("    equities control (%s): %d/%d arms FRAGILE" %
          (EQUITY_CONTROL, sum(1 for e in eq if e["fragile_offsets"]), len(eq)))
    print("    non-equity universes:  %d/%d arms FRAGILE" %
          (sum(1 for e in non_eq if e["fragile_offsets"]), len(non_eq)))
    for e in non_eq:
        if e["fragile_offsets"]:
            print("      %s / %s / %s: fragile at %s (max |delta| = %.2f)"
                  % (e["Universe"], e["Window"], e["Model"],
                     e["fragile_offsets"], e["max_abs_delta"]))


if __name__ == "__main__":
    main()


