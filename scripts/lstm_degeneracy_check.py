"""
================================================================================
  LSTM DEGENERACY CHECK — same policy, or coincidentally same Sharpe?  (TASK D)
================================================================================
  Across the 10 baseline seeds the LSTM arm produces a *single distinct Sharpe*
  in 4 of 6 universes (US_ETFs, US_MegaCap_PIT, India_Nifty_50,
  Forex_Commodities). A single Sharpe is consistent with two very different
  stories:

    (1) the seeds converge to the SAME policy  -> the arm is degenerate, and its
        +/-0.000 is not precision;
    (2) different policies happen to land on the same Sharpe to float precision
        -> unlikely, but it is what the Sharpe alone can tell us.

  This script settles it by dumping what the strategy actually does day by day
  for 3 seeds on one such universe:
      * raw sigmoid P(up)   -- the network output, before thresholding
      * signal              -- 1 if P(up) > 0.5 else 0
      * target weights      -- what evaluate_lstm_strategy allocates:
                               0.8/N per asset when risk-on, 0.2/N when risk-off
  If the signal series are bit-identical across seeds while the sigmoid series
  differ, the networks are genuinely different but the *policy* is the same
  constant -- i.e. story (1), and the thresholding is what destroys the seed
  variance.

  Usage:  venv/Scripts/python.exe scripts/lstm_degeneracy_check.py
  Output: data/lstm_degeneracy_check.csv         (per-day, per-seed dump)
          data/lstm_degeneracy_check.json        (summary)
================================================================================
"""

import os
import sys
import json

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

from scripts.canonical_evaluation import train_lstm_on_split  # noqa: E402
from scripts.kaggle_axiom_10seed import UNIVERSES as KAGGLE_UNIVERSES  # noqa: E402
from scripts.baseline_multiseed import load_universe, split_60_20_20  # noqa: E402

UNIVERSE = "US_ETFs"          # LSTM: 1 distinct Sharpe across all 10 seeds
SEEDS = [42, 101, 202]
LOOKBACK = 20
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "lstm_degeneracy_check.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", "lstm_degeneracy_check.json")


def dump_policy(model, prices_df, lookback=LOOKBACK):
    """Reproduce evaluate_lstm_strategy's decision path and record it."""
    prices = prices_df.values[:, :10]
    T, N = prices.shape
    rets = np.diff(np.log(np.maximum(1e-4, prices)), axis=0)
    probs, sigs, weights = [], [], []
    for t in range(lookback, T - 1):
        x = torch.FloatTensor(rets[t - lookback:t]).unsqueeze(0)
        with torch.no_grad():
            p = float(torch.sigmoid(model(x)).item())
        s = 1 if p > 0.5 else 0
        probs.append(p)
        sigs.append(s)
        weights.append(np.full(N, (0.8 if s == 1 else 0.2) / N))
    return np.array(probs), np.array(sigs), np.vstack(weights)


def main():
    print("=" * 100)
    print("  LSTM DEGENERACY CHECK (TASK D) — universe %s, seeds %s" % (UNIVERSE, SEEDS))
    print("=" * 100)

    df = load_universe(UNIVERSE, KAGGLE_UNIVERSES[UNIVERSE])
    train, oos, _ = split_60_20_20(df)
    print("  train %d days / oos %d days\n" % (len(train), len(oos)))

    res = {}
    rows = []
    for seed in SEEDS:
        model = train_lstm_on_split(train, seed=seed)
        probs, sigs, w = dump_policy(model, oos)
        res[seed] = {"probs": probs, "sigs": sigs, "w": w}
        # network weights fingerprint: are the trained parameters themselves different?
        flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        print("  seed %-4d  P(up) mean %.6f  min %.6f  max %.6f  | signal: %d risk-on / %d days"
              " | param L2 %.6f"
              % (seed, probs.mean(), probs.min(), probs.max(),
                 int(sigs.sum()), len(sigs), float(flat.norm())))
        res[seed]["param_l2"] = float(flat.norm())
        for i in range(len(sigs)):
            rows.append({"Seed": seed, "Day": i, "P_up": probs[i],
                         "Signal": sigs[i], "Weight_per_asset": w[i, 0],
                         "Total_equity_exposure": float(w[i].sum())})

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("\n  PAIRWISE COMPARISON ACROSS SEEDS")
    print("    %-14s %14s %14s %14s %10s"
          % ("pair", "signal ident?", "weights ident?", "max |dP(up)|", "param L2 gap"))
    pairs = {}
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            sig_same = bool(np.array_equal(res[a]["sigs"], res[b]["sigs"]))
            w_same = bool(np.array_equal(res[a]["w"], res[b]["w"]))
            dp = float(np.abs(res[a]["probs"] - res[b]["probs"]).max())
            dl2 = abs(res[a]["param_l2"] - res[b]["param_l2"])
            pairs["%d vs %d" % (a, b)] = {
                "signal_identical": sig_same, "weights_identical": w_same,
                "max_abs_prob_diff": dp, "param_l2_gap": dl2,
                "n_days_signal_differs": int((res[a]["sigs"] != res[b]["sigs"]).sum()),
            }
            print("    %-14s %14s %14s %14.6f %10.4f"
                  % ("%d vs %d" % (a, b), sig_same, w_same, dp, dl2))

    all_const = {s: bool(len(set(res[s]["sigs"].tolist())) == 1) for s in SEEDS}
    verdict = (all(p["signal_identical"] for p in pairs.values())
               and all(all_const.values()))
    print("\n  Per-seed signal constant over the whole window? %s" % all_const)
    print("  VERDICT: %s" % ("DEGENERATE — every seed emits one identical constant "
                             "allocation for every day of the window"
                             if verdict else
                             "NOT fully degenerate — see pairwise table"))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "universe": UNIVERSE, "seeds": SEEDS, "window": "OOS",
            "n_days": int(len(res[SEEDS[0]]["sigs"])),
            "per_seed": {str(s): {"prob_mean": float(res[s]["probs"].mean()),
                                  "prob_min": float(res[s]["probs"].min()),
                                  "prob_max": float(res[s]["probs"].max()),
                                  "n_risk_on_days": int(res[s]["sigs"].sum()),
                                  "signal_constant": all_const[s],
                                  "param_l2": res[s]["param_l2"],
                                  "constant_weight_per_asset": float(res[s]["w"][0, 0])}
                         for s in SEEDS},
            "pairwise": pairs,
            "verdict_degenerate": verdict,
        }, f, indent=2)
    print("\n  Wrote %s" % os.path.relpath(OUT_CSV, PROJECT_ROOT))
    print("  Wrote %s" % os.path.relpath(OUT_JSON, PROJECT_ROOT))


if __name__ == "__main__":
    main()
