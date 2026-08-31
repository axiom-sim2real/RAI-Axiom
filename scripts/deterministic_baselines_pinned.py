"""
================================================================================
  DETERMINISTIC BASELINES ON THE PINNED WINDOW + AXIOM COST MODEL   (TASK B)
================================================================================
  The published README table mixed evaluation bases: Axiom / LSTM / XGBoost were
  on the pinned window (2016-08-20 or 2021-08-20 -> 2026-08-20), while Asset-0
  B&H (then mislabelled "SPY B&H"), Equal Weight, SMA 50/200, Risk Parity, 60/40
  and [v6] Fast were still on
  canonical_evaluation.py's own relative download window and, for the rule-based
  arms, on a zero-cost basis. This script re-runs all six on the pinned window
  and, where the strategy actually trades, charges Axiom's cost model
  (5 bps fee + 0.02% slippage on turnover, gated by a 3% portfolio-drift
  threshold -- identical to evaluate_on_real_data in kaggle_axiom_10seed.py).

  Cost treatment per arm:
    SPY B&H (fixed reference)   the real SPY, bought once at t=0 -- see below.
    Asset-0 B&H / EW (1/N) / 60/40
                                buy once at t=0 and never trade again. Axiom's
                                entry allocation is also free, so the
                                cost-charged and zero-cost curves are IDENTICAL
                                by construction. Reported as such, not as a
                                cost-robustness result.
    Risk Parity                 monthly (21-day) inverse-vol rebalance -> real
                                turnover. Cost-charged version added here, with
                                the same 3% drift gate Axiom uses.
    SMA 50/200                  binary in/out of the first asset -> full-wealth
                                turnover on each crossover flip.
    [v6] Fast                   FastTradingNet through Axiom's own evaluator, so
                                already on Axiom's cost model; a zero-cost
                                variant is also produced for the sensitivity
                                column.

  Deterministic arms have no seed to vary: SPY/Asset-0/EW/60/40/SMA/RiskParity
  are functions of the price series alone, and [v6] Fast is a single checkpoint.
  So these are point estimates, not 10-seed means -- the table must say so.

  Known warm-up artefact (FIXED 2026-08-29, TASK B): the deterministic evaluators
  originally saw only prices inside the evaluation window, so SMA 50/200 had no
  signal for its first 200 in-window days and defaulted to holding asset 0. On
  the 5y universes the OOS window is ~250 days, so almost all of it was that
  default, and the resulting curve was **bit-identical to asset-0 buy-and-hold in
  5 of the 12 universe-windows** (Forex/Holdout, Global_Indices/Holdout,
  India/Holdout, India/OOS, US_ETFs/Holdout). This script now also evaluates a
  warm-up-corrected arm, ``SMA 50/200 (warm-up)``, which feeds the indicator the
  prices immediately *preceding* the evaluation window (from the train split for
  OOS, and from train+OOS for the holdout) so the 50/200 means are defined from
  the window's first day. No look-ahead is introduced: the warm-up slice is
  strictly earlier than the window, and wealth is only accumulated inside the
  window. The uncorrected arm is kept in the output as ``SMA 50/200`` so the
  published numbers remain reproducible and the size of the artefact is visible.

  Asset-0 labelling (corrected 2026-08-29): ``evaluate_buy_hold_first`` and
  ``evaluate_sma_crossover`` both act on **column 0** of each universe, which is
  yfinance's alphabetically-first ticker -- *not* SPY, in any of the six
  universes. Column 0 is EEM (US_ETFs and Global_Indices -- which is why those
  two rows are identical), AAPL (US_MegaCap), AXISBANK.NS (India), AUDUSD=X
  (Forex) and BCH-USD (Crypto). The arms are therefore reported as
  "Asset-0 B&H (in-universe, <ticker>)" and "SMA 50/200 (asset 0)"; the earlier
  "SPY B&H" label was wrong.

  Fixed SPY reference (added 2026-08-30): because column 0 is never SPY, the
  project had never actually evaluated "does Axiom match buy-and-hold on SPY"
  outside US_ETFs / Global_Indices, even though that comparison is the one the
  paper's framing invokes. A separate arm, ``SPY B&H (fixed reference)``, buys
  and holds the **real SPY** in every universe, independent of that universe's
  ticker set or column order. For India, Forex and Crypto it is deliberately an
  OUT-OF-UNIVERSE reference point, not a swap-in for the in-universe asset-0
  arm, so both rows are kept. SPY is forward-filled onto each universe's own
  trading calendar (which is what a SPY holder marks to on a day the NYSE is
  shut but the NSE or the crypto market is open); the number of padded days and
  the Sharpe recomputed on SPY's own NYSE sessions are both reported, because
  padded zero-return days deflate an annualised sqrt(252) Sharpe. In US_ETFs and
  Global_Indices SPY *is* a constituent, so the fixed reference is cross-checked
  against buy-and-hold of that universe's own SPY column; the check is printed.

  Usage:  venv/Scripts/python.exe scripts/deterministic_baselines_pinned.py
  Output: data/deterministic_baselines_pinned.csv
          data/deterministic_baselines_pinned.json
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

from scripts.canonical_evaluation import (  # noqa: E402
    compute_metrics,
    evaluate_buy_hold_first, evaluate_equal_weight, evaluate_60_40,
    evaluate_risk_parity, evaluate_sma_crossover,
    load_spy_reference, evaluate_spy_reference, evaluate_spy_native_calendar,
    SPY_REFERENCE_TICKER,
    DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_PCT, DEFAULT_REBAL_THRESH,
)
from scripts.kaggle_axiom_10seed import (  # noqa: E402
    evaluate_on_real_data, UNIVERSES as KAGGLE_UNIVERSES,
)
from scripts.train_v6_fast import FastTradingNet  # noqa: E402
from scripts.baseline_multiseed import load_universe, split_60_20_20  # noqa: E402

FAST_CKPT = os.path.join(PROJECT_ROOT, "data", "v0.6_rl_checkpoints", "rai_v6_fast.pt")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "deterministic_baselines_pinned.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", "deterministic_baselines_pinned.json")
AXIOM_CSV = os.path.join(PROJECT_ROOT, "data", "axiom_per_seed_results.csv")


# ==============================================================================
#  COST-CHARGED VARIANTS OF THE REBALANCING RULE-BASED ARMS
# ==============================================================================
def evaluate_risk_parity_costed(prices, lb=60, initial_cash=10000.0,
                                fee_bps=DEFAULT_FEE_BPS,
                                slippage_pct=DEFAULT_SLIPPAGE_PCT,
                                rebal_thresh=DEFAULT_REBAL_THRESH):
    """Inverse-vol portfolio, monthly rebalance, charged Axiom's cost model.

    Mirrors evaluate_risk_parity() exactly (same 21-day schedule, same 60-day
    vol lookback, same first-10-asset slice) and adds: (a) a 3% drift gate, and
    (b) fee_rate * turnover deducted from wealth at each executed rebalance.
    """
    p = prices.values[:, :10] if isinstance(prices, pd.DataFrame) else prices[:, :10]
    T, N = p.shape
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0
    sh = (initial_cash / N) / p[0]          # free entry, as in Axiom's evaluator
    eq = [initial_cash]
    rebal = 0
    for t in range(1, T):
        w = float(np.sum(sh * p[t]))
        if t % 21 == 0 and t >= lb:
            v = np.std(np.diff(np.log(np.maximum(1e-4, p[t - lb:t + 1])), axis=0), axis=0)
            iv = 1.0 / np.maximum(1e-8, v)
            wts = iv / iv.sum()
            cur_w = (sh * p[t]) / max(1e-4, w)
            drift = float(np.sum(np.abs(cur_w - wts)))
            if drift > rebal_thresh:
                turnover = float(np.sum(np.abs(sh * p[t] - w * wts)))
                net = max(1e-4, w - turnover * fee_rate)
                sh = (net * wts) / np.maximum(1e-4, p[t])
                w = net
                rebal += 1
        eq.append(w)
    return eq, rebal


def evaluate_sma_crossover_costed(prices, sw=50, lw=200, initial_cash=10000.0,
                                  fee_bps=DEFAULT_FEE_BPS,
                                  slippage_pct=DEFAULT_SLIPPAGE_PCT):
    """SMA 50/200 in/out of the first asset, charged Axiom's cost model.

    A flip moves 100% of wealth, so drift is 2.0 and always clears the 3%
    gate -- the gate is therefore not a free parameter here. Turnover on a flip
    is the full portfolio value.
    """
    p = prices.values[:, :10] if isinstance(prices, pd.DataFrame) else prices[:, :10]
    T = p.shape[0]
    spy = p[:, 0]
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0
    wealth = initial_cash
    eq = [initial_cash]
    in_spy = True                            # free entry, as in Axiom's evaluator
    flips = 0
    for t in range(1, T):
        want = in_spy
        if t >= lw:
            want = bool(np.mean(spy[t - sw:t]) > np.mean(spy[t - lw:t]))
        if want != in_spy:
            wealth = max(1e-4, wealth - wealth * fee_rate)
            in_spy = want
            flips += 1
        dr = (spy[t] / spy[t - 1] - 1.0) if in_spy else 0.0
        wealth *= (1.0 + dr)
        eq.append(wealth)
    return eq, flips


def _sma_core(series, n_hist, sw, lw, fee_rate, initial_cash=10000.0):
    """Shared SMA 50/200 engine with an explicit warm-up prefix.

    ``series`` is the asset-0 price path covering ``n_hist`` pre-window days
    followed by the evaluation window. Wealth is accumulated only over the window
    (indices ``n_hist ..``), but the moving averages at window index ``t`` are
    taken over the concatenated series ending at global index ``n_hist + t``, so
    with ``n_hist >= lw`` the indicator is live from the window's first day.

    Returns (equity_curve, n_flips, n_signal_days, n_default_days).
    ``n_default_days`` counts in-window days where the long lookback is still
    undefined and the arm falls back to holding asset 0 -- the artefact this
    replaces. With a full warm-up it is 0.
    """
    T_win = len(series) - n_hist
    wealth = initial_cash
    eq = [initial_cash]
    n_sig = n_def = flips = 0

    # Opening position: decided from the warm-up prefix alone if it is long
    # enough, otherwise the original default (hold asset 0). Entry is free, as in
    # Axiom's evaluator, so an "out" opening simply starts the curve flat.
    if n_hist >= lw:
        in_pos = bool(np.mean(series[n_hist - sw:n_hist]) > np.mean(series[n_hist - lw:n_hist]))
    else:
        in_pos = True

    for t in range(1, T_win):
        g = n_hist + t
        if g >= lw:
            in_new = bool(np.mean(series[g - sw:g]) > np.mean(series[g - lw:g]))
            n_sig += 1
        else:
            in_new = in_pos
            n_def += 1
        if in_new != in_pos:
            wealth = max(1e-4, wealth - wealth * fee_rate)
            in_pos = in_new
            flips += 1
        dr = (series[g] / series[g - 1] - 1.0) if in_pos else 0.0
        wealth *= (1.0 + dr)
        eq.append(wealth)

    return eq, flips, n_sig, n_def


def _asset0(prices):
    return (prices.values[:, 0] if isinstance(prices, pd.DataFrame) else prices[:, 0]).astype(float)


def evaluate_sma_warmup(window, hist, sw=50, lw=200, initial_cash=10000.0,
                        fee_bps=DEFAULT_FEE_BPS, slippage_pct=DEFAULT_SLIPPAGE_PCT):
    """SMA 50/200 on asset 0 with a pre-window warm-up prefix.

    ``hist`` is the price frame immediately preceding ``window`` (train split for
    the OOS window; train+OOS for the holdout). Only its last ``lw`` rows matter.
    Returns (eq_zero_cost, eq_costed, flips_costed, n_signal_days, n_default_days,
    n_warmup_days).
    """
    h = _asset0(hist)[-lw:] if len(hist) else np.empty(0)
    series = np.concatenate([h, _asset0(window)])
    n_hist = len(h)
    fee_rate = fee_bps / 10000.0 + slippage_pct / 100.0
    eq0, _, n_sig, n_def = _sma_core(series, n_hist, sw, lw, 0.0, initial_cash)
    eqc, flips, _, _ = _sma_core(series, n_hist, sw, lw, fee_rate, initial_cash)
    return eq0, eqc, flips, n_sig, n_def, n_hist


def load_fast_model():
    model = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
    state = torch.load(FAST_CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    print("=" * 100)
    print("  DETERMINISTIC BASELINES RE-RUN ON THE PINNED WINDOW (TASK B)")
    print("  cost model: %d bps fee + %.2f%% slippage, %.0f%% drift gate"
          % (DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_PCT, DEFAULT_REBAL_THRESH * 100))
    print("=" * 100)

    fast = load_fast_model()
    rows = []

    # The fixed external reference is loaded ONCE: it is the same real SPY series
    # for every universe, which is the whole point of the arm.
    spy_ref = load_spy_reference()
    print("  Fixed SPY reference: %d closes %s..%s (source: %s)"
          % (len(spy_ref), spy_ref.index[0].date(), spy_ref.index[-1].date(),
             spy_ref.attrs.get("source", "?")), flush=True)
    spy_checks = []

    A0_ARM = "Asset-0 B&H (in-universe)"
    SPY_ARM = "SPY B&H (fixed reference)"

    for name, info in KAGGLE_UNIVERSES.items():
        df = load_universe(name, info)
        train, oos, fut = split_60_20_20(df)
        asset0 = str(df.columns[0])
        spy_in_universe = bool(SPY_REFERENCE_TICKER in df.columns)
        print("\n  --- %s --- %d days (train %d / oos %d / holdout %d)  %s..%s  [asset 0 = %s, SPY %s]"
              % (name, len(df), len(train), len(oos), len(fut),
                 df.index[0].date(), df.index[-1].date(), asset0,
                 "in universe" if spy_in_universe else "OUT of universe"), flush=True)

        # price frame strictly preceding each evaluation window, for SMA warm-up
        hist_for = {"OOS": train, "Holdout": pd.concat([train, oos])}

        for win_label, win in (("OOS", oos), ("Holdout", fut)):
            arms = {}
            extra = {}

            # --- FIXED EXTERNAL REFERENCE: the real SPY, identical series in all
            #     six universes, independent of ticker set and column order.
            eq_spy, n_pad = evaluate_spy_reference(win, spy_ref)
            eq_nat, n_nat = evaluate_spy_native_calendar(win, spy_ref)
            arms[SPY_ARM] = (eq_spy, eq_spy, 0,
                             "fixed reference, %s; no trading after free entry"
                             % ("in-universe constituent" if spy_in_universe
                                else "OUT-OF-UNIVERSE reference"))
            extra[SPY_ARM] = {
                "SPY in universe": spy_in_universe,
                "Calendar padded days": n_pad,
                "Window days": int(len(win)),
                "Sharpe (native NYSE calendar)": compute_metrics(eq_nat)["sharpe"],
                "Native NYSE days": n_nat,
            }
            # Cross-check: where SPY IS a constituent the fixed reference must
            # reproduce buy-and-hold of that universe's own SPY column.
            if spy_in_universe:
                col = win[SPY_REFERENCE_TICKER].to_numpy(float)
                eq_iu = 10000.0 * (col / col[0])
                a_ref = np.asarray(eq_spy, dtype=float)
                s_ref = compute_metrics(eq_spy)["sharpe"]
                s_iu = compute_metrics(eq_iu)["sharpe"]
                spy_checks.append({"Universe": name, "Window": win_label,
                                   "fixed_reference_sharpe": s_ref,
                                   "in_universe_column_sharpe": s_iu,
                                   "abs_diff": abs(s_ref - s_iu),
                                   "max_abs_rel_price_diff": float(np.max(np.abs(
                                       a_ref / a_ref[0] - col / col[0])))})

            # --- buy-once-hold arms: cost-charged == zero-cost by construction
            for arm, fn in ((A0_ARM, evaluate_buy_hold_first),
                            ("EW (1/N)", evaluate_equal_weight),
                            ("60/40", evaluate_60_40)):
                eq = fn(win)
                arms[arm] = (eq, eq, 0, "no trading after free entry")

            # --- rebalancing rule-based arms
            eq0 = evaluate_risk_parity(win)
            eqc, nreb = evaluate_risk_parity_costed(win)
            arms["Risk Parity"] = (eq0, eqc, nreb, "monthly inverse-vol")

            eq0 = evaluate_sma_crossover(win)
            eqc, nflip = evaluate_sma_crossover_costed(win)
            arms["SMA 50/200"] = (eq0, eqc, nflip,
                                  "asset 0, NO warm-up (as published): first %d "
                                  "in-window days have no signal" % min(200, len(win)))

            eq0w, eqcw, nflipw, n_sig, n_def, n_warm = evaluate_sma_warmup(win, hist_for[win_label])
            arms["SMA 50/200 (warm-up)"] = (
                eq0w, eqcw, nflipw,
                "asset 0, %d pre-window warm-up days -> %d/%d in-window days with "
                "a live signal (%d default)" % (n_warm, n_sig, len(win) - 1, n_def))

            # --- learned policy, single checkpoint
            eqc = evaluate_on_real_data(fast, win.values)
            eq0 = evaluate_on_real_data(fast, win.values, fee_bps=0, slippage_pct=0.0)
            arms["[v6] Fast"] = (eq0, eqc, -1, "FastTradingNet, 1 checkpoint")

            bh_costed = compute_metrics(arms[A0_ARM][1])["sharpe"]
            for arm, (eq0, eqc, ntr, note) in arms.items():
                m0, mc = compute_metrics(eq0), compute_metrics(eqc)
                label = ("Asset-0 B&H (in-universe, %s)" % asset0) if arm == A0_ARM else arm
                row = {
                    "Universe": name, "Window": win_label, "Arm": arm,
                    "Arm label": label,
                    "Asset 0": asset0,
                    "Sharpe (zero-cost)": m0["sharpe"],
                    "Sharpe (Axiom cost model)": mc["sharpe"],
                    "Return % (Axiom cost model)": mc["return_pct"],
                    "MaxDD % (Axiom cost model)": mc["max_dd"],
                    "Trades": ntr,
                    "Identical to Asset-0 B&H": bool(abs(mc["sharpe"] - bh_costed) < 1e-9),
                    "Note": note,
                }
                row.update(extra.get(arm, {}))
                rows.append(row)
                print("    %-8s %-30s zero-cost %+.3f -> costed %+.3f  (delta %+.3f, trades %s)%s"
                      % (win_label, label, m0["sharpe"], mc["sharpe"],
                         mc["sharpe"] - m0["sharpe"], ntr if ntr >= 0 else "n/a",
                         "  [== asset-0 B&H]"
                         if abs(mc["sharpe"] - bh_costed) < 1e-9 and arm != A0_ARM else ""),
                      flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print("\n  Wrote %s (%d rows)" % (os.path.relpath(OUT_CSV, PROJECT_ROOT), len(out)))

    # ---- summary: mean across the 6 universes, per arm and window
    summary = {}
    for win in ("OOS", "Holdout"):
        sub = out[out["Window"] == win]
        summary[win] = {}
        for arm in sub["Arm"].unique():
            a = sub[sub["Arm"] == arm]
            summary[win][arm] = {
                "mean_sharpe_zero_cost": float(a["Sharpe (zero-cost)"].mean()),
                "mean_sharpe_costed": float(a["Sharpe (Axiom cost model)"].mean()),
                "per_universe_costed": {r["Universe"]: float(r["Sharpe (Axiom cost model)"])
                                        for _, r in a.iterrows()},
            }
        # the fixed-SPY arm carries two extra descriptive series
        sp = sub[sub["Arm"] == SPY_ARM]
        if len(sp):
            summary[win][SPY_ARM]["per_universe_native_nyse"] = {
                r["Universe"]: float(r["Sharpe (native NYSE calendar)"]) for _, r in sp.iterrows()}
            summary[win][SPY_ARM]["mean_sharpe_native_nyse"] = float(
                sp["Sharpe (native NYSE calendar)"].mean())
            summary[win][SPY_ARM]["padded_days"] = {
                r["Universe"]: [int(r["Calendar padded days"]), int(r["Window days"])]
                for _, r in sp.iterrows()}
            summary[win][SPY_ARM]["spy_in_universe"] = {
                r["Universe"]: bool(r["SPY in universe"]) for _, r in sp.iterrows()}

    # Axiom reference (10-seed mean per universe, already on the pinned window)
    ax = pd.read_csv(AXIOM_CSV)
    summary["axiom_reference"] = {
        "OOS": {"mean": float(ax.groupby("Universe")["OOS Sharpe"].mean().mean()),
                "per_universe": {k: float(v) for k, v in
                                 ax.groupby("Universe")["OOS Sharpe"].mean().items()}},
        "Holdout": {"mean": float(ax.groupby("Universe")["Future Sharpe"].mean().mean()),
                    "per_universe": {k: float(v) for k, v in
                                     ax.groupby("Universe")["Future Sharpe"].mean().items()}},
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated": "2026-08-30",
                   "cost_model": {"fee_bps": DEFAULT_FEE_BPS,
                                  "slippage_pct": DEFAULT_SLIPPAGE_PCT,
                                  "drift_thresh": DEFAULT_REBAL_THRESH},
                   "note": "Deterministic arms have no seed; [v6] Fast is one checkpoint. "
                           "Point estimates, not 10-seed means.",
                   "sma_warmup_note":
                       "'SMA 50/200' is the as-published arm with no pre-window warm-up: "
                       "its first min(200, window) days have no 200-day mean and it defaults "
                       "to holding asset 0, which made it bit-identical to asset-0 B&H in 5 of "
                       "12 universe-windows. 'SMA 50/200 (warm-up)' feeds the indicator the "
                       "prices immediately preceding the window (train split for OOS, "
                       "train+OOS for holdout), so the signal is live from day 1. No "
                       "look-ahead: the warm-up slice is strictly earlier than the window.",
                   "asset0_note":
                       "'Asset-0 B&H (in-universe)' and both SMA arms act on column 0 of each "
                       "universe, which is yfinance's alphabetically-first ticker and is NOT SPY "
                       "in any universe: " + ", ".join(
                           "%s=%s" % (u, a) for u, a in
                           out.drop_duplicates("Universe").set_index("Universe")["Asset 0"].items()
                       ) + ". The earlier 'SPY B&H' label was wrong. The 'Arm label' column "
                           "carries the ticker-qualified label; 'Arm' is the grouping key.",
                   "spy_reference_note":
                       "'SPY B&H (fixed reference)' buys and holds the REAL SPY in every "
                       "universe, from a single cached series (data/pinned_universes/"
                       "_spy_reference.csv) that is independent of each universe's ticker set "
                       "and column order. It is an in-universe constituent only in US_ETFs and "
                       "Global_Indices; for US_MegaCap_PIT, India_Nifty_50, Forex_Commodities "
                       "and Crypto_PIT it is deliberately an OUT-OF-UNIVERSE reference point, "
                       "not a swap-in for the asset-0 arm -- both rows are kept. SPY is "
                       "forward-filled onto each universe's own trading calendar; "
                       "'Calendar padded days' counts the window dates that are not NYSE "
                       "sessions, and 'Sharpe (native NYSE calendar)' recomputes the same "
                       "buy-and-hold on SPY's own sessions inside the window's date span, "
                       "because padded zero-return days deflate an annualised sqrt(252) Sharpe. "
                       "Like the other buy-once arms it never trades after a free entry, so its "
                       "cost-charged and zero-cost curves are identical by construction.",
                   "spy_in_universe_crosscheck": spy_checks,
                   "summary": summary}, f, indent=2)
    print("  Wrote %s" % os.path.relpath(OUT_JSON, PROJECT_ROOT))

    print("\n  MEAN SHARPE ACROSS 6 UNIVERSES (pinned window)")
    print("    %-30s %10s %10s %8s" % ("arm", "zero-cost", "costed", "delta"))
    for win in ("OOS", "Holdout"):
        print("   [%s]" % win)
        for arm, v in summary[win].items():
            print("    %-30s %+10.3f %+10.3f %+8.3f"
                  % (arm, v["mean_sharpe_zero_cost"], v["mean_sharpe_costed"],
                     v["mean_sharpe_costed"] - v["mean_sharpe_zero_cost"]))
        print("    %-30s %10s %+10.3f" % ("Axiom (10 seeds)", "-",
                                          summary["axiom_reference"][win]["mean"]))

    # ---- fixed-SPY reference diagnostics
    print("\n  FIXED SPY REFERENCE (real SPY, same series in every universe)")
    print("    %-18s %-8s %8s %8s %9s %8s %-9s"
          % ("universe", "window", "SPY", "asset-0", "native", "padded", "SPY in U?"))
    sp = out[out["Arm"] == SPY_ARM].set_index(["Universe", "Window"])
    a0 = out[out["Arm"] == A0_ARM].set_index(["Universe", "Window"])
    for win in ("OOS", "Holdout"):
        for u in [x for x in out["Universe"].unique()]:
            r = sp.loc[(u, win)]
            print("    %-18s %-8s %+8.3f %+8.3f %+9.3f %4d/%-3d %-9s"
                  % (u[:18], win, float(r["Sharpe (Axiom cost model)"]),
                     float(a0.loc[(u, win)]["Sharpe (Axiom cost model)"]),
                     float(r["Sharpe (native NYSE calendar)"]),
                     int(r["Calendar padded days"]), int(r["Window days"]),
                     "yes" if bool(r["SPY in universe"]) else "OUT"))
    if spy_checks:
        print("\n    in-universe cross-check (SPY is a real constituent here; the fixed")
        print("    reference must reproduce buy-and-hold of the universe's own SPY column):")
        for c in spy_checks:
            print("      %-18s %-8s fixed %+.4f vs in-universe %+.4f  |diff| %.2e  %s"
                  % (c["Universe"][:18], c["Window"], c["fixed_reference_sharpe"],
                     c["in_universe_column_sharpe"], c["abs_diff"],
                     "OK" if c["abs_diff"] < 1e-6 else "MISMATCH"))

    # ---- SMA warm-up diagnostic: how many universe-windows collapsed onto B&H
    print("\n  SMA 50/200 WARM-UP AUDIT (Axiom cost model)")
    print("    %-18s %-8s %9s %9s %7s %-14s" % ("universe", "window", "no-warm", "warm-up",
                                                "delta", "no-warm == B&H"))
    sma = out[out["Arm"].isin(["SMA 50/200", "SMA 50/200 (warm-up)", A0_ARM])]
    n_collapsed = 0
    for (u, w), g in sma.groupby(["Universe", "Window"], sort=True):
        gi = g.set_index("Arm")["Sharpe (Axiom cost model)"]
        nw, wu = float(gi["SMA 50/200"]), float(gi["SMA 50/200 (warm-up)"])
        coll = bool(g[g["Arm"] == "SMA 50/200"]["Identical to Asset-0 B&H"].iloc[0])
        n_collapsed += int(coll)
        print("    %-18s %-8s %+9.3f %+9.3f %+7.3f %-14s"
              % (u[:18], w, nw, wu, wu - nw, "YES" if coll else "-"))
    print("    universe-windows where the un-warmed SMA was bit-identical to asset-0 B&H: "
          "%d of 12" % n_collapsed)


if __name__ == "__main__":
    main()
