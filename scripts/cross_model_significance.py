"""
================================================================================
  CROSS-MODEL SIGNIFICANCE TESTS  (TASK C)
================================================================================
  Axiom's 10 seeds are *training-initialisation* seeds; the LSTM/XGBoost seeds
  are *model-fitting* seeds on real data. Seed 42 of Axiom and seed 42 of the
  LSTM share nothing but the integer, so the samples are NOT paired and a paired
  test would be invalid. This script therefore runs two independent-sample
  tests on the 10-vs-10 seed samples:

    * Welch's t-test          (unequal variances, does not assume equal n or SD)
    * Mann-Whitney U          (rank-based, no normality assumption)

  Both are run on the pinned window under Axiom's cost model, which is the only
  basis on which the three arms are measured the same way:
    Axiom      data/axiom_per_seed_results.csv      (always cost-charged)
    LSTM/XGB   data/baseline_per_seed_results.csv   ("... Sharpe (costed)")

  Reported at three levels:
    per-universe   10 Axiom seeds vs 10 baseline seeds -> a clean two-sample
                   test, 6 per comparison per window.
    pooled         all 60 vs all 60. Convenient for a headline number but it
                   treats 60 observations as 60 independent draws, which they are
                   not: scripts/aggregate_ci.py measures ICC 0.85 OOS / 0.95
                   holdout, i.e. seeds within a universe are near-replicates and
                   the effective sample size is closer to 6 than to 60. The
                   pooled p-values are therefore anti-conservative. Retained
                   verbatim (they are the published numbers) but never as the
                   sole evidence.
    cluster-       added 2026-08-29. Three corrections, all reported next to the
    corrected      pooled numbers rather than replacing them:
                     (a) collapse to one mean per universe (n=6 per arm) and run
                         a PAIRED t-test on the 6 differences. Pairing here is by
                         *universe*, which is legitimate -- both arms are measured
                         on the same six price series. This is a different thing
                         from pairing by seed, which remains invalid.
                     (b) Wilcoxon signed-rank on the same 6 paired differences
                         (exact; the smallest attainable two-sided p at n=6 is
                         0.03125, so no cluster-corrected result can be more
                         significant than that), plus an exact sign test.
                     (c) OLS of Sharpe on an arm dummy over all 120 seed-level
                         observations with cluster-robust (CR1) standard errors,
                         clustered on universe, and p-values from t with G-1 = 5
                         degrees of freedom. This keeps every observation but
                         prices in the within-universe correlation.

  Degenerate-sample note: where the LSTM produces a single distinct Sharpe
  across all 10 seeds its sample variance is 0. Welch's t is still defined
  (it reduces to a one-sample t against a constant) but Mann-Whitney sees a
  fully tied group. Both are reported and the zero-variance arms are flagged.

  Fixed-SPY arm (added 2026-08-30). A third comparison, Axiom vs the real SPY
  buy-and-hold reference, is reported ALONGSIDE the two above -- it does not
  replace them. Its design differs in one way that must be stated rather than
  papered over: **SPY has no seed.** It is a deterministic function of one price
  series, so there is exactly one number per universe-window (read from
  data/deterministic_baselines_pinned.csv, arm "SPY B&H (fixed reference)",
  column "Sharpe (Axiom cost model)"), not a 10-seed sample. Consequences:

    per-universe   a two-sample test is impossible. What is reported is a
                   ONE-SAMPLE t (and an exact one-sample Wilcoxon) of Axiom's 10
                   seeds against the SPY constant. This asks "is Axiom's seed
                   distribution centred on SPY's value in this market?" and it
                   treats SPY's number as exact -- correct for this window and
                   this market, but it prices no uncertainty in SPY itself.
    pooled         60 Axiom observations vs 6 SPY observations. Unbalanced, and
                   the 6 SPY values are one-per-cluster rather than replicates,
                   so this row is even more anti-conservative than the 60-vs-60
                   one. Reported for format-parity with the LSTM/XGBoost table
                   and flagged, never used on its own.
    cluster-       the *valid* test, and identical in construction to the one
    corrected      used for LSTM/XGBoost: collapse Axiom to one mean per universe
                   and pair the six differences by universe (paired t, exact
                   Wilcoxon, exact sign test). The CR1 OLS variant is also run,
                   over 60 + 6 = 66 observations, but with singleton SPY clusters
                   the cluster-robust SE is only partly identified, so the paired
                   n=6 tests are the primary cluster-corrected evidence for this
                   comparison and the OLS row is annotated as such.

  Calendar sensitivity: SPY trades the NYSE calendar and three universes do not,
  so the primary SPY Sharpe is computed with SPY forward-filled onto each
  universe's own index. The cluster-corrected block is therefore also re-run
  against the native-NYSE-calendar Sharpe ("[native NYSE]"), which is the same
  buy-and-hold measured on SPY's own sessions, to show whether the verdict turns
  on the padded zero-return days. See §21 of docs/consolidation_report.md.

  Usage:  venv/Scripts/python.exe scripts/cross_model_significance.py
  Output: data/cross_model_significance.csv
          data/cross_model_significance.json
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
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

AXIOM_CSV = os.path.join(PROJECT_ROOT, "data", "axiom_per_seed_results.csv")
BASE_CSV = os.path.join(PROJECT_ROOT, "data", "baseline_per_seed_results.csv")
DET_CSV = os.path.join(PROJECT_ROOT, "data", "deterministic_baselines_pinned.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "cross_model_significance.csv")
OUT_JSON = os.path.join(PROJECT_ROOT, "data", "cross_model_significance.json")

UNIVERSE_ORDER = ["US_ETFs", "US_MegaCap_PIT", "Global_Indices",
                  "India_Nifty_50", "Forex_Commodities", "Crypto_PIT"]

WINDOWS = [("OOS", "OOS Sharpe", "OOS Sharpe (costed)"),
           ("Holdout", "Future Sharpe", "Future Sharpe (costed)")]

SEEDED_MODELS = ["LSTM (real)", "XGBoost (real)"]
SPY_ARM = "SPY B&H (fixed reference)"
SPY_ARM_NATIVE = "SPY B&H (fixed reference) [native NYSE]"
ALL_ARMS = SEEDED_MODELS + [SPY_ARM]
SHORT = {"LSTM (real)": "LSTM", "XGBoost (real)": "XGBoost",
         SPY_ARM: "SPY (fixed ref)", SPY_ARM_NATIVE: "SPY (native NYSE)"}


def load_spy_constants():
    """Per-universe SPY Sharpe on the pinned window under Axiom's cost model.

    Returns `(primary, native)`, each `{window: {universe: sharpe}}`. `primary`
    is SPY forward-filled onto the universe's own trading calendar; `native` is
    the same buy-and-hold measured on SPY's own NYSE sessions inside the window
    (the calendar-sensitivity variant). One value per universe-window: SPY is
    deterministic and has no seed.
    """
    d = pd.read_csv(DET_CSV)
    d = d[d["Arm"] == SPY_ARM]
    if d.empty:
        raise RuntimeError("no '%s' rows in %s — run "
                           "scripts/deterministic_baselines_pinned.py first"
                           % (SPY_ARM, os.path.basename(DET_CSV)))
    primary, native = {}, {}
    for win in ("OOS", "Holdout"):
        s = d[d["Window"] == win].set_index("Universe")
        primary[win] = {u: float(s.loc[u, "Sharpe (Axiom cost model)"]) for u in UNIVERSE_ORDER}
        native[win] = {u: float(s.loc[u, "Sharpe (native NYSE calendar)"]) for u in UNIVERSE_ORDER}
    return primary, native


def test_vs_constant(a, c):
    """Axiom's 10 seeds against a single deterministic value.

    A two-sample test is not available: `c` is one number, not a sample. This is
    a one-sample t (plus an exact one-sample Wilcoxon signed-rank) on the seed
    deviations `a - c`, which treats `c` as exact. Row schema is kept compatible
    with `test_pair` so both kinds of comparison land in one CSV.
    """
    a = np.asarray(a, float)
    d = a - float(c)
    res = {
        "n_axiom": len(a), "n_base": 1,
        "mean_axiom": float(a.mean()), "mean_base": float(c),
        "sd_axiom": float(a.std(ddof=1)), "sd_base": float("nan"),
        "diff": float(a.mean() - float(c)),
        "cohens_d": (float(d.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0
                     else float("nan")),
        "base_zero_variance": False,
        "base_is_deterministic_constant": True,
        "test": "one-sample t of 10 Axiom seeds vs the SPY constant",
    }
    if a.std(ddof=1) > 0:
        t, p = stats.ttest_1samp(a, float(c))
        res["welch_t"], res["welch_p"] = float(t), float(p)
    else:
        res["welch_t"], res["welch_p"] = float("nan"), float("nan")
    nz = d[d != 0]
    if len(nz) >= 1:
        try:
            w, pw = stats.wilcoxon(nz, alternative="two-sided", method="exact")
        except Exception:
            w, pw = stats.wilcoxon(nz, alternative="two-sided")
        res["mwu_u"], res["mwu_p"] = float(w), float(pw)
    else:
        res["mwu_u"], res["mwu_p"] = float("nan"), float("nan")
    return res



def cohens_d(a, b):
    """Pooled-SD effect size; undefined (nan) if both samples are constant."""
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / np.sqrt(sp2))


def test_pair(a, b):
    """Welch t + Mann-Whitney U for Axiom sample a vs baseline sample b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    res = {
        "n_axiom": len(a), "n_base": len(b),
        "mean_axiom": float(a.mean()), "mean_base": float(b.mean()),
        "sd_axiom": float(a.std(ddof=1)), "sd_base": float(b.std(ddof=1)),
        "diff": float(a.mean() - b.mean()),
        "cohens_d": cohens_d(a, b),
        "base_zero_variance": bool(b.std(ddof=1) == 0.0),
    }
    if a.std(ddof=1) == 0 and b.std(ddof=1) == 0:
        res["welch_t"], res["welch_p"] = float("nan"), float("nan")
    else:
        t, p = stats.ttest_ind(a, b, equal_var=False)
        res["welch_t"], res["welch_p"] = float(t), float(p)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    res["mwu_u"], res["mwu_p"] = float(u), float(pu)
    return res


def _paired_stats(a_u, b_u):
    """Paired-by-universe statistics on the six (Axiom, comparator) means.

    Shared by the seeded-baseline and fixed-SPY comparisons so that both are
    computed by exactly the same code path: a paired t-test on the six
    differences, an exact Wilcoxon signed-rank and an exact sign test, a paired
    effect size d_z, and a t(G-1) interval on the mean difference.
    """
    d = np.asarray(a_u, float) - np.asarray(b_u, float)
    G = len(UNIVERSE_ORDER)
    res = {
        "n_clusters": G,
        "universe_means_axiom": {u: float(v) for u, v in zip(UNIVERSE_ORDER, a_u)},
        "universe_means_base": {u: float(v) for u, v in zip(UNIVERSE_ORDER, b_u)},
        "paired_diffs": {u: float(v) for u, v in zip(UNIVERSE_ORDER, d)},
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)),
        "se_diff": float(d.std(ddof=1) / np.sqrt(G)),
        "n_universes_axiom_ahead": int((d > 0).sum()),
    }

    # (a) paired t on the 6 universe means
    t, p = stats.ttest_rel(np.asarray(a_u, float), np.asarray(b_u, float))
    res["paired_t"], res["paired_t_p"] = float(t), float(p)
    res["paired_dz"] = (float(d.mean() / d.std(ddof=1))
                        if d.std(ddof=1) > 0 else float("nan"))
    tcrit = float(stats.t.ppf(0.975, G - 1))
    res["paired_ci95"] = [float(d.mean() - tcrit * res["se_diff"]),
                          float(d.mean() + tcrit * res["se_diff"])]

    # (b) exact Wilcoxon signed-rank + exact sign test on the same 6 pairs
    nz = d[d != 0]
    if len(nz) >= 1:
        try:
            w, pw = stats.wilcoxon(nz, alternative="two-sided", zero_method="wilcox",
                                   method="exact")
        except Exception:
            w, pw = stats.wilcoxon(nz, alternative="two-sided")
        res["wilcoxon_w"], res["wilcoxon_p"] = float(w), float(pw)
        res["wilcoxon_n_used"] = int(len(nz))
        npos = int((nz > 0).sum())
        res["sign_test_p"] = float(stats.binomtest(npos, len(nz), 0.5).pvalue)
    else:
        res["wilcoxon_w"] = res["wilcoxon_p"] = float("nan")
        res["wilcoxon_n_used"] = 0
        res["sign_test_p"] = float("nan")
    return res


def _cluster_ols(long, G, res):
    """CR1 cluster-robust OLS of Sharpe on an arm dummy, clustered on universe.

    p-values from t at G-1 df, because with six clusters the normal
    approximation is far too optimistic. Mutates and returns `res`.
    """
    long = long.copy()
    long["y"] = long["y"].astype(float)
    try:
        import statsmodels.api as sm
        X = sm.add_constant(long[["is_axiom"]].to_numpy(float))
        fit = sm.OLS(long["y"].to_numpy(float), X).fit(
            cov_type="cluster",
            cov_kwds={"groups": long["Universe"].to_numpy(),
                      "use_correction": True, "df_correction": True},
        )
        beta = float(fit.params[1])
        se = float(fit.bse[1])
        tstat = beta / se if se > 0 else float("nan")
        p_t = float(2.0 * stats.t.sf(abs(tstat), G - 1)) if se > 0 else float("nan")
        res.update({
            "cluster_ols_n_obs": int(len(long)),
            "cluster_ols_beta": beta,
            "cluster_robust_se": se,
            "cluster_ols_t": float(tstat),
            "cluster_ols_p_tG1": p_t,
            "cluster_ols_p_normal": float(fit.pvalues[1]),
            "naive_ols_se": float(sm.OLS(long["y"].to_numpy(float), X).fit().bse[1]),
        })
    except Exception as e:      # statsmodels missing or singular design
        res["cluster_ols_error"] = repr(e)
    return res


def cluster_corrected(ax, bl, ax_col, bl_col, model):
    """Cluster-aware alternatives to the pooled 60-vs-60 test.

    Returns a dict with (a) a paired t-test on the six universe means, (b) an
    exact Wilcoxon signed-rank and an exact sign test on the same six paired
    differences, and (c) a cluster-robust OLS on all 120 seed-level observations
    with universe as the cluster.

    Pairing is by universe, not by seed: both arms are evaluated on the same six
    price series, so the six differences are legitimately paired. Seed-level
    pairing would not be (Axiom's seeds are training-init seeds, the baselines'
    are fitting seeds) and is not used anywhere.
    """
    a_u = np.array([ax[ax["Universe"] == u][ax_col].mean() for u in UNIVERSE_ORDER], float)
    b_u = np.array([bl[(bl["Universe"] == u) & (bl["Model"] == model)][bl_col].mean()
                    for u in UNIVERSE_ORDER], float)
    res = _paired_stats(a_u, b_u)

    # (c) cluster-robust OLS on all 120 seed-level observations
    a_all = ax[["Universe", ax_col]].rename(columns={ax_col: "y"}).copy()
    a_all["is_axiom"] = 1.0
    b_all = bl[bl["Model"] == model][["Universe", bl_col]].rename(columns={bl_col: "y"}).copy()
    b_all["is_axiom"] = 0.0
    return _cluster_ols(pd.concat([a_all, b_all], ignore_index=True),
                        len(UNIVERSE_ORDER), res)


def cluster_corrected_const(ax, ax_col, const_by_universe):
    """Same cluster-corrected block, against a deterministic comparator.

    `const_by_universe` is `{universe: sharpe}` — one number per universe, no
    seed dimension (the fixed-SPY arm). The paired n=6 block is computed by the
    identical code path as for the seeded baselines, which is the point: the
    valid test is the same test. The CR1 OLS is still run, over 60 Axiom
    observations plus 6 singleton comparator observations, but with one
    observation per cluster on the comparator side the within-cluster variance of
    that arm is unidentified, so the OLS row is a cross-check and the paired n=6
    tests are primary. Flagged in the returned dict.
    """
    a_u = np.array([ax[ax["Universe"] == u][ax_col].mean() for u in UNIVERSE_ORDER], float)
    b_u = np.array([float(const_by_universe[u]) for u in UNIVERSE_ORDER], float)
    res = _paired_stats(a_u, b_u)
    res["comparator_is_deterministic"] = True
    res["comparator_n_per_universe"] = 1

    a_all = ax[["Universe", ax_col]].rename(columns={ax_col: "y"}).copy()
    a_all["is_axiom"] = 1.0
    b_all = pd.DataFrame({"Universe": UNIVERSE_ORDER, "y": b_u, "is_axiom": 0.0})
    res = _cluster_ols(pd.concat([a_all, b_all], ignore_index=True),
                       len(UNIVERSE_ORDER), res)
    res["cluster_ols_caveat"] = (
        "66 obs: 60 Axiom seed-level + 6 singleton SPY. With one comparator "
        "observation per cluster the arm's within-cluster variance is not "
        "identified, so this SE is a cross-check on the paired n=6 tests, not a "
        "replacement for them.")
    return res



def main():
    ax = pd.read_csv(AXIOM_CSV)
    bl = pd.read_csv(BASE_CSV)
    spy_primary, spy_native = load_spy_constants()

    print("=" * 100)
    print("  CROSS-MODEL SIGNIFICANCE TESTS (TASK C) — seed-level samples are NOT paired;")
    print("  the cluster-corrected block pairs by UNIVERSE (n=6), which is valid")
    print("  basis: pinned window, Axiom cost model (5bps + 0.02% slip, 3% drift)")
    print("  the fixed-SPY arm has NO seed: 1 deterministic value per universe-window")
    print("=" * 100)

    rows = []
    out = {"note": ("Independent-sample tests at the seed level. Axiom seeds are "
                    "training-init seeds, baseline seeds are fitting seeds; they are "
                    "not pairable, so no seed-paired test is reported. The "
                    "cluster_corrected block pairs by UNIVERSE instead, which is "
                    "legitimate, and additionally reports cluster-robust OLS."),
           "spy_note": ("'%s' is deterministic — one value per universe-window, no "
                        "seed. Its per-universe row is a ONE-SAMPLE t of Axiom's 10 "
                        "seeds against that constant, its pooled row is an "
                        "unbalanced 60-vs-6 (flagged, anti-conservative), and its "
                        "cluster-corrected block is the same paired-by-universe n=6 "
                        "test used for the seeded baselines — which is the primary "
                        "evidence for this comparison. '%s' repeats the "
                        "cluster-corrected block against SPY's Sharpe on its own "
                        "NYSE sessions, as a calendar-alignment sensitivity."
                        % (SPY_ARM, SPY_ARM_NATIVE)),
           "basis": "pinned window, Axiom cost model",
           "icc_from_aggregate_ci": {"OOS": 0.847, "Holdout": 0.953},
           "spy_constants": {"forward_filled_to_universe_calendar": spy_primary,
                             "native_nyse_calendar": spy_native},
           "per_universe": {}, "pooled": {}, "cluster_corrected": {}}

    for win_label, ax_col, bl_col in WINDOWS:
        out["per_universe"][win_label] = {}
        out["cluster_corrected"][win_label] = {}
        for model in SEEDED_MODELS:
            out["per_universe"][win_label][model] = {}
            print("\n  [%s]  Axiom vs %s" % (win_label, model))
            print("    %-18s %8s %8s %8s   %9s %9s   %9s   %s"
                  % ("universe", "axiom", "base", "diff", "welch t", "welch p",
                     "MWU p", "flag"))
            for u in UNIVERSE_ORDER:
                a = ax[(ax["Universe"] == u)][ax_col].to_numpy(float)
                b = bl[(bl["Universe"] == u) & (bl["Model"] == model)][bl_col].to_numpy(float)
                r = test_pair(a, b)
                r.update({"Universe": u, "Window": win_label, "Comparison":
                          "Axiom vs %s" % model})
                rows.append(r)
                out["per_universe"][win_label][model][u] = r
                flag = "base SD=0" if r["base_zero_variance"] else ""
                print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   %s"
                      % (u, r["mean_axiom"], r["mean_base"], r["diff"],
                         r["welch_t"], r["welch_p"], r["mwu_p"], flag))

            # pooled across all 60 observations
            a = ax[ax_col].to_numpy(float)
            b = bl[bl["Model"] == model][bl_col].to_numpy(float)
            rp = test_pair(a, b)
            rp.update({"Universe": "POOLED (60 vs 60)", "Window": win_label,
                       "Comparison": "Axiom vs %s" % model})
            rows.append(rp)
            out["pooled"].setdefault(win_label, {})[model] = rp
            sig = sum(1 for u in UNIVERSE_ORDER
                      if out["per_universe"][win_label][model][u]["welch_p"] < 0.05)
            rp["n_universes_welch_p_lt_05"] = sig
            print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   "
                  "%d/6 universes p<0.05 (pooled ignores clustering)"
                  % ("POOLED 60v60", rp["mean_axiom"], rp["mean_base"], rp["diff"],
                     rp["welch_t"], rp["welch_p"], rp["mwu_p"], sig))

            # cluster-corrected: collapse to n=6 universe means, plus CR1 OLS
            rc = cluster_corrected(ax, bl, ax_col, bl_col, model)
            out["cluster_corrected"][win_label][model] = rc
            rows.append({
                "Universe": "CLUSTER-CORRECTED (n=6 universes)", "Window": win_label,
                "Comparison": "Axiom vs %s" % model,
                "n_axiom": rc["n_clusters"], "n_base": rc["n_clusters"],
                "mean_axiom": float(np.mean(list(rc["universe_means_axiom"].values()))),
                "mean_base": float(np.mean(list(rc["universe_means_base"].values()))),
                "diff": rc["mean_diff"],
                "sd_axiom": float("nan"), "sd_base": float("nan"),
                "cohens_d": rc["paired_dz"],
                "base_zero_variance": False,
                "welch_t": rc["paired_t"], "welch_p": rc["paired_t_p"],
                "mwu_u": rc["wilcoxon_w"], "mwu_p": rc["wilcoxon_p"],
                "sign_test_p": rc["sign_test_p"],
                "paired_ci95_lo": rc["paired_ci95"][0],
                "paired_ci95_hi": rc["paired_ci95"][1],
                "cluster_ols_beta": rc.get("cluster_ols_beta", float("nan")),
                "cluster_robust_se": rc.get("cluster_robust_se", float("nan")),
                "naive_ols_se": rc.get("naive_ols_se", float("nan")),
                "cluster_ols_p_tG1": rc.get("cluster_ols_p_tG1", float("nan")),
                "n_universes_axiom_ahead": rc["n_universes_axiom_ahead"],
            })
            print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   "
                  "paired-t / Wilcoxon on n=6 universe means; sign p = %.4g; "
                  "Axiom ahead in %d/6"
                  % ("CLUSTER n=6", rows[-1]["mean_axiom"], rows[-1]["mean_base"],
                     rc["mean_diff"], rc["paired_t"], rc["paired_t_p"],
                     rc["wilcoxon_p"], rc["sign_test_p"],
                     rc["n_universes_axiom_ahead"]))
            if "cluster_ols_beta" in rc:
                print("    %-18s beta %+.3f  CR1 SE %.3f (naive SE %.3f, x%.2f)  "
                      "t(5) = %+.2f  p = %.4g"
                      % ("CR1 OLS 120 obs", rc["cluster_ols_beta"],
                         rc["cluster_robust_se"], rc["naive_ols_se"],
                         rc["cluster_robust_se"] / rc["naive_ols_se"]
                         if rc["naive_ols_se"] > 0 else float("nan"),
                         rc["cluster_ols_t"], rc["cluster_ols_p_tG1"]))
            else:
                print("    %-18s unavailable: %s"
                      % ("CR1 OLS 120 obs", rc.get("cluster_ols_error", "?")))

        # ---- fixed-SPY reference: deterministic, no seed dimension -------------
        cvals = spy_primary[win_label]
        out["per_universe"][win_label][SPY_ARM] = {}
        print("\n  [%s]  Axiom vs %s   (SPY has no seed: 1 value per universe)"
              % (win_label, SPY_ARM))
        print("    %-18s %8s %8s %8s   %9s %9s   %9s   %s"
              % ("universe", "axiom", "SPY", "diff", "1-samp t", "1-samp p",
                 "Wilcox p", "flag"))
        for u in UNIVERSE_ORDER:
            a = ax[ax["Universe"] == u][ax_col].to_numpy(float)
            r = test_vs_constant(a, cvals[u])
            r.update({"Universe": u, "Window": win_label,
                      "Comparison": "Axiom vs %s" % SPY_ARM})
            rows.append(r)
            out["per_universe"][win_label][SPY_ARM][u] = r
            print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   %s"
                  % (u, r["mean_axiom"], r["mean_base"], r["diff"],
                     r["welch_t"], r["welch_p"], r["mwu_p"],
                     "comparator is a constant"))

        # pooled: 60 Axiom obs vs 6 SPY obs — unbalanced AND one-per-cluster
        a = ax[ax_col].to_numpy(float)
        b = np.array([cvals[u] for u in UNIVERSE_ORDER], float)
        rp = test_pair(a, b)
        rp.update({"Universe": "POOLED (60 vs 6)", "Window": win_label,
                   "Comparison": "Axiom vs %s" % SPY_ARM,
                   "unbalanced_no_seed_comparator": True})
        rows.append(rp)
        out["pooled"].setdefault(win_label, {})[SPY_ARM] = rp
        sig = sum(1 for u in UNIVERSE_ORDER
                  if out["per_universe"][win_label][SPY_ARM][u]["welch_p"] < 0.05)
        rp["n_universes_welch_p_lt_05"] = sig
        print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   "
              "%d/6 universes p<0.05 (60-vs-6, one obs per cluster: anti-conservative)"
              % ("POOLED 60v6", rp["mean_axiom"], rp["mean_base"], rp["diff"],
                 rp["welch_t"], rp["welch_p"], rp["mwu_p"], sig))

        for arm_label, arm_vals in ((SPY_ARM, cvals), (SPY_ARM_NATIVE, spy_native[win_label])):
            rc = cluster_corrected_const(ax, ax_col, arm_vals)
            out["cluster_corrected"][win_label][arm_label] = rc
            rows.append({
                "Universe": "CLUSTER-CORRECTED (n=6 universes)", "Window": win_label,
                "Comparison": "Axiom vs %s" % arm_label,
                "n_axiom": rc["n_clusters"], "n_base": rc["n_clusters"],
                "mean_axiom": float(np.mean(list(rc["universe_means_axiom"].values()))),
                "mean_base": float(np.mean(list(rc["universe_means_base"].values()))),
                "diff": rc["mean_diff"],
                "sd_axiom": float("nan"), "sd_base": float("nan"),
                "cohens_d": rc["paired_dz"],
                "base_zero_variance": False,
                "base_is_deterministic_constant": True,
                "welch_t": rc["paired_t"], "welch_p": rc["paired_t_p"],
                "mwu_u": rc["wilcoxon_w"], "mwu_p": rc["wilcoxon_p"],
                "sign_test_p": rc["sign_test_p"],
                "paired_ci95_lo": rc["paired_ci95"][0],
                "paired_ci95_hi": rc["paired_ci95"][1],
                "cluster_ols_beta": rc.get("cluster_ols_beta", float("nan")),
                "cluster_robust_se": rc.get("cluster_robust_se", float("nan")),
                "naive_ols_se": rc.get("naive_ols_se", float("nan")),
                "cluster_ols_p_tG1": rc.get("cluster_ols_p_tG1", float("nan")),
                "n_universes_axiom_ahead": rc["n_universes_axiom_ahead"],
            })
            print("    %-18s %+8.3f %+8.3f %+8.3f   %+9.2f %9.2g   %9.2g   "
                  "paired-t / Wilcoxon on n=6 universe pairs; sign p = %.4g; "
                  "Axiom ahead in %d/6   [%s]"
                  % ("CLUSTER n=6", rows[-1]["mean_axiom"], rows[-1]["mean_base"],
                     rc["mean_diff"], rc["paired_t"], rc["paired_t_p"],
                     rc["wilcoxon_p"], rc["sign_test_p"],
                     rc["n_universes_axiom_ahead"],
                     "ffill calendar" if arm_label == SPY_ARM else "native NYSE"))
            if "cluster_ols_beta" in rc:
                print("    %-18s beta %+.3f  CR1 SE %.3f (naive SE %.3f, x%.2f)  "
                      "t(5) = %+.2f  p = %.4g   [singleton SPY clusters: cross-check only]"
                      % ("CR1 OLS 66 obs", rc["cluster_ols_beta"],
                         rc["cluster_robust_se"], rc["naive_ols_se"],
                         rc["cluster_robust_se"] / rc["naive_ols_se"]
                         if rc["naive_ols_se"] > 0 else float("nan"),
                         rc["cluster_ols_t"], rc["cluster_ols_p_tG1"]))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n  Wrote %s (%d rows)" % (os.path.relpath(OUT_CSV, PROJECT_ROOT), len(df)))
    print("  Wrote %s" % os.path.relpath(OUT_JSON, PROJECT_ROOT))

    print("\n  HEADLINE (pooled, cost-matched, clustering caveat applies)")
    for win_label, _, _ in WINDOWS:
        for model in ALL_ARMS:
            r = out["pooled"][win_label][model]
            print("    %-8s Axiom %+.2f vs %-18s %+.2f  diff %+.2f  "
                  "%-7s p = %.3g  rank p = %.3g  d = %.2f%s"
                  % (win_label, r["mean_axiom"], SHORT[model], r["mean_base"], r["diff"],
                     "1-samp" if model == SPY_ARM else "Welch",
                     r["welch_p"], r["mwu_p"], r["cohens_d"],
                     "  [60 vs 6, no seed]" if model == SPY_ARM else ""))

    print("\n" + "=" * 100)
    print("  SIDE BY SIDE: POOLED (ignores clustering)  vs  CLUSTER-CORRECTED")
    print("  ICC = 0.847 OOS / 0.953 holdout, so pooled p-values are anti-conservative.")
    print("  Neither replaces the other; the pooled column is the published number.")
    print("=" * 100)
    print("  %-8s %-18s | %9s %9s | %9s %9s %9s | %9s %9s"
          % ("window", "comparison", "pooled", "pooled", "paired-t", "Wilcox", "sign",
             "CR1 OLS", "SE infl"))
    print("  %-8s %-18s | %9s %9s | %9s %9s %9s | %9s %9s"
          % ("", "", "Welch p", "rank p", "p (n=6)", "p (n=6)", "p (n=6)",
             "p t(5)", "x naive"))
    print("  " + "-" * 100)
    for win_label, _, _ in WINDOWS:
        for model in ALL_ARMS + [SPY_ARM_NATIVE]:
            c = out["cluster_corrected"][win_label][model]
            r = out["pooled"][win_label].get(
                model if model != SPY_ARM_NATIVE else SPY_ARM, {})
            infl = (c["cluster_robust_se"] / c["naive_ols_se"]
                    if c.get("naive_ols_se") else float("nan"))
            print("  %-8s %-18s | %9.2g %9.2g | %9.4g %9.4g %9.4g | %9.4g %9.2f"
                  % (win_label, SHORT[model],
                     r.get("welch_p", float("nan")), r.get("mwu_p", float("nan")),
                     c["paired_t_p"], c["wilcoxon_p"], c["sign_test_p"],
                     c.get("cluster_ols_p_tG1", float("nan")), infl))
    print("  (SPY rows: the pooled columns are 60-vs-6 with one comparator observation")
    print("   per cluster and are reported for format parity only; the paired n=6 block")
    print("   is the valid test. The 'native NYSE' row is the same paired test against")
    print("   SPY measured on its own sessions, and repeats the pooled columns of the row")
    print("   above it because no pooled variant is defined for it.)")
    print("\n  mean difference in Sharpe with a universe-level 95% CI (n=6 clusters):")
    for win_label, _, _ in WINDOWS:
        for model in ALL_ARMS + [SPY_ARM_NATIVE]:
            c = out["cluster_corrected"][win_label][model]
            print("    %-8s Axiom - %-18s %+.3f  [%+.3f, %+.3f]  "
                  "d_z = %+.2f  Axiom ahead in %d/6 universes"
                  % (win_label, SHORT[model], c["mean_diff"],
                     c["paired_ci95"][0], c["paired_ci95"][1], c["paired_dz"],
                     c["n_universes_axiom_ahead"]))
    print("\n  Exact-test floor: at n=6 the smallest attainable two-sided Wilcoxon /")
    print("  sign-test p is 2/64 = 0.03125, so no cluster-corrected p can go below it.")

    print("\n  WHAT THE CORRECTION CHANGES (direction, not just magnitude)")
    for win_label, _, _ in WINDOWS:
        for model in ALL_ARMS:
            r = out["pooled"][win_label][model]
            c = out["cluster_corrected"][win_label][model]
            p_pool, p_clu = r["welch_p"], c["paired_t_p"]
            pool_sig = p_pool < 0.05
            clu_sig = p_clu < 0.05 and c["wilcoxon_p"] < 0.05
            if pool_sig and not clu_sig:
                verdict = "WEAKENED: pooled significant, cluster-corrected not (or rank test fails)"
            elif clu_sig and not pool_sig:
                verdict = "STRENGTHENED: pooled saw a tie, cluster-corrected is significant"
            elif clu_sig and pool_sig:
                verdict = "survives both"
            else:
                verdict = "not significant either way"
            print("    %-8s %-18s pooled p %9.3g -> cluster p %9.4g  (ratio %8.3g)  %s"
                  % (win_label, SHORT[model], p_pool, p_clu,
                     p_clu / p_pool if p_pool > 0 else float("nan"), verdict))

    print("\n  AXIOM vs THE REAL FIXED SPY — direction per universe (cluster-corrected basis)")
    for win_label, _, _ in WINDOWS:
        c = out["cluster_corrected"][win_label][SPY_ARM]
        print("   [%s]  Axiom %+.3f  SPY %+.3f  diff %+.3f  (Axiom ahead in %d/6)"
              % (win_label,
                 float(np.mean(list(c["universe_means_axiom"].values()))),
                 float(np.mean(list(c["universe_means_base"].values()))),
                 c["mean_diff"], c["n_universes_axiom_ahead"]))
        for u in UNIVERSE_ORDER:
            print("      %-18s axiom %+7.3f  SPY %+7.3f  diff %+7.3f  %s"
                  % (u, c["universe_means_axiom"][u], c["universe_means_base"][u],
                     c["paired_diffs"][u],
                     "axiom" if c["paired_diffs"][u] > 0 else "SPY"))


if __name__ == "__main__":
    main()
