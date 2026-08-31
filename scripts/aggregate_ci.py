"""
================================================================================
  AGGREGATE STATISTIC FOR THE "OVERALL" ROW
================================================================================
  Problem this fixes
  ------------------
  The "Overall" row of the Axiom CI table was originally reported as
  "+0.98 +/- 0.67" (OOS) and "+0.62 +/- 1.10" (holdout). Those +/- values are
  the standard deviation of the SIX per-universe means. That is cross-universe
  dispersion (how much Sharpe varies market-to-market), not the seed-level
  estimation uncertainty of the aggregate mean.

  What this script computes, from the 60 raw seed-level observations
  (10 seeds x 6 universes) in data/axiom_per_seed_results.csv:

    [A] The old statistic, reproduced, so the diagnosis is verifiable:
        SD of the 6 per-universe means.
    [B] Naive pooled SE over all 60 observations (ignores clustering) --
        reported only to show why it is also wrong.
    [C] Stratified / hierarchical bootstrap: resample the 10 seeds WITH
        replacement inside each universe, recompute the mean of the 6 universe
        means. Universes are held fixed. -> seed-level uncertainty.
    [D] Two-level cluster bootstrap: resample universes with replacement AND
        seeds within them. -> uncertainty if the 6 universes are themselves a
        sample from a population of markets.
    [E] Mixed-effects estimate: Sharpe ~ 1 + (1 | universe), REML via
        statsmodels MixedLM, plus the closed-form balanced-design equivalent
        SE(grand mean) = sqrt((var_u + var_e/n) / k).

  Usage:  python scripts/aggregate_ci.py
  Output: console table + data/axiom_aggregate_stats.json
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(PROJECT_ROOT, "data", "axiom_per_seed_results.csv")
OUT = os.path.join(PROJECT_ROOT, "data", "axiom_aggregate_stats.json")

N_BOOT = 10000
BOOT_SEED = 20260829

WINDOWS = {"OOS": "OOS Sharpe", "Holdout": "Future Sharpe"}


def per_universe_matrix(df, col):
    """Return (universe_names, matrix[k universes x n seeds]) for one metric."""
    names = list(dict.fromkeys(df["Universe"]))
    rows = []
    for u in names:
        vals = df.loc[df["Universe"] == u, col].to_numpy(dtype=float)
        rows.append(vals)
    n = min(len(r) for r in rows)
    if any(len(r) != n for r in rows):
        raise ValueError("Unbalanced seed counts per universe: " +
                         str({u: len(r) for u, r in zip(names, rows)}))
    return names, np.vstack(rows)


def stratified_bootstrap(mat, n_boot=N_BOOT, rng=None):
    """Resample seeds with replacement within each universe; universes fixed."""
    rng = rng or np.random.default_rng(BOOT_SEED)
    k, n = mat.shape
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=(k, n))
        resampled = np.take_along_axis(mat, idx, axis=1)
        draws[b] = resampled.mean(axis=1).mean()
    return draws


def cluster_bootstrap(mat, n_boot=N_BOOT, rng=None):
    """Two-level: resample universes with replacement, then seeds within them."""
    rng = rng or np.random.default_rng(BOOT_SEED + 1)
    k, n = mat.shape
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        u_idx = rng.integers(0, k, size=k)
        s_idx = rng.integers(0, n, size=(k, n))
        resampled = np.take_along_axis(mat[u_idx], s_idx, axis=1)
        draws[b] = resampled.mean(axis=1).mean()
    return draws


def mixed_effects(df, col):
    """Random-intercept model: value ~ 1 + (1 | Universe). REML."""
    out = {}
    work = df[["Universe", col]].rename(columns={col: "y"}).dropna()
    try:
        import statsmodels.formula.api as smf
        md = smf.mixedlm("y ~ 1", work, groups=work["Universe"])
        fit = md.fit(reml=True)
        beta = float(fit.params["Intercept"])
        se = float(fit.bse["Intercept"])
        ci = fit.conf_int().loc["Intercept"].tolist()
        out["statsmodels"] = {
            "grand_mean": beta,
            "se": se,
            "ci95": [float(ci[0]), float(ci[1])],
            "var_universe": float(fit.cov_re.iloc[0, 0]),
            "var_residual": float(fit.scale),
            "converged": bool(fit.converged),
        }
    except Exception as exc:  # pragma: no cover - dependency/convergence guard
        out["statsmodels"] = {"error": repr(exc)}

    # Closed-form balanced-design equivalent (method of moments / ANOVA).
    names, mat = per_universe_matrix(df, col)
    k, n = mat.shape
    grand = float(mat.mean())
    u_means = mat.mean(axis=1)
    ms_between = n * float(np.var(u_means, ddof=1))
    ms_within = float(np.mean(np.var(mat, axis=1, ddof=1)))
    var_u = max(0.0, (ms_between - ms_within) / n)
    se = float(np.sqrt((var_u + ms_within / n) / k))
    # equals sd(u_means)/sqrt(k) whenever var_u > 0
    out["closed_form"] = {
        "grand_mean": grand,
        "se": se,
        "ci95": [grand - 1.96 * se, grand + 1.96 * se],
        "var_universe": var_u,
        "var_residual": ms_within,
        "df_satterthwaite_note": "k-1 = %d universes; normal approx used" % (k - 1),
        "icc": float(var_u / (var_u + ms_within)) if (var_u + ms_within) > 0 else 0.0,
    }
    return out


def summarize(df, label, col):
    names, mat = per_universe_matrix(df, col)
    k, n = mat.shape
    flat = mat.reshape(-1)
    u_means = mat.mean(axis=1)

    old_stat = float(np.std(u_means, ddof=1))
    old_stat_pop = float(np.std(u_means, ddof=0))
    naive_se = float(np.std(flat, ddof=1) / np.sqrt(flat.size))

    strat = stratified_bootstrap(mat)
    clust = cluster_bootstrap(mat)
    mixed = mixed_effects(df, col)

    res = {
        "window": label,
        "metric_column": col,
        "n_universes": int(k),
        "n_seeds_per_universe": int(n),
        "n_observations": int(flat.size),
        "universes": names,
        "per_universe_mean": {u: float(m) for u, m in zip(names, u_means)},
        "per_universe_sd": {u: float(s) for u, s in zip(names, mat.std(axis=1, ddof=1))},
        "grand_mean_unweighted": float(u_means.mean()),
        "grand_mean_pooled": float(flat.mean()),
        "A_cross_universe_dispersion_sd": old_stat,
        "A_cross_universe_dispersion_sd_ddof0": old_stat_pop,
        "B_naive_pooled_se_ignoring_clusters": naive_se,
        "B_naive_pooled_ci95": [float(flat.mean() - 1.96 * naive_se),
                                 float(flat.mean() + 1.96 * naive_se)],
        "C_stratified_bootstrap": {
            "n_boot": N_BOOT,
            "mean": float(strat.mean()),
            "se": float(strat.std(ddof=1)),
            "ci95": [float(np.percentile(strat, 2.5)), float(np.percentile(strat, 97.5))],
            "interpretation": "seed-level uncertainty, universes treated as fixed",
        },
        "D_cluster_bootstrap": {
            "n_boot": N_BOOT,
            "mean": float(clust.mean()),
            "se": float(clust.std(ddof=1)),
            "ci95": [float(np.percentile(clust, 2.5)), float(np.percentile(clust, 97.5))],
            "interpretation": "universes resampled too; generalises to other markets",
        },
        "E_mixed_effects": mixed,
    }
    return res


def fmt_ci(ci):
    return "[%+.2f, %+.2f]" % (ci[0], ci[1])


def main():
    df = pd.read_csv(CSV)
    print("=" * 96)
    print("  AGGREGATE STATISTIC FOR THE 'OVERALL' ROW - Axiom v0.9")
    print("  source: %s" % os.path.relpath(CSV, PROJECT_ROOT))
    print("  rows: %d   universes: %d   seeds/universe: %d"
          % (len(df), df["Universe"].nunique(),
             len(df) // max(1, df["Universe"].nunique())))
    print("=" * 96)

    results = {}
    for label, col in WINDOWS.items():
        r = summarize(df, label, col)
        results[label] = r

        print("\n" + "-" * 96)
        print("  WINDOW: %s   (column '%s')" % (label, col))
        print("-" * 96)
        print("  per-universe means: " +
              ", ".join("%s=%+.3f" % (u, m) for u, m in r["per_universe_mean"].items()))
        print("  grand mean (unweighted mean of 6 universe means) = %+.4f" % r["grand_mean_unweighted"])
        print("  grand mean (pooled over all %d obs)              = %+.4f"
              % (r["n_observations"], r["grand_mean_pooled"]))
        print()
        print("  [A] OLD STAT  cross-universe dispersion  SD(6 universe means, ddof=1) = %.4f" % r["A_cross_universe_dispersion_sd"])
        print("                (ddof=0 variant = %.4f)" % r["A_cross_universe_dispersion_sd_ddof0"])
        print("  [B] naive pooled SE over 60 obs (WRONG - ignores clustering) = %.4f  CI %s"
              % (r["B_naive_pooled_se_ignoring_clusters"], fmt_ci(r["B_naive_pooled_ci95"])))
        c = r["C_stratified_bootstrap"]
        print("  [C] stratified bootstrap (seeds within universe, %d iters)"
              "\n        mean=%+.4f  SE=%.4f  95%% CI %s" % (c["n_boot"], c["mean"], c["se"], fmt_ci(c["ci95"])))
        d = r["D_cluster_bootstrap"]
        print("  [D] two-level cluster bootstrap (universes + seeds, %d iters)"
              "\n        mean=%+.4f  SE=%.4f  95%% CI %s" % (d["n_boot"], d["mean"], d["se"], fmt_ci(d["ci95"])))
        sm = r["E_mixed_effects"].get("statsmodels", {})
        cf = r["E_mixed_effects"]["closed_form"]
        if "error" in sm:
            print("  [E] mixed effects (statsmodels): FAILED %s" % sm["error"])
        else:
            print("  [E] mixed effects REML  y ~ 1 + (1|universe)"
                  "\n        grand mean=%+.4f  SE=%.4f  95%% CI %s  var_universe=%.4f  var_resid=%.4f"
                  % (sm["grand_mean"], sm["se"], fmt_ci(sm["ci95"]),
                     sm["var_universe"], sm["var_residual"]))
        print("      closed-form balanced equivalent  SE=%.4f  95%% CI %s  ICC=%.3f"
              % (cf["se"], fmt_ci(cf["ci95"]), cf["icc"]))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "generated": "2026-08-29",
            "source_csv": os.path.relpath(CSV, PROJECT_ROOT),
            "n_boot": N_BOOT,
            "bootstrap_seed": BOOT_SEED,
            "windows": results,
        }, f, indent=2)
    print("\n  Wrote %s" % os.path.relpath(OUT, PROJECT_ROOT))

    print("\n" + "=" * 96)
    print("  RECOMMENDED REPORTING LINES")
    print("=" * 96)
    for label in WINDOWS:
        r = results[label]
        c = r["C_stratified_bootstrap"]
        d = r["D_cluster_bootstrap"]
        sm = r["E_mixed_effects"].get("statsmodels", {})
        print("  %-8s mean %+.2f | seed-level CI (stratified boot) %s | "
              "market-level CI (cluster boot) %s | mixed-effects CI %s | dispersion SD %.2f"
              % (label, r["grand_mean_unweighted"], fmt_ci(c["ci95"]), fmt_ci(d["ci95"]),
                 fmt_ci(sm["ci95"]) if "ci95" in sm else "n/a",
                 r["A_cross_universe_dispersion_sd"]))


if __name__ == "__main__":
    main()
