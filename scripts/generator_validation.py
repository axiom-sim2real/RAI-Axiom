"""
================================================================================
  P3 TASK 10: Level 6 Generator Empirical Validation
  
  Compares Level6_CombinedRealistic synthetic output against real market data on:
    1. Autocorrelation of squared returns (volatility clustering — Ljung-Box test)
    2. Estimated tail index vs. Student-t df=4 target (Hill estimator)
    3. Empirical jump frequency vs. Poisson lambda=0.02 target
    4. Return autocorrelation (near-zero for efficient markets)
    5. Distribution kurtosis comparison (synthetic vs. real)
  
  Output: numeric fit report + plots saved to data/generator_validation/
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
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from scipy import stats
    from scipy.stats import kurtosis, skew
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.download_data import ensure_real_market_checkpoints

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "generator_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Synthetic Level 6 Generator (inline, self-contained) ─────────────────────
def generate_level6_synthetic(n_assets=10, n_steps=2000, seed=42):
    """
    Level6_CombinedRealistic:
    Student-t fat tails (df=4) + GARCH(1,1) volatility clustering
    + Cholesky cross-asset correlation + Poisson jump-diffusion.
    """
    rng = np.random.RandomState(seed)

    # GARCH(1,1) parameters
    omega, alpha_g, beta_g = 0.00001, 0.09, 0.90
    # Poisson jump parameters
    jump_lambda = 0.02   # average 2% of days have a jump
    jump_mu     = -0.02  # mean jump size (negative = downward jumps)
    jump_sigma  = 0.04   # std of jump size

    # Random correlation structure
    A = rng.randn(n_assets, n_assets) * 0.3
    corr_mat = A @ A.T
    np.fill_diagonal(corr_mat, 1.0)
    # Normalize to correlation matrix
    D = np.sqrt(np.diag(corr_mat))
    corr_mat = corr_mat / np.outer(D, D)
    np.fill_diagonal(corr_mat, 1.0)
    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(corr_mat + np.eye(n_assets) * 1e-6)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets)

    # Initialize
    prices = np.ones((n_steps + 1, n_assets)) * 100.0
    sigma2 = np.ones(n_assets) * 0.0002   # initial variance
    base_vol = 0.015
    returns_all = []

    for t in range(1, n_steps + 1):
        # Draw correlated Student-t innovations
        z = rng.standard_t(df=4, size=n_assets) / np.sqrt(4 / (4 - 2))
        z_corr = (L @ z) * base_vol

        # GARCH(1,1) update
        eps_prev = z_corr
        sigma2 = omega + alpha_g * eps_prev ** 2 + beta_g * sigma2
        sigma2 = np.clip(sigma2, 1e-6, 0.25)
        garch_z = z_corr * np.sqrt(sigma2 / (base_vol ** 2))

        # Poisson jump process
        n_jumps = rng.poisson(jump_lambda, n_assets)
        jump_returns = np.where(n_jumps > 0,
                                rng.normal(jump_mu, jump_sigma, n_assets) * n_jumps,
                                0.0)

        # Combined log-return
        log_ret = 0.0002 + garch_z + jump_returns   # drift + GARCH + jumps
        prices[t] = prices[t - 1] * np.exp(log_ret)
        returns_all.append(log_ret)

    return prices, np.array(returns_all)


def ljung_box_test(returns_sq, nlags=10):
    """Ljung-Box test on squared returns for volatility clustering."""
    n = len(returns_sq)
    acf_vals = [np.corrcoef(returns_sq[:-k], returns_sq[k:])[0, 1] for k in range(1, nlags + 1)]
    lb_stat = n * (n + 2) * sum(r ** 2 / (n - k) for k, r in enumerate(acf_vals, 1))
    # p-value from chi-squared(nlags)
    if HAS_SCIPY:
        p_val = 1 - stats.chi2.cdf(lb_stat, df=nlags)
    else:
        p_val = float('nan')
    return lb_stat, p_val, acf_vals


def hill_estimator(returns, tail_fraction=0.10):
    """Hill estimator for tail index (alpha). Student-t df=4 → tail index ~4."""
    abs_returns = np.abs(returns)
    k = max(int(len(abs_returns) * tail_fraction), 10)
    sorted_r = np.sort(abs_returns)[::-1]
    if sorted_r[k - 1] <= 0:
        return float('nan')
    alpha_hat = k / np.sum(np.log(sorted_r[:k] / sorted_r[k - 1]))
    return alpha_hat


def empirical_jump_frequency(returns, threshold_sigma=3.0):
    """Fraction of days where |return| > threshold_sigma standard deviations."""
    mu, sig = np.mean(returns), np.std(returns)
    if sig < 1e-10:
        return 0.0
    return float(np.mean(np.abs(returns - mu) > threshold_sigma * sig))


def analyze_stylized_facts(returns_2d, label):
    """
    Analyze all assets, return dict of stylized-fact metrics.
    returns_2d: (T, N) array of daily log-returns
    """
    T, N = returns_2d.shape
    all_results = {}

    lb_stats, lb_pvals = [], []
    hill_alphas = []
    jump_freqs = []
    kurtoses = []
    return_autocorrs = []

    for i in range(N):
        r = returns_2d[:, i]
        # 1. Ljung-Box on squared returns
        lb, p, _ = ljung_box_test(r ** 2, nlags=10)
        lb_stats.append(lb)
        lb_pvals.append(p)
        # 2. Hill estimator (tail index)
        hill_alphas.append(hill_estimator(r))
        # 3. Jump frequency
        jump_freqs.append(empirical_jump_frequency(r, threshold_sigma=3.0))
        # 4. Kurtosis
        if HAS_SCIPY:
            kurtoses.append(float(kurtosis(r, fisher=True)))
        else:
            kurtoses.append(float(np.mean((r - np.mean(r)) ** 4) / np.std(r) ** 4 - 3))
        # 5. Return autocorrelation (lag-1)
        if len(r) > 2:
            return_autocorrs.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))

    all_results = {
        "label": label,
        "n_assets": N,
        "n_days": T,
        "lb_stat_mean": float(np.nanmean(lb_stats)),
        "lb_pval_mean": float(np.nanmean(lb_pvals)),
        "lb_clustering_detected": float(np.mean(np.array(lb_pvals) < 0.05)),
        "hill_alpha_mean": float(np.nanmean(hill_alphas)),
        "hill_alpha_std": float(np.nanstd(hill_alphas)),
        "target_tail_alpha_student_t_df4": 4.0,
        "jump_freq_mean": float(np.nanmean(jump_freqs)),
        "target_jump_freq_poisson_lambda002": empirical_jump_frequency(
            np.random.normal(0, 1, 100000), threshold_sigma=3.0),
        "kurtosis_mean": float(np.nanmean(kurtoses)),
        "kurtosis_std": float(np.nanstd(kurtoses)),
        "return_autocorr_mean": float(np.nanmean(return_autocorrs)),
    }
    return all_results


def print_report(r):
    print(f"\n  {'─'*70}")
    print(f"  STYLIZED FACT ANALYSIS: {r['label']}")
    print(f"  Assets: {r['n_assets']}, Days: {r['n_days']}")
    print(f"  {'─'*70}")
    print(f"  1. VOLATILITY CLUSTERING (Ljung-Box on squared returns)")
    print(f"     LB Statistic (mean):  {r['lb_stat_mean']:.2f}")
    print(f"     LB p-value (mean):    {r['lb_pval_mean']:.4f}  (< 0.05 = clustering present)")
    print(f"     Assets with p<0.05:   {r['lb_clustering_detected']*100:.0f}%  (target: >80% for GARCH)")
    print(f"  2. TAIL INDEX (Hill estimator, tail fraction=10%)")
    print(f"     Estimated alpha:      {r['hill_alpha_mean']:.2f} ± {r['hill_alpha_std']:.2f}")
    print(f"     Target (Student-t df=4): ~4.0  (lower = fatter tails)")
    fat_tail_ok = r['hill_alpha_mean'] < 6.0
    print(f"     Fat-tail match:       {'PASS' if fat_tail_ok else 'FAIL'} (alpha < 6.0 indicates heavy tails)")
    print(f"  3. JUMP FREQUENCY (|return| > 3 sigma)")
    print(f"     Empirical jump freq:  {r['jump_freq_mean']*100:.2f}%")
    print(f"     Target (Poisson l=0.02 × 3σ threshold): ~0.27% for N(0,1)")
    print(f"     Note: target ~{0.02*100:.1f}% of days expect a jump per Poisson process")
    print(f"  4. KURTOSIS (excess, Fisher definition)")
    print(f"     Kurtosis (mean):      {r['kurtosis_mean']:.2f} ± {r['kurtosis_std']:.2f}")
    print(f"     Gaussian baseline:    0.0 (positive = fat tails)")
    print(f"  5. RETURN AUTOCORRELATION (lag-1)")
    print(f"     Autocorr (mean):      {r['return_autocorr_mean']:.4f}  (target: ~0 for efficient markets)")
    print(f"  {'─'*70}")


def main():
    ensure_real_market_checkpoints()

    print("=" * 75, flush=True)
    print("  P3 TASK 10: LEVEL 6 GENERATOR EMPIRICAL VALIDATION", flush=True)
    print("  Comparing synthetic G6 output vs. real market stylized facts", flush=True)
    print("=" * 75, flush=True)

    # ── Real market data ──────────────────────────────────────────────────────
    train_csv = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "train_prices.csv")
    real_df = pd.read_csv(train_csv, index_col=0, parse_dates=True).dropna()
    real_prices = real_df.values
    real_returns = np.diff(np.log(np.maximum(real_prices, 1e-8)), axis=0)
    real_results = analyze_stylized_facts(real_returns, "Real Market Data (2010-2019)")
    print_report(real_results)

    # ── Synthetic Level 6 ─────────────────────────────────────────────────────
    n_assets = real_prices.shape[1]
    syn_prices, syn_returns = generate_level6_synthetic(n_assets=n_assets, n_steps=2000)
    syn_results = analyze_stylized_facts(syn_returns, "Level 6 Synthetic Generator")
    print_report(syn_results)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n  {'='*75}")
    print(f"  SIDE-BY-SIDE FIT COMPARISON")
    print(f"  {'='*75}")
    print(f"  {'Metric':<40} | {'Real':>10} | {'Synthetic':>10} | {'Match?':>8}")
    print(f"  {'-'*72}")

    def match(rv, sv, tol_pct=30):
        if rv == 0:
            return "N/A"
        return "PASS" if abs(rv - sv) / (abs(rv) + 1e-8) < tol_pct / 100 else "FAIL"

    rows = [
        ("LB Stat mean (vol. clustering)", real_results['lb_stat_mean'], syn_results['lb_stat_mean']),
        ("% Assets with LB p<0.05", real_results['lb_clustering_detected']*100, syn_results['lb_clustering_detected']*100),
        ("Hill alpha (tail index)", real_results['hill_alpha_mean'], syn_results['hill_alpha_mean']),
        ("Jump freq (|r|>3sig) %", real_results['jump_freq_mean']*100, syn_results['jump_freq_mean']*100),
        ("Excess kurtosis mean", real_results['kurtosis_mean'], syn_results['kurtosis_mean']),
        ("Return autocorr (lag-1)", real_results['return_autocorr_mean'], syn_results['return_autocorr_mean']),
    ]

    for name, rv, sv in rows:
        m_txt = match(rv, sv)
        print(f"  {name:<40} | {rv:>10.3f} | {sv:>10.3f} | {m_txt:>8}")

    print(f"  {'-'*72}")

    # ── Save report ───────────────────────────────────────────────────────────
    import json
    report = {"real": real_results, "synthetic_level6": syn_results}
    out_path = os.path.join(OUTPUT_DIR, "generator_validation_report.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to: {out_path}", flush=True)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='#0f1117')
        fig.suptitle('Level 6 Generator vs Real Market — Stylized Facts', color='white', fontsize=13)

        # (1) Return distribution comparison
        ax = axes[0]
        ax.set_facecolor('#1a1d27')
        r_flat = real_returns.flatten()
        s_flat = syn_returns.flatten()
        lim = np.percentile(np.abs(r_flat), 99)
        ax.hist(r_flat, bins=100, density=True, alpha=0.6, color='#4fc3f7', label='Real', range=(-lim, lim))
        ax.hist(s_flat, bins=100, density=True, alpha=0.6, color='#81c784', label='Synthetic G6', range=(-lim, lim))
        ax.set_title('Return Distribution', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1d27', labelcolor='white')

        # (2) Squared-return autocorrelation (vol clustering)
        ax = axes[1]
        ax.set_facecolor('#1a1d27')
        max_lag = 20
        real_sq_acf = [np.corrcoef(r_flat[:-k] ** 2, r_flat[k:] ** 2)[0, 1] for k in range(1, max_lag + 1)]
        syn_sq_acf  = [np.corrcoef(s_flat[:-k] ** 2, s_flat[k:] ** 2)[0, 1] for k in range(1, max_lag + 1)]
        lags = range(1, max_lag + 1)
        ax.plot(lags, real_sq_acf, color='#4fc3f7', marker='o', markersize=4, label='Real')
        ax.plot(lags, syn_sq_acf,  color='#81c784', marker='s', markersize=4, label='Synthetic G6')
        ax.axhline(0, color='white', alpha=0.3)
        ax.set_title('Squared-Return ACF (Vol Clustering)', color='white')
        ax.set_xlabel('Lag', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1d27', labelcolor='white')

        # (3) Hill estimator across tail fractions
        ax = axes[2]
        ax.set_facecolor('#1a1d27')
        fracs = np.linspace(0.05, 0.20, 10)
        real_hills = [hill_estimator(r_flat, f) for f in fracs]
        syn_hills  = [hill_estimator(s_flat, f) for f in fracs]
        ax.plot(fracs * 100, real_hills, color='#4fc3f7', marker='o', markersize=4, label='Real')
        ax.plot(fracs * 100, syn_hills,  color='#81c784', marker='s', markersize=4, label='Synthetic G6')
        ax.axhline(4.0, color='#ffb74d', linestyle='--', label='Student-t df=4 target')
        ax.set_title('Hill Tail Index by Tail Fraction', color='white')
        ax.set_xlabel('Tail fraction (%)', color='white')
        ax.tick_params(colors='white')
        ax.legend(facecolor='#1a1d27', labelcolor='white')

        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#333')
            ax.spines['bottom'].set_color('#333')

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "generator_validation_plots.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
        plt.close()
        print(f"  Plots saved to: {plot_path}", flush=True)

    print(f"\n  INTERPRETATION:")
    print(f"  - Good fit on all metrics → Level 6 generator is sufficiently realistic.")
    print(f"  - FAIL on key metrics → cite in paper as a limitation / calibrate parameters.")


if __name__ == "__main__":
    main()
