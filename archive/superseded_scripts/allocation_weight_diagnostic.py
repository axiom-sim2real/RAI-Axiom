"""
================================================================================
  P0 TASK 1 SUPPLEMENT: Allocation Weight Diagnostic
  
  Plots RAI v6 cash allocation weight vs. SPY price over the test period.
  Checks whether the policy de-risks into cash before drawdowns (smart)
  or allocates uniformly regardless of market conditions (dumb).
  
  Output: data/diagnostics/allocation_vs_spy.png
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.download_data import ensure_real_market_checkpoints
from scripts.train_v6_fast import FastTradingNet
from scripts.eval_vs_standard_ai import metrics

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "diagnostics")
os.makedirs(OUTPUT_DIR, exist_ok=True)

V6_VARIANTS = {
    "RAI v6 Fast":       "rai_v6_fast.pt",
    "RAI v6 Alpha":      "rai_v6_alpha.pt",
    "RAI v6 Pro-Growth": "rai_v6_pro_growth.pt",
}


def extract_cash_weights(model, df):
    """Run model over DataFrame, return (dates, cash_weights, wealth_series)."""
    prices_raw = df.values[:, :10]
    T, N = prices_raw.shape

    cash = 5000.0
    shares = (5000.0 / N) / prices_raw[30]
    peak = 10000.0
    wealth_hist = [10000.0]
    cash_weights = []
    dates_out = []

    obs_history = []
    for t in range(30):
        p = prices_raw[t]
        p_prev = prices_raw[max(0, t - 1)]
        obs_history.append(np.concatenate([
            p / prices_raw[30],
            np.log(p / np.maximum(1e-4, p_prev)),
            [0.5, 0.0]
        ]).astype(np.float32))

    for t in range(30, T):
        flat_obs = np.concatenate(obs_history).astype(np.float32)
        act = model.get_action(flat_obs, deterministic=True)
        cl = np.clip(act[0] - 2.5, -8.0, 3.0)
        target_cash = float(1.0 / (1.0 + np.exp(-cl)))
        target_stock = 1.0 - target_cash
        ea = np.exp(act[1:] - np.max(act[1:]))
        target_aw = (ea / np.sum(ea)) * target_stock

        p = prices_raw[t]
        w = max(1e-4, cash + np.sum(shares * p))
        caw = (shares * p) / w
        ccf = cash / w
        drift = abs(ccf - target_cash) + np.sum(np.abs(caw - target_aw))

        if drift > 0.03:
            tv = abs(cash - w * target_cash) + np.sum(np.abs(shares * p - w * target_aw))
            net = max(1e-4, w - tv * 5 / 10000.0)
            cash = net * target_cash
            shares = (net * target_aw) / np.maximum(1e-4, p)

        nw = cash + np.sum(shares * p)
        peak = max(peak, nw)
        wealth_hist.append(nw)
        cash_weights.append(target_cash)
        dates_out.append(df.index[t])

        p_prev = prices_raw[t - 1]
        obs_history.pop(0)
        obs_history.append(np.concatenate([
            p / prices_raw[30],
            np.log(p / np.maximum(1e-4, p_prev)),
            [cash / nw, np.clip((nw - peak) / peak, -1, 0)]
        ]).astype(np.float32))

    return np.array(dates_out), np.array(cash_weights), np.array(wealth_hist)


def plot_diagnostic(models_dict, df, period_name, out_path):
    spy = df['SPY'].values if 'SPY' in df.columns else df.values[:, 0]
    spy_norm = spy / spy[0]
    dates_spy = df.index

    n_variants = len(models_dict)
    fig, axes = plt.subplots(n_variants + 1, 1, figsize=(16, 4 * (n_variants + 1)),
                              sharex=True, facecolor='#0f1117')
    fig.suptitle(f'RAI v6 Allocation Diagnostic — {period_name}',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    # Top panel: SPY price
    ax0 = axes[0]
    ax0.set_facecolor('#1a1d27')
    ax0.plot(dates_spy, spy_norm * 100, color='#4fc3f7', linewidth=1.5, label='SPY (normalized to 100)')
    ax0.fill_between(dates_spy, spy_norm * 100, 100, where=spy_norm < 1.0, color='#ef5350', alpha=0.3, label='SPY below start')
    ax0.set_ylabel('SPY (indexed=100)', color='white')
    ax0.tick_params(colors='white')
    ax0.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
    ax0.spines['bottom'].set_color('#333')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('#333')

    colors = ['#81c784', '#ffb74d', '#ce93d8']

    for idx, (label, (ckpt_path, model)) in enumerate(models_dict.items()):
        dates_m, cash_w, wealth = extract_cash_weights(model, df)
        m = metrics(wealth.tolist())

        ax = axes[idx + 1]
        ax.set_facecolor('#1a1d27')

        color = colors[idx % len(colors)]
        ax.plot(dates_m, cash_w * 100, color=color, linewidth=1.2, label=f'Cash % — {label}')
        ax.axhline(y=np.mean(cash_w) * 100, color=color, linestyle='--', alpha=0.5,
                   label=f'Mean cash: {np.mean(cash_w)*100:.1f}%')
        ax.fill_between(dates_m, cash_w * 100, alpha=0.15, color=color)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Cash Allocation (%)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333')

        info_txt = (f"Return: {m['return_pct']:+.1f}%  |  Sharpe: {m['sharpe']:.2f}  |  "
                    f"MaxDD: {m['max_dd']:+.1f}%  |  "
                    f"Ckpt: {os.path.basename(ckpt_path)}")
        ax.set_title(info_txt, color='#aaa', fontsize=9, pad=4)
        ax.legend(facecolor='#1a1d27', labelcolor='white', fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(color='white', rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    print(f"  Saved diagnostic plot: {out_path}", flush=True)


def main():
    ensure_real_market_checkpoints()
    test_csv = os.path.join(PROJECT_ROOT, "data", "real_market_checkpoints", "test_prices.csv")
    test_df = pd.read_csv(test_csv, index_col=0, parse_dates=True)

    ckpt_dir = os.path.join(PROJECT_ROOT, "data", "v0.6_rl_checkpoints")

    print("=" * 80, flush=True)
    print("  RAI v6 — ALLOCATION WEIGHT DIAGNOSTIC", flush=True)
    print("  Checks: Does cash allocation rise before SPY drawdowns?", flush=True)
    print("=" * 80, flush=True)

    loaded = {}
    for label, fname in V6_VARIANTS.items():
        path = os.path.join(ckpt_dir, fname)
        if os.path.exists(path):
            m = FastTradingNet(history_len=30, features_per_step=22, action_dim=11)
            m.load_state_dict(torch.load(path, weights_only=True))
            m.eval()
            loaded[label] = (path, m)
            print(f"  Loaded: {label} from {path}", flush=True)
        else:
            print(f"  MISSING: {path}", flush=True)

    if not loaded:
        print("  No checkpoints found. Train models first.", flush=True)
        return

    # Full OOS period
    plot_path = os.path.join(OUTPUT_DIR, "allocation_vs_spy.png")
    plot_diagnostic(loaded, test_df, "2020–2024 Out-of-Sample", plot_path)

    # Sub-period: COVID crash (2020)
    df_covid = test_df.loc["2020-01-01":"2020-12-31"]
    plot_path_covid = os.path.join(OUTPUT_DIR, "allocation_vs_spy_covid_2020.png")
    plot_diagnostic(loaded, df_covid, "2020 COVID Crash", plot_path_covid)

    # Sub-period: 2022 rate hike bear
    df_2022 = test_df.loc["2022-01-01":"2022-12-31"]
    plot_path_2022 = os.path.join(OUTPUT_DIR, "allocation_vs_spy_2022_bear.png")
    plot_diagnostic(loaded, df_2022, "2022 Rate-Hike Bear Market", plot_path_2022)

    print("\n  Summary Statistics", flush=True)
    print(f"  {'Model':<30} | {'Mean Cash%':>10} | {'Max Cash%':>10} | {'Min Cash%':>10}", flush=True)
    print(f"  {'-'*68}", flush=True)
    for label, (path, model) in loaded.items():
        _, cash_w, _ = extract_cash_weights(model, test_df)
        print(f"  {label:<30} | {np.mean(cash_w)*100:>9.1f}% | {np.max(cash_w)*100:>9.1f}% | {np.min(cash_w)*100:>9.1f}%", flush=True)

    print(f"\n  Plots saved to: {OUTPUT_DIR}", flush=True)
    print("  Review whether cash peaks align with SPY drawdown troughs.", flush=True)


if __name__ == "__main__":
    main()
