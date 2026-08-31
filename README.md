# RAI: Relational Artificial Intelligence from Artificial Worlds

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Can a portfolio allocation policy trained **entirely on synthetic market data** generalise to real financial markets?

---

## Overview

**RAI** trains a deep RL agent inside computer-generated markets. The agent never sees real-world price data during training. After training, the policy is frozen and deployed zero-shot on real financial markets across multiple geographies and asset classes.

This repository is the research codebase for a study on **zero-shot sim-to-real transfer in portfolio management**. The contribution is demonstrating that a policy trained on 0% real data can produce risk-adjusted returns statistically indistinguishable from a real-data-trained LSTM *when both are charged the same transaction costs on the same window* (Welch p = 0.51 OOS, and still a tie at p = 0.55 after correcting for clustering by universe) — not that it outperforms buy-and-hold in absolute return, and not that it holds up out of sample: in the holdout it is beaten by monthly Risk Parity and loses to the LSTM in every universe where the difference is significant (cluster-corrected, in all six, p = 0.0053). Measured against a **real, fixed SPY buy-and-hold** — added 2026-08-30, the same SPY series in all six universes — it is a tested tie OOS (+0.98 vs +0.90, cluster-corrected p = 0.76) and **0.69 Sharpe behind on the holdout point estimate** (+0.62 vs +1.31, SPY ahead in 4 of 6 universes, too dispersed to resolve at n = 6: p = 0.23). All arms are now on one basis; see [Baseline Comparison](#baseline-comparison--single-basis-pinned-window-axiom-cost-model) and [Significance Tests](#significance-tests--axiom-vs-the-two-learned-baselines), and note that the LSTM arm is a confirmed degenerate constant allocation.

---

## Canonical Results (August 2026)

> Axiom was previously referred to as 'RAI v6 Alpha' in earlier local reports — renamed to avoid confusion with the v6/v7/v8.2 numbering used in the separate GitHub repository maintained by other contributors. Currently v0.9.

> **Architecture**: All Axiom results use **`AxiomNet`** — Conv1d ×2 + 1-layer Transformer + flatten, 289,527 params. `[v6] Fast` results use a *different* network, **`FastTradingNet`** (mean-pool, 51,703 params). Both were formerly named `DeepEndToEndTradingNet`; see [Architecture](#architecture).  
> **Axiom status**: **CI-verified** — 10 seeds, 6 universes, 60 evaluations. LSTM and XGBoost baselines are also 10-seed as of 2026-08-29; the deterministic baselines (Risk Parity, EW, **SPY B&H (fixed reference)**, Asset-0 B&H, SMA, 60/40) have no seed to vary.  
> **Cost model**: 5 bps fee + 0.02% slippage + 3% portfolio-drift rebalancing threshold. All arms in the baseline table are now evaluated on this same model and the same pinned window.  
> **Split**: 60% train / 20% OOS / 20% untouched future holdout (chronological, no look-ahead).  
> **Universes**: Point-in-time construction. Sources documented in [`canonical_evaluation.py`](scripts/canonical_evaluation.py).

### Axiom v0.9 — CI-Verified Sharpe (10 seeds, mean ± SD across seeds, bootstrap 95% CI)

| Universe | OOS Mean ± SD | OOS 95% CI | Holdout Mean ± SD | Holdout 95% CI |
|---|---|---|---|---|
| US ETFs | **+1.35 ± 0.17** | [+1.24, +1.44] | **+1.43 ± 0.16** | [+1.34, +1.53] |
| US Mega-Cap (PIT) | **+1.90 ± 0.41** | [+1.67, +2.14] | **+1.69 ± 0.21** | [+1.56, +1.80] |
| Global Indices | **+1.08 ± 0.23** | [+0.94, +1.20] | **+0.91 ± 0.13** | [+0.84, +0.99] |
| India Nifty 50 | +0.04 ± 0.36 | [-0.17, +0.26] | -0.46 ± 0.41 | [-0.70, -0.22] |
| Forex & Commodities | +0.40 ± 0.27 | [+0.26, +0.57] | **+1.34 ± 0.34** | [+1.13, +1.53] |
| Crypto (PIT) | **+1.15 ± 0.17** | [+1.04, +1.24] | -1.18 ± 0.18 | [-1.28, -1.08] |
| **Overall — seed-level 95% CI** | **+0.98** | **[+0.92, +1.05]** | **+0.62** | **[+0.56, +0.68]** |
| **Overall — market-level 95% CI** | **+0.98** | **[+0.45, +1.52]** | **+0.62** | **[-0.31, +1.56]** |
| *Cross-universe dispersion (descriptive)* | *SD = 0.67* | — | *SD = 1.17* | — |

> **Note on "Overall" — three different questions, three different numbers.** All are computed from
> the same 60 seed-level observations (10 seeds × 6 universes) in [`data/axiom_per_seed_results.csv`](data/axiom_per_seed_results.csv);
> reproduce with `python scripts/aggregate_ci.py` → [`data/axiom_aggregate_stats.json`](data/axiom_aggregate_stats.json).
>
> - **Seed-level 95% CI** — stratified bootstrap, 10,000 iterations, seeds resampled within each universe, universes treated as fixed (SE = 0.034 OOS / 0.032 holdout). Answers *"would different random seeds change this?"* — no.
> - **Market-level 95% CI** — mixed-effects REML with universe as a random intercept (`y ~ 1 + (1|universe)`); a two-level cluster bootstrap agrees ([+0.48, +1.47] OOS / [-0.29, +1.42] holdout). Answers *"would different markets change this?"* — substantially. ICC = 0.85 OOS / 0.95 holdout, i.e. most of the variance is between markets, not between seeds. **The holdout interval crosses zero: the holdout result does not generalise beyond the six markets tested.**
> - **Cross-universe dispersion SD** — descriptive spread of the six per-universe means. Not an interval on the mean and not a standard error.
>
> An earlier version of this document reported "±0.67 / ±1.10" as though they were uncertainty on the
> aggregate mean; they were dispersion. Verified directly: `std({1.35, 1.90, 1.08, 0.04, 0.40, 1.15}, ddof=1) = 0.668`.
> The holdout dispersion figure was also wrong — it is **1.17** (ddof=1), not 1.10.


### Baseline Comparison — single basis: pinned window, Axiom cost model

Every arm below is evaluated on the **same** pinned window (2016-08-20 / 2021-08-20 → 2026-08-20),
the **same** chronological 60/20/20 split, and the **same** cost model (5 bps + 0.02% slippage above
a 3% drift gate). Earlier versions of this table mixed bases — Axiom/LSTM/XGBoost on the pinned
window, the rule-based arms and `[v6] Fast` on `canonical_evaluation.py`'s own relative download
window at zero cost. Re-run 2026-08-29 by
[`scripts/deterministic_baselines_pinned.py`](scripts/deterministic_baselines_pinned.py) →
[`data/deterministic_baselines_pinned.csv`](data/deterministic_baselines_pinned.csv); the fixed-SPY
column was added by the same script on 2026-08-30.

**OOS Sharpe**

| Universe | Axiom | LSTM (real) | XGBoost (real) | Risk Parity | EW (1/N) | [v6] Fast | 60/40 | SPY B&H (fixed ref) | Asset-0 B&H | SMA 50/200 |
|---|---|---|---|---|---|---|---|---|---|---|
| US ETFs | **+1.35** | +1.11 | +0.00 | +0.79 | +0.74 | +0.81 | +0.17 | +1.00 | +0.39 | -0.05 |
| US Mega-Cap (PIT) | **+1.90** | +1.74 | +0.69 | +1.30 | +1.49 | +1.22 | +0.53 | +1.00 | +0.64 | +0.30 |
| Global Indices | +1.08 | +0.86 | -0.20 | +0.72 | +0.63 | **+1.32** | +0.60 | +1.00 | +0.39 | -0.05 |
| India Nifty 50 | +0.04 | -0.25 | -0.72 | +0.13 | +0.08 | **+0.99** | +0.47 | +0.82 | -0.25 | -0.75 |
| Forex & Commodities | +0.40 | +0.90 | -0.51 | **+1.16** | +0.65 | +0.35 | -0.54 | +0.87 | -0.28 | -0.95 |
| Crypto (PIT) | +1.15 | +1.08 | +1.17 | **+1.51** | +1.24 | +0.45 | +0.92 | +0.73 | +0.88 | -0.14 |
| **Mean** | **+0.98** | +0.91 | +0.07 | +0.94 | +0.81 | +0.86 | +0.36 | +0.90 | +0.29 | -0.27 |

**Holdout Sharpe**

| Universe | Axiom | LSTM (real) | XGBoost (real) | Risk Parity | EW (1/N) | [v6] Fast | 60/40 | SPY B&H (fixed ref) | Asset-0 B&H | SMA 50/200 |
|---|---|---|---|---|---|---|---|---|---|---|
| US ETFs | +1.43 | **+1.63** | +0.38 | +1.50 | +1.58 | +1.13 | +0.98 | +1.14 | +1.18 | +1.06 |
| US Mega-Cap (PIT) | +1.69 | **+1.86** | +1.01 | +1.76 | +1.71 | +1.34 | +1.01 | +1.14 | +0.75 | +0.28 |
| Global Indices | +0.91 | +1.09 | +0.23 | +1.05 | +1.04 | +0.90 | **+1.22** | +1.14 | +1.18 | +1.06 |
| India Nifty 50 | -0.46 | -0.43 | -1.43 | -0.45 | -0.29 | +0.29 | +0.61 | **+1.50** | +0.73 | +0.29 |
| Forex & Commodities | +1.34 | +1.46 | -0.04 | **+2.61** | +1.01 | +0.36 | +0.87 | +1.61 | +1.14 | +1.14 |
| Crypto (PIT) | -1.17 | -1.11 | -1.08 | -0.94 | -1.01 | -1.88 | -0.72 | **+1.32** | -0.89 | -0.17 |
| **Mean** | +0.62 | +0.75 | -0.15 | +0.92 | +0.67 | +0.36 | +0.66 | **+1.31** | +0.68 | +0.61 |

> **Three labelling / coverage corrections** — none of them changed any number reported for Axiom:
>
> - **`SMA 50/200` is now warm-up-corrected** (2026-08-29). The column above is the *fixed* arm. Previously the
>   evaluator saw only in-window prices, so the 200-day mean was undefined for the first 200 in-window
>   days and the rule silently defaulted to holding asset 0 — see finding 4 below and
>   [`docs/consolidation_report.md` §20](docs/consolidation_report.md).
> - **`Asset-0 B&H` was previously labelled `SPY B&H`** (relabelled 2026-08-29). Both this arm and the SMA act on **column 0**
>   of each universe, which is yfinance's alphabetically-first ticker and is **not SPY in any of the
>   six universes**: EEM (US ETFs *and* Global Indices — which is why those two rows are identical),
>   AAPL (US Mega-Cap), AXISBANK.NS (India), AUDUSD=X (Forex), BCH-USD (Crypto). Only the label
>   changed; the numbers are unchanged.
> - **`SPY B&H (fixed reference)` is a genuinely new arm** (added 2026-08-30). It holds the **real SPY** —
>   the same price series in every universe, independent of that universe's tickers and column order —
>   on the same pinned windows and the same cost model. SPY is a real constituent only in US ETFs and
>   Global Indices; for the other four it is deliberately an **out-of-universe reference point**, not a
>   swap-in for `Asset-0 B&H`, which is why both rows are kept. The three 10-year universes share one
>   SPY value because they share identical window date spans. Validated where it can be: in US ETFs and
>   Global Indices it reproduces buy-and-hold of that universe's own `SPY` column to
>   |ΔSharpe| < 1e-6. See [`docs/consolidation_report.md` §21](docs/consolidation_report.md).

Five things this single-basis table makes visible that the old mixed-basis one hid:

1. **Risk Parity beats Axiom in the holdout** (+0.92 vs +0.62) and is essentially level with it OOS
   (+0.94 vs +0.98). It uses no ML, no training data and no synthetic markets — just monthly
   inverse-volatility weights. On the holdout window the zero-shot policy does not beat a textbook
   rule.
2. **A real fixed-SPY buy-and-hold is the strongest holdout arm in the table** (+1.31 vs Axiom's
   +0.62 and Risk Parity's +0.92), while OOS it is level with Axiom (+0.90 vs +0.98). The arm the
   repo previously called "SPY" was column 0 and scored +0.29 / +0.68 — roughly a third of the real
   thing, so replacing it makes the passive benchmark materially harder. Significance in
   [Axiom vs the real fixed SPY](#axiom-vs-the-real-fixed-spy-2026-08-30): a tie OOS, and a holdout
   deficit too dispersed to resolve at n = 6.
3. **SPY / Asset-0 B&H / EW (1/N) / 60/40 are cost-invariant *by construction*, not by robustness.** They buy
   once at t=0 and never trade, and Axiom's evaluator charges nothing for the entry allocation, so
   their cost-charged and zero-cost curves are byte-identical. Do not read their unchanged numbers as
   evidence that costs don't matter.
4. **Deterministic arms have no seed.** Risk Parity, EW, SPY B&H, Asset-0 B&H, SMA and 60/40 are functions of
   the price series alone, and `[v6] Fast` is a single `FastTradingNet` checkpoint. These are point
   estimates, not 10-seed means, and carry no CI. Only Axiom, LSTM and XGBoost are 10-seed.
5. **SMA 50/200's published numbers were a 200-day warm-up artefact — now fixed.** The evaluator saw
   only in-window prices, so the 200-day mean was undefined for the first 200 in-window days and the
   rule fell back to holding asset 0. On the ~250-day 5-year windows that was almost the entire
   window, and the curve came out **bit-identical to asset-0 buy-and-hold in 5 of 12 universe-windows**
   (Forex/holdout, Global Indices/holdout, US ETFs/holdout, India/holdout, India/OOS). The arm now
   receives the 200 trading days immediately *preceding* each evaluation window as indicator warm-up
   (train split for OOS, train+OOS for the holdout), which is available for all 12 universe-windows —
   **0 default days remain**. No look-ahead: the warm-up slice is strictly earlier than the window and
   wealth accrues only inside it. The correction moves the arm **down**, because the artefact had been
   substituting buy-and-hold for a rule that mostly loses:

   | | OOS mean | Holdout mean | universe-windows == asset-0 B&H |
   |---|---|---|---|
   | SMA 50/200, no warm-up (as published) | +0.13 | +0.77 | 5 of 12 |
   | **SMA 50/200, warm-up-corrected (now reported)** | **-0.27** | **+0.61** | **1 of 12** |

   The one remaining tie (Forex/holdout, +1.14) is genuine, not an artefact: with a live signal the
   rule stays long for the whole window and executes 0 flips. Per-universe deltas and flip counts:
   [`data/deterministic_baselines_pinned.csv`](data/deterministic_baselines_pinned.csv); both arms are
   kept in that file so the published figures stay reproducible.

Cost sensitivity, mean Sharpe across the six universes (pinned window, same rows as above):

| Arm | OOS zero-cost | OOS costed | Δ | Trades |
|---|---|---|---|---|
| Risk Parity | +0.944 | +0.937 | -0.006 | 8–15 rebalances |
| SMA 50/200 (warm-up-corrected) | -0.261 | -0.272 | -0.011 | 2–5 flips |
| SMA 50/200 (no warm-up, as published) | +0.136 | +0.131 | -0.004 | 0–3 flips |
| [v6] Fast | +0.887 | +0.857 | -0.030 | drift-gated |
| SPY B&H (fixed ref) / Asset-0 B&H / EW / 60/40 | — | — | 0.000 | none after entry |
| XGBoost (real) | +1.046 | +0.070 | **-0.976** | daily |
| LSTM (real) | +0.941 | +0.906 | -0.035 | ~none (see below) |

For continuity with the previously published numbers, the zero-cost harness figures for the two
learned baselines are retained here: LSTM **+0.94** OOS / +0.78 holdout, XGBoost **+1.05** OOS /
+0.79 holdout, with seed-level 95% CIs [+0.93, +0.95] and [+1.01, +1.08] and market-level CIs
[+0.41, +1.36] and [+0.67, +1.42] (Axiom's own: [+0.92, +1.05] and [+0.45, +1.52]). Those are *not*
the basis for any claim below — the cost-matched columns are.

**LSTM and XGBoost are 10-seed means** (seeds 42/101/202/303/404/505/606/707/808/909 — the same
seeds as Axiom), re-run after the seed-propagation fix. Per-seed data:
[`data/baseline_per_seed_results.csv`](data/baseline_per_seed_results.csv); aggregates:
[`data/baseline_multiseed_summary.json`](data/baseline_multiseed_summary.json). Reproduce with
`python scripts/baseline_multiseed.py`.


### Significance Tests — Axiom vs the two learned baselines

Axiom's 10 seeds are *training-initialisation* seeds; the LSTM/XGBoost seeds are *model-fitting*
seeds on real data. Seed 42 of Axiom and seed 42 of the LSTM share nothing but the integer, so the
two samples are **not paired** and a paired test would be invalid. Reported instead: Welch's t
(unequal variances) and Mann-Whitney U (rank-based) on the independent 10-vs-10 samples, on the
pinned window under Axiom's cost model — the only basis on which all three arms are measured the same
way. [`scripts/cross_model_significance.py`](scripts/cross_model_significance.py) →
[`data/cross_model_significance.csv`](data/cross_model_significance.csv).

| Comparison | Axiom | Baseline | Diff | Welch p | MWU p | Cohen's d |
|---|---|---|---|---|---|---|
| OOS, Axiom vs LSTM | +0.98 | +0.91 | +0.08 | 0.51 | 0.184 | 0.12 |
| OOS, Axiom vs XGBoost | +0.98 | +0.07 | +0.91 | **2.2e-11** | **8.3e-10** | 1.35 |
| Holdout, Axiom vs LSTM | +0.62 | +0.75 | -0.13 | 0.534 | 0.208 | -0.11 |
| Holdout, Axiom vs XGBoost | +0.62 | -0.15 | +0.78 | **4.4e-05** | **9.3e-06** | 0.78 |

#### Cluster-corrected, reported beside the pooled numbers (2026-08-29)

The four rows above pool 10 seeds × 6 universes as 60 independent observations. They are not
independent: the mixed-effects analysis below puts **ICC at 0.847 OOS / 0.953 holdout**, so the
effective sample size is nearer 6 than 60 and every pooled *p* above is anti-conservative. Three
corrections are reported *next to* the pooled numbers rather than replacing them — the pooled column
is what was previously published, and both belong on the page.

Pairing by **universe** is legitimate (both arms are scored on the same six price series); pairing by
**seed** stays invalid for the reason in the paragraph above the table. So the correction collapses
each arm to one mean per universe (n = 6) and runs a paired *t*, an exact Wilcoxon signed-rank and an
exact sign test on the six differences, plus a cluster-robust (CR1) OLS of Sharpe on an arm dummy over
all 120 seed-level observations, clustered on universe, with *p* from *t* at G−1 = 5 df.

| Comparison | pooled Welch p (n=60) | pooled MWU p | paired t p (n=6) | exact Wilcoxon p | sign-test p | CR1 OLS p, t(5) | CR1 SE ÷ naive SE | Verdict |
|---|---|---|---|---|---|---|---|---|
| OOS, vs LSTM | 0.51 | 0.184 | 0.549 | 0.4375 | 0.219 | 0.551 | 1.03× | tie either way |
| OOS, vs XGBoost | **2.2e-11** | **8.3e-10** | **0.0072** | 0.0625 | 0.219 | **0.0073** | 1.70× | **weakened ~3×10⁸** — significant by *t*, *not* by the exact rank test |
| Holdout, vs LSTM | 0.534 | 0.208 | **0.0053** | **0.03125** | **0.03125** | **0.0054** | 0.13× | **strengthened ~100×, against Axiom** |
| Holdout, vs XGBoost | **4.4e-05** | **9.3e-06** | **0.0127** | 0.0625 | 0.219 | **0.0129** | 1.12× | **weakened ~290×** — significant by *t*, not by the exact rank test |

Mean Sharpe difference with a *universe-level* 95% CI (n = 6 clusters), which is the interval that
answers "would another market change this?":

| Comparison | Axiom − baseline | 95% CI (n=6) | d_z | Axiom ahead in |
|---|---|---|---|---|
| OOS, vs LSTM | +0.077 | [-0.231, +0.385] | +0.26 | 5 of 6 |
| OOS, vs XGBoost | +0.913 | [+0.376, +1.450] | +1.78 | 5 of 6 |
| Holdout, vs LSTM | **-0.127** | **[-0.196, -0.058]** | -1.92 | **0 of 6** |
| Holdout, vs XGBoost | +0.777 | [+0.251, +1.303] | +1.55 | 5 of 6 |

Four things this makes visible that the pooled row hid:

1. **The XGBoost headline loses eight orders of magnitude.** OOS *p* goes 2.2e-11 → 0.0072. The
   direction survives, the "overwhelming" magnitude does not: most of that 2.2e-11 was 10 near-replicate
   seeds per universe being counted as 10 independent draws.
2. **At n = 6, the exact rank tests cannot reach 0.05 on a 5-of-6 split.** The smallest attainable
   two-sided Wilcoxon *p* is 2/64 = **0.03125**, and 5-of-6 gives 0.0625. So both XGBoost comparisons
   are significant on the parametric paired *t* and *not* on the distribution-free test. That is a
   power limit of six clusters, not evidence of no effect — but it means "Axiom beats XGBoost" rests on
   a normality assumption over six points.
3. **The holdout LSTM comparison reverses from "tie" to "significant against Axiom."** Pooling hid it:
   between-universe variance (Sharpe from +1.86 to −1.11) swamped a small but *perfectly consistent*
   within-universe deficit. Clustering prices that consistency instead — the LSTM is ahead in **6 of 6**
   holdout universes, the CR1 SE *shrinks* to 0.13× the naive SE, and the universe-level CI
   [-0.196, -0.058] excludes zero. This is the one place where correcting for clustering makes a
   result stronger, and it goes against Axiom.
4. **SE inflation is not uniform.** 1.70× (OOS XGBoost) to 0.13× (holdout LSTM). Cluster correction is
   not a blanket penalty; it re-weights toward *consistency across markets* and away from
   *magnitude within one market*.

Pooled tests treat all 60 observations as independent, which they are not. The
per-universe 10-vs-10 tests are the clean seed-level ones, and they are not uniformly favourable:

- **vs LSTM, OOS**: Axiom significantly ahead in 4 of 6 (US ETFs p=0.0022, Global Indices p=0.014,
  India p=0.031, and — against it — **Forex, where Axiom loses by -0.50, p=0.00021**). Mega-Cap
  (p=0.25) and Crypto (p=0.30) are ties.
- **vs LSTM, holdout**: **every significant result goes against Axiom** — US ETFs -0.20 (p=0.0044),
  Mega-Cap -0.18 (p=0.028), Global Indices -0.17 (p=0.0018); India, Forex and Crypto are ties.
- **vs XGBoost**: Axiom significantly ahead in 5 of 6 in both windows; Crypto is the sole tie
  (OOS p=0.71, holdout p=0.18).

### Axiom vs the Real Fixed SPY (2026-08-30)

The passive benchmark the paper's own framing invokes — "competitive with data-trained baselines, not
outperforming buy-and-hold" — was never actually tested against SPY for 5 of 6 universes: the arm
called "SPY B&H" held column 0. The real fixed-SPY arm is now evaluated on the same pinned windows and
the same cost model, and the tests below are run with the **same paired-by-universe methodology** as
the LSTM/XGBoost block above, reported beside it rather than replacing it.

One asymmetry has to be stated rather than hidden: **SPY has no seed.** It is a deterministic function
of one price series, so there is one number per universe-window, not a 10-seed sample. The pooled row
is therefore an unbalanced 60-vs-6 and is anti-conservative (it prices no uncertainty in SPY itself);
it is shown only for format parity. **The valid test is the cluster-corrected one** — collapse Axiom
to one mean per universe and pair the six differences. The CR1 OLS column runs over 66 observations
(60 Axiom seed-level + 6 singleton SPY clusters), so it is a cross-check on the paired n = 6 tests,
not a replacement.

| Comparison | pooled Welch p (n=60 vs 6) | pooled MWU p | paired t p (n=6) | exact Wilcoxon p | sign-test p | CR1 OLS p, t(5) |
|---|---|---|---|---|---|---|
| OOS, vs LSTM | 0.51 | 0.184 | 0.549 | 0.4375 | 0.219 | 0.551 |
| OOS, vs XGBoost | **2.2e-11** | **8.3e-10** | **0.0072** | 0.0625 | 0.219 | **0.0073** |
| **OOS, vs real SPY** | 0.411 | 0.332 | **0.756** | 0.844 | 0.688 | 0.758 |
| OOS, vs real SPY (native NYSE calendar) | — | — | 0.815 | 0.844 | 0.688 | 0.817 |
| Holdout, vs LSTM | 0.534 | 0.208 | **0.0053** | **0.03125** | **0.03125** | **0.0054** |
| Holdout, vs XGBoost | **4.4e-05** | **9.3e-06** | **0.0127** | 0.0625 | 0.219 | **0.0129** |
| **Holdout, vs real SPY** | 0.00015 | 0.270 | **0.234** | 0.563 | 0.688 | 0.237 |
| Holdout, vs real SPY (native NYSE calendar) | — | — | 0.232 | 0.563 | 0.688 | 0.235 |

Mean Sharpe difference with a *universe-level* 95% CI (n = 6 clusters):

| Comparison | Axiom − benchmark | 95% CI (n=6) | d_z | Axiom ahead in |
|---|---|---|---|---|
| **OOS, − real SPY** | **+0.082** | [-0.563, +0.727] | +0.13 | 4 of 6 |
| OOS, − real SPY (native NYSE) | +0.060 | [-0.566, +0.686] | +0.10 | 4 of 6 |
| **Holdout, − real SPY** | **-0.688** | [-1.993, +0.618] | -0.55 | 2 of 6 |
| Holdout, − real SPY (native NYSE) | -0.740 | [-2.139, +0.659] | -0.56 | 2 of 6 |

Per-universe direction (Axiom 10-seed mean vs the SPY constant):

| Universe | OOS Axiom | OOS SPY | OOS winner | Holdout Axiom | Holdout SPY | Holdout winner |
|---|---|---|---|---|---|---|
| US ETFs | +1.345 | +1.000 | Axiom | +1.433 | +1.143 | Axiom |
| US Mega-Cap (PIT) | +1.896 | +1.000 | Axiom | +1.688 | +1.143 | Axiom |
| Global Indices | +1.076 | +1.000 | Axiom (+0.08) | +0.914 | +1.143 | SPY |
| India Nifty 50 | +0.035 | +0.817 | SPY | -0.462 | +1.503 | SPY (-1.97) |
| Forex & Commodities | +0.403 | +0.866 | SPY | +1.340 | +1.614 | SPY |
| Crypto (PIT) | +1.145 | +0.725 | Axiom | -1.175 | +1.316 | SPY (-2.49) |

What this does and does not establish:

1. **OOS is a tested tie.** +0.98 vs +0.90, difference +0.08, universe-level CI [-0.56, +0.73],
   p = 0.76. "Competitive with buy-and-hold on SPY" now has a real SPY behind it — and it cuts both
   ways: there is no evidence Axiom beats SPY out-of-sample either.
2. **The holdout point estimate is well behind SPY** (+0.62 vs +1.31, -0.69) and SPY wins 4 of 6
   universes, but the CI [-1.99, +0.62] spans zero, so at n = 6 the deficit is not statistically
   resolved. That is an underpowered comparison, not a draw: the honest statement is *"behind SPY by
   0.69 Sharpe on the holdout, not statistically resolvable with six markets"*, with the point
   estimate attached.
3. **The pooled 0.00015 must not be quoted.** It counts 60 Axiom seeds against 6 singleton SPY values
   when ICC is 0.95; the cluster correction moves it to 0.234.
4. **Calendar padding does not drive any of it.** SPY trades the NYSE calendar and four of the six
   universes do not, so SPY is forward-filled onto each universe's own index. Recomputing on SPY's
   *native* NYSE sessions moves the holdout difference further **against** Axiom (-0.688 → -0.740) and
   changes no verdict. Only crypto is materially affected (115 of 365 padded days), where the padding
   *understates* SPY.
5. **Scope limit.** For four of six universes SPY is not investable within that universe's asset set —
   Axiom allocating over Nifty-50 constituents cannot hold SPY. The SPY row answers "would an investor
   have done better in SPY?", not "did Axiom pick badly from what it was given". `Asset-0 B&H` is kept
   beside it as the in-universe passive arm. Full detail:
   [`docs/consolidation_report.md` §21](docs/consolidation_report.md).

### LSTM Degeneracy — Confirmed Directly

The LSTM's ±0.000 across seeds is degeneracy, not precision. Dumping the day-by-day policy for seeds
42/101/202 on US ETFs OOS
([`scripts/lstm_degeneracy_check.py`](scripts/lstm_degeneracy_check.py) →
[`data/lstm_degeneracy_check.csv`](data/lstm_degeneracy_check.csv)) shows the three trained networks
genuinely differ (parameter-L2 8.448 / 8.707 / 8.682, max |ΔP(up)| up to 0.0071 between pairs) but
their raw sigmoid output never leaves **[0.549, 0.568]** — so it never crosses the 0.5 threshold. All
three emit **risk-on on 481 of 481 days**, and the thresholded weight vectors are *bit-identical*
across all three seeds and constant over the whole window (0.08 per asset, 80% invested). The arm
reduces to a static 80% equal-weight buy-and-hold, which also explains why costs barely move it
(+0.94 → +0.91): it never trades. Its interval is an interval on a constant.

> **Interpretation**: Axiom (0% real data, 10 seeds) reaches OOS mean Sharpe **+0.98**
> [seed-level 95% CI: +0.92, +1.05] on the pinned window under its own cost model. Against the same
> cost model, a real-data-trained LSTM reaches **+0.91** and XGBoost **+0.07**. Axiom is
> **statistically indistinguishable from the LSTM** (Welch p = 0.51, MWU p = 0.18, d = 0.12; still a
> tie after cluster correction, paired p = 0.55) and
> **ahead of XGBoost** (pooled Welch p = 2.2e-11, MWU p = 8.3e-10, d = 1.35 — but **p = 0.0072** once
> clustering is priced in, and the exact rank test at n = 6 gives 0.0625, i.e. not significant). Three
> things
> qualify that: the LSTM arm is a degenerate constant allocation (confirmed above), so "indistinguishable
> from the LSTM" means indistinguishable from a static 80% buy-and-hold; XGBoost's collapse is a
> cost effect — at zero cost it scores +1.05; and the pooled p-values assume 60 independent draws when
> ICC is 0.85/0.95. Against a **real fixed SPY buy-and-hold** (+0.90 OOS) the OOS result is likewise a
> tested tie: difference +0.08, universe-level CI [-0.56, +0.73], p = 0.76. In the holdout Axiom achieves **+0.62** [+0.56, +0.68]
> seed-level but **[-0.31, +1.56]** market-level, is **behind the LSTM in every universe where the
> difference is significant** — and cluster-corrected, behind it in **all six**, universe-level CI
> [-0.196, -0.058], p = 0.0053 — is beaten by monthly Risk Parity (+0.92), and is **0.69 Sharpe behind
> the real fixed SPY** (+0.62 vs +1.31, SPY ahead in 4 of 6 universes; CI [-1.99, +0.62], p = 0.23, so
> the deficit is not statistically resolvable at n = 6, but the point estimate is large and must be
> quoted with it). The honest summary: a
> policy trained on 0% real data is competitive with real-data-trained baselines *and with SPY* under
> matched costs OOS, is not better than simple rules or SPY out of sample in the holdout, and does not
> generalise beyond these six markets.


---

## What Axiom Learned (Equity Universes)

In equity-class universes (US ETFs, US Mega-Cap, Global Indices), the Axiom policy exhibits:
- **Conservative allocation**: Higher cash fraction than Fast, reducing drawdowns
- **Multi-asset diversification**: Spreads weights across available assets rather than concentrating
- **Low turnover**: Rebalances infrequently (1–3 times per evaluation window), keeping cost impact minimal

> [!IMPORTANT]
> **Every Axiom-vs-Fast comparison in this repo is a comparison of two different architectures, not
> two seeds or two checkpoints of one.** Axiom is `AxiomNet` (289,527 params, flatten over the 30-day
> window); Fast is `FastTradingNet` (51,703 params, mean-pool over time). They share no state_dict
> and a cross-load is rejected. This applies to the bullets above, the crypto-concentration forensics
> below, and the `[v6] Fast` column of the offset ablation. Any behavioural difference between them —
> including the crypto concentration failure — may reflect architecture, training run, or both, and
> this repo cannot separate the two: there is no `AxiomNet` checkpoint trained under Fast's recipe and
> no `FastTradingNet` checkpoint from the Kaggle run. Until 2026-08-29 both classes were named
> `DeepEndToEndTradingNet`, which is why earlier text treated them as one model.

**Boundary failure mode (Crypto/Fast)**: These properties do not hold universally. In the crypto universe, v6 Fast concentrates **98.9% of wealth into a single altcoin (BCH-USD)** with only 1.5% cash buffer, resulting in -86.5% return during the holdout bear market. This is documented as a known failure mode — see [consolidation report §5](docs/consolidation_report.md). Note that Fast's mean-pool head cannot represent position-in-window information at all, so this is a plausible architectural contributor and not only a training artefact.

The action-constant ablation (cash-logit offset = -2.5) has now been re-run **outside** equities —
on India Nifty 50, Forex & Commodities and Crypto (PIT), with US ETFs as an in-scope control, across
all 10 Axiom (`AxiomNet`) seeds plus the single Fast (`FastTradingNet`) checkpoint, on both windows
([`scripts/action_constant_ablation_multiuniverse.py`](scripts/action_constant_ablation_multiuniverse.py)).
**"Not fragile" holds:** sweeping the offset over [-3.5, -2.5, -1.5, -0.5, 0.0] moves mean Sharpe by
at most **0.06** in any of the 16 arms (threshold for "fragile" is 0.15), including crypto. The Fast
arm there is one checkpoint of a *different* network, so it carries no seed variance and is not a
seed-level replication of the Axiom arms.

That is a weaker statement than it sounds, and the same sweep shows why: Sharpe is a ratio and is
nearly invariant to scaling exposure, but the offset *is* the exposure dial. Over the same sweep the
mean cash fraction moves from ~6% to ~56%, and in the crypto holdout Axiom's return moves from
**-52% to -28%** with max drawdown from **-59% to -33%**. The constant does not change the
risk-adjusted ranking, but it materially sets the level of risk taken — and the hand-picked -2.5 is
the most aggressive end of the range tested. See [consolidation report §3](docs/consolidation_report.md).

---

## Architecture

Until 2026-08-29 two structurally different networks in this repo shared the class name
`DeepEndToEndTradingNet`, and the diagram published here described the wrong one. They are now
renamed and separated. **The 10 CI-verified Axiom checkpoints were trained with `AxiomNet`; every
`[v6] Fast` / `rai_v6_alpha.pt` result comes from `FastTradingNet`.** The two are not
state_dict-compatible — a cross-load is rejected — and they differ by 5.6× in parameter count.

**Axiom — `AxiomNet`** ([`scripts/kaggle_axiom_10seed.py`](scripts/kaggle_axiom_10seed.py))

```
Input: 30-day window × 22 features
    [Conv1d 22→32, k=3] → [Conv1d 32→64, k=3]  →  [Transformer Encoder] (d=64, h=2, 1 layer)
    →  flatten(64 × 30 = 1920)  →  [Dense 128, LayerNorm]  →  [Actor 11] + [Critic 1]
Parameters: 289,527 | 25 state_dict tensors (conv1.*, conv2.*, fc_features.0/2, actor, critic)
Training: PPO with GAE, 1024-step rollouts | Checkpoints: checkpoints/axiom_multiseed/axiom_seed*.pt
```

**Fast — `FastTradingNet`** ([`scripts/train_v6_fast.py`](scripts/train_v6_fast.py))

```
Input: 30-day window × 22 features
    [Conv1D Block] (22 → 32 → 64, k=3)  →  [Transformer Encoder] (d=64, h=2, 1 layer)
    →  mean-pool over time (64)  →  [Dense 128]  →  [Actor 11] + [Critic 1]
Parameters: 51,703 | 23 state_dict tensors (conv1d.0/2, fc_features.0, actor_head, critic_head)
Checkpoints: data/v0.6_rl_checkpoints/{rai_v6_fast,rai_v6_alpha,axiom_v0_prototype_fasttradingnet}.pt
```

> [!NOTE]
> `axiom_v0_prototype_fasttradingnet.pt` was named `axiom.pt` until 2026-08-29 and was loaded under
> the bare label **"Axiom"** by `scripts/canonical_evaluation.py`. It is a `FastTradingNet`
> checkpoint, byte-identical to `rai_v6_alpha.pt` (sha256 prefix `6dfb41b7f4e2d8a0`, 23 tensors,
> 51,703 params) — not an Axiom model. The bare label "Axiom" now resolves only to
> `checkpoints/axiom_multiseed/axiom_seed*.pt`. See
> [`docs/consolidation_report.md` §19](docs/consolidation_report.md).

The published "~51,700 parameters" figure was Fast's, not Axiom's. `mean-pool` vs `flatten` is the
substantive difference: Fast collapses the 30-day window to a single 64-d vector before the policy
head, so it cannot represent position-in-window information; Axiom keeps all 30 timesteps, which is
where 245,888 of its 289,527 parameters live (`Linear(1920, 128)`).

Other architectures in this repo:
- **v7** (`SpatioTemporalTradingNet`): Multi-scale conv + 2-layer Transformer, 4 heads. No checkpoint available.
- **v8.2** (`MultiScaleRiskAwareNet`): Kaggle walk-forward architecture. 120 checkpoints available in `checkpoints/kaggle_import/`. Evaluated with 10-seed CI — **showed no improvement over baselines** (see [§7 in consolidation report](docs/consolidation_report.md)).
- **`DeepTransformerTradingNet`** ([`scripts/train_v6_deep_transformer.py`](scripts/train_v6_deep_transformer.py)): 60-day window, d=128, 4 heads, 2 layers, 361,431 params. No checkpoint; never evaluated.
- **`LegacyV6TradingNet`** ([`rai/learning/v6_model.py`](rai/learning/v6_model.py)): same tensor shapes as `AxiomNet` (289,527 params) but different attribute names, so it loads neither checkpoint family. Not imported by any evaluation path; retained only as a record.


---

## Quickstart

```bash
git clone https://github.com/axiom-sim2real/RAI-Axiom.git && cd RAI-Axiom
python -m venv venv && .\venv\Scripts\activate
pip install -r requirements.txt

# Fetch the pinned price windows. No price data is redistributed in this repo
# (see data/README.md), so this step is required before anything below runs.
# It downloads the 6 universes plus the SPY reference into data/pinned_universes/
# and then verifies them against data/pinned_universes_manifest.json.
python scripts/fetch_pinned_universes.py

# Run canonical evaluation (all universes, all baselines)
python scripts/canonical_evaluation.py

# Single universe
python scripts/canonical_evaluation.py --universe us_etf

# List universes
python scripts/canonical_evaluation.py --list-universes
```

### Reproducing the price inputs

The committed results in [`data/`](data/README.md) were computed against a window pinned
to explicit dates — `2016-08-20` (10-year universes) and `2021-08-20` (5-year universes),
both ending `2026-08-20`, with `auto_adjust=True`. `scripts/fetch_pinned_universes.py`
re-downloads exactly that window, so the CSVs can be rebuilt from scratch:

```bash
python scripts/fetch_pinned_universes.py            # download + auto-verify
python scripts/fetch_pinned_universes.py --verify   # verify existing caches only
python scripts/fetch_pinned_universes.py --force    # re-download, overwriting
```

`data/pinned_universes_manifest.json` fingerprints each file the results were computed
from — row/column counts, first and last date, **column order**, byte checksum, and a
platform-independent float fingerprint at 6 dp. `--verify` reports `OK`, `DRIFTED`
(Yahoo revised that window after 2026-08-30 — the evaluation still runs, but the last
decimals will differ), or a `column_order CHANGED` failure, which is the one that
invalidates reproduction outright.

Two things are load-bearing and neither is cosmetic: the **dates must stay pinned**
(a `period=`-relative download drifts every day), and the **column order must stay as
`yfinance` returns it** — alphabetical by ticker, not the order the ticker lists are
written in. The policy's per-asset logits are position-dependent, so re-sorting the
columns changes every number in `data/`.

### Reproducing the tables

Reproducing the tables above, once `data/pinned_universes/` has been fetched (everything
below reads the caches; nothing re-downloads):

```bash
python scripts/aggregate_ci.py                      # Axiom seed- and market-level CIs
python scripts/baseline_multiseed.py                # LSTM / XGBoost, 10 seeds, both cost bases
python scripts/deterministic_baselines_pinned.py    # rule-based arms + Fast + fixed-SPY reference, pinned window
python scripts/cross_model_significance.py          # Welch t + MWU + cluster-corrected (n=6) + CR1 OLS, vs LSTM / XGBoost / real SPY
python scripts/lstm_degeneracy_check.py             # per-day policy dump, 3 LSTM seeds
```

---

## Limitations

1. **The LSTM baseline is a confirmed degenerate policy**: LSTM and XGBoost have been re-run across the same 10 seeds as Axiom after the seed-propagation fix (`scripts/baseline_multiseed.py`, 2026-08-29). The LSTM returns a *single distinct Sharpe* across all 10 seeds in 4 of 6 universes, and a direct per-day policy dump for 3 seeds (`scripts/lstm_degeneracy_check.py`) confirms **bit-identical weight vectors, constant over the entire window** — P(up) stays inside [0.549, 0.568] and never crosses the 0.5 threshold, so the arm is a static 80% equal-weight buy-and-hold. "Axiom is indistinguishable from the LSTM" therefore means indistinguishable from that constant, not from a functioning learned baseline.
2. **Generator GARCH gap**: Level 6 synthetic data under-reproduces volatility clustering (LB stat 27× lower than real).
3. **Crypto concentration failure**: v6 Fast concentrates 98.9% into BCH-USD with 1.5% cash buffer — see [consolidation report §5](docs/consolidation_report.md). Raising the cash-logit offset to 0.0 lifts the cash buffer to ~23% but still leaves 74% in one asset: the concentration lives in the asset logits, not the cash constant. Fast is `FastTradingNet`, a different architecture from Axiom's `AxiomNet` (see limitation 11), so this failure cannot be attributed to training alone.
4. **v8.2 null result**: The newer architecture (MultiScaleRiskAwareNet), evaluated with 10-seed CI on Kaggle (4 universes), showed no improvement over baselines (mean Sharpe diff < ±0.05). See [consolidation report §7](docs/consolidation_report.md).
5. **Significance tests are unpaired at the seed level by necessity, and not uniformly favourable**: Axiom's seeds are training-initialisation seeds and the baselines' are model-fitting seeds, so no seed-paired test is valid; Welch's t and Mann-Whitney U on the independent 10-vs-10 samples give, cost-matched, **p = 0.51 / 0.184 vs LSTM (tie)** and **p = 2.2e-11 / 8.3e-10 vs XGBoost (Axiom ahead)** OOS. Per-universe, Axiom **loses** Forex to the LSTM OOS (-0.50, p = 0.00021) and loses US ETFs, Mega-Cap and Global Indices to the LSTM in the holdout (p = 0.0044 / 0.028 / 0.0018). Those pooled tests also treat 60 observations as independent when **ICC is 0.847 / 0.953**; the cluster-corrected versions (universe-paired, n = 6, plus CR1 cluster-robust OLS) are now reported beside them and change two conclusions: the OOS XGBoost result drops from 2.2e-11 to **p = 0.0072** and fails the exact rank test at n = 6 (0.0625), while the holdout LSTM comparison goes from "tie" to **significant against Axiom** (p = 0.0053, Wilcoxon 0.03125, LSTM ahead in 6 of 6). With only six clusters no distribution-free test can return a two-sided p below 0.03125, so the cluster-corrected evidence is genuinely low-powered — that is a limitation of six markets, not a null result.
6. **Result does not generalise beyond the six markets tested**: with universe as a random effect, the market-level 95% CI is [+0.45, +1.52] OOS and **[-0.31, +1.56]** in the holdout — the holdout interval crosses zero. ICC 0.85 / 0.95 means market choice dominates seed choice.
7. **India OOS**: OOS CI crosses zero [-0.17, +0.26] — Axiom's advantage is not significant in this universe.
8. **India holdout failure confirmed**: Multi-seed holdout Sharpe -0.46 ± 0.41, 95% CI [-0.70, -0.22] — the CI lies entirely below zero (9 of 10 seeds negative; the exception is seed 42 at +0.21). India is the only universe negative in *both* windows. Both real-data baselines also fail India on the same window (LSTM -0.24 OOS / -0.42 holdout; XGBoost +0.31 / -0.34), so this is at least partly a hard-market effect rather than a purely zero-shot one.
9. **Crypto holdout failure confirmed**: Multi-seed holdout Sharpe -1.18 ± 0.18, 95% CI [-1.28, -1.08] — negative for all 10 seeds without exception.
10. **The cash-logit offset sets risk level even though Sharpe is insensitive to it**: sweeping it changes mean cash from ~6% to ~56% and crypto-holdout drawdown from -59% to -33% while mean Sharpe moves ≤0.06. "Not fragile" therefore means "does not change the ranking", not "does not matter".
11. **Axiom and Fast are different architectures, and the two effects cannot be separated**: `AxiomNet` (289,527 params, flatten) and `FastTradingNet` (51,703 params, mean-pool) were both named `DeepEndToEndTradingNet` until 2026-08-29. Every Axiom-vs-Fast difference in this repo therefore confounds architecture with training run; no checkpoint exists that would let the two be disentangled. The originally published single-seed "+1.17" Axiom figure was a `FastTradingNet` checkpoint, so the revision to +0.98 is not purely a seed effect.
12. **Axiom does not beat a rule-based baseline in the holdout**: on the single pinned/cost-matched basis, monthly inverse-volatility Risk Parity scores **+0.92** holdout mean vs Axiom's **+0.62**, and is level with it OOS (+0.94 vs +0.98). Risk Parity uses no training data of any kind. A real fixed-SPY buy-and-hold is higher still at **+1.31** (limitation 15).
13. **Deterministic arms are point estimates, not 10-seed means**: Risk Parity, EW, SPY B&H (fixed reference), Asset-0 B&H, SMA and 60/40 have no seed to vary, and `[v6] Fast` is a single checkpoint, so no CI is available for them. The buy-once arms (SPY B&H, Asset-0 B&H, EW, 60/40) are additionally cost-invariant *by construction* — Axiom's evaluator does not charge the entry allocation. The SMA 50/200 arm's 200-day warm-up bug was fixed on 2026-08-29 (pre-window history now feeds the indicator, 0 default days in all 12 universe-windows); the correction moved it from +0.13/+0.77 to **-0.27/+0.61** and reduced its bit-identical-to-buy-and-hold cases from 5 of 12 to 1 of 12.
14. **"Asset-0 B&H" is not SPY buy-and-hold — and until 2026-08-30 no arm was**: that arm, and both SMA arms, act on column 0 of each universe — EEM (US ETFs and Global Indices, hence their identical rows), AAPL, AXISBANK.NS, AUDUSD=X, BCH-USD. It was labelled "SPY B&H" until 2026-08-29, which was wrong in all six universes; the numbers were unchanged, only the label. **Resolved 2026-08-30**: a real fixed-SPY arm was added beside it (`SPY B&H (fixed reference)`, [consolidation report §21](docs/consolidation_report.md)), so every "vs SPY" claim now points at the real series. The two rows are kept separate on purpose — SPY is out-of-universe for 4 of 6 universes and is not a swap-in for the in-universe passive arm.
15. **Axiom is behind a real fixed SPY buy-and-hold in the holdout**: +0.62 vs **+1.31**, a -0.69 gap with SPY ahead in 4 of 6 universes. The universe-level CI [-1.99, +0.62] spans zero, so at n = 6 the deficit is not statistically resolved — but it is a large point estimate and must be reported with it, not as a draw. OOS the two are a tested tie (+0.98 vs +0.90, p = 0.76, CI [-0.56, +0.73]). Two scope qualifications: SPY is not investable inside four of the six universes, so this answers "would an investor have done better in SPY?" rather than "did Axiom choose badly from its own asset set"; and the three 10-year universes share one SPY number because their window date spans are identical, so the six clusters are less independent than the count suggests. Recomputing on SPY's native NYSE calendar instead of forward-filling moves the holdout gap further against Axiom (-0.74).

---

## Citation

```bibtex
% Jason Pandian and Balamurugan P G contributed equally to this work.
% The author order is alphabetical by surname and does not encode seniority.
@article{pandian_balamurugan_rai_2026,
  title   = {RAI: Relational Artificial Intelligence from Artificial Worlds},
  author  = {Jason Pandian and Balamurugan P G},
  note    = {Equal contribution},
  year    = {2026},
  url     = {https://github.com/axiom-sim2real/RAI-Axiom}
}
```

> **Authorship**: **Jason Pandian** and **Balamurugan P G** — **equal contribution**. Listed
> alphabetically by surname; the ordering carries no seniority. For the same reason the repository is
> owned by the shared `axiom-sim2real` organisation rather than by either author's personal account,
> with **both authors holding Owner rights** — neither depends on the other for access.

Licensed under the MIT License — full text in [`LICENSE`](LICENSE), `Copyright (c) 2026 Jason
Pandian and Balamurugan P G`. The badge at the top of this file links to that same file.

The licence covers the **code and documentation in this repository**. It does not extend to the
market price data the evaluation runs against: those series are fetched from Yahoo Finance at
runtime and are not redistributed here (see [`data/README.md`](data/README.md)).

