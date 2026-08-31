# RAI Repository Consolidation Report v5

> **Date**: 2026-08-29 (statistics + baseline re-run update; supersedes the 2026-08-20 v4 revision)  
> **Harness**: canonical_evaluation.py v2; baseline re-run via `scripts/baseline_multiseed.py`  
> **Axiom**: CI-verified (10 seeds × 6 universes). **Baselines: also 10-seed as of 2026-08-29** — the LSTM/XGBoost seed-propagation bug (§12) is fixed *and re-run*. §14 carries the re-run numbers; the single-seed tables in §2 are retained as the historical record and are explicitly marked.

---

## 1. Baselines Confirmed in Canonical Harness (Step 2)

All 5 requested baselines are now first-class comparison arms:

| Arm | Type | Source | Grid-Search Tuning |
|---|---|---|---|
| SPY B&H (fixed reference) | Passive | The **real SPY**, held in every universe regardless of that universe's tickers; out-of-universe for 4 of 6 (added 2026-08-30 — §21) | N/A |
| Asset-0 B&H (in-universe) | Passive | Price of the first ticker in column order — **not SPY**; previously mislabelled "SPY Buy & Hold" (§20) | N/A |
| Equal Weight (1/N) | Passive | Uniform allocation | N/A |
| 60/40 Portfolio | Passive | 60% column 0 / 40% column 1 — only in US ETFs is that literally SPY/TLT (§21) | N/A |
| Risk Parity | Rule-based | Inverse-vol, monthly rebal | lb=60 |
| SMA 50/200 | Rule-based | Golden/death cross on **column 0** of the universe (not SPY — see §20) | sw=50, lw=200 |
| LSTM Return Predictor | ML (real-trained) | 2-layer LSTM, trained on train split | hidden=32, lr=1e-3 (3x3 grid, val Sharpe=0.7150) |
| XGBoost Classifier | ML (real-trained) | GBT direction predictor | depth=5, est=100 (3x3 grid, val Acc=0.5291) |
| Axiom v0.9 | Zero-shot | `AxiomNet`, 0% real data, 289,527 params | N/A (frozen checkpoint) |
| RAI v6 Fast | Zero-shot | `FastTradingNet`, 0% real data, 51,703 params | N/A (frozen checkpoint) |

Both were listed as `DeepEndToEndTradingNet` until 2026-08-29. They are two different networks that
happened to share a class name; see §15.

---

## 2. Full Pairwise Win/Loss Summary (Step 3)

### OOS Window — Sharpe Win Counts (out of 8 opponents)

> [!NOTE]
> The column headed **`Asset-0`** was published as `SPY` until 2026-08-29. It is the buy-and-hold of
> each universe's **column 0**, which is never SPY (§20). A real fixed-SPY arm was only added on
> 2026-08-30 and does **not** appear in these single-seed tables — see §21 for it.

| Universe | Alpha | Fast | LSTM | XGBoost | RiskP | EW | Asset-0 | SMA | 60/40 |
|---|---|---|---|---|---|---|---|---|---|
| US ETFs | **8** | 3 | 6 | 7 | 5 | 4 | 2 | 1 | 0 |
| US Mega-Cap (PIT) | **8** | 3 | 7 | 6 | 4 | 5 | 2 | 0 | 1 |
| Global Indices | 7 | **8** | 6 | 3 | 5 | 4 | 1 | 0 | 2 |
| India Nifty 50 | 4 | **8** | 5 | 7 | 6 | 1 | 2 | 0 | 3 |
| Forex & Commodities | 6 | 1 | 3 | 4 | **8** | 7 | 2 | 0 | 5 |
| Crypto (PIT) | 5 | 0 | 6 | **8** | 7 | 4 | 2 | 1 | 3 |

### Future Holdout Window — Sharpe Win Counts

| Universe | Alpha | Fast | LSTM | XGBoost | RiskP | EW | Asset-0 | SMA | 60/40 |
|---|---|---|---|---|---|---|---|---|---|
| US ETFs | 7 | 0 | **8** | 4 | 5 | 6 | 2 | 3 | 1 |
| US Mega-Cap (PIT) | 5 | 3 | 7 | **8** | 6 | 4 | 1 | 0 | 2 |
| Global Indices | 1 | 2 | 5 | 0 | 4 | 3 | 6 | 7 | **8** |
| India Nifty 50 | 1 | 4 | 2 | **8** | 6 | 0 | 7 | 5 | 3 |
| Forex & Commodities | **8** | 0 | 5 | 2 | 4 | 6 | 3 | 1 | 7 |
| Crypto (PIT) | 1 | 0 | 2 | 6 | 4 | 3 | 5 | **8** | 7 |

### Cross-Universe Win-Count Summary

| Rank | Model | OOS Wins (/48) | Holdout Wins (/48) | Overall (/96) |
|---|---|---|---|---|
| 1 | Risk Parity | 35 | 29 | **64** |
| 2 | XGBoost (real) | 35 | 28 | 63 |
| 3 | LSTM (real) | 33 | 29 | 62 |
| 4 | Axiom v0.9 | **38** | 23 | 61 |
| 5 | Equal Weight | 25 | 22 | 47 |
| 6 | 60/40 | 14 | 28 | 42 |
| 7 | Asset-0 B&H (published as "SPY B&H") | 11 | 24 | 35 |
| 8 | RAI v6 Fast | 23 | 9 | 32 |
| 9 | SMA 50/200 | 2 | 24 | 26 |

### Mean Sharpe per Model (averaged across 6 universes)

Win counts treat a 0.01-Sharpe margin the same as a 1.0-Sharpe blowout. The table below shows actual mean Sharpe for each arm to give magnitude context:

| Model | OOS Mean Sharpe | Holdout Mean Sharpe | Overall Mean Sharpe |
|---|---|---|---|
| Axiom v0.9 | **+1.17** | +0.67 | +0.92 |
| XGBoost (real) | +1.08 | +0.68 | +0.88 |
| Risk Parity | +1.02 | **+0.78** | **+0.90** |
| LSTM (real) | +1.00 | +0.70 | +0.85 |
| Equal Weight | +0.83 | +0.61 | +0.72 |
| RAI v6 Fast | +0.76 | +0.29 | +0.52 |
| 60/40 | +0.50 | +0.62 | +0.56 |
| Asset-0 B&H (published as "SPY B&H") | +0.29 | +0.60 | +0.45 |
| SMA 50/200 | +0.14 | +0.60 | +0.37 |

> [!IMPORTANT]
> **This table and the win-count tables above are the original single-seed `canonical_evaluation.py` run and are superseded for every arm.** Axiom is now 10-seed (§11: OOS **+0.98**, not +1.17 — and note per §15 that the +1.17 checkpoint was a *different architecture*, `FastTradingNet`, not `AxiomNet`). LSTM/XGBoost are 10-seed on the Axiom-matched pinned window (§14: LSTM **+0.94**, XGBoost **+1.05** OOS; **+0.91 / +0.07** once charged Axiom's transaction costs). The six deterministic/Fast arms have now also been re-run on the pinned window under Axiom's cost model (§16), so **all nine arms are on one basis in §16** — that is the table to read, not this one. The pairwise win counts themselves have *not* been recomputed on the re-run; doing so requires re-running all nine arms through the pairwise harness, which has not been done. Read the win counts as descriptive of the single-seed, old-window run only. **No row in these tables is a real SPY benchmark**: the `Asset-0` / `Asset-0 B&H` entries (published as "SPY") hold column 0 of each universe (§20), and the actual fixed-SPY arm — a much harder benchmark, +0.90 OOS / +1.31 holdout mean Sharpe — was only added on 2026-08-30 (§21).

### Key Findings

> [!IMPORTANT]
> **Risk Parity** (inverse-volatility weighting — no ML, no training data of any kind) has the highest overall pairwise win count of all 9 arms (64/96), narrowly ahead of XGBoost (63) and LSTM (62), with Axiom v0.9 at 61. A rule-based strategy with zero learning outperforms or matches every data-trained model in this single-seed evaluation.

> [!WARNING]
> Axiom v0.9 is the **top-ranked OOS performer** (38/48 wins, mean Sharpe +1.17) but drops to **7th of 9 arms** in the Future Holdout (23/48 wins, mean Sharpe +0.67) — ahead of only Equal Weight (22/48) and RAI v6 Fast (9/48). This suggests the OOS advantage is window-specific and does not persist into genuinely untouched data for 4 of 6 universes.

### Holdout Ranking (explicit)

| Holdout Rank | Model | Holdout Wins (/48) | Holdout Mean Sharpe |
|---|---|---|---|
| #1 | LSTM (real) | 29 | +0.70 |
| #1 | Risk Parity | 29 | **+0.78** |
| #3 | XGBoost (real) | 28 | +0.68 |
| #3 | 60/40 | 28 | +0.62 |
| #5 | Asset-0 B&H (published as "SPY B&H") | 24 | +0.60 |
| #5 | SMA 50/200 | 24 | +0.60 |
| **#7** | **Axiom v0.9** | **23** | **+0.67** |
| #8 | Equal Weight | 22 | +0.61 |
| #9 | RAI v6 Fast | 9 | +0.29 |

---

## 3. CI-Only Labeling (Step 4)

Every result in `data/canonical_results.json` includes:
```json
"seed_status": "single-seed, not CI-verified"
```

Multi-seed evaluation requires:
1. Training multiple seeds (need GPU for RAI models, or loading Kaggle checkpoints)
2. For ML baselines: re-running with different random seeds for LSTM/XGBoost initialization

At the time of writing this was **not possible** in the CPU-only environment and was blocked on Kaggle
checkpoint import. **Both blockers are now cleared**: (1) the 10 Axiom checkpoints were imported and
evaluated (§11), and (2) the ML baselines were re-run across the same 10 seeds on CPU (§14). The
`seed_status` field in `data/canonical_results.json` still refers to that legacy single-seed run and
has not been rewritten; the CI-verified numbers live in `data/axiom_per_seed_results.csv` and
`data/baseline_per_seed_results.csv`.

### Re-scoped "Not Fragile" Claim (Step 4 of framing fixes) — RE-TESTED 2026-08-29

The action-constant ablation (cash-logit offset of -2.5) was originally validated as "not fragile"
in `scripts/action_constant_ablation.py` on **US equities / sector ETFs, 2020–2024 only**. It has now
been re-run outside equities by
[`scripts/action_constant_ablation_multiuniverse.py`](../scripts/action_constant_ablation_multiuniverse.py):
offsets `[-3.5, -2.5, -1.5, -0.5, 0.0]` × {India Nifty 50, Forex & Commodities, Crypto (PIT), plus
US ETFs as an in-scope control} × {OOS, holdout} × {10 Axiom seeds, 1 Fast checkpoint} = **440
evaluations**. The harness is an exact mirror of `kaggle_axiom_10seed.evaluate_on_real_data` with the
constant lifted into an argument, so the -2.5 column reproduces the published per-seed numbers to
within the repro-check tolerance (mean |err| 0.042 OOS / 0.021 holdout, `data/axiom_repro_check.csv`).
Raw output: `data/action_constant_ablation_multiuniverse.csv` / `.json`.

**Verdict: "not fragile" holds outside equities.** FRAGILE = mean Sharpe moves more than 0.15 from
the -2.5 default. **0 of 16 arms are fragile**; the largest movement anywhere is 0.06.

> [!NOTE]
> **The Axiom and Fast rows below are two different architectures, not two checkpoints of one** —
> `AxiomNet` (289,527 params, flatten) vs `FastTradingNet` (51,703 params, mean-pool); see §15. The
> Fast arm is additionally a single checkpoint (`n = 1`), so it carries no seed variance and its
> column is not a seed-level replication of the Axiom column above it. Any Axiom-vs-Fast difference in
> this table confounds architecture with training run.

| Universe | Window | Model | n | -3.5 | **-2.5** | -1.5 | -0.5 | 0.0 | max abs delta | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| US ETFs (control) | OOS | Axiom | 10 | 1.35 | **1.34** | 1.34 | 1.33 | 1.31 | 0.03 | not fragile |
| US ETFs (control) | OOS | Fast | 1 | 0.81 | **0.81** | 0.82 | 0.84 | 0.86 | 0.04 | not fragile |
| US ETFs (control) | Holdout | Axiom | 10 | 1.43 | **1.43** | 1.43 | 1.41 | 1.42 | 0.02 | not fragile |
| US ETFs (control) | Holdout | Fast | 1 | 1.13 | **1.13** | 1.14 | 1.14 | 1.15 | 0.01 | not fragile |
| India Nifty 50 | OOS | Axiom | 10 | 0.09 | **0.09** | 0.09 | 0.07 | 0.07 | 0.03 | not fragile |
| India Nifty 50 | OOS | Fast | 1 | 1.00 | **0.99** | 0.98 | 0.96 | 0.94 | 0.06 | not fragile |
| India Nifty 50 | Holdout | Axiom | 10 | -0.51 | **-0.52** | -0.52 | -0.53 | -0.54 | 0.02 | not fragile |
| India Nifty 50 | Holdout | Fast | 1 | 0.30 | **0.29** | 0.30 | 0.31 | 0.33 | 0.03 | not fragile |
| Forex & Commodities | OOS | Axiom | 10 | 0.41 | **0.40** | 0.39 | 0.37 | 0.35 | 0.05 | not fragile |
| Forex & Commodities | OOS | Fast | 1 | 0.35 | **0.35** | 0.36 | 0.38 | 0.39 | 0.05 | not fragile |
| Forex & Commodities | Holdout | Axiom | 10 | 1.35 | **1.33** | 1.33 | 1.31 | 1.27 | 0.06 | not fragile |
| Forex & Commodities | Holdout | Fast | 1 | 0.38 | **0.36** | 0.38 | 0.36 | 0.33 | 0.03 | not fragile |
| Crypto (PIT) | OOS | Axiom | 10 | 1.24 | **1.24** | 1.24 | 1.24 | 1.24 | 0.00 | not fragile |
| Crypto (PIT) | OOS | Fast | 1 | 0.45 | **0.45** | 0.44 | 0.41 | 0.39 | 0.06 | not fragile |
| Crypto (PIT) | Holdout | Axiom | 10 | -1.22 | **-1.22** | -1.22 | -1.22 | -1.22 | 0.00 | not fragile |
| Crypto (PIT) | Holdout | Fast | 1 | -1.87 | **-1.88** | -1.88 | -1.87 | -1.84 | 0.04 | not fragile |

#### But Sharpe insensitivity is not the same as the constant not mattering

Sharpe is a ratio and is close to invariant under scaling total exposure up or down — and the
cash-logit offset *is* the exposure dial. The same sweep, expressed in level metrics (swing from the
best to the worst offset within each arm), shows the constant sets how much risk is taken:

| Universe | Window | Model | Return swing | Max-DD swing | Cash-fraction swing |
|---|---|---|---|---|---|
| US ETFs | OOS | Axiom | 17.9 pp | 4.0 pp | 50.0 pp |
| US ETFs | Holdout | Axiom | 20.7 pp | 6.6 pp | 49.9 pp |
| India Nifty 50 | OOS | Axiom | 0.6 pp | 6.1 pp | 50.3 pp |
| India Nifty 50 | Holdout | Axiom | 3.4 pp | 8.5 pp | 50.4 pp |
| Forex & Commodities | OOS | Axiom | 1.9 pp | 4.1 pp | 50.0 pp |
| Forex & Commodities | Holdout | Axiom | 10.1 pp | 5.8 pp | 49.8 pp |
| **Crypto (PIT)** | **OOS** | **Axiom** | **56.0 pp** | **23.4 pp** | **50.6 pp** |
| **Crypto (PIT)** | **Holdout** | **Axiom** | **28.1 pp** | **30.1 pp** | **51.2 pp** |
| Crypto (PIT) | Holdout | Fast | 8.3 pp | 7.9 pp | 21.4 pp |

Concretely, in the crypto holdout the Axiom arm's mean return moves **-56.2% → -28.0%** and max
drawdown **-62.9% → -32.8%** as the offset goes from -3.5 to 0.0, while mean Sharpe does not move at
all (-1.22 throughout). Mean cash fraction over the same sweep goes from ~6% to ~56%. **The
hand-picked -2.5 is at the most aggressive end of the range tested.**

> [!WARNING]
> Revised scope of the claim: the -2.5 cash-logit offset **does not change the risk-adjusted ranking**
> in any universe tested, including crypto, forex and India (max |ΔSharpe| = 0.06 across 16 arms).
> It **does** materially set the level of risk taken — cash fraction, return and drawdown all move
> by tens of percentage points across the same sweep. "Not fragile" should therefore be cited as
> "does not change the ranking", never as "does not matter".

> [!NOTE]
> Two limits on this re-test. (1) The Fast arms are a **single checkpoint**, so no seed variance is
> available for that model and its verdicts rest on one draw. (2) Only the cash-logit offset was
> swept outside equities; the clip range `(-8, 3)` and the 3% rebalance threshold were swept only in
> the original equities-only ablation and remain untested elsewhere.

#### Superseded by this re-test

The earlier claim that the offset "does not generalise across universes without further tuning",
based on the Fast variant's 1.1–2.0% crypto cash fraction (§5), was an inference from a single
checkpoint's allocation forensics rather than a sweep. The sweep does not support it as stated: the
Fast crypto Sharpe moves 0.04 across the whole offset range. What §5 actually documents is a
*concentration* failure that the cash logit cannot fix — raising the offset to 0.0 lifts Fast's crypto
holdout cash buffer from 2.8% to 22.6% and its return from -85.5% to -77.2%, but top-asset weight
only falls from 89.5% to 74.4%. The concentration lives in the asset logits, not in the cash constant.


---

## 4. Verified Point-in-Time Universes (Step 5)

### US Mega-Cap (Jan 2015 S&P 500 Top-10)

**Primary source**: voronoiapp.com S&P 500 historical market cap rankings.  
**Cross-verified against**: Multiple independent web search results (macromicro.me, CRSP methodology documentation) confirming the same 10 companies in the top-10 for Jan 2015. All sources agree on the composition; minor ordering differences exist for ranks 5–10.  
**Reproducible verification path**: CRSP `dsp500list` table via WRDS, filtered to 2015-01-02, joined with CRSP Monthly Stock file for market cap. This is the paper-grade method; the web sources above serve as a cross-check.

| Rank | Verified Top-10 (Jan 2015) | In our list? | Note |
|---|---|---|---|
| 1 | AAPL | ✅ | |
| 2 | XOM | ✅ | |
| 3 | MSFT | ✅ | |
| 4 | GOOGL | ✅ | Added in v2 (was missing in v1) |
| 5 | JNJ | ✅ | |
| 6 | WFC | ✅ | |
| 7 | BRK-B | ❌ → CVX | BRK-B excluded (conglomerate); replaced with CVX (~#11–#13) |
| 8 | GE | ✅ | |
| 9 | PG | ✅ | Added in v2 (was missing in v1) |
| 10 | JPM | ✅ | |

**Result**: 9 of the verified top-10 are in our list. BRK-B is excluded because Berkshire Hathaway is a conglomerate (not a single-sector equity), and replaced with CVX as a reasonable fill. PFE (in the v1 list) was NOT in any top-10 source and has been removed.

> [!NOTE]
> For paper submission, the BRK-B → CVX substitution should be further justified or BRK-B included. The CRSP `dsp500list` path via WRDS is available for a definitive, citable verification; the current web-search cross-check is sufficient for development but not for a methods section.

### Crypto (Jan 2020 Top-10 by Market Cap)

**Source**: CoinMarketCap Jan 1 2020 historical snapshot, verified 2026-08-19.

**Selection rule**: Top-10 cryptocurrencies by market capitalisation as of January 1, 2020, excluding (a) stablecoins (assets pegged to fiat, which cannot be meaningfully allocated in a portfolio optimisation context) and (b) assets not continuously tradable on major exchanges through the end of the holdout window (August 2026). Excluded assets are replaced by the next-ranked non-excluded asset.

| Rank (Jan 2020) | Asset | Status | Exclusion Reason |
|---|---|---|---|
| 1 | BTC | ✅ Included | |
| 2 | ETH | ✅ Included | |
| 3 | XRP | ✅ Included | |
| 4 | USDT | ❌ Excluded | Stablecoin (USD-pegged) |
| 5 | BCH | ✅ Included | |
| 6 | LTC | ✅ Included | |
| 7 | EOS | ✅ Included | |
| 8 | BNB | ✅ Included | |
| 9 | BSV | ❌ Excluded | Delisted from Binance (Apr 2019), Kraken (2019), ShapeShift (2019) prior to universe construction date. Not continuously tradable through holdout. |
| 10 | XTZ | ✅ Included | |
| 13 | LINK | ✅ Fill | Replaces USDT (stablecoin) |
| 14 | TRX | ✅ Fill | Replaces BSV (not continuously tradable) |

> [!NOTE]
> BSV was delisted from multiple major exchanges in 2019, prior to the Jan 2020 universe construction date — this is not hindsight. The exclusion is based on tradability at the time of construction, not on subsequent price performance.

---

## 5. Fast Crypto Holdout Collapse Investigation (Step 6)

### Findings

RAI v6 Fast returned **-86.5% with -87.1% max drawdown** in the crypto future holdout.

**Allocation forensics** (from `data/crypto_fast_holdout_allocations.json`):

| Metric | Value |
|---|---|
| Total trading days | 335 |
| Rebalance events | **2** (out of 335 days) |
| Top allocation | **BCH-USD at 98.1-98.9%** |
| Cash buffer | **1.1-2.0% (mean 1.5%)** |
| Other asset allocations | All < 0.05% |

**Root cause**: The Fast checkpoint's learned policy concentrated essentially all wealth into a single altcoin (Bitcoin Cash) with virtually no diversification and negligible cash buffer. When BCH-USD dropped ~70% during the holdout window, the portfolio collapsed proportionally.

**Why the cash buffer failed**: The cash logit offset (-2.5) combined with the Fast variant's learned action outputs produces a sigmoid cash fraction of ~1.1-2.0%. This is far below the 3% drift threshold, so the portfolio never rebalanced away from the collapsing asset. See §3 above for the re-scoped "not fragile" claim.

**Contrast with Alpha**: Axiom v0.9 in the same window returned -45.5% (bad, but much less severe). Alpha rebalanced **9 times** vs Fast's **3**, suggesting Alpha's policy is more responsive to drawdown signals.

> [!IMPORTANT]
> **This contrast is confounded by architecture (added 2026-08-29, see §15).** Axiom is `AxiomNet`
> (289,527 params; keeps all 30 timesteps via `flatten`), Fast is `FastTradingNet` (51,703 params;
> `mean-pool` collapses the 30-day window to one 64-d vector before the policy head). They are not two
> seeds or two checkpoints of one network — they share no state_dict and a cross-load is rejected. So
> "Fast's policy concentrated" and "Alpha's policy is more responsive" may reflect the architecture as
> much as the training run: a mean-pooled representation cannot encode position-in-window information
> at all, which is a plausible mechanism for a policy that stops responding to a developing drawdown.
> This repo cannot separate the two effects — there is no `AxiomNet` checkpoint trained under Fast's
> recipe and no `FastTradingNet` checkpoint from the Kaggle run. Both classes were named
> `DeepEndToEndTradingNet` when this investigation was written, which is why the original text treated
> the difference as purely a policy/training difference.

### Re-scoped Diversification Claim

The README previously characterised the RAI policy as "diversified and low-turnover" without qualification. This holds in some universes:

- **US ETFs / Mega-Cap / Global Indices**: Alpha maintains multi-asset allocations with reasonable diversification across evaluation windows.
- **Crypto (Fast)**: **98.9% concentration in a single altcoin** with 1.5% cash buffer — the opposite of diversified.

The diversification characterisation has been scoped in the README to state that it applies to equity-class universes. The crypto/Fast concentration is documented as a boundary failure mode, not a general property of the policy — and, per the note above, a failure mode of a *different network* from the one carrying the CI-verified Axiom results.

---

## 5.1 India Holdout Failure (added 2026-08-29)

§5 documents the crypto collapse in detail; there was no equivalent entry for India, even though
India is the **only universe in which Axiom is negative in both evaluation windows**. Source:
`data/axiom_per_seed_results.csv`, aggregates in `data/axiom_aggregate_stats.json`.

| Window | Mean ± SD (10 seeds) | Bootstrap 95% CI | Seeds below zero |
|---|---|---|---|
| OOS | **+0.04 ± 0.36** | [-0.17, +0.26] | 5 of 10 |
| Holdout | **-0.46 ± 0.41** | **[-0.70, -0.22]** | **9 of 10** |

Per-seed holdout Sharpe: 42 → **+0.21**, 101 → -0.09, 202 → -0.63, 303 → -0.89, 404 → -0.25,
505 → -0.96, 606 → -0.06, 707 → -0.78, 808 → -0.34, 909 → -0.84.

Three ways this differs from the crypto failure:

1. **It is not regime-specific.** Crypto is strongly positive OOS (+1.15) and strongly negative in
   the holdout (-1.18) — a regime flip. India is indistinguishable from zero OOS (CI crosses zero)
   and negative in the holdout, i.e. a persistent transfer failure rather than a collapse.
2. **It is not unanimous across seeds.** The crypto holdout is negative for all 10 seeds without
   exception. India's holdout is negative for **9 of 10** — seed 42 reaches +0.21. It is the
   *confidence interval on the mean* that lies entirely below zero, not every individual seed. Any
   text claiming "negative across all 10 seeds" for India is wrong and has been corrected.
3. **It is at least partly a hard-market effect, not purely a zero-shot one.** Both real-data-trained
   baselines also fail India on the same pinned window (§14):

| Model | India OOS | India Holdout |
|---|---|---|
| Axiom v0.9 (0% real data, 10 seeds) | +0.04 | **-0.46** |
| LSTM (real-trained, 10 seeds) | -0.24 | **-0.42** |
| XGBoost (real-trained, 10 seeds) | +0.31 | **-0.34** |
| XGBoost under Axiom's cost model | -0.72 | **-1.43** |

   All three arms are negative in the India holdout regardless of how much real data they saw, and
   the real-data LSTM is *worse* than Axiom in OOS. This is a distinction the crypto section cannot
   draw: there, XGBoost's holdout (-0.71) is materially better than Axiom's (-1.18).

**Not established**: why India specifically fails. Candidate explanations not tested here include
INR-denominated return dynamics absent from the synthetic generator, the shorter India history
(1239 days total vs 2512 for the US universes, so 248-day OOS/holdout windows), and Nifty
constituent turnover. No experiment in this repo isolates any of these.

---

## 6. Kaggle Checkpoint Saving Fix (Step 7)

**Fixed in**: `scripts/kaggle_chronological_walk_forward_master_10seed.py`

Changes made:
1. **Checkpoint format**: Each `.pt` file now saves `{model_state_dict, architecture, universe, seed, dates, tickers, train/oos/future_days}` — not just bare `state_dict()`
2. **Per-seed results**: OOS Sharpe and OOS Max DD are now saved alongside return (previously only OOS Return was recorded; Sharpe and DD were discarded)
3. **Auditable output**: Script now produces `per_seed_results.csv` (every single data point) and `per_universe_summary.json` (mean/std/min/max per model per universe)

> [!IMPORTANT]
> The actual Kaggle re-run to produce these checkpoints is a **manual step for the user**:
> 1. Upload the updated script to Kaggle
> 2. Run with GPU accelerator (requires dual T4)
> 3. Download `rai_master_trained_models.zip` and `per_seed_results.csv` from Kaggle Output
> 4. Extract `.pt` files into `RAI/checkpoints/kaggle_import/`
> 5. Run: `python scripts/canonical_evaluation.py --checkpoint-dir checkpoints/kaggle_import`

---

## 7. v6-vs-v8.2 Comparison (Step 8) — COMPLETED

Kaggle notebook was run with GPU T4 ×2. 120 checkpoints downloaded (4 universes × 3 models × 10 seeds). Per-seed results in `data/kaggle_per_seed_results.csv`.

### v8.2 Kaggle Results (10-seed, Future Holdout Sharpe, mean ± std)

| Universe | LSTM-DNN (60% real) | Real-PPO (60% real) | **RAI v8.2 (0% real)** |
|---|---|---|---|
| Indian Nifty 50 | -0.269 ± 0.001 | -0.199 ± 0.153 | -0.246 ± 0.099 |
| US Tech & Benchmark | +0.969 ± 0.001 | +1.005 ± 0.092 | +0.966 ± 0.138 |
| Forex & Commodities | +1.460 ± 0.002 | +1.439 ± 0.176 | +1.435 ± 0.114 |
| Cryptocurrency | -1.309 ± 0.000 | -1.271 ± 0.065 | -1.306 ± 0.034 |
| **Overall (N=40)** | **+0.213 ± 1.095** | **+0.244 ± 1.081** | **+0.212 ± 1.088** |

### v8.2 vs Baselines (paired by seed)

| Universe | v8.2 vs LSTM-DNN | v8.2 vs Real-PPO |
|---|---|---|
| Indian Nifty 50 | v8.2 wins 7/10 (+0.023) | v8.2 wins 3/10 (-0.047) |
| US Tech | v8.2 wins 5/10 (-0.003) | v8.2 wins 5/10 (-0.039) |
| Forex | v8.2 wins 5/10 (-0.025) | v8.2 wins 6/10 (-0.005) |
| Crypto | v8.2 wins 6/10 (+0.003) | v8.2 wins 4/10 (-0.035) |
| **Total** | **23/40 (57.5%)** | **18/40 (45.0%)** |

> [!WARNING]
> **v8.2 is a null result.** The newer architecture (MultiScaleRiskAwareNet) with a different procedural world engine does not outperform its own real-data-trained baselines. Mean Sharpe differences are all within ±0.05. The 10-seed CI confirms this is not noise — the confidence intervals are tight.

### Why v6 and v8.2 are not directly comparable

The Kaggle benchmark uses different universes (Indian Nifty, US Tech, Forex, Crypto) with different tickers, different baseline implementations (LSTM-DNN, Real-PPO — not the XGBoost/Risk Parity/SMA used in the canonical harness), and a different cost model. A v6 Sharpe number cannot be directly subtracted from a v8.2 Sharpe number.

### Primary Model Decision

**Axiom is the primary model for the paper**, with these caveats stated explicitly:
1. ~~All v6 results are single-seed (not CI-verified)~~ — superseded: Axiom is 10-seed (§11) and the ML baselines are 10-seed (§14) as of 2026-08-29. The deterministic baselines have no seed to vary; the pairwise win counts in §2 remain single-seed.
2. Axiom's OOS advantage (#1 of 9) does not persist into holdout (#7 of 9)
3. Risk Parity (no ML) leads overall pairwise wins (64/96)
4. v8.2 (newer architecture, 10-seed CI) showed no improvement over baselines — this is an informative negative result that should appear in the paper
5. The aggregate holdout result does not generalise beyond the six markets tested — market-level 95% CI [-0.31, +1.56] crosses zero (§13)

---

## 8. README Version Labeling (Step 9)

Every results table in `README.md` now explicitly states:
- `[v6]` architecture tag on every RAI model row
- ~~"Architecture: v6 (DeepEndToEndTradingNet)" in the results section header~~ — superseded
  2026-08-29: that header named a class shared by two different networks and its diagram described
  the wrong one. It now names `AxiomNet` for Axiom and `FastTradingNet` for Fast, with separate
  diagrams and parameter counts. See §15.
- v7/v8.2 noted as available code with no local checkpoints

---

## 9. Framing Fixes Applied (this session)

| Fix | Section | What Changed |
|---|---|---|
| Holdout ranking | §2 | "middle-of-pack" → "7th of 9 arms", explicit ranking table added |
| Risk Parity finding | §2 | Explicit sentence stating Risk Parity has highest overall win count (64/96) |
| Re-scoped fragility | §3 | "Not fragile" claim scoped to equities/US-sector 2020–2024; crypto noted as untested |
| Re-scoped diversification | §5 | Diversification claim scoped to equity universes; crypto/Fast noted as failure mode |
| PIT Mega-Cap citation | §4 | Cross-verified with second search; CRSP `dsp500list` method documented for paper-grade |
| Crypto exclusion rule | §4 | "Delisted" → dated, explicit selection rule based on tradability at construction date |
| Mean Sharpe table | §2 | Added alongside win-count table to show magnitude, not just rank |
| Step 8 confirmation | §7 | Explicit statement that v8.2 was NOT touched |

### Claims That Could Not Be Cleanly Re-scoped (Flagged for Manual Review)

1. **BRK-B → CVX substitution**: The exclusion of BRK-B as a "conglomerate" is reasonable but should be further justified in a paper. CVX was ~#11–#13, not #10. If the reviewer objects, BRK-B should be included and CVX dropped.
2. **BSV delisting timeline**: The delistings from Binance and Kraken occurred in 2019, which is before the Jan 2020 construction date — so the exclusion is defensible. However, BSV remained tradable on some exchanges (e.g., Robinhood, some regional exchanges). The selection rule has been tightened to "continuously tradable on major exchanges through the holdout window" to address this.

---

## 10. Confirmation Checklist

- [x] Step 1: Backup at `backups/local_pre_framing_fix_2026-08-19_2029/` (99 files)
- [x] Step 2: Holdout ranking fixed (7th of 9, not "middle-of-pack")
- [x] Step 3: Risk Parity finding surfaced explicitly
- [x] Step 4: Fragility claim re-scoped to equities window
- [x] Step 5: Diversification claim re-scoped; crypto/Fast documented as boundary failure
- [x] Step 6: PIT Mega-Cap citation strengthened; CRSP path documented
- [x] Step 7: Crypto exclusion rule made explicit and dated
- [x] Step 8: Mean-Sharpe table added alongside win-count table
- [x] Step 8 (v8.2): **COMPLETED** — Kaggle run done, 120 checkpoints imported, v8.2 is a null result
- [x] Step 9: Report updated (this document)
- [x] Axiom renamed (v6 Alpha → Axiom v0.9), backup at `backups/local_pre-rename_2026-08-20_0925/`
- [x] LSTM/XGBoost seed bug found and fixed (seed not propagated + no shuffle)
- [x] Axiom multi-seed CI: 10 seeds × 6 universes, OOS mean +0.98, holdout mean +0.62
- [x] Primary model: **Axiom v0.9** (CI-verified, with caveats)

### 2026-08-29 additions

- [x] Backup at `backups/local_pre-claudecode_2026-08-29/` (134 files, 3.6 MB) taken before any edit
- [x] TASK 1: three distinctly-labelled aggregate statistics (seed-level CI / market-level CI / dispersion SD); ±0.67 confirmed as dispersion, holdout dispersion corrected 1.10 → 1.17 (§11, §13)
- [x] TASK 2: every "competitive with LSTM/comparable to baselines" claim site located by grep and given an inline caveat (§13 lists the sites)
- [x] TASK 3: India holdout failure documented as a full subsection (§5.1); the false "all 10 seeds negative" clause corrected to 9 of 10
- [x] TASK 4: 61 mojibake sequences repaired across `README.md` (11) and this file (50); no BOM was ever present in any `.md` (§13)
- [x] TASK 5: LSTM/XGBoost re-run across the same 10 seeds on the pinned window; cost-model asymmetry quantified; LSTM degeneracy identified (§14)
- [x] TASK 6: cash-logit offset sweep re-run outside equities — 0/16 arms fragile, but the constant sets the risk *level* (§3)
- [ ] Pairwise win counts (§2) recomputed on the pinned window with 10-seed arms — **not done**
- [x] TASK A: architecture naming collision resolved — `AxiomNet` / `FastTradingNet` / `DeepTransformerTradingNet` / `LegacyV6TradingNet`; 48 occurrences rewritten across 16 files; zero ambiguous references in `.py`; strict loads verified and cross-loads rejected (§15)
- [x] TASK B: deterministic baselines + `[v6] Fast` re-run on the pinned window under Axiom's cost model — all nine arms now on one basis; Risk Parity beats Axiom in the holdout (§16)
- [x] TASK C: unpaired significance tests run (Welch t + Mann-Whitney U, 10 vs 10 and pooled 60 vs 60), p-values now in the interpretation sentence (§17)
- [x] TASK D: LSTM degeneracy confirmed by per-day policy dump — bit-identical constant allocation across seeds (§18)
- [x] TASK A (2nd pass): checkpoint naming collision resolved — `data/v0.6_rl_checkpoints/axiom.pt` was a `FastTradingNet` byte-identical to `rai_v6_alpha.pt`, renamed `axiom_v0_prototype_fasttradingnet.pt`; the bare label "Axiom" now binds only to `checkpoints/axiom_multiseed/axiom_seed*.pt` (§19)
- [x] TASK B (2nd pass): SMA 50/200 warm-up bug fixed by feeding pre-window history — collapsed universe-windows 5/12 → 1/12; and the arm previously labelled "SPY B&H" relabelled **Asset-0 B&H** (column 0 is never SPY) (§20)
- [x] TASK C (2nd pass): cluster-corrected significance tests added beside the pooled ones — universe-paired t / exact Wilcoxon / sign test at n = 6 plus CR1 cluster-robust OLS over 120 observations; OOS XGBoost 2.2e-11 → 0.0072, holdout LSTM tie → significant against Axiom (§17.1)
- [x] TASK (3rd pass): a **real fixed-SPY** passive arm added — the same SPY series in all six universes, pinned windows, Axiom cost model, kept *beside* `Asset-0 B&H` rather than replacing it; Axiom ties it OOS (+0.98 vs +0.90, cluster-corrected p = 0.76) and is 0.69 Sharpe behind on the holdout point estimate (+0.62 vs +1.31) (§21)
- [ ] Architecture vs training-run confound between Axiom and Fast separated — **not possible with the checkpoints in this repo** (§15)
- [ ] Seed-level CIs for the deterministic arms — **not applicable**: they have no seed; `[v6] Fast` is one checkpoint (§16)
- [ ] *Seed*-paired cross-model significance test — **not possible as specified** (training seeds vs fitting seeds are unpaired). *Universe*-paired tests are valid and were run instead (§17.1).

---

## 11. Axiom v0.9 CI-Verified Results (2026-08-20)

10 seeds trained on Kaggle GPU T4×2. Per-seed data in `data/axiom_per_seed_results.csv`.

| Universe | OOS Sharpe (mean ± SD) | Bootstrap 95% CI | Holdout Sharpe (mean ± SD) | Bootstrap 95% CI |
|---|---|---|---|---|
| US ETFs | +1.35 ± 0.17 | [+1.24, +1.44] | +1.43 ± 0.16 | [+1.34, +1.53] |
| US Mega-Cap (PIT) | +1.90 ± 0.41 | [+1.67, +2.14] | +1.69 ± 0.21 | [+1.56, +1.80] |
| Global Indices | +1.08 ± 0.23 | [+0.94, +1.20] | +0.91 ± 0.13 | [+0.84, +0.99] |
| India Nifty 50 | +0.04 ± 0.36 | [-0.17, +0.26] | -0.46 ± 0.41 | [-0.70, -0.22] |
| Forex & Commodities | +0.40 ± 0.27 | [+0.26, +0.57] | +1.34 ± 0.34 | [+1.13, +1.53] |
| Crypto (PIT) | +1.15 ± 0.17 | [+1.04, +1.24] | -1.18 ± 0.18 | [-1.28, -1.08] |
| **Overall — seed-level 95% CI** | **+0.98** | **[+0.92, +1.05]** | **+0.62** | **[+0.56, +0.68]** |
| **Overall — market-level 95% CI** | **+0.98** | **[+0.45, +1.52]** | **+0.62** | **[-0.31, +1.56]** |
| *Cross-universe dispersion (descriptive)* | *SD = 0.67* | — | *SD = 1.17* | — |

The three "Overall" quantities answer different questions and are computed from the same 60
seed-level observations (10 seeds × 6 universes). Method and variance components in §13; reproduce
with `python scripts/aggregate_ci.py` → `data/axiom_aggregate_stats.json`.

- **Seed-level CI** — stratified bootstrap, seeds resampled within universe, universes fixed.
  *"Would different random seeds change this?"* No.
- **Market-level CI** — mixed-effects REML `y ~ 1 + (1|universe)`. *"Would different markets change
  this?"* Substantially. **The holdout interval crosses zero — the holdout result does not
  generalise beyond the six markets tested.**
- **Cross-universe dispersion SD** — descriptive spread of the six per-universe means. Not a
  standard error and not an interval on the mean.

**Single-seed was slightly optimistic**: OOS dropped from +1.17 → +0.98 (16% lower). Holdout stable: +0.67 → +0.62. **Caveat added 2026-08-29 (§15): the +1.17 checkpoint was `FastTradingNet` and the +0.98 checkpoints are `AxiomNet`, so this delta is not a pure seed effect — it mixes a seed effect with an architecture change and the two cannot be separated with the checkpoints in this repo.**

**Baseline comparison (updated 2026-08-29 — resolved, single basis, significance tested)**: LSTM and
XGBoost were re-run across the same 10 seeds on the Axiom-matched pinned window (§14), and the six
deterministic/Fast arms were re-run on that same window under Axiom's cost model (§16), so all nine
arms are now on one basis. Axiom's OOS mean **+0.98** [seed-level CI +0.92, +1.05] sits against
**+0.91** (LSTM) and **+0.07** (XGBoost) cost-matched — the unpaired significance tests (§17) make
Axiom **indistinguishable from the LSTM** (Welch p = 0.51, MWU p = 0.184, d = 0.12) and
**ahead of XGBoost** (pooled p = 2.2e-11 / 8.3e-10, d = 1.35; **p = 0.0072** once clustering by
universe is priced in, and 0.0625 on the exact rank test at n = 6 — §17.1). Four caveats belong in the
same
breath: the LSTM arm is a **confirmed constant allocation** — bit-identical weight vectors across
seeds, risk-on on 481 of 481 days (§18) — so "indistinguishable from the LSTM" means indistinguishable
from a static 80% buy-and-hold; **in the holdout every significant LSTM comparison goes against
Axiom** (§17), and cluster-corrected the LSTM is ahead in **all six** holdout universes
(p = 0.0053, universe-level CI [−0.196, −0.058], §17.1); and **Risk Parity, now on the same basis, beats
Axiom in the holdout** (+0.92 vs +0.62)
while being level OOS (+0.94 vs +0.98) with no training data at all (§16).

---

## 12. LSTM/XGBoost Seed Bug Fix (2026-08-20)

**Bug**: `train_lstm_on_split()` never called `torch.manual_seed()` and used full-batch chronological training. `train_xgboost_on_split()` had hardcoded `random_state=42`.

**Fix**: Both functions now accept a `seed` parameter. LSTM uses `DataLoader(shuffle=True, batch_size=128)`. XGBoost passes `seed` to `random_state`.

**Fix verified applied (2026-08-29)**: `verify_seed_fix()` in `scripts/baseline_multiseed.py` checks
the live source of `scripts/canonical_evaluation.py` for all seven conditions — `torch.manual_seed(seed)`,
`DataLoader(shuffle=True)`, seeded generator, `random_state=seed`, no hardcoded `random_state=42`, and
both train functions accepting `seed`. **7/7 pass** (recorded under `seed_fix_checks` in
`data/baseline_multiseed_summary.json`).

**Impact**: The single-seed baseline numbers in §2 came from the bugged pipeline — valid as
single-seed but with artificially zero variance. **The re-run is now done (§14)**, so this no longer
blocks promotion to v1.0. It produced one result the fix alone did not predict: LSTM variance is
*still* essentially zero in 4 of 6 universes even with the seeding correct, because that network
collapses to a constant signal. See §14 and `docs/consolidation_inventory.md` §6.

---

## 13. Statistical Methodology Correction (2026-08-20, extended 2026-08-29)

**What changed**: The "Overall" row previously reported ±0.67 (OOS) / ±1.10 (holdout), which was the standard deviation of the six per-universe means — measuring cross-universe dispersion, not seed-level estimation uncertainty.

**Old (incorrect)**: Overall OOS = +0.98 ± 0.67, Overall Holdout = +0.62 ± 1.10

Verified directly: `std({1.35, 1.90, 1.08, 0.04, 0.40, 1.15}, ddof=1) = 0.6684` — so the published
±0.67 was indeed the dispersion of the per-universe means. The holdout figure was additionally
**wrong**: the dispersion there is **1.166** (ddof=1) or 1.064 (ddof=0), and the published 1.10 matches
neither. It is corrected to **1.17** throughout.

**New (correct)** — three distinctly-labelled quantities from the same 60 seed-level observations
(10 seeds × 6 universes), all in `data/axiom_aggregate_stats.json`, reproduced by
`python scripts/aggregate_ci.py` (10,000 bootstrap iterations, bootstrap seed 20260829):

| Quantity | Method | OOS | Holdout | Answers |
|---|---|---|---|---|
| **Seed-level 95% CI** | Stratified bootstrap: seeds resampled within universe, universes fixed | **+0.98 [+0.92, +1.05]** (SE 0.0345) | **+0.62 [+0.56, +0.68]** (SE 0.0317) | "Would other seeds change this?" |
| **Market-level 95% CI** | Mixed-effects REML `y ~ 1 + (1\|universe)`, `statsmodels`, converged | **+0.98 [+0.45, +1.52]** (SE 0.2729) | **+0.62 [-0.31, +1.56]** (SE 0.4760) | "Would other markets change this?" |
| *Cross-universe dispersion SD* | Descriptive SD of the six per-universe means (ddof=1) | *0.67* | *1.17* | "How much does Sharpe vary market to market?" |

Supporting detail:

- **The naive pooled SE is invalid and is reported only to be rejected**: treating all 60 observations
  as independent gives SE 0.0867 / 0.1422 and CI [+0.81, +1.15] / [+0.34, +0.90]. It ignores clustering
  by universe entirely and sits between the two defensible intervals, resembling neither.
- **Two independent routes to the market-level interval agree.** A two-level cluster bootstrap
  (resample universes, then seeds within the resampled universes) gives OOS **[+0.48, +1.47]** and
  holdout **[-0.29, +1.42]**, against REML's [+0.45, +1.52] / [-0.31, +1.56].
- **Variance components**: OOS var(universe) 0.4388 vs var(residual) 0.0793 → **ICC 0.847**; holdout
  var(universe) 1.3525 vs var(residual) 0.0670 → **ICC 0.953**. Most of the variance is *between
  markets*, not between seeds — which is why the two intervals differ by roughly an order of
  magnitude in width.
- **Closed-form check**: for this balanced design (equal seeds per universe) the analytic SE
  reproduces `statsmodels` to 5 decimal places (0.27287 vs 0.27285 OOS; 0.47596 both, holdout), so the
  REML fit is not an artefact of the optimiser.
- **Degrees of freedom caveat**: the market-level interval rests on k = 6 universes (5 df). A normal
  approximation is used; a t-based interval would be wider still, so the stated interval is if
  anything optimistic.

**Substantive consequence**: the seed-level CI answers only *"would another seed change this?"*. It is
**not** evidence that the result generalises to other markets. The market-level holdout interval
**[-0.31, +1.56] crosses zero** — the holdout result does not generalise beyond the six markets
tested. Both README and this report now state that explicitly.

**Baseline caveat placement**: the inline caveat at every point where Axiom is compared to LSTM/XGBoost
has been updated from "(LSTM baseline pre-dates seed-propagation fix; comparison provisional)" — no
longer true — to the substantive one: baselines are now 10-seed, the LSTM arm is degenerate in 4 of 6
universes, and under Axiom's cost model XGBoost's apparent edge disappears (+1.05 → +0.07). Claim
sites found by grep and updated: `README.md` (overview, status block, comparison table, interpretation
block, Limitations 1/5), this report (§2, §11, §12, footer),
`docs/consolidation_inventory.md` §6, `docs/v6_vs_v82_comparison.md`, and
`RAI_UPGRADE_VALIDATION_REPORT.md`.

**India holdout limitation**: India's holdout CI [-0.70, -0.22] lies entirely below zero, parallel to
the Crypto holdout failure. Added as a README limitation and, as of 2026-08-29, as a full subsection
(§5.1) here. **Correction**: earlier text described India's holdout as "entirely below zero across all
10 seeds". That is false — **9 of 10** seeds are negative and seed 42 reaches +0.21. It is the interval
on the mean that excludes zero. The equivalent Crypto claim was checked and *is* accurate (all 10
seeds negative).

**Encoding**: an earlier revision of this section claimed "All 12 active .md files verified as clean
UTF-8 (no BOM). Mojibake fixed in consolidation_report.md." Both halves were misleading. Re-audited
2026-08-29 (`scripts/fix_md_encoding.py --check`): **no `.md` file in this repository has ever had a
BOM**, so there was nothing to fix there; and the mojibake had **not** in fact been repaired — 11
corrupted sequences remained in `README.md` and 50 in this file (em-dash, en-dash, ×, →, ✅ all
double-encoded via cp1252). All 61 were repaired on 2026-08-29 by `scripts/fix_md_encoding.py`, which
re-decodes only those non-ASCII runs that round-trip cleanly through cp1252 → UTF-8 (leaving
already-correct characters untouched, hence idempotent). Verified clean by
`scripts/verify_md_encoding.py`: 8 files, 0 residual markers, 0 BOMs.

---

## 14. Baseline 10-Seed Re-Run (2026-08-29)

Script: [`scripts/baseline_multiseed.py`](../scripts/baseline_multiseed.py). Per-seed data:
`data/baseline_per_seed_results.csv`. Aggregates: `data/baseline_multiseed_summary.json`.
Seeds 42/101/202/303/404/505/606/707/808/909 — the same ten as Axiom. The seed fix is verified
applied before anything is trained (§12, 7/7 checks).

**Window**: the pinned window that reproduces the Kaggle Axiom run — start 2016-08-20 (10y universes)
or 2021-08-20 (5y universes), end 2026-08-20 — not `canonical_evaluation.py`'s relative download
window used for the old single-seed numbers. Validated by re-evaluating the 10 saved Axiom checkpoints
on this window against the published `data/axiom_per_seed_results.csv`: mean |err| **0.042** OOS /
**0.021** holdout (`data/axiom_repro_check.csv`). Day counts: US ETFs / Mega-Cap / Global 2512
(1507/502/503), India 1239 (743/248/248), Forex 1255 (753/251/251), Crypto 1826 (1095/365/366).

### OOS Sharpe, old single-seed vs new 10-seed

| Universe | LSTM old (1 seed) | **LSTM new (10 seeds)** | XGB old (1 seed) | **XGB new (10 seeds)** |
|---|---|---|---|---|
| US ETFs | 1.05 | **+1.116 ± 0.000** | 1.21 | **+1.165 ± 0.075** |
| US Mega-Cap (PIT) | 1.69 | **+1.729 ± 0.000** | 1.62 | **+1.660 ± 0.062** |
| Global Indices | 0.85 | **+0.875 ± 0.031** | 0.59 | **+0.710 ± 0.061** |
| India Nifty 50 | 0.79 | **-0.237 ± 0.000** | 1.08 | **+0.306 ± 0.186** |
| Forex & Commodities | 0.20 | **+0.990 ± 0.000** | 0.41 | **+0.922 ± 0.231** |
| Crypto (PIT) | 1.43 | **+1.174 ± 0.080** | 1.56 | **+1.513 ± 0.096** |
| **Mean** | +1.00 | **+0.941** | +1.08 | **+1.046** |

Per-universe deltas mix a seed effect with the window change, so only the new column is like-for-like
against Axiom. The largest single move is India LSTM, +0.79 → **-0.24**.

### Aggregates, both cost models

| | LSTM (real) | XGBoost (real) | Axiom v0.9 (reference) |
|---|---|---|---|
| OOS mean, 10 seeds | +0.941 | +1.046 | **+0.983** |
| OOS seed-level 95% CI | [+0.93, +0.95] | [+1.01, +1.08] | [+0.92, +1.05] |
| OOS market-level 95% CI | [+0.41, +1.36] | [+0.67, +1.42] | [+0.45, +1.52] |
| OOS cross-universe dispersion SD | 0.65 | 0.51 | 0.67 |
| Holdout mean, 10 seeds | +0.776 | +0.793 | **+0.617** |
| Holdout seed-level 95% CI | [+0.77, +0.78] | [+0.73, +0.86] | [+0.56, +0.68] |
| Holdout market-level 95% CI | [-0.17, +1.61] | [-0.05, +1.58] | [-0.31, +1.56] |
| **OOS mean, Axiom cost model** | **+0.906** | **+0.070** | +0.983 (already costed) |
| **Holdout mean, Axiom cost model** | **+0.750** | **-0.154** | +0.617 (already costed) |

### Finding 1 — the cost model, not the seeds, was carrying the comparison

`evaluate_lstm_strategy` / `evaluate_xgb_strategy` in `canonical_evaluation.py` rebalance **daily at
zero transaction cost**. Axiom pays 5 bps + 0.02% slippage on turnover above a 3% drift threshold.
Charging the baselines Axiom's own model (`evaluate_signal_strategy_with_costs`) gives:

| Universe | XGB, zero-cost OOS | XGB, Axiom costs OOS | Delta |
|---|---|---|---|
| US ETFs | +1.165 | +0.001 | -1.16 |
| US Mega-Cap (PIT) | +1.660 | +0.686 | -0.97 |
| Global Indices | +0.710 | -0.202 | -0.91 |
| India Nifty 50 | +0.306 | -0.724 | -1.03 |
| Forex & Commodities | +0.922 | -0.507 | -1.43 |
| Crypto (PIT) | +1.513 | +1.169 | -0.34 |
| **Mean** | **+1.046** | **+0.070** | **-0.98** |

XGBoost's apparent edge over Axiom (+1.05 vs +0.98) was a free-trading artefact: it flips direction
daily and pays nothing for it. The LSTM barely moves (+0.941 → +0.906) because it trades rarely — which
is the same fact as Finding 2. **On the matched cost model Axiom (+0.98) is above both real-data
baselines OOS, and below both in the holdout (+0.62 vs +0.75 / -0.15 — LSTM wins the holdout).**

### Finding 2 — the LSTM's near-zero variance is degeneracy, not stability

Distinct Sharpe values across the 10 seeds, from `nunique()` on `data/baseline_per_seed_results.csv`:

| Universe | LSTM distinct values | XGBoost distinct values |
|---|---|---|
| US ETFs | **1** | 10 |
| US Mega-Cap (PIT) | **1** | 10 |
| Global Indices | 2 | 10 |
| India Nifty 50 | **1** | 10 |
| Forex & Commodities | **1** | 10 |
| Crypto (PIT) | 9 | 10 |

In 4 of 6 universes the LSTM yields exactly **one** Sharpe across ten independently seeded training
runs — with the seeding verifiably correct. The network converges to a constant risk-on or risk-off
signal, so the resulting strategy is seed-invariant by construction. Its ±0.000 must not be read as a
precisely estimated result, and its tight seed-level CI [+0.93, +0.95] is meaningless as a measure of
estimation uncertainty. This resolves the open question in `docs/consolidation_inventory.md` §6 in
favour of hypothesis 2 (trivially stable strategy) and rules out hypothesis 3 (residual seed bug).

**Confirmed directly on 2026-08-29 (§18)**: the inference above was made from the Sharpe distribution
alone. A per-day policy dump for 3 seeds now shows bit-identical weight vectors, constant over all 481
OOS days, with P(up) confined to [0.549, 0.568] and never crossing the 0.5 threshold. The arm is a
static 80% equal-weight buy-and-hold.

### What this re-run does not establish

- ~~**No paired cross-model significance test.**~~ **Resolved 2026-08-29 (§17, §17.1).** A *seed*-paired
  test remains invalid — Axiom's seeds are *training* seeds for a policy that never sees real data, the
  baselines' are *fitting* seeds on real data — but the correct unpaired tests have now been run:
  Welch's t and Mann-Whitney U on the independent 10-vs-10 samples, cost-matched. Cost-matched OOS,
  Axiom vs LSTM is a tie (p = 0.51 / 0.184) and Axiom vs XGBoost is significant (p = 2.2e-11 /
  8.3e-10). Model orderings are no longer merely descriptive; see §17 for the per-universe results,
  which are not uniformly favourable. A *universe*-paired test **is** valid and was added in §17.1;
  it cuts the XGBoost OOS p to 0.0072 and turns the holdout LSTM tie into a significant result against
  Axiom.
- ~~**The deterministic arms were not re-run on the pinned window.**~~ **Resolved 2026-08-29 (§16).**
  Risk Parity, EW, Asset-0 B&H, SMA 50/200, 60/40 and `[v6] Fast` were all re-run on the pinned window
  under Axiom's cost model. They still have no seed to vary, so they remain point estimates. **Extended
  2026-08-30 (§21):** a real fixed-SPY arm was added on the same basis, so the passive set is no longer
  limited to in-universe column-0 holds.
- **Pairwise win counts (§2) were not recomputed.** They remain single-seed, old-window.
- ~~**Only the two ML baselines were charged Axiom's cost model.**~~ **Resolved 2026-08-29 (§16).**
  All nine arms are now on one cost basis. Note that for Asset-0/SPY/EW/60/40 the cost-charged result is
  identical to the zero-cost one *by construction*, not by robustness.

---

## 15. Architecture Naming Collision Resolved (2026-08-29)

**The problem.** Two structurally different networks shared the class name
`DeepEndToEndTradingNet`, in two different modules. Every published architecture claim about "Axiom"
was therefore ambiguous, and the diagram in `README.md` described the wrong network.

| New class name | Module | Topology after the Transformer | Params | state_dict tensors | Checkpoints |
|---|---|---|---|---|---|
| **`AxiomNet`** | `scripts/kaggle_axiom_10seed.py` | `flatten(64 × 30 = 1920)` → `fc_features` + LayerNorm → `actor` / `critic` | **289,527** | 25 | `checkpoints/axiom_multiseed/axiom_seed*.pt` (10) |
| **`FastTradingNet`** | `scripts/train_v6_fast.py` | `mean-pool over time (64)` → `fc_features` → `actor_head` / `critic_head` | **51,703** | 23 | `data/v0.6_rl_checkpoints/{rai_v6_fast,rai_v6_alpha,axiom_v0_prototype_fasttradingnet}.pt` (3) |
| `DeepTransformerTradingNet` | `scripts/train_v6_deep_transformer.py` | 60-day window, d=128, 4 heads, 2 layers | 361,431 | 37 | none |
| `LegacyV6TradingNet` | `rai/learning/v6_model.py` | flatten, same shapes as `AxiomNet`, different attribute names | 289,527 | 25 | none (loads neither family) |

**Fix applied at the code level.** `scripts/rename_trading_nets.py` rewrote **48 occurrences across
16 files** (idempotent, with an explicit per-file target map rather than a blanket substitution,
because the correct target differs per file). `grep -rn "DeepEndToEndTradingNet" --include=*.py` now
returns only that refactor script itself, where the old name appears as *data* (its docstring and its
`OLD = "..."` constant).

Verification:

```
AxiomNet                   params=  289,527  tensors=25
FastTradingNet             params=   51,703  tensors=23
DeepTransformerTradingNet  params=  361,431  tensors=37
LegacyV6TradingNet         params=  289,527  tensors=25
axiom_seed42 -> AxiomNet strict load: OK
rai_v6_fast  -> FastTradingNet strict load: OK
CROSS-LOAD Fast->AxiomNet:      rejected (as expected)
CROSS-LOAD Axiom->FastTradingNet: rejected (as expected)
pytest tests/ -q -> 5 passed
```

**The substantive consequence, not just a documentation one.** The originally published single-seed
Axiom figure (**+1.17**) came from a `FastTradingNet` checkpoint; the 10-seed CI-verified figure
(**+0.98**) comes from `AxiomNet`. The −16% revision recorded in §11 and in
`docs/v6_vs_v82_comparison.md` therefore mixes a seed effect with an **architecture change**, and
cannot be attributed to seeds alone. Likewise every Axiom-vs-Fast contrast in this report — the
conservative-allocation bullets (§2 key findings), the crypto concentration forensics (§5), and the
`[v6] Fast` column of the offset ablation (§3) — is a comparison of two architectures, not two seeds.
No checkpoint exists that would separate the two effects: there is no `AxiomNet` trained under Fast's
recipe and no `FastTradingNet` from the Kaggle run.

Two related corrections: the banner comment in `scripts/kaggle_axiom_10seed.py` claiming the
architecture is "identical to train_v6_alpha.py" was **false** (`train_v6_alpha.py` uses
`FastTradingNet`) and has been fixed. `checkpoints/axiom_multiseed/rai-axiom.ipynb` was deliberately
**left unedited**: it is the verbatim record of the Kaggle run that produced the published
checkpoints, so it still contains the old class name by design.

Both classes expose the same `get_action(flat_obs, deterministic=True) → 11-dim` contract, which is
why they are interchangeable as *policies* in the evaluation harness even though they are not
interchangeable as *networks*.

---

## 16. Deterministic Baselines Re-Run on the Pinned Window (2026-08-29)

**Why.** After §14, three arms (Axiom, LSTM, XGBoost) were on the pinned window under Axiom's cost
model while six (Asset-0 B&H, EW 1/N, 60/40, Risk Parity, SMA 50/200, `[v6] Fast`) were still on
`canonical_evaluation.py`'s own relative download window, and the rule-based ones on a zero-cost
basis. Any table mixing them was mixing evaluation bases. All six have now been re-run on the pinned
window (2016-08-20 / 2021-08-20 → 2026-08-20) with Axiom's cost model — 5 bps fee + 0.02% slippage on
turnover, gated by a 3% portfolio-drift threshold.

Script: `scripts/deterministic_baselines_pinned.py` → `data/deterministic_baselines_pinned.csv`
(96 rows: 84 after §20 added the warm-up-corrected SMA arm, plus 12 for the fixed-SPY arm added in
§21) / `.json`. Cost-charged variants of the two
rebalancing rules (`evaluate_risk_parity_costed`, `evaluate_sma_crossover_costed`) mirror the
originals exactly — same 21-day schedule, same 60-day vol lookback, same first-10-asset slice — and
add only the drift gate and the `fee_rate × turnover` deduction.

> [!NOTE]
> Two arm-level defects found after this section was first written are fixed in **§20**: the SMA
> 50/200 warm-up bug, and the `SPY B&H` label (the arm holds **column 0** of each universe, which is
> never SPY). The tables below are left as originally computed, so the row now headed
> `Asset-0 B&H (was "SPY B&H")` is asset-0 buy-and-hold and `SMA 50/200` is "SMA on asset 0, no
> warm-up". §20 carries the corrected SMA figures. **§21 (2026-08-30) adds the arm this section never
> had — a real fixed SPY, on this same pinned window and cost model: +0.90 OOS / +1.31 holdout, i.e.
> roughly three times the asset-0 row below and ahead of every arm in this table on the holdout.**

### Mean Sharpe across the 6 universes — old (README, mixed basis) vs new (pinned, costed)

| Arm | OOS old | OOS new | Δ | Holdout new |
|---|---|---|---|---|
| Risk Parity | +1.02 | **+0.937** | -0.08 | **+0.922** |
| EW (1/N) | +0.83 | +0.806 | -0.02 | +0.673 |
| `[v6] Fast` | +0.76 | **+0.857** | +0.10 | +0.357 |
| 60/40 | +0.50 | **+0.358** | -0.14 | +0.662 |
| Asset-0 B&H (was "SPY B&H") | +0.29 | +0.294 | +0.00 | +0.681 |
| SMA 50/200 | +0.14 | +0.131 | -0.01 | +0.773 |
| **SPY B&H (fixed reference)** — added §21 | *not previously computed* | **+0.901** | — | **+1.310** |
| Axiom (10 seeds, reference) | +0.98 | +0.983 | — | +0.623 |

### Per-universe OOS Sharpe, pinned window, Axiom cost model

| Arm | US_ETFs | US_MegaCap | Global | India | Forex | Crypto |
|---|---|---|---|---|---|---|
| Risk Parity | +0.79 | +1.30 | +0.72 | +0.13 | +1.16 | +1.51 |
| EW (1/N) | +0.74 | +1.49 | +0.63 | +0.08 | +0.65 | +1.24 |
| `[v6] Fast` | +0.81 | +1.22 | +1.32 | +0.99 | +0.35 | +0.45 |
| 60/40 | +0.17 | +0.53 | +0.60 | +0.47 | -0.54 | +0.92 |
| Asset-0 B&H (was "SPY B&H") | +0.39 | +0.64 | +0.39 | -0.25 | -0.28 | +0.88 |
| SMA 50/200 | +0.15 | +0.22 | +0.15 | -0.25 | -0.32 | +0.83 |
| **SPY B&H (fixed reference)** — §21 | +1.00 | +1.00 | +1.00 | +0.82 | +0.87 | +0.73 |
| Axiom | +1.35 | +1.90 | +1.08 | +0.04 | +0.40 | +1.15 |

### Cost sensitivity, and what it does and does not show

| Arm | zero-cost | costed | Δ | trading activity |
|---|---|---|---|---|
| SPY (fixed ref) / Asset-0 / EW / 60/40 | = costed | = zero-cost | **exactly 0.000** | none after entry |
| Risk Parity | +0.944 | +0.937 | -0.006 | 8–15 rebalances per window |
| SMA 50/200 | +0.136 | +0.131 | -0.004 | 0–3 crossover flips |
| `[v6] Fast` | +0.887 | +0.857 | -0.030 | drift-gated |
| LSTM (real) | +0.941 | +0.906 | -0.035 | ~none (constant policy, §18) |
| XGBoost (real) | +1.046 | +0.070 | **-0.976** | daily |

The three buy-once arms are cost-invariant **by construction**: they allocate at t=0 and never trade,
and `evaluate_on_real_data` charges nothing for the entry allocation, so the two equity curves are
byte-identical. This must not be reported as cost robustness. The same applies to the fixed-SPY arm
added in §21.

### Findings

1. **Risk Parity beats Axiom in the holdout, +0.922 vs +0.623**, and is level with it OOS (+0.937 vs
   +0.983). A monthly inverse-volatility rule with no training data of any kind is the strongest arm
   in the holdout *among the arms in this section*. This was invisible while the arms were on
   different windows and cost bases, and it is now a stated limitation in `README.md`. **Superseded as
   the holdout leader by §21**: the fixed-SPY reference reaches +1.310, above Risk Parity's +0.922.
2. **The mixed basis was materially distorting three arms.** `[v6] Fast` gains +0.10 and 60/40 loses
   -0.14 purely from the window/cost realignment; India in particular moves a lot (Risk Parity
   +0.97 → +0.13, `[v6] Fast` +1.14 → +0.99, asset-0 B&H -0.43 → -0.25).
3. **SMA 50/200's warm-up artefact is confirmed, and is large.** The evaluator only sees in-window
   prices, so the rule has no signal for its first 200 days and holds asset 0 by default. Its curve is
   *identical to asset-0 B&H* in **5 of 12** universe-windows (0 crossovers), so its holdout mean of
   +0.773 was mostly buy-and-hold, not a signal. **Fixed in §20** — the indicator now receives
   pre-window warm-up, and the arm falls to -0.272 OOS / +0.610 holdout.
4. **Deterministic arms have no seed and no CI.** They are functions of the price series alone, and
   `[v6] Fast` is a single `FastTradingNet` checkpoint. Reported as point estimates. This is the one
   asymmetry that remains against the 10-seed arms and it cannot be removed without retraining.

---

## 17. Cross-Model Significance Tests (2026-08-29)

**Design.** A *seed*-paired test is invalid here and was not run: Axiom's 10 seeds are
*training-initialisation* seeds for a policy that never sees real data, while the LSTM/XGBoost seeds
are *model-fitting* seeds on real data — seed 42 of one shares nothing but the integer with seed 42 of
the other. Run instead on the independent 10-vs-10 samples: **Welch's t** (unequal variances) and
**Mann-Whitney U** (rank-based, no normality assumption), plus Cohen's d with pooled SD. Basis: pinned
window, Axiom cost model — the only basis on which the three arms are measured the same way
(`OOS Sharpe` / `Future Sharpe` from `data/axiom_per_seed_results.csv`, which are always cost-charged,
vs `OOS Sharpe (costed)` / `Future Sharpe (costed)` from `data/baseline_per_seed_results.csv`).

A *universe*-paired test **is** valid, and was added 2026-08-29 — see §17.1. Both arms are scored on
the same six price series, so the six per-universe differences are legitimately paired. That is a
different operation from pairing by seed and does not resurrect it.

Script: `scripts/cross_model_significance.py` → `data/cross_model_significance.csv` (32 rows) /
`.json`.

### Pooled, 60 vs 60

| Window | Comparison | Axiom | Baseline | Diff | Welch p | MWU p | Cohen's d |
|---|---|---|---|---|---|---|---|
| OOS | Axiom vs LSTM | +0.98 | +0.91 | +0.077 | **0.51** | **0.184** | 0.12 |
| OOS | Axiom vs XGBoost | +0.98 | +0.07 | +0.913 | **2.21e-11** | **8.34e-10** | 1.35 |
| Holdout | Axiom vs LSTM | +0.62 | +0.75 | -0.127 | **0.534** | **0.208** | -0.11 |
| Holdout | Axiom vs XGBoost | +0.62 | -0.15 | +0.777 | **4.43e-05** | **9.32e-06** | 0.78 |

Pooling treats all 60 observations as independent, which they are not — §13 puts ICC at 0.85 OOS /
0.95 holdout, i.e. most variance is between markets. The per-universe 10-vs-10 tests below are the
clean ones and are reported alongside, never replaced by the pooled number.

### Per-universe, 10 vs 10 (diff = Axiom − baseline; Welch p)

| Universe | OOS vs LSTM | OOS vs XGB | Holdout vs LSTM | Holdout vs XGB |
|---|---|---|---|---|
| US_ETFs | **+0.231 (p=0.0022)** | **+1.344 (p=4.7e-11)** | **-0.195 (p=0.0044)** | **+1.051 (p=2.9e-07)** |
| US_MegaCap_PIT | +0.158 (p=0.25) | **+1.210 (p=4.8e-06)** | **-0.176 (p=0.028)** | **+0.681 (p=4.7e-07)** |
| Global_Indices | **+0.221 (p=0.014)** | **+1.279 (p=4.7e-09)** | **-0.174 (p=0.0018)** | **+0.680 (p=7.1e-09)** |
| India_Nifty_50 | **+0.289 (p=0.031)** | **+0.759 (p=4.0e-05)** | -0.037 (p=0.78) | **+0.970 (p=4.1e-05)** |
| Forex_Commodities | **-0.501 (p=0.00021)** | **+0.910 (p=1.4e-07)** | -0.118 (p=0.30) | **+1.376 (p=4.6e-08)** |
| Crypto_PIT | +0.063 (p=0.30) | -0.024 (p=0.71) | -0.062 (p=0.31) | -0.098 (p=0.18) |

### Findings

1. **Cost-matched, Axiom and the LSTM are statistically indistinguishable overall** — OOS p = 0.51
   (Welch) / 0.184 (MWU), d = 0.12; holdout p = 0.534 / 0.208, d = −0.11. The wording "Axiom beats the
   LSTM" is not supported in either window. The OOS tie survives cluster correction (§17.1, paired
   p = 0.549); the *holdout* tie does not — cluster-corrected it becomes significant **against** Axiom.
2. **Axiom is significantly ahead of XGBoost in both windows** — OOS p = 2.2e-11, d = 1.35; holdout
   p = 4.4e-05, d = 0.78. But this is a cost effect: at zero cost XGBoost scores +1.046 OOS. The
   significant result is "Axiom beats daily-rebalancing XGBoost *once XGBoost pays for its trades*".
   The magnitude is also an artefact of pooling: cluster-corrected the OOS p is **0.0072**, not 2.2e-11,
   and the exact rank test at n = 6 returns 0.0625 (§17.1).
3. **The per-universe picture is not uniformly favourable.** Against the LSTM, Axiom is significantly
   *behind* in Forex OOS (−0.50, p = 0.00021) and in **all three** universes where the holdout
   difference is significant (US_ETFs, US_MegaCap, Global_Indices). Every significant holdout LSTM
   comparison goes against Axiom.
4. **Crypto is the one universe where Axiom cannot be distinguished from either baseline** in either
   window (all four p ≥ 0.18).
5. **Zero-variance baseline samples are flagged, not hidden.** Where the LSTM's 10 seeds produce a
   single Sharpe (`sd_base = 0.000`), Welch's t degenerates to a one-sample t against a constant and
   Mann-Whitney sees a fully tied group. Both are reported; §18 establishes that this is degeneracy.
6. **All four pooled p-values above are anti-conservative** and must be read next to §17.1.

### 17.1 Cluster-Corrected Tests (added 2026-08-29)

**The defect.** The pooled tests above treat 10 seeds × 6 universes as 60 independent observations.
§13 measures **ICC = 0.847 OOS / 0.953 holdout**: within a universe, seeds are near-replicates, so the
effective sample size is closer to 6 than to 60 and the pooled *p*-values are too small. Three
corrections are now reported *beside* the pooled numbers, never in place of them — the pooled column is
what was previously published, and the point of the exercise is that the difference be visible.

**Why pairing is now allowed.** Pairing by *seed* remains invalid for the reason in §17's Design
paragraph. Pairing by *universe* is a different operation and is valid: both arms are evaluated on the
same six price series, so the six per-universe differences are matched observations. Collapsing each
arm to one mean per universe gives n = 6 per arm and 6 paired differences.

**The three corrections.**

| | Method | Unit | n | Assumption it drops / adds |
|---|---|---|---|---|
| (a) | Paired *t* on the 6 universe means | universe | 6 | drops the independence-of-seeds assumption; still assumes normality of 6 differences |
| (b) | Exact Wilcoxon signed-rank + exact sign test on the same 6 differences | universe | 6 | drops normality too; floor two-sided *p* = 2/64 = **0.03125** |
| (c) | OLS of Sharpe on an arm dummy, CR1 cluster-robust SE clustered on universe, *p* from *t* at G−1 = 5 df | seed | 120 | keeps every observation; prices within-universe correlation instead of discarding it |

**Side by side.**

| Window | Comparison | pooled Welch p (n=60) | pooled MWU p | (a) paired t p (n=6) | (b) Wilcoxon p | (b) sign p | (c) CR1 OLS p, t(5) | CR1 SE ÷ naive SE |
|---|---|---|---|---|---|---|---|---|
| OOS | vs LSTM | 0.51 | 0.184 | 0.5492 | 0.4375 | 0.2188 | 0.5508 | 1.03× |
| OOS | vs XGBoost | **2.21e-11** | **8.34e-10** | **0.00722** | 0.0625 | 0.2188 | **0.00735** | 1.70× |
| Holdout | vs LSTM | 0.534 | 0.208 | **0.00527** | **0.03125** | **0.03125** | **0.00536** | 0.13× |
| Holdout | vs XGBoost | **4.43e-05** | **9.32e-06** | **0.0127** | 0.0625 | 0.2188 | **0.0129** | 1.12× |

**Universe-level effect sizes.** This is the interval that answers "would another *market* change the
answer?", the same distinction §13 draws between seed-level and market-level CIs.

| Window | Comparison | mean diff | 95% CI (n=6 clusters) | d_z | Axiom ahead in |
|---|---|---|---|---|---|
| OOS | vs LSTM | +0.077 | [−0.231, +0.385] | +0.26 | 5 of 6 |
| OOS | vs XGBoost | +0.913 | [+0.376, +1.450] | +1.78 | 5 of 6 |
| Holdout | vs LSTM | **−0.127** | **[−0.196, −0.058]** | −1.92 | **0 of 6** |
| Holdout | vs XGBoost | +0.777 | [+0.251, +1.303] | +1.55 | 5 of 6 |

**Findings.**

1. **The OOS XGBoost headline loses eight orders of magnitude** — 2.21e-11 → 0.00722. The direction
   survives; the "overwhelming" magnitude was 10 near-replicate seeds per universe being counted as 10
   independent draws. Same story in the holdout: 4.43e-05 → 0.0127.
2. **Neither XGBoost comparison passes the distribution-free test.** Axiom leads XGBoost in 5 of 6
   universes; at n = 6 that gives an exact Wilcoxon *p* of 0.0625 and a sign-test *p* of 0.2188, both
   above 0.05. So "Axiom beats XGBoost" is supported by the parametric paired *t* over six points and
   **not** by the rank test. With only six clusters no distribution-free test can return anything below
   0.03125, so this is a power ceiling imposed by having six markets — it is not evidence of no effect,
   but it does mean the claim rests on a normality assumption over n = 6.
3. **The holdout LSTM comparison reverses from "tie" to "significant against Axiom."** Pooled it was
   p = 0.534. Cluster-corrected: paired t p = 0.00527, exact Wilcoxon p = 0.03125 (its floor — the LSTM
   is ahead in **6 of 6**), sign test 0.03125, CR1 OLS p = 0.00536, universe-level CI [−0.196, −0.058]
   excluding zero. Pooling hid it because between-universe spread (holdout Sharpe runs +1.86 to −1.11)
   swamped a small but perfectly consistent within-universe deficit; clustering prices the consistency.
   This is the one comparison where correcting for clustering makes a result *stronger*, and it goes
   against Axiom. It converts §17 Finding 3 from "every *significant* holdout LSTM comparison goes
   against Axiom" into the stronger "the holdout LSTM deficit is significant overall".
4. **SE inflation is not a uniform penalty** — 1.70× (OOS XGBoost) down to 0.13× (holdout LSTM). CR1
   clustering re-weights toward consistency *across* markets and away from magnitude *within* one, so it
   can shrink an SE when the effect is near-constant across clusters.
5. **(a) and (c) agree to within 2% of each other** in all four comparisons (0.5492 vs 0.5508; 0.00722
   vs 0.00735; 0.00527 vs 0.00536; 0.0127 vs 0.0129), which is the expected behaviour for a balanced
   design — 10 seeds per universe per arm — and is a useful internal consistency check on the
   `statsmodels` CR1 implementation. They are not identical because CR1 estimates the cluster
   covariance from the 120 residuals rather than assuming the 6 cluster means are the whole story.

**What this does not change.** Axiom's own aggregate numbers (§11, §13) already used
cluster-aware machinery — the mixed-effects REML and two-level cluster bootstrap — so no Axiom CI moves.
The correction applies only to the *cross-model* comparisons in §17, whose pooled *p*-values were the
last place in the repository where 60 clustered observations were being treated as 60 independent ones.

---

## 18. LSTM Degeneracy Confirmed Directly (2026-08-29)

§14 Finding 2 inferred degeneracy from the Sharpe distribution alone (a single distinct value across
10 seeds in 4 of 6 universes). That is consistent with two stories: the seeds converge to the same
policy, or different policies coincidentally land on the same Sharpe. `docs/consolidation_inventory.md`
§6 recorded a per-seed allocation dump as the outstanding way to settle it. Now done.

Script: `scripts/lstm_degeneracy_check.py` → `data/lstm_degeneracy_check.csv` (per-day, per-seed) /
`.json`. Universe `US_ETFs` (one of the 4 with ±0.000), OOS window, 481 days, seeds 42 / 101 / 202. It
reproduces `evaluate_lstm_strategy`'s decision path exactly and records raw sigmoid P(up), the
thresholded signal, and the resulting weight vector for every day.

```
seed 42   P(up) mean 0.556632  min 0.552088  max 0.562370  | 481 risk-on / 481 days | param L2 8.448437
seed 101  P(up) mean 0.554351  min 0.548709  max 0.560898  | 481 risk-on / 481 days | param L2 8.706903
seed 202  P(up) mean 0.560200  min 0.553843  max 0.568005  | 481 risk-on / 481 days | param L2 8.682370

pair        signal ident?  weights ident?  max |dP(up)|  param L2 gap
42 vs 101            True            True      0.003379        0.2585
42 vs 202            True            True      0.005635        0.2339
101 vs 202           True            True      0.007107        0.0245

VERDICT: DEGENERATE — every seed emits one identical constant allocation for every day of the window
```

**Verdict: yes, degenerate — same policy, not coincidentally the same Sharpe.** The three trained
networks are genuinely different (parameter-L2 norms differ by up to 0.26; P(up) differs day-to-day by
up to 0.0071), but P(up) never leaves **[0.549, 0.568]** and so never crosses the 0.5 threshold. All
three seeds are risk-on on **481 of 481 days**, and the resulting weight vectors are *bit-identical*
across seeds and constant across time: 0.08 per asset, 80% invested, 20% cash, every day. The
thresholding is what destroys the seed variance — the network variation exists, it is just entirely
inside one side of the decision boundary.

Two consequences. The arm is a **static 80% equal-weight buy-and-hold**, so (a) its ±0.000 SD and
tight seed-level CI [+0.93, +0.95] are an interval on a constant and must not be read as precision,
and (b) it explains the cost insensitivity in §16 — it never trades, so charging it Axiom's cost model
moves it only +0.94 → +0.91. Every "indistinguishable from the LSTM" statement in §17 therefore means
*indistinguishable from a static buy-and-hold*, which is a much weaker claim than it sounds.

This closes `docs/consolidation_inventory.md` §6: hypothesis 2 (trivially stable strategy)
**confirmed by direct evidence**, hypothesis 3 (residual seed bug) **ruled out** — the seeding
demonstrably varies the network.

---

## 19. Checkpoint Naming Collision Resolved (2026-08-29)

§15 disambiguated the two *classes*. This section resolves the matching collision in the
*checkpoint files*, which was a separate defect: a `FastTradingNet` file was named `axiom.pt` and
was the default binding for the bare label **"Axiom"** in the canonical harness.

**Evidence that `data/v0.6_rl_checkpoints/axiom.pt` was not an Axiom model.**

| Property | `axiom.pt` (old name) | `rai_v6_alpha.pt` | `checkpoints/axiom_multiseed/axiom_seed42.pt` |
|---|---|---|---|
| sha256 prefix | `6dfb41b7f4e2d8a0` | `6dfb41b7f4e2d8a0` | different |
| state_dict tensors | 23 | 23 | 25 |
| Parameters | 51,703 | 51,703 | 289,527 |
| Distinctive keys | `conv1d.0/2`, `actor_head`, `critic_head` | same | `conv1.*`, `conv2.*`, `actor`, `critic` |
| Architecture | `FastTradingNet` | `FastTradingNet` | `AxiomNet` |

`axiom.pt` was **byte-identical to `rai_v6_alpha.pt`** — a copy of the Fast alpha checkpoint under an
Axiom-sounding name, created before the two architectures were distinguished. Its origin is
`scripts/train_v6_alpha.py`, which trained a `FastTradingNet`, labelled it "Axiom" throughout its
own output, and saved to `rai_axiom.pt`.

**Fixes applied.**

| Site | Before | After |
|---|---|---|
| `data/v0.6_rl_checkpoints/` | `axiom.pt` | `axiom_v0_prototype_fasttradingnet.pt` |
| `scripts/canonical_evaluation.py` loader | `for label, fname in [("RAI v6 Fast","rai_v6_fast.pt"), ("Axiom","axiom.pt")]` → both via `load_v6_model` | arch-tagged `manifest`; new `load_axiom_model()` builds `AxiomNet`; new `--axiom-checkpoint` flag defaulting to `checkpoints/axiom_multiseed/axiom_seed42.pt` |
| Label for the prototype | `"Axiom"` | `"[v0 prototype, Fast-arch]"` |
| `scripts/action_constant_ablation.py` `V6_VARIANTS` | `"Axiom v0.9" → axiom.pt` | `"[v0 prototype, Fast-arch]" → axiom_v0_prototype_fasttradingnet.pt` |
| `scripts/train_v6_alpha.py` | saves `rai_axiom.pt`; prints "Axiom" ×7 | saves `rai_v6_alpha.pt`; prints "RAI v6 Alpha"; docstring states it is **not** Axiom |
| `scripts/allocation_forensics.py` fallback list | `["rai_v6_fast.pt", "rai_axiom.pt", "rai_v6_pro_growth.pt"]` (2nd file did not exist) | `["rai_v6_fast.pt", "rai_v6_alpha.pt", "axiom_v0_prototype_fasttradingnet.pt", "rai_v6_pro_growth.pt"]` |
| `docs/model_selection_protocol.md` | "Primary reported model: `rai_axiom.pt`" | superseded banner; primary is `AxiomNet` × 10 seeds |
| `README.md`, `docs/consolidation_inventory.md`, `RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md`, `RAI_UPGRADE_VALIDATION_REPORT.md` | checkpoint lists naming `axiom.pt` / `rai_axiom.pt` | corrected, with the collision noted |

**Verification.** Cross-loads are rejected in both directions, so the binding cannot silently
regress:

```
"Axiom"   -> checkpoints/axiom_multiseed/axiom_seed42.pt          : AxiomNet, 289,527 params  OK
prototype -> FastTradingNet                                        : 51,703 params            OK
prototype -> AxiomNet                                              : rejected (as expected)
```

**Consequence for published numbers.** None of the 10-seed Axiom CI results change — they were
always produced by `scripts/kaggle_axiom_10seed.py` from the `axiom_multiseed` checkpoints, not by
the canonical harness's "Axiom" row. What changes is that any *older* single-number "Axiom" result
printed by `canonical_evaluation.py` (including the `+1.17` discussed in §15) was a
`FastTradingNet` figure and is now correctly labelled as the v0 prototype.

---

## 20. SMA 50/200 Warm-Up Bug Fixed, and the "SPY B&H" Label Corrected (2026-08-29)

### The bug

`evaluate_sma_crossover` was handed only the evaluation window. Its guard is `if t >= lw`, with `t`
indexed **from the window's first day**, so for the first 200 in-window days the 200-day mean did not
exist and the rule fell through to its initialisation, `in_spy = True` — hold asset 0. That is not
"SMA 50/200 with a slow start"; for those days it is buy-and-hold wearing the SMA's label.

Severity scales with window length. The 5-year universes have ~250-day OOS windows, so ~80% of the
window was the default. The observable signature: the SMA equity curve came out **bit-identical to
asset-0 buy-and-hold in 5 of the 12 universe-windows**, with 0 crossover flips.

| Universe / window | un-warmed SMA | asset-0 B&H | identical? |
|---|---|---|---|
| Forex & Commodities / holdout | +1.143 | +1.143 | **yes** |
| Global Indices / holdout | +1.178 | +1.178 | **yes** |
| US ETFs / holdout | +1.178 | +1.178 | **yes** |
| India Nifty 50 / holdout | +0.729 | +0.729 | **yes** |
| India Nifty 50 / OOS | -0.253 | -0.253 | **yes** |

### The fix

`scripts/deterministic_baselines_pinned.py` gains `_sma_core()` + `evaluate_sma_warmup()`, which take
an explicit warm-up prefix: the price rows immediately *preceding* the evaluation window — the train
split for the OOS window, train+OOS for the holdout. The moving averages at window index `t` are
computed over the concatenated series ending at global index `n_hist + t`; wealth accumulates only
over window indices. The opening position is set from the warm-up prefix alone rather than defaulting
to long.

No look-ahead is introduced: the warm-up slice is strictly earlier in time than the window, is used
only to evaluate the indicator, and contributes no return to the equity curve. A full 200-day prefix
is available for **all 12** universe-windows (the shortest preceding span is the 5-year universes'
~1,095-day train split), so the corrected arm has a live signal on **every** in-window day —
`n_default_days = 0` in all 12 cases, recorded per row in the CSV's `Note` column.

### Effect: the artefact was inflating the arm

| | OOS mean | Holdout mean | universe-windows == asset-0 B&H |
|---|---|---|---|
| `SMA 50/200` — no warm-up (as published) | +0.131 | +0.773 | 5 of 12 |
| **`SMA 50/200 (warm-up)` — corrected** | **-0.272** | **+0.610** | **1 of 12** |

Per universe, Axiom cost model (flip counts in parentheses):

| Universe | OOS no-warm | OOS warm-up | Δ | Holdout no-warm | Holdout warm-up | Δ |
|---|---|---|---|---|---|---|
| US ETFs | +0.153 (2) | -0.048 (3) | -0.201 | +1.178 (0) | +1.061 (2) | -0.117 |
| US Mega-Cap (PIT) | +0.218 (2) | +0.295 (5) | **+0.077** | +0.482 (2) | +0.283 (2) | -0.200 |
| Global Indices | +0.153 (2) | -0.048 (3) | -0.201 | +1.178 (0) | +1.061 (2) | -0.117 |
| India Nifty 50 | -0.253 (0) | -0.746 (2) | -0.493 | +0.729 (0) | +0.289 (2) | -0.439 |
| Forex & Commodities | -0.316 (2) | -0.949 (2) | -0.633 | +1.143 (0) | +1.143 (0) | 0.000 |
| Crypto (PIT) | +0.834 (2) | -0.136 (3) | **-0.969** | -0.074 (1) | -0.174 (3) | -0.100 |

Eleven of the twelve deltas are ≤ 0. The one increase (US Mega-Cap OOS, +0.077) is the only case where
the default long position was worse than the rule. The single remaining tie with buy-and-hold
(Forex/holdout) is **not** an artefact: with a live signal the rule stays long for the entire window
and executes 0 flips, which is a real decision rather than a missing one.

Both arms are retained in `data/deterministic_baselines_pinned.csv` (`SMA 50/200` and
`SMA 50/200 (warm-up)`), plus an `Identical to Asset-0 B&H` boolean column per row, so the published
figures stay reproducible and the size of the artefact is auditable. `README.md` now reports the
warm-up-corrected arm as the SMA column and footnotes the uncorrected one.

### The separate labelling defect

`evaluate_buy_hold_first` and both SMA arms act on **column 0** of each universe frame. Columns are
left in yfinance's alphabetical order (deliberately — see `load_universe`, since the policy's
per-asset logits are position-dependent), so column 0 is:

| Universe | column 0 |
|---|---|
| US ETFs | `EEM` |
| Global Indices | `EEM` |
| US Mega-Cap (PIT) | `AAPL` |
| India Nifty 50 | `AXISBANK.NS` |
| Forex & Commodities | `AUDUSD=X` |
| Crypto (PIT) | `BCH-USD` |

It is **not SPY in any of the six universes**, so the label `SPY B&H` was wrong everywhere it appeared.
The arm is now `Asset-0 B&H`. This also explains a coincidence visible in the published tables that
had no stated cause: US ETFs and Global Indices report identical `SPY B&H` and `SMA 50/200` figures
in both windows — both universes' column 0 is `EEM`. Only labels changed; no number for this arm
moved. **The label `SPY B&H (fixed reference)` was subsequently given to a genuinely new arm holding
the real SPY — see §21; the two rows coexist and must not be conflated.**

**What this does not change.** Axiom, LSTM, XGBoost, Risk Parity, EW, 60/40 and `[v6] Fast` are
untouched by both fixes. The SMA arm is the weakest baseline in the table before and after, so no
ranking involving Axiom changes — but the holdout row is affected in kind: the "SMA 50/200 +0.77
holdout mean" that previously sat above Axiom's +0.62 was largely buy-and-hold, and the corrected
figure is +0.61.

---

## 21. A Real, Fixed SPY Benchmark Arm (2026-08-30)

§20 corrected the *label* of the `SPY B&H` arm — it was never SPY, it was column 0 of each universe.
That fix left a substantive hole behind it: the paper's own framing invokes a comparison against
**buy-and-hold on SPY**, and outside US ETFs and Global Indices (where SPY happens to be a
constituent) that comparison had never actually been run. Renaming the arm to `Asset-0 B&H` made the
tables honest but did not supply the missing benchmark.

This section adds it. `SPY B&H (fixed reference)` buys and holds the **real SPY** in every universe,
from one cached series (`data/pinned_universes/_spy_reference.csv`, 2548 auto-adjusted closes
2016-07-01 → 2026-08-20, downloaded fresh from yfinance) that is **independent of each universe's
ticker set and column ordering**. Both rows are kept: for US Mega-Cap, India, Forex and Crypto the
SPY row is deliberately an **out-of-universe reference point**, not a swap-in replacement for the
in-universe asset-0 arm.

Implementation: `load_spy_reference` / `align_spy_to_index` / `evaluate_spy_reference` /
`evaluate_spy_native_calendar` in `scripts/canonical_evaluation.py`; wired into the pinned-window
table by `scripts/deterministic_baselines_pinned.py` and into the tests by
`scripts/cross_model_significance.py`. Same pinned windows, same 60/20/20 splits and the same cost
model (5 bps + 0.02% slippage, 3% drift gate) as Axiom, LSTM, XGBoost and the §16 arms.

### 21.1 Old (mislabelled asset-0) vs new (real SPY), per universe

Every "SPY" number previously published was this asset-0 arm. It is unchanged and still reported;
what follows is what the label had promised all along.

| Universe | asset 0 | OOS: "SPY" as published (= asset-0) | OOS: **real SPY** | Holdout: "SPY" as published (= asset-0) | Holdout: **real SPY** |
|---|---|---|---|---|---|
| US ETFs | `EEM` | +0.39 | **+1.00** | +1.18 | **+1.14** |
| US Mega-Cap (PIT) | `AAPL` | +0.64 | **+1.00** | +0.75 | **+1.14** |
| Global Indices | `EEM` | +0.39 | **+1.00** | +1.18 | **+1.14** |
| India Nifty 50 | `AXISBANK.NS` | −0.25 | **+0.82** | +0.73 | **+1.50** |
| Forex & Commodities | `AUDUSD=X` | −0.28 | **+0.87** | +1.14 | **+1.61** |
| Crypto (PIT) | `BCH-USD` | +0.88 | **+0.73** | −0.89 | **+1.32** |
| **Mean of 6** | — | **+0.29** | **+0.90** | **+0.68** | **+1.31** |

Sharpe under the Axiom cost model; zero-cost values are identical to 3 dp because this arm allocates
once and never trades and the harness charges nothing for entry — *by construction*, not as evidence
of cost robustness. Source: `data/deterministic_baselines_pinned.csv`
(arm `SPY B&H (fixed reference)`), `data/deterministic_baselines_pinned.json` §summary.

Two properties of the new row that are artefacts of the design, not findings:

- **The three 10-year universes share one SPY number** (+1.00 OOS, +1.14 holdout) because US ETFs,
  US Mega-Cap and Global Indices are all pinned to 2016-08-20 and therefore have *identical* OOS
  (2022-08-17 → 2024-08-15) and holdout (2024-08-16 → 2026-08-19) date spans. The five-year
  universes differ from them and from each other: India OOS 2024-08-26 → 2025-08-21, Forex
  2024-08-20 → 2025-08-19, Crypto 2024-08-19 → 2025-08-18. The SPY row varies across universes
  **only** through those date spans — the same reason the arm is a *reference*, not a competitor
  measured on the same assets.
- **Validation.** In US ETFs and Global Indices SPY is a genuine constituent, so the fixed reference
  must reproduce buy-and-hold of that universe's own `SPY` column. It does, to |ΔSharpe| < 1e-6 in
  both windows (`spy_in_universe_crosscheck` in the JSON: +0.9998 vs +0.9998 OOS, +1.1431 vs +1.1431
  holdout). This is the check that the arm is wired to the real series and not to a positional index.

### 21.2 Calendar alignment (a disclosure, not a correction)

SPY trades the NYSE calendar; the NSE, FX/futures and crypto markets do not. The primary row
forward-fills SPY onto each universe's own index — what a SPY holder actually marks to on a day the
NYSE is shut. Those padded days contribute exactly zero return, which deflates both the mean and the
SD of the daily series and therefore biases an annualised √252 Sharpe. The bias is quantified rather
than assumed away by recomputing the same buy-and-hold on **SPY's own sessions** inside each window:

| Universe | padded days (OOS) | OOS ffill | OOS native NYSE | padded days (holdout) | Holdout ffill | Holdout native NYSE |
|---|---|---|---|---|---|---|
| US ETFs / Mega-Cap / Global | 0 / 502 | +1.000 | +1.000 | 0 / 503 | +1.143 | +1.143 |
| India Nifty 50 | 10 / 248 | +0.817 | +0.808 | 8 / 248 | +1.503 | +1.543 |
| Forex & Commodities | 2 / 251 | +0.866 | +0.857 | 0 / 251 | +1.614 | +1.614 |
| Crypto (PIT) | **115 / 365** | +0.725 | +0.877 | **114 / 366** | +1.316 | +1.590 |
| **Mean of 6** | — | **+0.901** | **+0.923** | — | **+1.310** | **+1.363** |

Only crypto is materially affected (weekends: ~31% of its days are non-NYSE), and the padding
*understates* SPY there by 0.15–0.27 Sharpe. Both columns are reported; the forward-filled one is
primary because it is the series a holder experiences on the universe's own calendar, and every
significance test below is run on both.

### 21.3 Axiom vs the real SPY — significance

Design asymmetry that has to be stated rather than hidden: **SPY has no seed.** It is a
deterministic function of one price series, so there is one number per universe-window, not a
10-seed sample. Consequently the per-universe row is a *one-sample* t of Axiom's 10 seeds against
that constant (which prices no uncertainty in SPY itself), the pooled row is an unbalanced 60-vs-6
with one comparator observation per cluster, and **the valid test is the cluster-corrected one** —
identical in construction to §17.1: collapse Axiom to one mean per universe and pair the six
differences by universe. Source: `scripts/cross_model_significance.py`,
`data/cross_model_significance.{csv,json}`.

| Window | comparison | pooled Welch p | pooled rank p | paired-t p (n=6) | Wilcoxon p (n=6) | sign p (n=6) | CR1 OLS p, t(5) |
|---|---|---|---|---|---|---|---|
| OOS | Axiom vs LSTM | 0.51 | 0.18 | 0.5492 | 0.4375 | 0.2188 | 0.5508 |
| OOS | Axiom vs XGBoost | 2.2e-11 | 8.3e-10 | **0.0072** | 0.0625 | 0.2188 | 0.0073 |
| OOS | **Axiom vs real SPY** | 0.41 | 0.33 | **0.756** | 0.844 | 0.688 | 0.758 |
| OOS | Axiom vs real SPY (native NYSE) | — | — | 0.816 | 0.844 | 0.688 | 0.817 |
| Holdout | Axiom vs LSTM | 0.53 | 0.21 | **0.0053** | 0.03125 | 0.03125 | 0.0054 |
| Holdout | Axiom vs XGBoost | 4.4e-05 | 9.3e-06 | **0.0127** | 0.0625 | 0.2188 | 0.0129 |
| Holdout | **Axiom vs real SPY** | 0.00015 | 0.27 | **0.234** | 0.563 | 0.688 | 0.237 |
| Holdout | Axiom vs real SPY (native NYSE) | — | — | 0.232 | 0.563 | 0.688 | 0.235 |

The LSTM/XGBoost rows are unchanged from §17.1 and are reproduced here only so the SPY rows sit in
the same format. Mean differences with universe-level 95% intervals (n = 6 clusters):

| Window | Axiom − comparator | mean diff | 95% CI | d_z | Axiom ahead in |
|---|---|---|---|---|---|
| OOS | − real SPY | **+0.082** | [−0.563, +0.727] | +0.13 | 4 of 6 |
| OOS | − real SPY (native NYSE) | +0.060 | [−0.566, +0.686] | +0.10 | 4 of 6 |
| Holdout | − real SPY | **−0.688** | [−1.993, +0.618] | −0.55 | 2 of 6 |
| Holdout | − real SPY (native NYSE) | −0.740 | [−2.139, +0.659] | −0.56 | 2 of 6 |

Per-universe direction (Axiom 10-seed mean vs the SPY constant):

| Universe | OOS Axiom | OOS SPY | OOS winner | Holdout Axiom | Holdout SPY | Holdout winner |
|---|---|---|---|---|---|---|
| US ETFs | +1.345 | +1.000 | Axiom | +1.432 | +1.143 | Axiom |
| US Mega-Cap (PIT) | +1.896 | +1.000 | Axiom | +1.688 | +1.143 | Axiom |
| Global Indices | +1.076 | +1.000 | Axiom (+0.08) | +0.914 | +1.143 | SPY |
| India Nifty 50 | +0.035 | +0.817 | SPY | −0.462 | +1.503 | SPY (−1.97) |
| Forex & Commodities | +0.403 | +0.866 | SPY | +1.340 | +1.614 | SPY |
| Crypto (PIT) | +1.145 | +0.725 | Axiom | −1.175 | +1.316 | SPY (−2.49) |

**What this settles.**

1. **OOS: a statistical tie.** Axiom +0.98 vs real SPY +0.90, difference +0.08 with a
   universe-level CI spanning [−0.56, +0.73] and p = 0.76. "Competitive with buy-and-hold on SPY" is
   now a *tested* statement rather than an assumed one, and it survives — but as a tie, in both
   directions: there is no evidence Axiom beats SPY out-of-sample either.
2. **Holdout: the point estimate is well behind SPY** (+0.62 vs +1.31, −0.69) and SPY wins 4 of 6
   universes, yet the cluster-corrected test does **not** reject (p = 0.23) because the six
   differences are enormously dispersed — from +0.55 (Mega-Cap) to −2.49 (Crypto). That is a
   genuinely underpowered n = 6 comparison, not a clean draw: the honest statement is *"behind SPY
   by 0.69 Sharpe on the holdout point estimate, not statistically resolvable at n = 6"*, and it
   must not be reported as a tie without the point estimate attached.
3. **The pooled holdout p (0.00015) is exactly the artefact §17.1 warned about.** It treats 60 Axiom
   seeds as 60 independent draws against 6 singleton SPY values when ICC is 0.95; the correction
   moves it to 0.23, a factor of ~1600. It is retained only for format parity.
4. **Calendar padding does not drive any of this.** Every verdict is unchanged on SPY's native NYSE
   calendar (OOS p 0.76 → 0.82, holdout 0.234 → 0.232), and the crypto padding, the largest effect,
   moves the holdout difference *against* Axiom (−0.69 → −0.74).
5. **This is a harder benchmark than the arm it sits beside.** The real SPY (+0.90 / +1.31) beats
   the asset-0 arm (+0.29 / +0.68) in the aggregate of both windows, so replacing an implicit
   "SPY" that was really `EEM` / `AXISBANK.NS` / `AUDUSD=X` / `BCH-USD` with the real thing raises
   the bar Axiom is measured against — most sharply in the holdout, where a single passive US
   equity position outperforms the trained policy in 4 of 6 universes and by a mean of 0.69 Sharpe.

**Scope limit.** For four of six universes SPY is not investable *within* that universe's asset set;
Axiom allocating over Nifty-50 constituents cannot hold SPY. So the SPY row answers "would an
investor have done better in a US index fund than in this policy on that market's calendar?" — a
legitimate and standard opportunity-cost question — and **not** "did the policy allocate its own
universe well". The in-universe `Asset-0 B&H` row and the `EW (1/N)` row remain the within-universe
passive comparators, and both are kept for that reason.

---

**Primary model: Axiom v0.9 (CI-verified).** OOS mean Sharpe **+0.98** [seed-level 95% CI +0.92, +1.05;
market-level +0.45, +1.52]. On the single pinned/cost-matched basis (§16): statistically
indistinguishable from the real-data LSTM (Welch p = 0.51; cluster-corrected p = 0.55) and ahead of
XGBoost (pooled p = 2.2e-11, cluster-corrected **p = 0.0072**, exact rank test 0.0625 — §17.1) in OOS;
in the holdout, behind the LSTM in every universe where the difference is
significant — cluster-corrected, in **all six** (p = 0.0053) — and behind monthly Risk Parity overall
(+0.62 vs +0.92). Against a real fixed **SPY** benchmark (§21, added 2026-08-30 — the arm previously
labelled "SPY" was column 0, not SPY): a tie OOS (+0.98 vs +0.90, cluster-corrected p = 0.76) and
**0.69 Sharpe behind on the holdout point estimate** (+0.62 vs +1.31, SPY ahead in 4 of 6 universes),
a gap too dispersed to resolve at n = 6 (p = 0.23) but too large to report as a draw.
The LSTM comparison is weakened
by that arm being a confirmed constant allocation (§18), so "indistinguishable from the LSTM" means
indistinguishable from a static 80% buy-and-hold. Promotion to v1.0 is no longer blocked on the
baseline re-run (§14), the deterministic-arm re-run (§16) or the significance test (§17/§17.1); the
outstanding blockers are the market-level holdout interval crossing zero (§13), the India/Crypto
holdout failures (§5, §5.1), the holdout loss to Risk Parity (§16), the cluster-corrected holdout loss
to the LSTM in 6 of 6 universes (§17.1), the holdout deficit against the fixed SPY reference (§21),
and the architecture/training
confound between Axiom and Fast (§15).

