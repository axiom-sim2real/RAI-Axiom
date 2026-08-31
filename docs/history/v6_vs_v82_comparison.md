# RAI v6 vs v8.2 Comparison Report

> ## SUPERSEDED — historical record only
>
> Compares two model generations that both predate Axiom v0.9, on two evaluation
> harnesses that are not directly comparable to each other (see the caveats below,
> which still apply). Kept for provenance.
>
> - **Current headline benchmark tables**: [`README.md`](../../README.md)
> - **Current consolidated results record**: [`docs/consolidation_report.md`](../consolidation_report.md)

> **Date**: 2026-08-19 (§4/§5 revised 2026-08-29)  
> **v6 results**: Canonical harness, single-seed, 6 universes × 9 arms  
> **v8.2 results**: Kaggle 10-seed walk-forward, 4 universes × 3 arms  
> **These are NOT directly comparable** — different universes, different baselines, different splits. Read caveats below.

> [!IMPORTANT]
> **The v6/Axiom numbers in §2 and §4 below are the original single-seed run and have been superseded.**
> Axiom is now 10-seed (OOS mean Sharpe **+0.98**, not +1.17; holdout **+0.62**, not +0.67) and the
> LSTM/XGBoost baselines are 10-seed as of 2026-08-29. Current numbers live in
> [`consolidation_report.md`](../consolidation_report.md) §11 (Axiom) and §14 (baselines). The
> single-seed tables here are retained as the historical record of the v6-vs-v8.2 decision.

---

## 1. Kaggle v8.2 Results (10-seed, mean ± std)

### Per-Universe Future Holdout Sharpe

| Universe | LSTM-DNN (60% real) | Real-PPO (60% real) | **RAI v8.2 (0% real)** |
|---|---|---|---|
| Indian Nifty 50 | -0.269 ± 0.001 | -0.199 ± 0.153 | **-0.246 ± 0.099** |
| US Tech & Benchmark | +0.969 ± 0.001 | +1.005 ± 0.092 | **+0.966 ± 0.138** |
| Forex & Commodities | +1.460 ± 0.002 | +1.439 ± 0.176 | **+1.435 ± 0.114** |
| Cryptocurrency | -1.309 ± 0.000 | -1.271 ± 0.065 | **-1.306 ± 0.034** |
| **Overall** | **+0.213 ± 1.095** | **+0.244 ± 1.081** | **+0.212 ± 1.088** |

### Pairwise v8.2 vs Baselines (paired by seed, Future Sharpe)

| Universe | v8.2 vs LSTM-DNN | v8.2 vs Real-PPO |
|---|---|---|
| Indian Nifty 50 | **v8.2 wins 7/10** (+0.023) | v8.2 wins 3/10 (-0.047) |
| US Tech | v8.2 wins 5/10 (-0.003) | v8.2 wins 5/10 (-0.039) |
| Forex | v8.2 wins 5/10 (-0.025) | v8.2 wins 6/10 (-0.005) |
| Crypto | v8.2 wins 6/10 (+0.003) | v8.2 wins 4/10 (-0.035) |
| **Total** | **23/40 (57.5%)** | **18/40 (45.0%)** |

> [!IMPORTANT]
> **v8.2 is statistically indistinguishable from its baselines.** The mean Sharpe differences are all within ±0.05. v8.2 slightly beats LSTM in 23/40 paired comparisons but loses to Real-PPO in 22/40. There is no clear winner among the three Kaggle arms.

---

## 2. Canonical v6 Results (single-seed, for context)

### Axiom v0.9 — Mean Sharpe across 6 universes

| Window | Axiom v0.9 | LSTM (real) | XGBoost (real) | Risk Parity |
|---|---|---|---|---|
| OOS | **+1.17** | +1.00 | +1.08 | +1.02 |
| Holdout | +0.67 | +0.70 | +0.68 | **+0.78** |
| Overall | **+0.92** | +0.85 | +0.88 | +0.90 |

Axiom v0.9 achieves the highest OOS mean Sharpe (+1.17) across 6 universes, but drops to 7th of 9 arms in the holdout.

---

## 3. Why These Are Not Directly Comparable

| Dimension | Canonical v6 | Kaggle v8.2 |
|---|---|---|
| **Universes** | US ETFs, US Mega-Cap, Global Indices, India, Forex, Crypto (6) | Indian Nifty, US Tech, Forex, Crypto (4) |
| **Tickers** | Different per universe | Different per universe |
| **Baselines** | 7 (LSTM, XGBoost, Risk Parity, SMA, EW, Asset-0 B&H, 60/40) — the arm listed as "SPY" when this table was written was actually column 0 of each universe (`consolidation_report.md` §20); a real fixed-SPY arm was added 2026-08-30 (§21) and is **not** in the v6 numbers below | 2 (LSTM-DNN, Real-PPO) |
| **Seeds** | 1 | 10 |
| **Train/OOS/Future split** | 60/20/20 (chronological) | 60/20/20 (chronological) |
| **Cost model** | 5bps + slippage + 3% drift | Built into walk-forward (different implementation) |

The split methodology is similar, but different tickers and different baseline implementations mean **you cannot compare a v6 Sharpe number directly to a v8.2 Sharpe number**.

---

## 4. What We Can Say

### v8.2 (from Kaggle, 10-seed CI):
- **Does not outperform its own baselines.** In the Kaggle framework, v8.2 (0% real data) is indistinguishable from LSTM-DNN and Real-PPO (60% real data). Mean Sharpe overall: v8.2 = +0.212, LSTM = +0.213, Real-PPO = +0.244.
- **Has tight confidence intervals.** The 10-seed runs show low variance (std ~0.03–0.14 on Future Sharpe), so the non-result is not a noise artifact.
- **Has slightly better drawdown behavior** than baselines in crypto (-39.3% vs -42.7%/-45.0%).

### Axiom v0.9 (from canonical harness, single-seed):
- **Leads OOS** across 6 universes (mean Sharpe +1.17, 38/48 pairwise wins).
- **Drops to 7th of 9 in holdout** (mean Sharpe +0.67, 23/48 wins).
- ~~**Cannot be CI-verified** without multi-seed checkpoints (only 1 checkpoint exists).~~ **Superseded 2026-08-20/29**: 10 Axiom seeds were trained on Kaggle and imported. The CI-verified numbers are OOS **+0.98** [seed-level 95% CI +0.92, +1.05] and holdout **+0.62** [+0.56, +0.68] — i.e. the single-seed +1.17 above was optimistic by 16%. See `consolidation_report.md` §11. **Correction 2026-08-29: that 16% revision is not purely a seed effect.** The single-seed +1.17 checkpoint is a `FastTradingNet` (mean-pool, 51,703 params) and the 10 CI-verified checkpoints are `AxiomNet` (flatten, 289,527 params) — two different architectures that shared the class name `DeepEndToEndTradingNet` until 2026-08-29. The delta mixes a seed effect with an architecture change and the two cannot be separated with the checkpoints in this repo. See `consolidation_report.md` §15.

### The honest comparison:
- v8.2's 10-seed CI shows it's **competitive but not superior** to real-data baselines *under the Kaggle harness's own cost treatment*.
- ~~Axiom v0.9's single-seed OOS result is **strong but unverified** — the same checkpoint could be a lucky seed.~~ **Now verified across 10 seeds** and revised down to +0.98 OOS. The seed concern is resolved; three others replaced it. (i) The aggregate holdout **market-level** CI [-0.31, +1.56] crosses zero, so the holdout result does not generalise beyond the six markets tested (`consolidation_report.md` §13). (ii) The real-data baselines were re-run across 10 seeds on 2026-08-29 (§14) and the LSTM arm turned out **degenerate** — one distinct Sharpe across all 10 seeds in 4 of 6 universes, now confirmed by a per-day policy dump to be a static 80% equal-weight buy-and-hold (§18). (iii) With all nine arms finally on one basis (pinned window, Axiom cost model, §16), the unpaired significance tests (§17) make Axiom **indistinguishable from that LSTM** OOS (Welch p = 0.51, MWU p = 0.184) and **ahead of XGBoost** (pooled p = 2.2e-11 / 8.3e-10; cluster-corrected p = 0.0072, §17.1) — but in the holdout Axiom is behind the LSTM in every universe where the difference is significant (cluster-corrected, in all six, p = 0.0053), and behind Risk Parity overall (+0.62 vs +0.92). (iv) **Added 2026-08-30**: a real fixed-SPY buy-and-hold arm (`consolidation_report.md` §21) — the arm previously labelled "SPY" was column 0, not SPY — ties Axiom OOS (+0.98 vs +0.90, cluster-corrected p = 0.76) and is **0.69 Sharpe ahead of it in the holdout** (+1.31 vs +0.62, SPY ahead in 4 of 6 universes; CI [-1.99, +0.62] at n = 6, so not statistically resolved).
- **Neither model clearly dominates.** Both are now 10-seed; v6/Axiom has the better headline number on its own harness, v8.2 is a null result on its own harness, and the two harnesses are not comparable (§3 above).

---

## 5. Recommendation

> [!WARNING]
> **For the paper, use Axiom v0.9 as the primary model with these caveats:**
> 1. ~~All v6 results are single-seed. State this prominently.~~ Axiom is 10-seed (OOS +0.98, holdout +0.62); the LSTM/XGBoost baselines are 10-seed as of 2026-08-29; the deterministic baselines have no seed to vary; the §2 pairwise win counts were **not** recomputed and remain single-seed.
> 2. The OOS advantage (1st of 9) does not persist into holdout (7th of 9). State this.
> 3. v8.2 (the newer architecture) was evaluated with 10-seed CI on a separate benchmark and showed no improvement over data-trained baselines. Mention this as a negative result — it is informative.
> 4. Risk Parity (no ML) leads overall pairwise wins. State this. **Stronger as of 2026-08-29**: on the single pinned/cost-matched basis (§16), Risk Parity also **beats Axiom on holdout mean Sharpe, +0.92 vs +0.62**, and is level with it OOS (+0.94 vs +0.98).
> 5. The aggregate holdout does not generalise beyond the six markets tested (market-level CI crosses zero). ~~and no paired cross-model significance test is available — Axiom's seeds are training seeds, the baselines' are fitting seeds.~~ **Updated 2026-08-29**: a *seed*-paired test remains invalid for that reason, but the correct **unpaired** tests have been run on the independent 10-seed samples, cost-matched (§17): Welch t / Mann-Whitney U give **p = 0.51 / 0.184 vs the LSTM (tie)** and **p = 2.2e-11 / 8.3e-10 vs XGBoost (Axiom ahead, d = 1.35)** OOS. Per-universe, Axiom loses Forex to the LSTM OOS (p = 0.00021) and loses all three significant holdout comparisons against the LSTM. A *universe*-paired test is valid and was added in §17.1: cluster-corrected, the OOS XGBoost p is **0.0072** (0.0625 on the exact rank test at n = 6) and the holdout LSTM comparison becomes **significant against Axiom** (p = 0.0053, LSTM ahead in 6 of 6).
> 6. Axiom and `[v6] Fast` are **different architectures** (`AxiomNet`, 289,527 params vs `FastTradingNet`, 51,703 params), not two seeds of one — every Axiom-vs-Fast contrast, including the crypto concentration failure, confounds architecture with training run (§15).
> 7. **New 2026-08-30 (§21)**: the passive benchmark has been fixed. Until then no arm in the canonical harness held SPY — the row labelled "SPY B&H" was column 0 of each universe (EEM / AAPL / AXISBANK.NS / AUDUSD=X / BCH-USD). A real fixed-SPY arm on the same pinned window and cost model scores **+0.90 OOS / +1.31 holdout**, roughly three times the old asset-0 row (+0.29 / +0.68). Axiom ties it OOS (p = 0.76) and trails it by **-0.69** in the holdout (not resolvable at n = 6, CI [-1.99, +0.62]). Any "does not outperform buy-and-hold" sentence in the paper must cite this row, not the asset-0 one.

**Rationale**: Axiom v0.9's OOS result is the strongest evidence for the paper's thesis (zero-shot sim-to-real transfer produces competitive risk-adjusted returns) — with "competitive" meaning, precisely: OOS mean Sharpe +0.98 against +0.91 for a real-data LSTM, +0.07 for XGBoost and +0.90 for a real fixed-SPY buy-and-hold **once all are charged the same transaction costs on the same window** (`consolidation_report.md` §14, §21). Under the baselines' original zero-cost daily-rebalance harness XGBoost reaches +1.05 and the claim would be "comparable", not "ahead". The ordering is descriptive, not tested; the Axiom-vs-SPY OOS tie *is* tested (p = 0.76). v8.2's null result does not contradict this — it shows that a different architecture with a different procedural engine did not improve further. Both results should appear in the paper.

---

## 6. Step 8 Status

- [x] Kaggle checkpoints downloaded and extracted (120 .pt files)
- [x] Kaggle per-seed CSV analyzed (120 rows, 4 universes × 3 models × 10 seeds)
- [x] v8.2 vs baseline comparison complete
- [x] v6 vs v8.2 comparison documented with caveats
- [ ] Direct apple-to-apple comparison NOT possible without running v8.2 through canonical harness with same universes/baselines (requires matching the Kaggle architecture to our evaluation pipeline — possible future work)
