# RAI Repository Consolidation Inventory

> ## SUPERSEDED — historical record only
>
> A point-in-time inventory (2026-08-18) of the divergence between two earlier
> codebase copies. The consolidation it tracks is complete, so this file is kept for
> provenance and describes a state the published repository is no longer in.
>
> - **Outcome of the consolidation**: [`docs/consolidation_report.md`](../consolidation_report.md)
> - **Current overview and reproduction steps**: [`README.md`](../../README.md)

> **Purpose**: Complete inventory of divergence between Local and GitHub codebases.  
> **Generated**: 2026-08-18  
> **Status**: Step 2 of the 6-step consolidation plan

---

## 1. Backup Verification

| Backup | Path | Files | Status |
|---|---|---|---|
| Local pre-consolidation | `backups/local_pre-consolidation_2026-08-18_2005/` | 91 | ✅ Complete |
| GitHub import | `backups/github_import_2026-08-18_2005/` | 295 | ✅ Complete (remote tracking removed) |
| Kaggle checkpoints | N/A | 0 | ⚠️ NOT FOUND — No `.pt`/`.pth`/`.zip` files in GitHub repo |

> [!WARNING]
> The Kaggle-trained model checkpoints (`rai_master_trained_models.zip`) are **not included** in the GitHub repo. The only trained checkpoints available are the local `rai_v6_fast.pt` and `rai_axiom.pt` (0.21 MB each). The user must supply the Kaggle checkpoints separately before Step 4 can proceed.
>
> **Resolved 2026-08-29.** The Kaggle checkpoints were supplied and now live in
> `checkpoints/axiom_multiseed/axiom_seed*.pt` (10 × `AxiomNet`, 289,527 params). Both files named
> above are `FastTradingNet` (51,703 params) and neither is an Axiom model; `rai_axiom.pt` was
> renamed to `rai_v6_alpha.pt` and its `axiom.pt` copy to
> `axiom_v0_prototype_fasttradingnet.pt`. See `docs/consolidation_report.md` §15 / §19.

---

## 2. File-Level Divergence

### 2A. Scripts Directory — File Existence

| Category | Count | Details |
|---|---|---|
| **In both** | 14 | All 14 have different content (see §2B) |
| **Local only** (rigor fixes) | 6 | `action_constant_ablation.py`, `allocation_weight_diagnostic.py`, `cost_sensitivity_sweep.py`, `download_data.py`, `generator_validation.py`, `__init__.py` |
| **GitHub only** | 93 | 63 evaluation/benchmark scripts, 18 training scripts, 12 utility/test scripts |

### 2B. Shared Scripts — Content Differences

| Script | Local KB | GitHub KB | Diff | Nature of Divergence |
|---|---|---|---|---|
| `allocation_forensics.py` | 26.7 | 25.9 | +0.8 | Minor: local has UTF-8 encoding fix |
| **`compare_v6_vs_all_models.py`** | **13.2** | **7.4** | **+5.8** | **Major: local has multi-window OOS, bootstrap CI, cost disclosure, 4 variants. GitHub has single-window, no CI.** |
| `cross_domain_eval.py` | 5.9 | 5.4 | +0.5 | Minor differences |
| **`eval_multi_dataset_transfer.py`** | **15.8** | **10.6** | **+5.3** | **Major: local has point-in-time universes (survivorship fix). GitHub uses hindsight tickers.** |
| **`eval_vs_standard_ai.py`** | **11.7** | **26.0** | **-14.2** | **Major: GitHub has a much larger version (possibly with v7 baselines). Local has grid-search tuning docs.** |
| `honest_benchmark.py` | 29.1 | 29.0 | +0.2 | Minor differences |
| `rai_v6_robustness_experiment.py` | 42.8 | 42.4 | +0.3 | Minor differences |
| `real_train_vs_rai_zeroshot.py` | 14.6 | 14.5 | +0.2 | Minor differences |
| `synthetic_ablation_ladder.py` | 31.0 | 30.7 | +0.3 | Minor differences |
| **`train_v5_dual_head.py`** | **2.3** | **18.5** | **-16.2** | **Major: GitHub version is 8× larger** |
| `train_axiom.py` | 13.7 | 13.3 | +0.4 | Minor: local has encoding fixes |
| `train_v6_deep_transformer.py` | 20.5 | 20.0 | +0.5 | Minor differences |
| `train_v6_fast.py` | 15.4 | 14.8 | +0.5 | Minor differences |
| `train_v6_pro_growth.py` | 14.1 | 13.7 | +0.4 | Minor differences |

### 2C. GitHub-Only Structures

| Path | What It Contains |
|---|---|
| `src/rai/models/` | v8/v8.1 uncertainty networks |
| `src/rai/world_v8/` | Procedural engine v8/v8.1/v8.2 |
| `configs/base.yaml` | Base configuration |
| `v1.0_FROZEN/` | Frozen v1.0 archive |
| `archive/v0.1/`, `archive/v0.2/` | Legacy version archives |
| Root-level `kaggle_*.py` / `kaggle_*.ipynb` | **14 Kaggle notebooks/scripts** |
| `data/parallel_multi_dataset_benchmark/` | **Benchmark results JSON** (actual numbers) |

### 2D. Local-Only Structures

| Path | What It Contains |
|---|---|
| `docs/related_work.md` | Prior art and reviewer pre-emption |
| `docs/model_selection_protocol.md` | Pre-registered primary model documentation |
| `data/v0.6_rl_checkpoints/` | `FastTradingNet` checkpoints only: `rai_v6_fast.pt`, `rai_v6_alpha.pt`, `axiom_v0_prototype_fasttradingnet.pt` (the Axiom seeds live in `checkpoints/axiom_multiseed/`) |
| `data/diagnostics/` | Allocation weight diagnostic plots |
| `data/generator_validation/` | Generator validation report + plots |
| `data/real_market_checkpoints/` | Downloaded market data cache |
| `antigravity_rai_upgrade_prompt.md` | Upgrade directive |
| `RAI_UPGRADE_VALIDATION_REPORT.md` | Validation report |
| `RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md` | Comprehensive summary |

---

## 3. Result Table Contradictions

> [!CAUTION]
> These are the specific Sharpe/return numbers that **contradict each other** across the two codebases for nominally the same model. This is the core problem to resolve.

### 3A. "RAI v6" — at least 6 conflicting numbers

| Source | Universe | Window | Return | Sharpe | Max DD | Seeds | Cost Model |
|---|---|---|---|---|---|---|---|
| **Local README** | US Sector (SPY basket) | 2020-2024 | +27.4% | 0.50 | -24.5% | 1 (alpha ckpt) | 5bps + 0.02% slip |
| **Local README** | US Sector | 2020-2024 | +24.7% | 0.50 | -23.5% | 1 (fast ckpt) | 5bps + 0.02% slip |
| **Local Validation Report (original)** | US Sector | 2020-2024 | +13.73% | **0.34** | -22.13% | 1 (unknown ckpt) | **Undisclosed** |
| **GitHub README Table 1** | Unknown (master eval) | 2021-2026 | +27.37 ± 9.42% | **1.05 ± 0.07** | -7.04% | Multi-seed | **Not stated** |
| **GitHub README Table 2** (US ETFs) | SPY,QQQ,EEM… | OOS split | +32.68 ± 9.70% | 0.92 ± 0.06 | -9.18% | Multi-seed | **Not stated** |
| **GitHub README Table 5** (Multi-Dataset) | US ETFs | 1,459 days | +86.68% | **1.18** | N/A | Unknown | **Not stated** |
| **GitHub benchmark JSON** (US ETFs) | SPY,QQQ… | OOS | +32.68 ± 9.70% | 0.92 ± 0.06 | -9.18% | Programmatic | **Not stated** |

**Contradiction summary for RAI v6 Sharpe alone**:
- 0.34 (old local, discredited)
- 0.50 (corrected local)
- 0.92 (GitHub parallel benchmark)
- 1.05 (GitHub master table)
- 1.18 (GitHub multi-dataset eval)

### 3B. "RAI v7" — only in GitHub

| Source | Universe | Return | Sharpe | Max DD |
|---|---|---|---|---|
| GitHub README heading | 16 years 2010-2026 | +48.92% | — | — |
| GitHub README Table 1 | Master 2021-2026 | +48.92 ± 64.90% | 0.81 ± 0.25 | -8.79% |
| GitHub README Table 2 US ETFs | OOS | +18.16 ± 24.51% | **0.34** ± 0.32 | -10.70% |
| GitHub README Table 3 US Mega-Cap | OOS | +134.52 ± **187.46%** | 1.11 ± 0.16 | -20.34% |

> [!WARNING]
> RAI v7 results are **internally inconsistent** within the GitHub README alone:
> - US ETF Sharpe: 0.34 (terrible)
> - US Mega-Cap Sharpe: 1.11 (great)  
> - US Mega-Cap Return SD: ±187.46% (extreme variance — likely 1 or 2 outlier seeds dominating)
> - Master table: 0.81 (middling)
>
> The ±64.90% SD on the master return and ±187.46% on Mega-Cap suggest seed instability or a degenerate seed dominating the mean.

### 3C. Baseline Discrepancies

| Baseline | Local (2020-2024) | GitHub Master (2021-2026) | GitHub US ETFs |
|---|---|---|---|
| LSTM Sharpe | 0.56 | 1.01 ± 0.03 | 0.86 ± 0.03 |
| SPY/EW Return | +71.8% | +59.78% | +57.30% |

LSTM Sharpe differs by 0.45 to 0.81× — explained by different test windows, different universes, and potentially different LSTM architectures.

> [!WARNING]
> **The `SPY/EW Return` row cannot be resolved unambiguously and is left as recorded.** The local
> +71.8% is the real SPY (`compare_v6_vs_all_models.py` selected the column by name), but the two
> GitHub columns are labelled "SPY/EW" in the source they were transcribed from and it is not
> determinable from this repo whether they are SPY buy-and-hold or an equal-weight basket, nor on which
> cost basis. Do not cite this row as a SPY comparison. For SPY on the current basis use the
> `SPY B&H (fixed reference)` arm ([`consolidation_report.md` §21](../consolidation_report.md)):
> +0.90 OOS / +1.31 holdout mean Sharpe, pinned windows, Axiom cost model. Separately, in
> `canonical_evaluation.py` the arm labelled "SPY B&H" until 2026-08-29 was column 0 of each universe
> and is now `Asset-0 B&H` (§20) — any older table quoting a "SPY" Sharpe from that harness is the
> asset-0 arm, not SPY.

---

## 4. Methodology Disagreements

| Dimension | Local Codebase | GitHub Codebase |
|---|---|---|
| **Model version** | v6 only | v6, v7, v8/v8.1/v8.2 |
| **Architecture** | Conv1D + 1-layer Transformer (d=64, 2 heads) | v6: same; v7: SpatioTemporal (multi-scale conv, 2-layer, 4 heads); v8.2: MultiScaleRiskAwareNet |
| **Cost model disclosure** | ✅ Explicit: 5bps + 0.02% slippage + 3% drift threshold | ❌ Not disclosed in README tables; 0.1% on drift in Kaggle scripts |
| **Survivorship bias** | ✅ Fixed: point-in-time universes | ❌ Uses 2026 hindsight tickers (NVDA, META, GOOGL in "US Mega-Cap") |
| **Statistical reporting** | ✅ mean ± 95% CI, bootstrap | ⚠️ mean ± SD (not CI), no bootstrap |
| **OOS windows** | 3 non-overlapping windows | Single window per evaluation |
| **Train/test split** | 2010-2019 train / 2020-2024 OOS (fixed dates) | 60/20/20 chronological (rolling 5y) |
| **Significance tests** | ✅ Planned (paired t-test) | ❌ None |
| **Action constant ablation** | ✅ Completed, no FRAGILE flags | ❌ Not done |
| **Generator validation** | ✅ Completed, GARCH gap identified | ❌ Not done |
| **Baseline tuning** | ✅ Grid search documented | ❌ No documentation |
| **Seed count** | 1-2 checkpoints available locally | 10 seeds per Kaggle experiment |
| **GPU** | CPU only | Dual NVIDIA T4 |

---

## 5. Kaggle Walk-Forward Script Details

The **key script** from GitHub is `kaggle_chronological_walk_forward_master_10seed.py`:

- **Version**: RAI v8.2 (not v6 or v7 — despite being called from within the RAI repo)
- **Architecture**: `MultiScaleRiskAwareNet` (3-day + 7-day multi-scale conv, 2-layer Transformer, 4 heads)
- **4 Universes**: Indian Nifty 50, US Tech & SPY, Global Forex & Commodities, Crypto
- **Split**: 60% train / 20% OOS / 20% Future Holdout (genuine untouched window)
- **Seeds**: 10 × [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
- **Cost**: 0.1% proportional to portfolio drift (synthetic env and eval)
- **Baselines**: LSTM-DNN (60% real), Real-PPO (60% real), RAI v8.2 (0% real)
- **No Equal-Weight baseline** in this script (unlike README tables)
- **No hardcoded results** — all computed at runtime on Kaggle GPUs

### Universes Compared to Local

| Walk-Forward Universe | Equivalent in Local? | Survivorship Issue? |
|---|---|---|
| Indian Nifty 50 | ❌ Not in local | RELIANCE.NS etc. — generally stable tickers |
| US Tech & SPY | ⚠️ Partial overlap | **YES**: includes NVDA, META, GOOGL (2026 hindsight winners) |
| Global Forex & Commodities | ❌ Not in local | Forex pairs are stable, commodities too — no survivorship issue |
| Crypto | ⚠️ Different universe | **YES**: includes SOL-USD (launched 2020), AVAX-USD (launched 2020) |

---

## 6. LSTM Near-Zero-Variance Anomaly

From the GitHub benchmark JSON:
- **LSTM std_ret on US ETFs**: 1.10% (vs RAI v6's 9.70%)
- **LSTM std_ret on Mega-Cap**: 4.27% (vs RAI v7's 187.46%)
- **LSTM std_sh on US ETFs**: 0.03 (vs RAI v6's 0.06)
- **LSTM std_sh on Global**: 0.01 (vs RAI v6's 0.03)

The LSTM's cross-seed variance is **3-10× lower** than RL-based models. This could indicate:
1. LSTM converges to a near-identical policy regardless of seed (plausible for supervised learning)
2. LSTM learns a trivially stable strategy (e.g., near-equal-weight allocation always)
3. A bug in seed handling for LSTM training

### RESOLVED 2026-08-29 — hypothesis 2 confirmed, hypothesis 3 ruled out

A seed-handling bug **did** exist and was fixed (`consolidation_report.md` §12): `train_lstm_on_split()`
never called `torch.manual_seed()` and trained full-batch chronologically. The fix is verified applied
by `verify_seed_fix()` in `scripts/baseline_multiseed.py` — 7/7 source checks pass, recorded under
`seed_fix_checks` in `data/baseline_multiseed_summary.json`.

**The variance is still zero after the fix**, which rules out hypothesis 3 as the explanation. Distinct
Sharpe values across 10 correctly-seeded training runs (`nunique()` on
`data/baseline_per_seed_results.csv`):

| Universe | LSTM | XGBoost |
|---|---|---|
| US ETFs | **1** | 10 |
| US Mega-Cap (PIT) | **1** | 10 |
| Global Indices | 2 | 10 |
| India Nifty 50 | **1** | 10 |
| Forex & Commodities | **1** | 10 |
| Crypto (PIT) | 9 | 10 |

In 4 of 6 universes ten independently seeded runs produce exactly **one** Sharpe. XGBoost, seeded
through the same harness, produces 10 distinct values in every universe — so this is a property of the
LSTM arm, not of the seeding. The LSTM collapses to a constant risk-on/risk-off signal, making the
resulting strategy seed-invariant by construction: **hypothesis 2**. Consequence for reporting: the
LSTM's ±0.000 and its tight seed-level CI [+0.93, +0.95] are degeneracy, not precision, and must not be
cited as a well-estimated baseline. See `consolidation_report.md` §14 Finding 2.

> [!IMPORTANT]
> ~~**Must investigate before trusting**: Compare LSTM allocation patterns across seeds to determine if this is legitimate convergence or a trivial strategy.~~
> ~~Investigated 2026-08-29 via the 10-seed re-run: trivial strategy. Still not done: a direct
> allocation-pattern dump per seed.~~
> **CLOSED 2026-08-29 — the allocation dump is now done.** `scripts/lstm_degeneracy_check.py` dumps the
> day-by-day policy for seeds 42 / 101 / 202 on US_ETFs OOS (481 days) →
> `data/lstm_degeneracy_check.csv` / `.json`. Result: the three trained networks genuinely differ
> (parameter-L2 8.448 / 8.707 / 8.682; max |ΔP(up)| up to 0.0071 between pairs), but P(up) stays inside
> **[0.549, 0.568]** and never crosses the 0.5 threshold, so all three are risk-on on **481 of 481
> days** and their weight vectors are **bit-identical across seeds and constant across time** (0.08 per
> asset, 80% invested, 20% cash). The shape of the constant signal is therefore known: a static 80%
> equal-weight buy-and-hold. Thresholding is what destroys the seed variance — the network variation is
> real but lies entirely on one side of the decision boundary. Full write-up:
> `consolidation_report.md` §18.

---

## 7. GitHub README Framing Issues Flagged

| Issue | Location | Problem |
|---|---|---|
| 🏆 emoji on RAI v7 | Table 1, Table 3 | Implies "best" despite lower Sharpe than baselines in Table 1 (0.81 vs LSTM 1.01, Real PPO v5 1.05) |
| "RAI Zero-Shot Advantage" column | Was in old local README (now removed) | Reports return deltas while omitting Sharpe deltas that don't favor RAI |
| v7 US ETF Sharpe 0.34 | Table 2 | Worst in table — yet v7 is called "🏆" in Table 1 |
| v7 Mega-Cap ±187.46% SD | Table 3 | Extreme variance suggests degenerate seeds |
| Missing cost model in all README tables | Tables 1-5 | Reviewer cannot reproduce or assess |
| "US Mega-Cap" tickers include NVDA, META, GOOGL | Table 3 | Survivorship bias — same issue local already fixed |

---

## 8. Recommended Resolution Path (for Step 3)

### Keep from Local (rigor infrastructure):
- Point-in-time universe construction
- Disclosed cost model (5bps + slippage + drift threshold)
- Bootstrap 95% CI reporting
- Action constant ablation
- Generator validation
- Model selection protocol
- Related work documentation

### Keep from GitHub (data and experiments):
- 60/20/20 chronological 3-way split structure (genuine future holdout — better than fixed-date split)
- 4 additional universes (Indian Nifty 50, Forex & Commodities) — after fixing survivorship for US Tech and Crypto
- v8.2 ProceduralWorldEngine (more advanced than local's v6 generator)
- 10-seed parallel execution infrastructure
- Benchmark results JSON (actual raw numbers, not README claims)
- v7/v8 model architectures (for architectural ablation)

### Discard / Archive:
- All 93+ GitHub-only evaluation scripts that produce one-off results not reproducible through the canonical harness
- All README claims not backed by JSON/reproducible data
- The "🏆" framing and asymmetric advantage columns
