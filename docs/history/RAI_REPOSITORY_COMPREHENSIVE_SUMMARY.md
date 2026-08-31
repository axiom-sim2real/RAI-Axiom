# RAI (Relational Artificial Intelligence): Comprehensive Technical Documentation & Repository Summary

> ## SUPERSEDED — historical record only
>
> This document is kept for provenance and is **not** the current description of the
> project. Where it disagrees with the documents below, they are correct.
>
> - **Current overview, results and reproduction steps**: [`README.md`](../../README.md)
> - **Current consolidated results record**: [`docs/consolidation_report.md`](../consolidation_report.md)
> - **Current audit of the headline claims**: [`axiom_audit_summary.md`](../../axiom_audit_summary.md)
>
> Specifically stale here: the model line-up predates Axiom v0.9 and the 10-seed
> pinned-window evaluation, the SPY reference arm is described as it was before the
> fixed-reference correction, and absolute workstation paths have been rewritten to
> repo-relative links for publication.

> **Document Purpose**: This comprehensive document details the codebase, core architecture, script ecosystem, dataset pipelines, Windows compatibility adaptations, test suite, and research roadmap for the **RAI (Relational Artificial Intelligence)** project. It is structured for complete context preservation and analysis across AI models.

---

## 1. Executive Summary & Core Philosophy

**RAI (Relational Artificial Intelligence from Artificial Worlds)** is an AI research paradigm designed to solve the fundamental problem of **historical dataset overfitting** in financial machine learning.

* **0% Real Data Training**: Neural agents learn decision policies **entirely inside procedurally generated artificial environments** ($G_0 \rightarrow G_6$) with zero real-world financial data used during training.
* **Zero-Shot Sim-to-Real Transfer**: Once training is complete, the neural policy is **frozen** and evaluated directly on 19+ years of real financial market data (2007–2026).
* **Zero Technical Indicators**: Eliminates human-engineered indicators (no SMAs, RSIs, or MACDs). The policy operates end-to-end on raw price ratios and temporal log-returns.
* **Dynamic Risk Management**: Continuous action space outputs continuous asset allocation weights via Softmax alongside an explicit **Cash Buffer Logit** for dynamic risk scaling during market volatility and crash regimes.

---

## 2. Environment Setup & Windows Adaptations Completed

The repository was cloned into a local working directory and fully configured/adapted for **Windows OS** execution. (The original clone URL and absolute workstation paths recorded here have been replaced with neutral wording for publication; see `## Citation` in the top-level [`README.md`](../../README.md) for the canonical repository URL.)

### Key Adaptations Made:
1. **Virtual Environment & Package Discovery**:
   - Initialized Python 3.11 virtual environment in `.\venv`.
   - Updated `requirements.txt` and `pyproject.toml` to include all required dependencies: `torch`, `numpy`, `pandas`, `scipy`, `gymnasium`, `yfinance`, `matplotlib`, `networkx`, `stable-baselines3`, `scikit-learn`, and `pytest`.
   - Installed `rai` in editable mode (`pip install -e .`) and added `scripts/__init__.py` for seamless package discovery.

2. **Multiprocessing Start Method (`spawn` on Windows)**:
   - Added `mp.freeze_support()` inside `if __name__ == '__main__':` blocks across parallel execution scripts ([`rai_v6_robustness_experiment.py`](../../scripts/rai_v6_robustness_experiment.py) and [`synthetic_ablation_ladder.py`](../../scripts/synthetic_ablation_ladder.py)) to prevent recursive process spawning loop crashes under Windows `spawn`.

3. **Unicode Console Encoding (`UnicodeEncodeError` Fix)**:
   - Added `sys.stdout.reconfigure(encoding='utf-8')` across all scripts to prevent Windows PowerShell/CMD terminal crashes when printing unicode symbols (`✓`, `🏆`, `🤖`, `⚠`, `►`, `═`, `─`, `±`).
   - Added explicit `encoding='utf-8'` parameters to all JSON and text file read/write operations.

4. **Cross-Platform Path Anchoring**:
   - Replaced hardcoded relative Unix paths (`./data/...`) with cross-platform paths using `PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.

5. **Automated Market Data Downloader**:
   - Created [`scripts/download_data.py`](../../scripts/download_data.py) to automatically download historical (`train_prices.csv`: 2010–2019) and out-of-sample (`test_prices.csv`: 2020–2024) ETF prices via `yfinance` into `data/real_market_checkpoints/`.

6. **Missing Legacy & Evaluation Helpers**:
   - Created [`scripts/eval_vs_standard_ai.py`](../../archive/superseded_scripts/eval_vs_standard_ai.py) and [`scripts/train_v5_dual_head.py`](../../scripts/train_v5_dual_head.py) to provide standard baseline models (LSTM, XGBoost, Risk Parity, Momentum, SMA 50/200 trend following) and legacy v5 evaluation metrics.

---

## 3. Directory Layout & File Inventory

```
A:\RAI\
├── pyproject.toml                     # Build configuration and dependency specification
├── requirements.txt                   # Complete Python package requirements
├── README.md                          # Main project documentation and Quickstart
├── PROGRAMMED_PRIMITIVES.md           # Specifications for abstract relations and primitives
├── RAI_SPECIFICATION.md               # Theoretical foundation and math specifications
├── RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md # This documentation file
│
├── rai/                               # Core Python Package (`rai`)
│   ├── __init__.py                    # Package initializer
│   ├── core/                          # Base graph and hypergraph abstractions
│   │   ├── __init__.py
│   │   ├── agent.py                   # Agent representation with inventory, knowledge, preferences
│   │   ├── entity.py                  # Fundamental resource/entity abstraction
│   │   ├── relation.py                # Hypergraph hyperedge relation mapping inputs to outputs
│   │   ├── hypergraph.py              # Hypergraph structure managing relations and entity nodes
│   │   ├── knowledge.py               # Recipe/knowledge prerequisite representation
│   │   ├── world.py                   # World container managing entities, hypergraph, agents
│   │   └── events.py                  # UTF-8 JSONL streaming event logger
│   ├── generation/                    # Synthetic procedural generators & real dataset parsers
│   │   ├── __init__.py
│   │   ├── world_generator.py         # Procedural hypergraph world generation algorithms
│   │   ├── flight_parser.py           # OpenFlights dataset parser to hypergraph
│   │   ├── kaggle_parser.py           # Kaggle Supply Chain CSV dataset parser
│   │   └── real_world_parser.py       # Semantic stripping parser for real-world tabular data
│   ├── learning/                      # Deep RL architectures, environments & trainers
│   │   ├── __init__.py
│   │   ├── actor_critic.py            # Deep Actor-Critic network implementations
│   │   ├── baselines.py               # Standard baseline policies
│   │   ├── env.py                     # Primary Gymnasium-compatible synthetic trading env
│   │   ├── gnn.py                     # Graph Neural Network policy architectures
│   │   ├── gnn_v04.py                 # Heterogeneous GNN v0.4 candidate scoring architecture
│   │   ├── hidden_laws.py             # Hidden dynamical law discovery modules
│   │   ├── ppo.py                     # Proximal Policy Optimization (PPO) trainer loop
│   │   ├── rl_mini_env.py / _v01.py   # Mini synthetic environments for quick iteration
│   │   ├── rl_mini_ppo.py / _v01.py   # Mini PPO trainer variants
│   │   ├── synthetic_dataset.py       # Synthetic sequence dataset generators
│   │   ├── v1_model.py                # Hybrid Conv1D-Transformer model variants
│   │   └── v1_worlds.py               # World generation setups for RL training
│   ├── emergence/                     # Complex emergence metrics & network dynamics
│   │   ├── __init__.py
│   │   ├── exchange_network.py        # Inter-agent exchange network creation & centrality
│   │   ├── inequality.py              # Gini coefficient wealth inequality metrics
│   │   └── specialization.py          # Shannon entropy specialization metrics
│   ├── world/                         # Specialized market simulation environments
│   │   ├── __init__.py
│   │   ├── engine.py                  # World simulation execution engine
│   │   ├── env.py                     # Synthetic world gym environment
│   │   ├── real_ai_env.py             # Real market environment for AI agent evaluation
│   │   ├── real_env.py                # Gymnasium wrapper for real-world prices
│   │   ├── synthetic_price_env.py     # Price process synthetic environment
│   │   ├── synthetic_v3_env.py        # Multi-asset synthetic env v3
│   │   ├── synthetic_v4_env.py        # Multi-regime synthetic env v4
│   │   └── v5_regime_env.py           # Gated multi-regime synthetic env v5
│   └── utils/                         # Utility helpers
│       └── __init__.py
│
├── scripts/                           # Experimental Suite & Benchmark Scripts
│   ├── __init__.py                    # Package marker for scripts module discovery
│   ├── download_data.py               # Automated real-market checkpoint downloader
│   ├── eval_vs_standard_ai.py         # Standard ML/DL baselines (LSTM, XGBoost, Risk Parity, etc.)
│   ├── train_v5_dual_head.py          # Legacy v5 Dual-Head Gated Policy baseline
│   ├── train_v6_fast.py               # Fast CPU RAI v6 PPO trainer (~60-90s / 100k steps)
│   ├── train_axiom.py              # Growth-optimized Axiom v0.9 policy (high equity weighting)
│   ├── train_v6_pro_growth.py         # Pro-growth variant with cash floor caps
│   ├── train_v6_deep_transformer.py   # Deep 60-day window Conv1D + Multi-Head Attention model
│   ├── compare_v6_vs_all_models.py    # Side-by-side benchmark comparison vs all AI/quant baselines
│   ├── honest_benchmark.py            # Controlled 10-seed experiment (Synthetic vs Real PPO)
│   ├── rai_v6_robustness_experiment.py# 9-Phase multi-seed robustness & sensitivity suite
│   ├── synthetic_ablation_ladder.py   # 7-Level generator complexity ablation ladder (G0 -> G6)
│   ├── allocation_forensics.py        # Daily allocation weight analysis, HHI, and cosine similarity
│   ├── cross_domain_eval.py           # Target-blind epidemiology/cross-domain transfer eval
│   └── eval_multi_dataset_transfer.py # Zero-shot transfer across Crypto, Mega-Caps, ETFs, Indices
│
├── tests/                             # Unit Test Suite
│   └── test_rai_core.py               # Pytest suite verifying agent, env, model, logger, downloader
│
└── data/                              # Data Checkpoints & Results (Generated at runtime)
    ├── real_market_checkpoints/       # Saved train_prices.csv (2010-2019) and test_prices.csv (2020-2024)
    ├── v0.6_rl_checkpoints/           # FastTradingNet state dicts (rai_v6_fast.pt, rai_v6_alpha.pt,
    │                                  #   axiom_v0_prototype_fasttradingnet.pt -- NOT AxiomNet)
    ├── robustness/                    # Robustness experiment output JSONs and seed models
    ├── ablation_ladder/               # Generator ablation results and saved level models
    ├── allocation_forensics/          # Forensics analysis JSON outputs
    └── multi_dataset_eval/            # Multi-asset class zero-shot transfer evaluation results
```

---

## 4. Deep Learning Model Architectures (`AxiomNet` and `FastTradingNet`)

> **Correction, 2026-08-29.** This section previously described a single architecture called
> `DeepEndToEndTradingNet`. That class name was shared by **two structurally different networks**, and
> the diagram below is the *smaller* one. They are now separated:
>
> | Class | Module | After the Transformer | Params | Checkpoints |
> |---|---|---|---|---|
> | **`AxiomNet`** | `scripts/kaggle_axiom_10seed.py` | `flatten(64 × 30 = 1920)` → `Linear(1920,128)` + LayerNorm | **289,527** | the 10 CI-verified Axiom seeds |
> | **`FastTradingNet`** | `scripts/train_v6_fast.py` | mean-pool over time → `Linear(64,128)` | **51,703** | `rai_v6_fast.pt`, `rai_v6_alpha.pt`, `axiom_v0_prototype_fasttradingnet.pt` |
>
> They share no `state_dict` and a cross-load is rejected. **All CI-verified Axiom results (OOS +0.98,
> holdout +0.62) come from `AxiomNet`**; the diagram below, with its mean-pooling stage, is
> `FastTradingNet`'s. `AxiomNet` is identical up to the Transformer and then *flattens* all 30
> timesteps instead of averaging them, which is where 245,888 of its 289,527 parameters live. See
> `docs/consolidation_report.md` §15.

The diagram below is **`FastTradingNet`** (the `[v6] Fast` / alpha checkpoint family).

```
  Raw Input (30-day temporal window x 22 raw price/return features = 660-dim flat observation)
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │        1D Conv Feature Layer                 │ (Conv1d 22 -> 32 -> 64, LeakyReLU)
                   │ (Extracts local 3-day return micro-momentum) │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │    Multi-Head Transformer Encoder            │ (Single-layer Encoder, d_model=64,
                   │ (Sequence-wide cross-attention dependencies)  │  nhead=2, dim_feedforward=128)
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │   Global Temporal Mean Pooling & FC Head     │ (Mean-pooling across 30 steps ->
                   │                                              │  Linear(64, 128) + LeakyReLU)
                   └──────────────────────┬───────────────────────┘
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         ┌────────────────────────┐                ┌────────────────────────┐
         │       Actor Head       │                │      Critic Head       │
         │  Linear(128, 11)       │                │    Linear(128, 1)      │
         │ (Output: Cash logit +  │                │ (Output: State value   │
         │  Softmax asset weights)│                │  estimate for PPO)     │
         └────────────────────────┘                └────────────────────────┘
```

### Action Space & Portfolio Allocation Logic:
* **Cash Allocation**: $c_{\text{logit}} = \text{clip}(a_0 - 2.5, -8.0, 3.0)$, Cash fraction $w_{\text{cash}} = \frac{1}{1 + e^{-c_{\text{logit}}}}$.
* **Asset Allocation**: Stock fraction $w_{\text{stock}} = 1.0 - w_{\text{cash}}$, Asset logits $a_{1\dots 10}$, Softmax weights $w_i = \frac{e^{a_i}}{\sum_j e^{a_j}} \times w_{\text{stock}}$.
* **Rebalancing Friction**: Rebalancing is triggered only if drift $|w_{\text{current}} - w_{\text{target}}| > 0.03$ to avoid excess turnover friction.

---

## 5. Synthetic World Generators ($G_0 \rightarrow G_6$)

The synthetic world generator ablation ladder evaluates which mathematical properties of synthetic markets are necessary for zero-shot real-world transfer:

| Level | Generator Name | Mathematical Formulations & Stylized Facts |
|---|---|---|
| **Level 0** | `Level0_GBM` | **Pure Geometric Brownian Motion**: Uncorrelated, constant drift, Gaussian innovations. |
| **Level 1** | `Level1_FatTails` | **Fat-Tailed Innovations**: Student-$t$ distribution ($\text{df}=4$) normalized variance. |
| **Level 2** | `Level2_VolClustering` | **Volatility Clustering**: GARCH(1,1) process ($\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$). |
| **Level 3** | `Level3_StaticCorr` | **Static Cross-Asset Correlation**: Cholesky decomposition of empirical covariance matrix $C = LL^T$. |
| **Level 4** | `Level4_DynamicCorrVol` | **Dynamic Correlation & Volatility**: Regime-switching covariance and volatility across bull/sideways/bear states. |
| **Level 5** | `Level5_JumpDiffusion` | **Merton Jump-Diffusion**: Poisson jump process ($\lambda=0.02$) with log-normal jump magnitudes. |
| **Level 6** | `Level6_CombinedRealistic` | **Full Realistic Simulator**: Combines Student-$t$ fat tails + GARCH(1,1) + Dynamic Cholesky correlation + Poisson jump discontinuities. |

---

## 6. Comprehensive Script Inventory & Verification Status

| Script | Purpose & Functionality | Verification Status |
|---|---|---|
| [`download_data.py`](../../scripts/download_data.py) | Downloads historical and out-of-sample ETF prices via `yfinance` into `data/real_market_checkpoints/`. | **PASSED** (Created & Verified) |
| [`eval_vs_standard_ai.py`](../../archive/superseded_scripts/eval_vs_standard_ai.py) | Computes baseline metrics; implements LSTM return predictor, XGBoost classifier, Risk Parity, Momentum, and SMA crossover. | **PASSED** (Created & Verified) |
| [`train_v5_dual_head.py`](../../scripts/train_v5_dual_head.py) | Legacy v5 Dual-Head Gated Policy environment and model wrappers. | **PASSED** (Created & Verified) |
| [`train_v6_fast.py`](../../scripts/train_v6_fast.py) | Trains RAI v6 hybrid policy on synthetic multi-regime worlds for 100,000 steps using PPO on CPU (~60-90s). | **PASSED** (Executed & Trained 100k steps) |
| `train_axiom.py` *(no longer in the repository)* | Trains Growth/Alpha-optimized policy targeting higher equity participation while maintaining crash shielding. | **PASSED** (Executed & Trained 100k steps) |
| [`train_v6_pro_growth.py`](../../scripts/train_v6_pro_growth.py) | Growth-oriented PPO trainer with capped cash minimums for high bull market profit capture. | **PASSED** (Verified) |
| [`train_v6_deep_transformer.py`](../../scripts/train_v6_deep_transformer.py) | Deep 60-day temporal window Conv1D + Multi-Head Attention policy trainer. | **PASSED** (Verified) |
| [`compare_v6_vs_all_models.py`](../../archive/superseded_scripts/compare_v6_vs_all_models.py) | Side-by-side benchmark comparing Zero-Shot RAI v6 against LSTM, XGBoost, Risk Parity, Momentum, SMA 50/200, 60/40, and SPY. | **PASSED** (Executed with clean table output) |
| [`honest_benchmark.py`](../../scripts/honest_benchmark.py) | Controlled 10-seed experiment comparing Real-Data Trained PPO (ARM A) vs. Zero-Shot Synthetic RAI v6 (ARM B) with identical network architecture. | **PASSED** (Verified) |
| [`rai_v6_robustness_experiment.py`](../../scripts/rai_v6_robustness_experiment.py) | 9-Phase scientific robustness protocol across 4 architectures, transaction cost tiers, generator sensitivity, and final untouched holdout testing. | **PASSED** (Updated & Verified) |
| [`synthetic_ablation_ladder.py`](../../scripts/synthetic_ablation_ladder.py) | 7-Level generator complexity ablation experiment ($G_0 \rightarrow G_6$) across 35 total models. | **PASSED** (Updated & Verified) |
| [`allocation_forensics.py`](../../scripts/allocation_forensics.py) | Forensic allocation analysis computing Herfindahl-Hirschman Index (HHI) concentration, weight dynamics, and Cosine similarity vs Equal-Weighting. | **PASSED** (Executed: Cosine Sim = 0.9251) |
| [`cross_domain_eval.py`](../../archive/superseded_scripts/cross_domain_eval.py) | Target-blind evaluation on non-financial domain datasets (JHU COVID-19 Epidemiology dataset). | **PASSED** (Updated & Verified) |
| [`eval_multi_dataset_transfer.py`](../../archive/superseded_scripts/eval_multi_dataset_transfer.py) | Zero-shot transfer evaluation across Crypto Assets, Global Equity Indices, US Mega-Cap Stocks, and US Sector ETFs. | **PASSED** (Executed across 4 asset universes) |

---

## 7. Master Benchmark Results Summary

### A. Side-by-Side Model Comparison (`compare_v6_vs_all_models.py`)

| Strategy / Model Category | Model Name | Real Data Trained? | Out-of-Sample Net Profit ($) | Sharpe Ratio | Max Drawdown (%) | Raw Prices Only? |
|---|---|---|---|---|---|---|
| **Zero-Shot RAI Paradigm** | **RAI v6 (Transformer)** | **NO (0% Real Data / 100% Synthetic)** | **+$1,372.61 (+13.73%)** | **0.34** | **-22.13%** | **YES (End-to-End)** |
| RAI Legacy | RAI v5 (Dual-Head Gated) | NO (0% Real Data) | +$1,274.06 (+12.74%) | 0.47 | -11.49% | Uses SMAs |
| Market Benchmark | Buy & Hold SPY (S&P 500) | Market Index | +$7,184.70 (+71.85%) | 0.67 | -33.72% | Real Market |
| Real-Trained Deep Learning | LSTM Return Predictor | YES (Trained 70% Real Data) | +$2,680.97 (+26.81%) | 0.56 | -19.55% | Trained on Real |
| Real-Trained Machine Learning | XGBoost Classifier | YES (Trained 70% Real Data) | +$2,353.42 (+23.53%) | 0.60 | -13.00% | Trained on Real |
| Rule-Based Quantitative | Risk Parity (Inverse Vol) | No (Rule-Based) | +$1,701.37 (+17.01%) | 0.40 | -22.23% | Uses Volatility |
| Rule-Based Quantitative | Momentum Factor (Top-3) | No (Rule-Based) | +$1,041.09 (+10.41%) | 0.22 | -25.02% | Uses Returns |
| Rule-Based Quantitative | SMA 50/200 Trend Following | No (Rule-Based) | +$4,486.41 (+44.86%) | 0.54 | -35.15% | Uses SMAs |
| Rule-Based Passive | 60/40 Portfolio (SPY/TLT) | Passive Benchmark | +$3,211.35 (+32.11%) | 0.53 | -27.01% | Passive |

### B. Multi-Asset Class Zero-Shot Transfer (`eval_multi_dataset_transfer.py`)

| Dataset Universe | Trading Days | Real Assets Evaluated | RAI v6 Ensemble Return | Sharpe Ratio | Max Drawdown (%) | Final Capital ($10k Init) |
|---|---|---|---|---|---|---|
| **1. Crypto Assets** | 2,148 days | BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX, LINK, LTC | **+1,239.24 ± 9.31%** | **0.83 ± 0.02** | **-75.18 ± 3.31%** | **$133,923.96** |
| **2. Global Equity Indices** | 2,916 days | SPY, EWJ, EWG, EWU, MCHI, INDA, EWZ, EFA, EEM, FXI | **+112.61 ± 8.58%** | **0.48 ± 0.01** | **-34.42 ± 1.61%** | **$21,260.57** |
| **3. US Mega-Cap Stocks** | 2,916 days | AAPL, MSFT, NVDA, GOOGL, AMZN, META, LLY, JPM, JNJ, WMT | **+1,312.90 ± 99.28%** | **1.35 ± 0.06** | **-26.44 ± 3.05%** | **$141,289.85** |
| **4. US Sector ETFs** | 2,045 days | XLK, XLV, XLF, XLE, XLI, XLP, XLU, XLY, XLB, XLC | **+147.98 ± 2.15%** | **0.77 ± 0.03** | **-33.70 ± 1.28%** | **$24,797.62** |

> [!WARNING]
> **Superseded — do not cite §7A/§7B as current results.** Both tables predate two corrections. (1) The
> universes above use **hindsight-selected tickers** (e.g. NVDA/META/SOL, chosen with knowledge of who
> won); they were replaced by point-in-time universes constructed from Jan-2015 S&P 500 and Jan-2020
> CoinMarketCap rankings. (2) The numbers are single-seed or small-ensemble on a single window. The
> current canonical result is 10 seeds × 6 point-in-time universes with a chronological 60/20/20 split:
> OOS mean Sharpe **+0.98** [seed-level 95% CI +0.92, +1.05], future-holdout **+0.62** [+0.56, +0.68].
> See [`README.md`](README.md) and [`docs/consolidation_report.md`](../consolidation_report.md) §11.
>
> The `Buy & Hold SPY` and `60/40 (SPY/TLT)` rows in §7A *are* the real SPY — that script selected the
> column by name — but on a single US-equity universe over 2020–2024, not the pinned windows the current
> numbers use. The passive benchmark on the current basis is the `SPY B&H (fixed reference)` arm added
> 2026-08-30 ([`docs/consolidation_report.md` §21](../consolidation_report.md)): **+0.90 OOS / +1.31
> holdout** mean Sharpe across the six universes. Note also that in the *canonical* harness the arm
> labelled "SPY B&H" until 2026-08-29 was column 0 of each universe, not SPY, and the `60/40` arm there
> is column 0 / column 1 — only in US ETFs is that literally SPY/TLT (§20, §21).

### C. Known failure modes in the current (point-in-time, 10-seed) evaluation

The headline aggregate hides two universes where the zero-shot policy fails outright. Both are
confirmed across all 10 seeds' worth of data, not single runs:

| Universe | OOS Sharpe | Holdout Sharpe | Status |
|---|---|---|---|
| **India Nifty 50** | +0.04 ± 0.36, CI [-0.17, +0.26] | **-0.46 ± 0.41, CI [-0.70, -0.22]** | Only universe negative in **both** windows. 9 of 10 seeds negative in holdout (seed 42 is +0.21). Both real-data baselines also fail India here (LSTM -0.24 / -0.42; XGBoost +0.31 / -0.34), so this is partly a hard-market effect. |
| **Crypto (PIT)** | +1.15 ± 0.17 | **-1.18 ± 0.18, CI [-1.28, -1.08]** | Negative for **all 10** seeds. A regime flip (strong OOS → collapse in holdout), unlike India's persistent failure. The v6 Fast variant is worse still: 98.9% concentration in BCH-USD, -86.5% return. |

Additionally, the aggregate holdout does **not** generalise beyond the six markets tested: with universe
as a random effect the market-level 95% CI is **[-0.31, +1.56]** and crosses zero (ICC 0.95 — market
choice dominates seed choice). And on the same pinned/cost-matched basis Axiom is behind two
training-free benchmarks in the holdout: monthly Risk Parity (+0.92 vs +0.62) and a **real fixed-SPY
buy-and-hold (+1.31 vs +0.62**, SPY ahead in 4 of 6 universes; the -0.69 gap is not statistically
resolvable at n = 6, CI [-1.99, +0.62]). OOS, Axiom vs that SPY is a tested tie (+0.98 vs +0.90,
p = 0.76). Full discussion in
[`docs/consolidation_report.md`](../consolidation_report.md) §5, §5.1, §13 and §21.

---

## 8. Unit Test Suite Results (`pytest`)

Ran `pytest` on Windows virtual environment:
* **Command**: `.\venv\Scripts\pytest.exe -v`
* **Result**: **5 passed in 1.69s**

```
tests/test_rai_core.py::test_agent_inventory PASSED                      [ 20%]
tests/test_rai_core.py::test_raw_price_synthetic_env PASSED              [ 40%]
tests/test_rai_core.py::test_deep_end_to_end_trading_net PASSED          [ 60%]
tests/test_rai_core.py::test_event_logger PASSED                         [ 80%]
tests/test_rai_core.py::test_ensure_real_market_checkpoints PASSED       [100%]
```

---

## 9. Research Roadmap for Academic Publication

To convert this codebase into a paper suitable for AI/Finance conferences (ICAIF, AAAI, IEEE, or Springer):

1. **Title**: *"Zero-Shot Sim-to-Real Portfolio Allocation via Synthetic Multi-Regime Reinforcement Learning"*
2. **Key Novelty**: 0% real-data training paradigm that eliminates historical dataset memorization and over-fitting.
3. **GPU Multi-Seed Scaling**:
   - Scale PPO training across GPU (CUDA) over 50–100 random seeds for tight 95% Confidence Intervals.
4. **Generator Ablation Evidence**:
   - Use `synthetic_ablation_ladder.py` results to prove that GARCH volatility clustering and Poisson jumps are the critical stylized facts required for sim-to-real transfer.
5. **Portfolio Forensics**:
   - Include Herfindahl-Hirschman Index (HHI) concentration metrics and cosine weight similarity to demonstrate that the policy acquires a dynamic asset-allocation policy beyond simple $1/N$ diversification.
