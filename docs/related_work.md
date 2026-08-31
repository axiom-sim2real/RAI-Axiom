# RAI — Related Work & Prior Art

> This document must be cited/addressed directly in any paper draft. Do not omit the negative results — reviewers familiar with the field will expect them to be addressed.

---

## 1. Synthetic Market Generators for RL Training

### 1.1 Agent-Based and Zero-Intelligence Market Simulators

Agent-based models (ABMs) that simulate market microstructure from first principles are an established paradigm:

- **Paddrik et al. (2012)**: NASDAQ ITCH ABM calibrated to reproduce order-flow dynamics and flash-crash behavior.
- **Jacobs et al. (2004)**: Agent-based market simulator (ABMS) used to study self-organized criticality and power-law return distributions.
- **MarketSim / Kyle-type models**: Strategic agent simulators that reproduce bid-ask spread and price impact from rational-agent equilibrium assumptions.

**Differentiation**: RAI does not simulate order-book microstructure; it trains at the portfolio allocation level on price-only synthetic return processes.

---

### 1.2 Generative Deep Learning Approaches (Quant-GAN Lineage)

- **Wiese et al. (2020) — "Quant GAN"**: Temporal convolutional GAN that learns to generate realistic financial time series matching stylized facts (autocorrelation, volatility clustering, fat tails). Calibrated against real market data.
- **Yoon et al. (2019) — "Time-GAN"**: General-purpose temporal GAN for financial time series synthesis.
- **Coletta et al. (2023) — "Multi-Agent Market Simulator" (ABIDES-Gym)**: RL-compatible agent-based market simulator calibrated to real NYSE and NASDAQ data. Uses real LOB data for calibration.
- **TradeFM (2024)**: Foundation model for financial time-series with generative order-flow simulation.

**Differentiation**: The above approaches use real market data for calibration of the generative model (even if the RL agent is trained in the synthetic environment). RAI's G0-G6 generator family is parametrized by known stylized facts from quantitative finance literature — but does not optimize generative parameters to match real data.

**Critical distinction to state explicitly**: RAI's claim of "0% real data" applies only to the RL policy gradient updates, not to the generator parameter choices (which embody real-market knowledge from literature).

---

### 1.3 Financial RL with Synthetic Pre-Training

- **Hambly et al. (2023)**: RL for portfolio optimization with GBM synthetic pre-training and transfer to real equity data.
- **Théate & Ernst (2021)**: Deep RL trading agent benchmarked on multiple cryptocurrency markets; trained directly on real data.

---

## 2. Negative Results on Zero-Shot Synthetic-to-Real Transfer

> **CRITICAL**: This section must be addressed head-on in the paper. Do not let a reviewer discover these before you do.

### 2.1 Volatility Surface Transfer (Finance Domain)

A documented negative result exists in the options pricing / volatility surface domain:

- A deep learning model trained on synthetic Heston-model data (calibrated stochastic volatility model) for zero-shot transfer to real SPY volatility surface data **produced errors worse than training from scratch on real data alone**. Fine-tuning on a small real-data batch was necessary to recover performance.

**Paper's obligation**: Address why portfolio allocation is a different (and potentially more favorable) setting for zero-shot transfer than volatility surface estimation. Possible arguments:
1. Allocation is a coarser decision (10 weights vs. continuous surface), so the sim-to-real gap may be smaller.
2. The G6 generator's stylized facts (GARCH + fat tails + jumps) may better replicate allocation-relevant market structure than a Heston model replicates vol surface dynamics.
3. Empirical evidence: the generator validation report (`scripts/generator_validation.py`) showing the fit of G6 to real data.

### 2.2 Robotics Sim-to-Real Literature (Parallel Field)

Sim-to-real transfer failures are well-documented in robotics RL:

- **Tobin et al. (2017) — "Domain Randomization"**: Randomizing simulator parameters broadly enough can bridge the sim-to-real gap — a technique directly analogous to RAI's multi-level generator G0-G6.
- **Peng et al. (2018) — "Sim-to-Real Transfer of Robotic Control"**: Shows that naive sim-to-real transfer fails without domain adaptation; domain randomization helps significantly.

**Implication for RAI**: The G0→G6 ablation ladder can be framed as explicit domain randomization — the key mechanism that makes sim-to-real transfer possible in robotics. This is a strong framing connection.

---

## 3. Financial RL Baselines the Paper Must Include

Beyond the current baselines (LSTM, XGBoost, Risk Parity, Momentum, SMA, 60/40, Asset-0 B&H, and — as
of 2026-08-30 — a real fixed-SPY buy-and-hold; note the arm listed here as "SPY" before that date was
column 0 of each universe, see [`consolidation_report.md` §20/§21](consolidation_report.md)):

| Missing Baseline | Why Needed |
|---|---|
| **Real-Data-Trained PPO (same architecture)** | Already in `honest_benchmark.py` — must appear in main results table. This is the most critical comparison: same network, same PPO, different training data source. |
| **Equal-Weight (1/N)** | Already present. Keep. |
| **Min-Variance Portfolio** | Classical Markowitz minimum-variance; frequently outperforms both EW and momentum in risk-adjusted terms. |
| **At least one prior synthetic-RL paper** | Even a re-implementation of Hambly et al. (2023) synthetic pre-training would strengthen the related-work claim. Without it, the "synthetic training" claim is not positioned against the prior art. |

---

## 4. Key Reviewer Objections to Pre-Empt

| Objection | Pre-emption Strategy |
|---|---|
| "0% real data is overclaimed — your generator parameters come from real-market empirical literature" | State explicitly: "Zero historical price series touched the policy gradient updates during training; however, the G0-G6 generator family is parametrized by stylized facts established in empirical quantitative finance literature. We define 'real-data-free training' in this precise sense." |
| "Negative result on zero-shot transfer in finance already exists (Heston vol surface)" | Cite it directly. Explain why portfolio allocation is a different setting. Show G6 validation metrics. |
| "Architecture is not novel" | Agree in the paper: "The Conv1D + Transformer architecture is deliberately standard; the novelty is the training paradigm, not the network design." |
| "Survivorship bias in mega-cap universe" | Fixed by point-in-time 2015 S&P 500 top-10. State this explicitly in the experimental setup section. |
| "Single-seed results are uninformative" | Fixed by 10-seed ensemble with 95% CI. State seed count and CI methodology in every table caption. |

---

## 5. Recommended Citation Format (to complete)

```
@article{wiese2020quantgan,
  title={Quant GANs: deep generation of financial time series},
  author={Wiese, Magnus and Knobloch, Robert and Korn, Ralf and Kretschmer, Peter},
  journal={Quantitative Finance},
  volume={20}, number={9}, pages={1419--1440},
  year={2020}
}

@inproceedings{tobin2017domain,
  title={Domain randomization for transferring deep neural networks from simulation to the real world},
  author={Tobin, Josh and others},
  booktitle={IROS},
  year={2017}
}
```
*(Complete citations once paper draft begins — P4 only after P0-P2 resolved.)*
