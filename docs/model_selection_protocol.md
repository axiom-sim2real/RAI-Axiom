# RAI Model Selection Protocol

> [!WARNING]
> **Superseded on 2026-08-29 — this document describes the pre-Axiom-multiseed protocol.**
> The checkpoint it names as primary, `rai_axiom.pt`, was a **`FastTradingNet`** file (51,703
> params, mean-pool), not an Axiom model. It no longer exists under that name: the Fast-arch
> checkpoint that carried the "axiom" label is now
> `data/v0.6_rl_checkpoints/axiom_v0_prototype_fasttradingnet.pt`, and
> `scripts/train_v6_alpha.py` now saves to `rai_v6_alpha.pt`. The currently reported primary
> model is **Axiom v0.9 = `AxiomNet`**, 10 seeds in `checkpoints/axiom_multiseed/axiom_seed*.pt`
> (289,527 params, flatten), evaluated by `scripts/kaggle_axiom_10seed.py`. Every mention of
> `rai_axiom.pt` below is retained as a historical record, not as current practice. See
> `docs/consolidation_report.md` §15 and §19.

## Pre-Registered Primary Model

**Primary reported model: `rai_axiom.pt`** *(historical; see the warning above)*

### Rationale
Selected prior to final evaluation based on:
1. **Architecture rationale**: `alpha` variant uses the same Conv1D+Transformer architecture as `fast` but with a slower, more conservative learning rate, reducing overfit to the synthetic training distribution.
2. **In-distribution validation**: `alpha` was selected on synthetic validation Sharpe, not on real OOS data.
3. **No cherry-picking**: The primary model was chosen before the 2020-2024 OOS period was evaluated.

### All Variants — Reported Side-by-Side

Every evaluation table in this repository reports ALL available variants as separate rows:

| Variant | Checkpoint File | Role |
|---|---|---|
| **Axiom v0.9** | `rai_axiom.pt` | **Primary (pre-registered)** |
| RAI v6 Fast | `rai_v6_fast.pt` | Ablation comparison |
| RAI v6 Pro-Growth | `rai_v6_pro_growth.pt` | Ablation comparison (train if needed) |

### Disclosure Rules (for paper)

1. All variants are trained on **identical synthetic data** (Level 6 generator, same G0-G6 curriculum), with only training hyperparameters varying.
2. The primary model (`alpha`) was designated **before** running any real-data OOS evaluation.
3. If any variant is not reported, an explicit note states why (e.g., checkpoint not available, training not yet completed).
4. **No post-hoc selection**: After OOS results are known, we do not re-designate the primary model.

### Why Not Best-Seed Selection?

Cherry-picking the best single seed from multiple training runs without disclosure is a form of selection bias. Our protocol:
- Reports **ensemble mean ± 95% CI** across all available seeds as the primary number.
- Separately reports the best-seed result, clearly labeled as such.
- States N_seeds in every table caption.
