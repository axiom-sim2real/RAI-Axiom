# `data/` — what ships here and what does not

## Shipped: computed results only

Every file in this directory is **output of this repository's own scripts**. 25 files,
~1.5 MB: 21 CSV/JSON result tables and 4 PNG diagnostic plots.

| File | Produced by | What it holds |
|---|---|---|
| `axiom_per_seed_results.csv` | `scripts/kaggle_axiom_10seed.py` | Axiom v0.9, 6 universes x 10 seeds, per-window metrics |
| `axiom_per_universe_summary.json` | `scripts/kaggle_axiom_10seed.py` | per-universe mean / SD / bootstrap CI |
| `axiom_aggregate_stats.json` | `scripts/aggregate_ci.py` | pooled CI and ICC (0.847 OOS / 0.953 holdout) |
| `axiom_repro_check.csv` | `scripts/baseline_multiseed.py --repro-only` | local re-evaluation of the 10 checkpoints vs the Kaggle run |
| `baseline_per_seed_results.csv` | `scripts/baseline_multiseed.py` | LSTM and XGBoost baselines over the same 10 seeds |
| `baseline_multiseed_summary.json` | `scripts/baseline_multiseed.py` | baseline aggregates |
| `deterministic_baselines_pinned.{csv,json}` | `scripts/deterministic_baselines_pinned.py` | 8 seedless arms incl. `SPY B&H (fixed reference)` |
| `cross_model_significance.{csv,json}` | `scripts/cross_model_significance.py` | paired-by-universe tests, Wilcoxon, cluster-robust OLS |
| `canonical_results.json` | `scripts/canonical_evaluation.py` | single-seed canonical harness output |
| `lstm_degeneracy_check.{csv,json}` | `scripts/lstm_degeneracy_check.py` | whether the LSTM baseline collapses to a constant allocation |
| `action_constant_ablation_multiuniverse.{csv,json}` | `scripts/action_constant_ablation_multiuniverse.py` | is the policy's action actually state-dependent |
| `kaggle_per_seed_results.csv`, `kaggle_per_universe_summary.json` | `scripts/kaggle_chronological_walk_forward_master_10seed.py` | v8.2 walk-forward run; written as `./per_seed_results.csv` / `./per_universe_summary.json` on Kaggle and renamed on import |
| `crypto_fast_holdout_allocations.json` | `scripts/canonical_evaluation.py` | per-day allocation trace, Crypto holdout |
| `allocation_forensics/forensics_results.json` | `scripts/allocation_forensics.py` | allocation concentration / turnover forensics |
| `generator_validation/generator_validation_report.json` + `_plots.png` | `scripts/generator_validation.py` | do the synthetic generators match real return moments |
| `multi_dataset_eval/multi_dataset_eval_results.json` | `archive/superseded_scripts/eval_multi_dataset_transfer.py` | transfer across datasets |
| `diagnostics/allocation_vs_spy{,_2022_bear,_covid_2020}.png` | `archive/superseded_scripts/allocation_weight_diagnostic.py` | allocation vs the reference arm, full window / 2022 bear / COVID |
| `pinned_universes_manifest.json` | `scripts/fetch_pinned_universes.py --write-manifest` | fingerprints of the price inputs (see below) |

Two of these are produced by scripts that now live in `archive/superseded_scripts/` rather
than `scripts/`. Those scripts are shipped for exactly that reason — every result file
here has its producer in the repository — but they are superseded and should not be
treated as the current harness.

**Caveat on the three `diagnostics/*.png` filenames.** They say `spy`, but they were
generated before the fixed-reference correction and plot the *in-universe asset-0
buy-and-hold* arm, which was mislabelled SPY at the time. Column 0 is never SPY in any
universe (it is `EEM`, `AAPL`, `EEM`, `AXISBANK.NS`, `AUDUSD=X`, `BCH-USD`
respectively). The corrected `SPY B&H (fixed reference)` arm lives in
`deterministic_baselines_pinned.csv`, not in these plots. Filenames were left as-is so
they still match the figure references in the reports.

## Not shipped: raw price data

**No Yahoo Finance price data is redistributed here.** `yfinance` carries no data
licence, and Yahoo's terms permit personal use while restricting redistribution of the
underlying quotes. The following are therefore absent, by decision rather than by
oversight:

- `data/pinned_universes/*.csv` — the 6 universe price windows and the SPY reference
- `data/real_market_checkpoints/{train,test}_prices.csv` — the legacy 2010-2019 /
  2020-2024 windows used by `scripts/download_data.py`

### Regenerating them

```bash
python scripts/fetch_pinned_universes.py           # download, then auto-verify
python scripts/fetch_pinned_universes.py --verify  # check existing caches
```

The window is pinned to explicit dates (`2016-08-20` / `2021-08-20` → `2026-08-20`,
`auto_adjust=True`), and **column order is left exactly as `yfinance` returns it**
— alphabetical by ticker, which is not the order the ticker lists are written in. The
policy's per-asset logits are position-dependent, so re-sorting the columns changes
every number in this directory.

`pinned_universes_manifest.json` records, per file, the row/column count, first and
last date, column order, a raw byte checksum, an LF-normalised checksum, and a float
fingerprint (every value at 6 dp). `--verify` compares against the fingerprint, which
is the only one of the three that is independent of platform and line endings.

Yahoo revises history. If `--verify` reports `DRIFTED` with shape and column order
intact, the vendor changed that window after 2026-08-30; the evaluation will still run
but will not reproduce these files to the last decimal. A `column_order CHANGED` report
is the serious one — results will not reproduce at all.

## Not shipped: model checkpoints beyond the 10 Axiom seeds

`checkpoints/axiom_multiseed/axiom_seed*.pt` (10 files, **11.13 MiB** total, 1.11 MiB
each) are the reproducibility-critical weights and are included. The 120 v8.2
checkpoints, the two archive zips (114.87 MB and 10.21 MB) and
`data/v0.6_rl_checkpoints/*.pt` are not — purely a size decision; they are our own
weights, trained on synthetic data, with no third-party rights attached.

> Correction to `docs/prepublish_hygiene_report.md` §4.4, which put these ten files at
> "~2.9 MB": the actual total is 11,675,669 bytes (11.13 MiB). The conclusion is
> unaffected — they are still small enough to commit directly — but the figure was wrong.
