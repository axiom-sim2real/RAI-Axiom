# Pre-Publish Repository Hygiene Report

> **Date**: 2026-08-30
> **Scope**: read-only audit of `.gitignore` coverage, leaked local paths, credential patterns,
> bundled data provenance, and the citation block. Produced by
> [`scripts/prepublish_hygiene_scan.py`](../scripts/prepublish_hygiene_scan.py) plus the `git`
> queries recorded below.
> **No git operations were run** — nothing was committed, pushed, staged or deleted.

> ## Audited the pre-publication working copy — most findings are now resolved
>
> This report describes the **private working repository as it stood on 2026-08-30**, before the
> public copy was assembled. It is kept because it is the record of *why* this repository ships what
> it ships — in particular the data-redistribution reasoning in §4.1, which is what
> [`data/README.md`](../data/README.md) implements. Read it as an audit trail, not as a description of
> the repository you are looking at.
>
> | Finding | Status in this public repository |
> |---|---|
> | §1.1 `backups/` not ignored | **resolved** — `backups/` is gitignored and no snapshot was copied |
> | §1.3 scratch files would be staged | **resolved** — `scratch_*.py`, `*.tmp.txt`, `**/.virtual_documents/` ignored; none present |
> | §1.4 389 MB of checkpoints, one file over GitHub's 100 MB limit | **resolved** — only the ten Axiom seed weights ship (11.13 MiB); no `.zip` |
> | §1.5 `data/` ignored, so cited evidence would be missing | **resolved** — the result files are tracked; only vendor price caches are excluded |
> | §1.6 suggested `.gitignore` additions | **applied** — see [`.gitignore`](../.gitignore) |
> | §2 19 `file:///a:/…` absolute paths | **resolved** — rewritten to repo-relative links by `scripts/strip_local_paths.py` |
> | §4.1 yfinance price data needs a rights decision | **decided** — option (a): nothing redistributed, regenerate with `scripts/fetch_pinned_universes.py` |
> | §4.3 `diagnostics/*.png` filenames say `spy` but plot asset-0 | **documented, not renamed** — see [`data/README.md`](../data/README.md) |
> | §4.5 notebooks may carry embedded output | **resolved** — neither notebook is included |
> | §5.2 MIT asserted with no `LICENSE` file | **resolved** — [`LICENSE`](../LICENSE) exists, `Copyright (c) 2026 Jason Pandian and Balamurugan P G` |
> | §5.1 URL placeholder | **resolved** — `https://github.com/axiom-sim2real/RAI-Axiom` in all three sites |
>
> One figure below is wrong and is corrected in [`data/README.md`](../data/README.md): §4.4 puts the
> ten Axiom checkpoints at "~2.9 MB"; the true total is 11.13 MiB. The conclusion — small enough to
> commit — is unaffected.


---

## 1. `.gitignore` Coverage

### 1.1 What the current file covers

`.gitignore` is 40 lines. It covers `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `*.egg`,
`venv/ .venv/ env/ ENV/`, `*.pt`, `*.zip`, `data/`, `covid.csv`, `*.jsonl`, and the usual IDE/OS
entries.

| Requested exclusion | Present? | Pattern |
|---|---|---|
| `__pycache__` | **yes** | `__pycache__/` (line 2) |
| `.egg-info` | **yes** | `*.egg-info/` (line 19) — matches `rai.egg-info/` |
| `venv/` | **yes** | `venv/` (line 24) |
| **`backups/`** | **NO** | *no pattern matches it* |
| Large checkpoint binaries | **yes, by extension** | `*.pt` (line 30) and `*.zip` (line 31) |

### 1.2 A `git check-ignore` false positive worth knowing about

`git check-ignore -q backups/` **exits 0**, which looks like confirmation that the directory is
ignored. It is not. With `-v` the reported match is `.gitignore:35:` — an **empty pattern**, because
line 35 of the file is a blank line. Any argument ending in `/` matches it, including a directory
that does not exist:

```
$ git check-ignore -v zzz_nonexistent/
.gitignore:35:	zzz_nonexistent/
```

The authoritative check is `git status` / `git ls-files --others --exclude-standard`, and both show
`backups/`, `checkpoints/`, `docs/`, `tests/` and `archive/` as **untracked but not ignored**. Do not
rely on `check-ignore` with a trailing slash on this repository.

### 1.3 What `git add -A` would actually stage today

```
files: 604      total: 5.09 MB
```

| Top-level dir | Files | Assessment |
|---|---|---|
| `backups/` | **560** | **Should be excluded.** Timestamped snapshots of prior versions of the same docs and scripts (7 snapshot sets). Publishing them ships every superseded number the reports were written to correct. |
| `scripts/` | 18 | Wanted. |
| `checkpoints/` | 6 | Non-binary residue only (`.pt`/`.zip` are ignored): 2 notebooks, per-seed CSV/JSON. One is Kaggle editor cruft — `checkpoints/kaggle_extract/.virtual_documents/__notebook_source__.ipynb`. |
| `archive/` | 6 | Superseded scripts — deliberate, keep. |
| `docs/` | 5 | Wanted. |
| `(root)` | 5 | Includes two working files that are arguably not publishable: `scratch_lstm_test.py` and `antigravity_rai_upgrade_prompt.md`. |
| `rai/` | 2 | `v6_model.py`, `v7_model.py` — wanted. |
| `tests/` | 1 | Wanted. |

### 1.4 Checkpoint size

```
checkpoints/          389 MB      259 files
  kaggle_extract/rai_master_trained_models.zip     114.87 MB
  axiom_multiseed/axiom_checkpoints.zip             10.21 MB
  120 × rai_v82_*.pt                                 1.46 MB each
all_outputs.zip (repo root)                        229.89 MB
```

**Flagged**: `rai_master_trained_models.zip` at **114.87 MB exceeds GitHub's 100 MB hard per-file
limit** (the warning threshold is 50 MB), and `all_outputs.zip` at **229.89 MB** is more than double
it. Neither would be staged today — `*.zip` and `*.pt` are both ignored — so this is a latent hazard
rather than a live one. It becomes live the moment anyone force-adds a checkpoint or removes the
extension patterns. Total binary weight (622 MB across 291 data files) is well past what belongs in
plain git; a release attachment, Zenodo deposit or Git LFS is the appropriate channel.

Secondary: 120 checkpoint filenames contain emoji (`rai_v82_🇮🇳_Indian_Nifty_50_Equities_seed42.pt`,
`…🇺🇸_US_Tech_&_Benchmark_Index…`). Non-ASCII plus `&` in filenames is a portability problem across
filesystems and tooling. They are ignored by git today, but any archive built from them inherits the
names.

### 1.5 The inverse problem: `data/` is ignored but the docs cite it as evidence

`data/` is excluded wholesale by line 32, and `*.jsonl` by line 34. That means **every result file
the README and the consolidation report link to as evidence is absent from the published
repository** — 37 files, 4.80 MB, including `data/deterministic_baselines_pinned.csv`,
`data/cross_model_significance.json`, `data/axiom_aggregate_stats.json`,
`data/baseline_per_seed_results.csv` and the six `data/pinned_universes/*.csv` price caches that pin
the evaluation windows.

This is a reproducibility problem, not a hygiene one, and it points the opposite way from every other
item in this section: those files need to be **added**, not excluded. See §4 for the redistribution
question that governs which of them can be.

### 1.6 Suggested `.gitignore` additions — **not applied**

No change was made to `.gitignore`; deciding what ships is a publication decision. The additions the
audit supports:

```gitignore
# Timestamped local snapshots -- never publish
backups/

# Kaggle notebook editor residue
**/.virtual_documents/

# Local scratch
scratch_*.py
*.tmp.txt
```

Plus a negation for the result files, if §4 clears them:

```gitignore
!data/*.csv
!data/*.json
!data/pinned_universes/*.csv
```

---

## 2. Local Filesystem Paths and Personal Identifiers

Full-repo scan of `.md .py .toml .txt .json .cfg .ini .yaml .yml .ipynb .sh .bat`, excluding
`venv/ backups/ .git/ __pycache__/ .pytest_cache/ rai.egg-info/`.

**Listed for manual review; nothing was removed.**

| # | File:line | Hit | Note |
|---|---|---|---|
| 1 | [RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:29](history/RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md) | `file:///a:/RAI/scripts/rai_v6_robustness_experiment.py`, `…/synthetic_ablation_ladder.py` | absolute `file://` URI to this workstation's `A:` drive |
| 2 | RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:39 | `file:///a:/RAI/scripts/download_data.py` | same |
| 3 | RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:42 | `file:///a:/RAI/scripts/eval_vs_standard_ai.py`, `…/train_v5_dual_head.py` | same |
| 4 | RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:215-228 | 14 further `file:///a:/RAI/scripts/*.py` links | one per script in the file-inventory table |

**19 hits, all in one untracked file**, all the same class: `file:///a:/RAI/...` links that resolve
only on the authoring machine and render as dead links anywhere else. Repo-relative markdown links
are the fix.

**No hits** for `C:\Users\…`, `/home/…`, `AppData\`, or the workstation account name in any file
under audit. (The scanner's own source matches its own patterns — `scripts/prepublish_hygiene_scan.py`
lines 100/105/116 — which is expected and is the only reason those categories are non-zero in the raw
output.)

`/kaggle/input` and `/kaggle/working` absolute paths: **no hits** in the scanned set.

Two non-path items in the same class, both **now changed** (see §5): `README.md` carried
`https://github.com/PandiaJason/RAI` in the citation block and as the clone URL, and
`RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:20` carried it too. Both are resolved in this public copy —
the README now points at `https://github.com/axiom-sim2real/RAI-Axiom`, and the summary's line was
rewritten when the document moved to
[`docs/history/`](history/RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md), because it named the authoring
workstation's absolute path on the same line.

---

## 3. Credential-Shaped Strings

**Zero hits.** Twelve patterns were tested across the same file set:

`aws_access_key_id` (`AKIA…`/`ASIA…`), `github_token` (`ghp_`/`gho_`/`ghu_`/`ghs_`/`ghr_`),
`anthropic_key` (`sk-ant-…`), `openai_key` (`sk-…`), `slack_token` (`xox[abprs]-…`),
`google_api_key` (`AIza…`), `private_key_block` (`-----BEGIN … PRIVATE KEY-----`), `jwt` (`eyJ….….…`),
`bearer_literal` (`Bearer <20+ chars>`), `assigned_secret` (any `api_key`/`secret`/`token`/`password`/
`credential` assigned a quoted literal of 8+ chars), `kaggle_credentials` (the `kaggle.json`
`{"username":…,"key":…}` shape).

This is consistent with the codebase's data access model: every market-data call is
`yfinance.download(...)`, which is unauthenticated. No `.env`, `.netrc`, `kaggle.json`,
`credentials.json` or `*.pem` file exists in the repository.

---

## 4. Bundled Data Files — Provenance for Redistribution Review

622 MB across 291 files ship on disk. Grouped by provenance, with the redistribution question each
group raises. **None of this was changed; it is a list for a rights check.**

### 4.1 Real market price data — **the one group that needs a rights decision**

| File | MB | Source |
|---|---|---|
| `data/pinned_universes/US_ETFs.csv` | 0.463 | yfinance / Yahoo Finance, `auto_adjust=True` |
| `data/pinned_universes/US_MegaCap_PIT.csv` | 0.466 | same |
| `data/pinned_universes/Global_Indices.csv` | 0.466 | same |
| `data/pinned_universes/Crypto_PIT.csv` | 0.335 | same |
| `data/pinned_universes/Forex_Commodities.csv` | 0.232 | same |
| `data/pinned_universes/India_Nifty_50.csv` | 0.222 | same |
| `data/pinned_universes/_spy_reference.csv` | 0.073 | same — 2548 SPY closes, 2016-07-01 → 2026-08-20 (§21) |
| `data/real_market_checkpoints/train_prices.csv` | 0.467 | same, 2010-2019 window (`scripts/download_data.py`) |
| `data/real_market_checkpoints/test_prices.csv` | 0.205 | same, 2020-2024 window |

**All nine are derived from Yahoo Finance via `yfinance`.** Yahoo's terms permit personal use and
restrict redistribution of the underlying quote data; `yfinance` itself carries no data licence and
its own README states it is for research/education and is not affiliated with Yahoo. **Redistributing
these caches is the single item in this audit most likely to need legal review.** Practical options,
in decreasing order of caution: (a) ship none of them and rely on the download path, accepting that
`period=`-relative reproducibility drifts; (b) ship only the derived returns/metrics, not price
levels; (c) ship the caches with an explicit provenance note and accept the risk. Note that (a) has a
real cost here — the pinned windows exist *specifically* to make the Axiom run reproducible, and
`data/axiom_repro_check.csv` is the evidence that they do.

### 4.2 Computed results — repo's own output, no third-party rights

24 files, ~0.66 MB total. Every one is generated by a script in this repository from the §4.1 inputs:
`canonical_results.json`, `deterministic_baselines_pinned.{csv,json}`,
`cross_model_significance.{csv,json}`, `axiom_per_seed_results.csv`,
`axiom_per_universe_summary.json`, `axiom_aggregate_stats.json`, `axiom_repro_check.csv`,
`baseline_per_seed_results.csv`, `baseline_multiseed_summary.json`, `lstm_degeneracy_check.{csv,json}`,
`action_constant_ablation_multiuniverse.{csv,json}`, `crypto_fast_holdout_allocations.json`,
`kaggle_per_seed_results.csv`, `kaggle_per_universe_summary.json`,
`allocation_forensics/forensics_results.json`, `multi_dataset_eval/multi_dataset_eval_results.json`,
`generator_validation/generator_validation_report.json`.

**Redistributable.** These are also exactly the files the docs cite as evidence and that §1.5 shows
are currently excluded.

### 4.3 Plots — repo's own output

`data/diagnostics/allocation_vs_spy{,_2022_bear,_covid_2020}.png` (0.586 MB),
`data/generator_validation/generator_validation_plots.png` (0.139 MB). Own output, redistributable.
Note the filenames say `spy` but predate §20/§21 — they plot allocations against the arm that was
then mislabelled SPY. Worth renaming or captioning before publication.

### 4.4 Model checkpoints — own weights

| Group | Files | MB |
|---|---|---|
| `checkpoints/axiom_multiseed/axiom_seed*.pt` | 10 | ~2.9 |
| `checkpoints/kaggle_import/`, `checkpoints/kaggle_extract/saved_models/` (`rai_v82_*.pt`) | 120 | ~175 |
| `checkpoints/kaggle_extract/rai_master_trained_models.zip` | 1 | 114.87 |
| `checkpoints/axiom_multiseed/axiom_checkpoints.zip` | 1 | 10.21 |
| `data/v0.6_rl_checkpoints/*.pt` | 3 | 0.615 |
| `all_outputs.zip` (root) | 1 | 229.89 |

**Own weights, trained on synthetic data — no third-party data rights attach.** These are purely a
size problem (§1.4), not a licensing one. The Axiom seeds are the reproducibility-critical ones and
are small enough (~2.9 MB for all ten) to publish directly; the 120 v8.2 checkpoints and both zips
belong in a release asset or archive deposit.

### 4.5 Notebooks

`checkpoints/axiom_multiseed/rai-axiom.ipynb` (0.034 MB) and
`checkpoints/kaggle_extract/.virtual_documents/__notebook_source__.ipynb` (0.030 MB). Own work.
**Check both for embedded outputs before publishing** — Kaggle notebooks routinely carry execution
output, and the second is editor residue that should simply be dropped.

---

## 5. Citation Block — Updated

Applied to [`README.md`](../README.md) `## Citation` and, for consistency,
[`pyproject.toml`](../pyproject.toml) `[project].authors`.

**Before** (README):

```bibtex
@article{jason_rai_2026,
  title   = {RAI: Relational Artificial Intelligence from Artificial Worlds},
  author  = {Jason, Pandia},
  year    = {2026},
  url     = {https://github.com/PandiaJason/RAI}
}
```

**After** (README) — equal contribution stated in a BibTeX comment, in a `note` field, and in prose
beneath the block, with the URL replaced by a marked placeholder:

```bibtex
% Jason Pandian and Balamurugan P G contributed equally to this work.
% The author order is alphabetical by surname and does not encode seniority.
% TODO(url): replace the placeholder below once the final repository
% destination is decided; it is not yet the canonical location of this work.
@article{pandian_balamurugan_rai_2026,
  title   = {RAI: Relational Artificial Intelligence from Artificial Worlds},
  author  = {Jason Pandian and Balamurugan P G},
  note    = {Equal contribution},
  year    = {2026},
  url     = {https://example.invalid/PLACEHOLDER-PENDING-REPO-DESTINATION}
}
```

Three notes on the mechanics:

- `author = {Jason Pandian and Balamurugan P G}` is given verbatim as requested. BibTeX will parse
  `Balamurugan P G` as first name `Balamurugan`, middle `P`, surname `G`. If that is wrong, the fix is
  explicit brace-grouping (`{Balamurugan P G}`) or `Last, First` form — a decision for the authors,
  not something to guess at.
- The citation key changed from `jason_rai_2026` to `pandian_balamurugan_rai_2026` to reflect both
  authors. Any existing reference to the old key needs updating; a grep found none in this repo.
- `pyproject.toml` gained the same authors, `license = {text = "MIT"}`, and a placeholder
  `urls.Homepage`, each with a comment pointing at the README block so they cannot drift apart.
  Verified to parse with `tomllib`.

### 5.1 URL placeholder — three sites, all marked

> **Resolved.** All three placeholders were replaced with
> `https://github.com/axiom-sim2real/RAI-Axiom` before the initial public commit. A shared
> organisation owns the repository rather than either author's personal account, matching the
> equal-contribution claim. The table below records the placeholder state this audit found.

| Site | Value |
|---|---|
| `README.md` `## Citation` BibTeX `url` | `https://example.invalid/PLACEHOLDER-PENDING-REPO-DESTINATION` |
| `README.md` Quickstart `git clone` | same, with a comment pointing at the citation block |
| `pyproject.toml` `[project].urls.Homepage` | same, with a `TODO(url)` comment |

`example.invalid` is the RFC 2606 reserved TLD, so the placeholder could never accidentally resolve
to a live host. One site was deliberately **not** changed at audit time:
`RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md:20` named the old GitHub destination. That line was
subsequently neutralised when the document was moved to
[`docs/history/`](history/RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md), because it also carried the
authoring workstation's absolute path.

### 5.2 Also surfaced: the MIT licence is asserted but absent

`README.md` carries an MIT badge and the line "Licensed under the MIT License", and `pyproject.toml`
now declares `license = {text = "MIT"}` — but **there is no `LICENSE` file in the repository**. The
README line now says so explicitly. Add the file before publishing; an MIT assertion with no licence
text and no copyright holder line is not an effective grant.

---

## 6. Summary of Findings

| # | Finding | Severity | Action taken |
|---|---|---|---|
| 1 | `backups/` is **not** gitignored — 560 files, 7 snapshot sets of superseded docs | **high** | reported; `.gitignore` **not** modified |
| 2 | `git check-ignore -q <dir>/` returns a false positive from a blank `.gitignore` line | medium | documented (§1.2) |
| 3 | `rai_master_trained_models.zip` 114.87 MB > GitHub's 100 MB limit; `all_outputs.zip` 229.89 MB | medium (latent) | reported; both currently ignored by `*.zip` |
| 4 | `data/` wholly ignored, so all cited evidence files would be missing from the publish | **high** | reported (§1.5); no change |
| 5 | 19 `file:///a:/RAI/...` absolute paths in `RAI_REPOSITORY_COMPREHENSIVE_SUMMARY.md` | medium | listed for review, not removed |
| 6 | No credential-shaped strings anywhere | — | clean |
| 7 | 9 yfinance-derived price CSVs need a redistribution-rights decision | **high** | flagged (§4.1) |
| 8 | 120 checkpoint filenames contain emoji and `&` | low | flagged |
| 9 | `LICENSE` file missing while MIT is asserted in three places | **high** | README now states it; file not created |
| 10 | Scratch files would be staged (`scratch_lstm_test.py`, `antigravity_rai_upgrade_prompt.md`, `.virtual_documents/`) | low | flagged, `.gitignore` suggestion in §1.6 |
| 11 | Citation block: authors, equal-contribution note, URL placeholder | — | **applied** to `README.md` + `pyproject.toml` |

Items 1, 4, 7, 9 and 10 are all decisions about what ships rather than defects to be fixed, so none
of them were acted on. No `git` command that writes was run at any point in this audit.
