# Axiom (formerly RAI v6): An Audit Trail, Not a Highlight Reel

## Why this document exists

This is not a "look how good my numbers are" summary. Several of the numbers below are worse than what was originally published. The point of this document is to show the *process* that found and fixed that — because in research, the ability to locate and honestly report findings that don't favor your own model is a stronger signal of quality than any single performance metric.

Everything here is independently verifiable against the codebase's own scripts, CSVs, and JSON outputs. Nothing is asserted without a script that reproduces it.

---

## The starting point

The original evaluation of the zero-shot sim-to-real portfolio policy (then called "RAI v6") had:

- A single, undisclosed training run reported as *the* result, with no seed count, no cost model stated, and no confidence interval
- A "mega-cap" test universe built from 2026's biggest tech winners (NVIDIA, Meta, Alphabet) — selected with the benefit of hindsight, which inflates results by construction
- A benchmark comparison ("RAI Zero-Shot Advantage") that reported return advantages while quietly omitting the Sharpe ratio in the cases where it didn't favor the model
- No statistical testing anywhere in the pipeline

None of that is unusual for an early-stage research codebase. What follows is the process of finding and correcting it.

---

## What the audit found, in the order it was found

| # | Issue found | Why it mattered | Resolution |
|---|---|---|---|
| 1 | Single-run, undisclosed cost/checkpoint result presented as "the" finding | Cannot be trusted or reproduced by a reviewer | Multi-seed protocol established as the standard going forward |
| 2 | Survivorship bias in the mega-cap test universe | Testing on stocks selected *because* they later became winners inflates results independent of any model skill | Rebuilt using point-in-time (pre-selection) constituent lists, verified against real historical market-cap data — corrected twice after the first attempt still had an unverified entry |
| 3 | "RAI Zero-Shot Advantage" column selectively reported favorable metrics | Presentation bias — a reviewer would catch this immediately | Column removed; every comparison now reports return **and** Sharpe, symmetrically |
| 4 | An "upgrade" report showed a large, unexplained jump in results (+80% relative) with no controlled comparison isolating the cause | A number getting better without a clear reason is exactly what should raise suspicion, not confidence | Flagged and held until later rounds could isolate true causes (see #9 below) |
| 5 | Level 6 synthetic market generator failed 4 of 6 empirical realism checks against real market data (volatility clustering, kurtosis) | The core "realistic simulator" premise was partly unvalidated, not just uncalibrated | Documented as an open limitation rather than papered over; flagged as needing a fix, not just a disclosure |
| 6 | A parallel GitHub codebase (faculty-maintained) reported at least 6 mutually contradictory Sharpe values for "the same" model | Two versions of the truth cannot both be cited in a paper | Full consolidation: backed up both codebases, built a single canonical evaluation harness combining the best of each |
| 7 | Full pairwise comparison across 9 strategies revealed **Risk Parity — a simple volatility-weighting rule with no machine learning at all — had the highest overall win count** | An honest project reports this rather than hiding it | Documented explicitly, not omitted |
| 8 | A checkpoint variant ("Fast") catastrophically concentrated 98.9% of a crypto portfolio into a single altcoin, losing -86.6% in a holdout crash | A silent failure like this, left unremarked in a results table, is a red flag for the whole project's honesty | Root-caused via direct allocation forensics: the cash-buffer mechanism never engaged because a hand-tuned constant was set for a different market regime |
| 9 | A staff-developed alternative architecture (v8.2) was evaluated with proper 10-seed statistical testing | Fair, rigorous comparison to a colleague's parallel work, not a dismissal | Result: a genuine statistical null — v8.2 showed no improvement over baselines. Reported as a real finding, not hidden because it wasn't "our" model |
| 10 | The model was renamed from "RAI v6 Alpha" to **Axiom**, deliberately separating this line of work from the shared repository's version numbering | Prevents future confusion between two diverging codebases under one name | Full rename executed and verified across the entire codebase with zero stray references |
| 11 | The published "Overall" confidence interval was miscomputed — it measured variation *between markets*, not statistical *uncertainty in the mean* | These answer different questions; reporting one as the other overstates confidence | Recomputed properly with bootstrap and mixed-effects methods; the corrected holdout interval **crosses zero**, meaning the result does not confidently generalize past the six markets tested |
| 12 | XGBoost's apparent performance edge over Axiom turned out to be a **zero-transaction-cost artifact** | This one finding reverses a "real-data training beats synthetic training" narrative that had stood since the very first review | Under Axiom's own (realistic) cost model, XGBoost collapses from a leading result to one of the weakest in the table |
| 13 | The LSTM baseline's suspiciously low variance across random seeds was investigated directly, not just noted | A baseline that never actually varies isn't a real comparison | Confirmed via a direct day-by-day allocation dump: the LSTM is a static 80%-invested buy-and-hold that never crosses its own decision threshold — a degenerate policy, not a functioning learned strategy |
| 14 | Two structurally different neural network architectures had been sharing one class name in the code | The original single-seed "+1.17" headline result and the new 10-seed "+0.98" result turned out to be **two different architectures**, not the same model measured more carefully | Both renamed and separated (`AxiomNet` vs `FastTradingNet`); the confound is now explicitly documented rather than hidden inside a misleading "more seeds = lower number" narrative |
| 15 | Pooled significance tests treated 10 correlated training seeds as 10 independent observations | This is textbook pseudoreplication — it makes results look far more statistically certain than they are | Cluster-corrected tests added alongside the original ones. This reversed a conclusion: what had read as "a tie" against the LSTM baseline in the holdout period became a **statistically significant loss**, consistent across all six markets tested |
| 16 | The "SPY buy-and-hold" benchmark row was discovered to hold whichever ticker sorted alphabetically first in each universe — **never actually SPY**, in any of the six markets tested | The project's own central claim ("competitive with baselines, not necessarily beating buy-and-hold") had never actually been tested against the benchmark it named | A genuine, fixed SPY reference was added across all six universes. Result: SPY is the **strongest performer in the holdout period**, beating Axiom by a large margin — not statistically resolvable with only six markets, but too large a gap to describe as a tie |
| 17 | A rule-based moving-average baseline was found to be silently defaulting to buy-and-hold in 5 of 12 test windows due to a missing warm-up period | Another baseline had been running incorrectly without anyone noticing | Fixed with proper historical warm-up data; the corrected version performs worse than the buggy version did, and that's reported directly |

---

## The current, honest state of the results

- **Out-of-sample**: Axiom is statistically indistinguishable from a real-data-trained LSTM and from real SPY buy-and-hold, and significantly ahead of a real-data-trained XGBoost (though that gap shrinks substantially once seed correlation is properly accounted for).
- **Holdout (the genuinely unseen future window)**: Axiom is **behind** a simple Risk Parity rule, behind the LSTM baseline (now confirmed to be a trivial constant strategy) in every market where the difference is statistically significant, and behind real SPY by a large, if not fully statistically resolvable, margin.
- **The result does not generalize with confidence beyond the six markets tested** — this is stated explicitly in the project's own documentation, not left for a reader to discover.
- Two architectures were confounded in the project's history, meaning part of the improvement between an early published result and the current one cannot be cleanly attributed to more rigorous testing versus a genuinely different model.

None of this is a hidden failure. It is the current, fully disclosed, fully reproducible state of the evidence.

---

## What this demonstrates

Any project can publish a favorable number once. What's harder — and what this audit trail actually shows — is a process that:

1. Went looking for its own mistakes rather than waiting to be caught
2. Found real ones, including several that made its own headline result look worse
3. Fixed them with reproducible code and verification, not just re-worded claims
4. Reported the corrected, less flattering picture anyway

That's the actual deliverable here.

---

## Open items (not yet resolved, by design)

- Whether and how the staff-developed v8.2 architecture is credited or included in any resulting paper
- Final authorship and citation framing, given the deliberate divergence between this line of work and the shared repository
- Where the canonical, audited codebase will be hosted once that decision is made
