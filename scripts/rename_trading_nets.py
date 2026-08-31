"""
One-off refactor: resolve the DeepEndToEndTradingNet naming collision.
=====================================================================
Three structurally distinct networks shared the class name
``DeepEndToEndTradingNet``. Verified against the actual checkpoints:

  AxiomNet  (scripts/kaggle_axiom_10seed.py)
      conv1/conv2 -> transformer -> flatten(embed*history) -> fc_features
      -> actor/critic.  state_dict keys: conv1.*, conv2.*, fc_features.0/2,
      actor, critic.  Matches checkpoints/axiom_multiseed/axiom_seed*.pt
      (25 tensors, 289,527 params).

  FastTradingNet  (scripts/train_v6_fast.py + 5 inlined duplicates)
      conv1d Sequential -> transformer -> mean-pool -> fc_features
      -> actor_head/critic_head.  state_dict keys: conv1d.0/2,
      fc_features.0, actor_head, critic_head.  Matches
      data/v0.6_rl_checkpoints/rai_v6_fast.pt (23 tensors, 51,703 params)
      and data/v0.6_rl_results/seeds/rai_v6_seed_*.pt.

  DeepTransformerTradingNet  (scripts/train_v6_deep_transformer.py)
      history_len=60, embed_dim=128, nhead=4, 2 encoder layers, extra
      Linear(256).  No checkpoint in the repository.

  LegacyV6TradingNet  (rai/learning/v6_model.py)
      flatten-based but named conv1d/fc/actor_head — compatible with
      neither checkpoint family.  Not imported anywhere.

Run once:  venv/Scripts/python.exe scripts/rename_trading_nets.py
Idempotent: re-running is a no-op once no occurrences remain.
"""

import io
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "DeepEndToEndTradingNet"

# file -> new class name for that file's occurrences
TARGETS = {
    # --- Axiom family (conv1/conv2 + flatten) ---
    "scripts/kaggle_axiom_10seed.py": "AxiomNet",
    "scripts/baseline_multiseed.py": "AxiomNet",
    "scripts/action_constant_ablation_multiuniverse.py": "AxiomNet",
    # --- Fast family (conv1d Sequential + mean-pool) ---
    "scripts/train_v6_fast.py": "FastTradingNet",
    "scripts/train_v6_alpha.py": "FastTradingNet",
    "scripts/train_v6_pro_growth.py": "FastTradingNet",
    "scripts/action_constant_ablation.py": "FastTradingNet",
    "scripts/canonical_evaluation.py": "FastTradingNet",
    "scripts/allocation_forensics.py": "FastTradingNet",
    "scripts/honest_benchmark.py": "FastTradingNet",
    "scripts/rai_v6_robustness_experiment.py": "FastTradingNet",
    "scripts/real_train_vs_rai_zeroshot.py": "FastTradingNet",
    "scripts/synthetic_ablation_ladder.py": "FastTradingNet",
    "tests/test_rai_core.py": "FastTradingNet",
    # --- structurally distinct singletons ---
    "scripts/train_v6_deep_transformer.py": "DeepTransformerTradingNet",
    "rai/learning/v6_model.py": "LegacyV6TradingNet",
    # --- archived superseded scripts: three import from train_v6_fast (so the
    #     rename would otherwise leave them with a broken import) and
    #     eval_multi_dataset_transfer.py carries its own inlined copy of the
    #     mean-pool topology. All four are the Fast family.
    "archive/superseded_scripts/allocation_weight_diagnostic.py": "FastTradingNet",
    "archive/superseded_scripts/compare_v6_vs_all_models.py": "FastTradingNet",
    "archive/superseded_scripts/cost_sensitivity_sweep.py": "FastTradingNet",
    "archive/superseded_scripts/eval_multi_dataset_transfer.py": "FastTradingNet",
}

# Deliberately NOT renamed:
#   checkpoints/axiom_multiseed/rai-axiom.ipynb  -- verbatim record of the Kaggle
#       run that produced the published checkpoints; disclosed, not rewritten.
#   backups/**                                  -- frozen pre-change snapshots.
#   antigravity_rai_upgrade_prompt.md           -- the user's own input brief.

# Lines that must be rewritten before the blanket rename because they already
# alias the old name to one of the new ones.
PRE_FIXES = {
    "scripts/action_constant_ablation_multiuniverse.py": [
        (
            "from scripts.train_v6_fast import DeepEndToEndTradingNet as FastTradingNet",
            "from scripts.train_v6_fast import FastTradingNet",
        ),
    ],
}


def main() -> int:
    total = 0
    for rel, new in TARGETS.items():
        path = os.path.join(PROJECT_ROOT, rel)
        if not os.path.exists(path):
            print(f"  SKIP (missing) {rel}")
            continue
        with io.open(path, encoding="utf-8") as fh:
            src = fh.read()
        original = src
        for old_line, new_line in PRE_FIXES.get(rel, []):
            src = src.replace(old_line, new_line)
        src, n = re.subn(r"\b%s\b" % OLD, new, src)
        if src == original:
            print(f"  unchanged      {rel}")
            continue
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(src)
        total += n
        print(f"  {n:3d} -> {new:26s} {rel}")
    print(f"\nrewrote {total} occurrences across {len(TARGETS)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
