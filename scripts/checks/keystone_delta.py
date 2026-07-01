"""M2: SAPLMA L15 PRR on OLD vs NEW prompts, on the same FREE yardstick, in-process.

Trains the SAPLMA MLP at layer 15 on each namespace's cached features and computes PRR with
results.prr, writing NOTHING (so the frozen old-prompt scores are not disturbed). The delta is the
prompt change alone, because the label function is identical on both sides:
  - AlignScore (correctness_alignscore) for all three datasets;
  - string-match for short-form (OLD keeps it in correctness_strmatch; NEW's 02_label writes it to
    correctness), same deterministic function either way.

Requires the NEW-prompt features+labels to exist in the pdnew namespace first (the generation +
AlignScore + string-match steps of slurm/keystone_newprompts.sbatch).

    python scripts/checks/keystone_delta.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache, probe, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B"

# (dataset, yardstick label, OLD label field, NEW label field)
COMPARISONS = [
    ("sciq",       "alignscore", "correctness_alignscore", "correctness_alignscore"),
    ("trivia_qa",  "alignscore", "correctness_alignscore", "correctness_alignscore"),
    ("pubmed_qa",  "alignscore", "correctness_alignscore", "correctness_alignscore"),
    ("sciq",       "strmatch",   "correctness_strmatch",   "correctness"),
    ("trivia_qa",  "strmatch",   "correctness_strmatch",   "correctness"),
]


def prr_for(model, dataset, regime, label_field, layer):
    """SAPLMA L15 PRR for one namespace, or None if features/labels are missing."""
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    key = cache.run_key(model, dataset, "ID")
    try:
        feat = cache.load_features(cfg.cache_dir, key, "saplma")
        records = cache.load_records(cfg.cache_dir, key)
    except FileNotFoundError:
        return None
    y = np.array([r.get(label_field, np.nan) for r in records], dtype=float)
    split = np.array([r["split"] for r in records])
    tr, te = split == "train", split == "test"
    if np.isnan(y[tr]).any() or np.isnan(y[te]).any():
        return None
    clf = probe.train_probe_mlp(feat[tr, layer, :], y[tr])
    return results.prr(list(y[te]), probe.uncertainty(clf, feat[te, layer, :]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--new-regime", default="pdnew")
    args = ap.parse_args()

    print(f"SAPLMA L{args.layer} PRR -- OLD (frozen) vs NEW ({args.new_regime}) prompts\n")
    print(f"  {'dataset':12s} {'yardstick':11s} {'OLD':>7} {'NEW':>7} {'delta':>7}  note")
    for dataset, name, old_f, new_f in COMPARISONS:
        # pubmed AlignScore is near-dead (short yes/no answers can't entail the abstract; ~82% score
        # ~0), so its delta cannot resolve a prompt change -- do NOT read "cosmetic" from it. The
        # informative pubmed yardstick is the judge (a small mini relabel), only worth it if the
        # trustworthy sciq/trivia string-match deltas move materially.
        note = ""
        if dataset == "pubmed_qa" and name == "alignscore":
            note = "UNINFORMATIVE (near-dead label, needs judge)"
        old = prr_for(args.model, dataset, "", old_f, args.layer)
        new = prr_for(args.model, dataset, args.new_regime, new_f, args.layer)
        if old is None or new is None:
            miss = "OLD" if old is None else "NEW"
            print(f"  {dataset:12s} {name:11s}   ({miss} features/labels missing)  {note}")
            continue
        print(f"  {dataset:12s} {name:11s} {old:+7.3f} {new:+7.3f} {new - old:+7.3f}  {note}")


if __name__ == "__main__":
    main()
