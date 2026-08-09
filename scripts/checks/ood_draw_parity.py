"""Cross-model OOD-draw parity: do capped source draws select the SAME EXAMPLES for both models?

Why this exists (2026-08-09): Llama's pubmed_qa and xsum records hold the same examples as Qwen's
in a DIFFERENT within-split order (same membership, general permutation — the two oldest Llama
caches predate the current prompt library). `sampled_train_idx` draws by POSITION (a seeded
permutation of positions), so on any rung that draws a capped subset of pubmed_qa/xsum as an OOD
source, the two models train on different subsets of the identical pool. Eval rows and ID train
pools use baked-in splits untouched and are NOT affected.

This script is both the EVIDENCE and the ACCEPTANCE GATE for the fix:
  * run before the draw is made order-independent -> the affected triples FAIL (sets differ);
  * run after -> every capped triple must PASS (drawn example sets identical across models).

Examples are compared by a MODEL-INDEPENDENT content key: sha256(prompt + NUL + gold). The gold
(target) is part of the key, the model's ANSWER never is — an answer-dependent key would differ
across models on every row by construction. Prompt+gold parity across the two models is separately
proven row-by-row by prompt_parity_qwen_llama.py. Drawn pools are compared as MULTISETS (Counter),
not sets: cnn_dailymail carries 35 genuinely duplicated (prompt, gold) pairs and med_quad one
(source-data duplicates, multiplicity 2), and two identical examples are interchangeable as
training data — multiset equality is exactly "the same training data", while set comparison would
either abort on the collision or silently under-count.

Read-only, CPU. Needs both models' record files (run where both exist).

    python scripts/checks/ood_draw_parity.py
    python scripts/checks/ood_draw_parity.py --seeds 1,2,3
"""
import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

import probe_drift_long as P  # noqa: E402
from probe_drift_long.splits import sampled_train_idx  # noqa: E402

REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def load(model, dataset):
    cfg = Config(model_name=model, dataset=dataset, ood_setting="ID",
                 prompt_regime=REGIME.get(dataset, ""))
    return cache.load_records(cfg.cache_dir, cache.run_key(model, dataset, "ID"))


def content_keys(records, key):
    """Model-independent content key per row, in file order — the order the ladder drivers use.
    Duplicates are allowed (see module docstring); comparison downstream is by multiset.

    key="prompt+gold" is the strict identity, but on pubmed_qa/xsum the Llama caches carry a
    PRE-REVISION prompt TEMPLATE (same examples, older wording — verified 2026-08-09, every
    gold-matched row differs in wording while the judge input is identical), so prompt-keyed
    comparison is disjoint across models there BY CONSTRUCTION and cannot certify a draw fix.
    key="gold" identifies an example by its target alone, which is template-invariant — use it
    to verify that a content-keyed draw selects the same EXAMPLES for both models."""
    if key == "prompt+gold":
        return np.array([hashlib.sha256((r["prompt"] + "\x00" + str(r["target"])).encode()).hexdigest()
                         for r in records])
    return np.array([hashlib.sha256(str(r["target"]).encode()).hexdigest() for r in records])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--model-b", default="Qwen/Qwen2.5-14B")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--key", default="prompt+gold", choices=["prompt+gold", "gold"],
                    help="example identity for comparison; use 'gold' to verify a draw fix on the "
                         "template-divergent datasets (see content_keys docstring)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    datasets = list(P.LONG_DATASETS)
    loaded = {}
    for m in (args.model_a, args.model_b):
        for d in datasets:
            try:
                recs = load(m, d)
            except Exception as e:
                print(f"SKIP {d} ({m}): {type(e).__name__}: {e}")
                continue
            loaded[(m, d)] = (np.array([r["split"] for r in recs]), content_keys(recs, args.key))

    cells = P.cells_long({d: None for d in datasets}, datasets)
    print(f"\nA = {args.model_a}\nB = {args.model_b}\nseeds = {seeds}   key = {args.key}\n")
    hdr = f"{'rung':<16}{'eval':<15}{'source':<15}{'cap':>5}{'pool':>6}  verdict (per seed)"
    print(hdr); print("-" * len(hdr))
    n_pass = n_fail = 0
    fail_rows = []
    for tag, X, spec in cells:
        for d, capn in spec:
            if d == X or capn is None:
                continue                                    # ID cell / uncapped: not a draw
            ka = loaded.get((args.model_a, d))
            kb = loaded.get((args.model_b, d))
            if ka is None or kb is None:
                continue
            (sa, keys_a), (sb, keys_b) = ka, kb
            pool = int((sa == "train").sum()) or len(sa)
            if capn >= pool:
                continue                                    # whole pool: order-immune by construction
            verdicts = []
            for seed in seeds:
                da = Counter(keys_a[sampled_train_idx(sa, seed, capn)])
                db = Counter(keys_b[sampled_train_idx(sb, seed, capn)])
                inter = sum((da & db).values())
                verdicts.append("PASS" if da == db else f"DIFF({inter}/{capn})")
            ok = all(v == "PASS" for v in verdicts)
            n_pass += ok; n_fail += (not ok)
            if not ok:
                fail_rows.append((tag, X, d))
            print(f"{tag:<16}{X:<15}{d:<15}{capn:>5}{pool:>6}  {' '.join(verdicts)}")
    print(f"\ncapped triples checked: {n_pass + n_fail}  identical: {n_pass}  differing: {n_fail}")
    if n_fail:
        print("*** DRAW PARITY FAILED — capped OOD source draws select different examples per model ***")
        raise SystemExit(1)
    print("DRAW PARITY OK — every capped OOD source draw selects the identical example set for both models.")


if __name__ == "__main__":
    main()
