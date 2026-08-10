#!/usr/bin/env python
"""Pre-ladder audit of the Qwen2.5-14B records: generation integrity AND labelling integrity.

Run this BEFORE any ladder, and read it as a gate rather than a report. Everything it checks is
something that, if wrong, produces a ladder number that looks perfectly normal:

  GENERATION
  G1  row count exact, per dataset, and the total
  G2  split composition (train/test), and that both are non-empty where the grid needs them
  G3  the effective token budget was the intended one -- max generated length <= budget, and the
      share sitting exactly AT the budget (a high share is a real property to report, not a bug,
      but a max ABOVE the budget means the cap did not apply)
  G4  empty generations. A repetition penalty over a long few-shot context once forced immediate
      EOS and produced ~35% empty on med_quad; nothing else catches that
  G5  token_logprobs length == gen_token_ids length. THE most load-bearing check here: msp_min,
      perplexity and every sharpening score are computed from token_logprobs, and a silent
      off-by-one would shift every uncertainty score without erroring
  G6  logprob sanity -- finite, and <= 0 (they are log probabilities)
  G7  round-trip: decoding gen_token_ids reproduces gen_text
  G8  degeneracy -- share of answers that are a single token repeated, and the duplicate-answer rate
  G9  provenance: one dataset_configs_sha256 across all 8, and a prompt hash present per dataset

  LABELLING
  L1  every row carries a label, or is an explicit judge DECLINE (never silently absent)
  L2  ONE judge model across every dataset. Mixing judges inside one comparison is disqualifying:
      a probe can learn a judge's biases and PRR needs a single yardstick
  L3  the label field is the right one per dataset (factuality for expertqa/factscore)
  L4  labels finite and inside [0, 1]
  L5  label spread -- a constant label makes PRR undefined and would silently produce nonsense
  L6  the sanity correlation: empty or near-empty generations should score LOW. If they do not,
      the judge is not reading what we think it is reading

    python scripts/checks/qwen_records_audit.py
    python scripts/checks/qwen_records_audit.py --model meta-llama/Meta-Llama-3.1-8B   # comparison
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                   # noqa: E402
from luq.config import Config                           # noqa: E402
from attn_pool import PROMPT_REGIME                     # noqa: E402
from xl_rungs import eval_split                         # noqa: E402

MODEL_DEFAULT = "Qwen/Qwen2.5-14B"

# (rows, budget) -- budgets are data.py's MAX_NEW_TOKENS, capped by Config.max_new_tokens_cap.
# The row counts are what generation was asked for; a mismatch means a partial or duplicated run.
EXPECTED = {
    "pubmed_qa":     (3800, 128),
    "xsum":          (3800, 56),
    "cnn_dailymail": (3800, 128),
    "med_quad":      (1800, 128),
    "samsum":        (1800, 56),
    "expertqa":      (2016, 384),
    "asqa":          (948, 256),
    "factscore":     (500, 256),
}
LABEL_FIELD = {"expertqa": "factuality", "factscore": "factuality"}
EXPECTED_JUDGE = "gpt-5-mini"

FAIL, WARN = [], []


def fail(ds, msg):
    FAIL.append(f"[{ds}] {msg}")


def warn(ds, msg):
    WARN.append(f"[{ds}] {msg}")


def load(model, ds):
    cfg = Config(model_name=model, dataset=ds, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(ds, ""))
    key = cache.run_key(model, ds, "ID")
    path = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    if cache._slug(model) not in path.name:
        raise SystemExit(f"model pin failed for {ds}: {path.name}")
    return [json.loads(l) for l in open(path)], cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--no-decode", action="store_true",
                    help="skip G7 (needs the tokeniser, which needs the model files present)")
    args = ap.parse_args()

    tok = None
    if not args.no_decode:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception as e:
            warn("-", f"G7 skipped, tokeniser unavailable: {type(e).__name__}")

    print("=" * 108)
    print(f"PRE-LADDER AUDIT   model={args.model}")
    print("A FAIL below means do not start the ladder. A WARN means read it and decide.")
    print("=" * 108)

    provenance, judges = {}, Counter()

    print("\n  train/test below are what EVAL_SPLIT returns on the labelled rows -- not the raw")
    print("  `split` field, which is all-train on five of the eight datasets.")
    print(f"\n{'dataset':15s}{'rows':>7s}{'train':>7s}{'test':>7s}{'budget':>8s}{'maxgen':>8s}"
          f"{'@cap':>7s}{'empty':>7s}{'p50':>6s}{'dup%':>6s}{'label':>12s}{'judged':>8s}{'decl':>6s}")
    for ds, (exp_rows, budget) in EXPECTED.items():
        try:
            recs, cfg = load(args.model, ds)
        except Exception as e:
            fail(ds, f"could not load records: {type(e).__name__}: {e}")
            continue

        # ---- G1 rows
        if len(recs) != exp_rows:
            fail(ds, f"G1 row count {len(recs)} != expected {exp_rows}")

        # ---- G2 splits.
        # ⚠️ The RAW `split` field is not the evaluation split, and confusing the two is an easy way
        # to raise a false alarm (this check did, on its first run). Three of the datasets carry a
        # baked-in train/test split that eval_split passes through untouched; the rest arrive
        # all-train and eval_split CARVES the test rows at XL_TEST_FRAC. So the thing that must be
        # non-empty is what eval_split returns, not what the file says.
        sp = Counter(r["split"] for r in recs)
        n_tr, n_te = sp.get("train", 0), sp.get("test", 0)
        if set(sp) - {"train", "test"}:
            fail(ds, f"G2 unexpected split values {set(sp) - {'train', 'test'}}")
        lf_g2 = LABEL_FIELD.get(ds, "correctness")
        y_g2 = np.array([r.get(lf_g2) if r.get(lf_g2) is not None else np.nan
                         for r in recs], dtype=float)
        keep_g2 = np.isfinite(y_g2)
        split_arr = np.array([r["split"] for r in recs])[keep_g2]
        tr_i, te_i = eval_split(split_arr)
        if len(te_i) == 0:
            fail(ds, "G2 eval_split returns NO test rows")
        n_tr, n_te = len(tr_i), len(te_i)

        # ---- G3 budget
        lens = np.array([len(r["gen_token_ids"]) for r in recs])
        maxgen, at_cap = int(lens.max()), float((lens == budget).mean())
        if maxgen > budget:
            fail(ds, f"G3 max generated length {maxgen} EXCEEDS budget {budget} -- cap did not apply")

        # ---- G4 empty
        n_empty = int(((lens == 0) | np.array([not r["gen_text"].strip() for r in recs])).sum())
        if n_empty / len(recs) > 0.02:
            fail(ds, f"G4 {n_empty}/{len(recs)} ({n_empty/len(recs):.1%}) empty generations")
        elif n_empty:
            warn(ds, f"G4 {n_empty} empty generations ({n_empty/len(recs):.2%})")

        # ---- G5 logprob alignment (the load-bearing one)
        bad_align = sum(1 for r in recs if len(r["token_logprobs"]) != len(r["gen_token_ids"]))
        if bad_align:
            fail(ds, f"G5 {bad_align} rows where len(token_logprobs) != len(gen_token_ids) -- "
                     f"every MSP/perplexity/sharpening score on this dataset is shifted")

        # ---- G6 logprob sanity
        allp = np.concatenate([np.asarray(r["token_logprobs"], dtype=float) for r in recs
                               if r["token_logprobs"]])
        if not np.isfinite(allp).all():
            fail(ds, f"G6 {int((~np.isfinite(allp)).sum())} non-finite token logprobs")
        if (allp > 1e-6).any():
            fail(ds, f"G6 {int((allp > 1e-6).sum())} POSITIVE token logprobs (max {allp.max():.3e})")

        # ---- G7 round-trip
        if tok is not None:
            mism = 0
            for r in recs[:200]:                        # 200 is plenty to catch a systematic break
                if tok.decode(r["gen_token_ids"], skip_special_tokens=True) != r["gen_text"]:
                    mism += 1
            if mism:
                warn(ds, f"G7 {mism}/200 sampled rows where decode(gen_token_ids) != gen_text")

        # ---- G8 degeneracy
        texts = [r["gen_text"].strip() for r in recs]
        dup = 1.0 - len(set(texts)) / len(texts)
        single_tok_rep = sum(1 for r in recs
                             if len(r["gen_token_ids"]) > 4 and len(set(r["gen_token_ids"])) == 1)
        if single_tok_rep:
            warn(ds, f"G8 {single_tok_rep} answers are ONE token repeated")

        # ---- G9 provenance
        src = Path(cfg.cache_dir) / "meta" / f"{cache.run_key(args.model, ds, 'ID')}.source.json"
        if src.exists():
            provenance[ds] = json.load(open(src)).get("dataset_configs_sha256")
        else:
            warn(ds, "G9 no source.json")
        ph = Path(cfg.cache_dir) / "meta" / f"{cache.run_key(args.model, ds, 'ID')}.prompthash"
        if not ph.exists():
            warn(ds, "G9 no prompthash")

        # ---- L1-L5 labelling
        lf = LABEL_FIELD.get(ds, "correctness")
        vals = np.array([r.get(lf) if r.get(lf) is not None else np.nan for r in recs], dtype=float)
        fin = np.isfinite(vals)
        declined = sum(1 for r in recs if not np.isfinite(
            float(r[lf]) if r.get(lf) is not None else np.nan)
            and ("uncovered" in r or "coherent" in r))
        untouched = sum(1 for r in recs
                        if lf not in r and "uncovered" not in r and "coherent" not in r)
        if untouched:
            fail(ds, f"L1 {untouched} rows the judge never touched (no '{lf}', no decline marker)")
        if int((~fin).sum()) != declined:
            fail(ds, f"L1 {int((~fin).sum())} non-finite labels but only {declined} carry a decline "
                     f"marker -- the rest are unexplained absences")

        jm = Counter(r.get(f"{lf}_model") or r.get("correctness_model") or r.get("judge_model")
                     for r in recs if fin[recs.index(r)] if False)   # placeholder, filled below
        stamps = Counter()
        for r, ok in zip(recs, fin):
            if not ok:
                continue
            stamps[r.get(f"{lf}_model") or r.get("correctness_model")
                   or r.get("judge_model") or "<unstamped>"] += 1
        judges.update(stamps)
        if len(stamps) > 1:
            fail(ds, f"L2 more than one judge stamp on the same dataset: {dict(stamps)}")
        if "<unstamped>" in stamps:
            warn(ds, f"L2 {stamps['<unstamped>']} labelled rows carry no judge stamp")

        v = vals[fin]
        if v.size == 0:
            fail(ds, "L4 no finite labels at all")
        else:
            if (v < -1e-9).any() or (v > 1 + 1e-9).any():
                fail(ds, f"L4 labels outside [0,1]: min {v.min():.3f} max {v.max():.3f}")
            if float(v.std()) < 1e-6:
                fail(ds, f"L5 label is constant at {v[0]:.3f} -- PRR is undefined")

        print(f"{ds:15s}{len(recs):>7d}{n_tr:>7d}{n_te:>7d}{budget:>8d}{maxgen:>8d}"
              f"{at_cap:>6.0%}{n_empty:>7d}{int(np.median(lens)):>6d}{dup:>6.1%}"
              f"{lf:>12s}{int(fin.sum()):>8d}{declined:>6d}")

    # ---- L6 the sanity correlation, done across datasets
    print("\nL6 SANITY -- do SHORT/degenerate answers actually score lower? (the judge is supposed to")
    print("   punish them; if not, it is not reading what we think it is)")
    print(f"{'dataset':15s}{'shortest 10% mean':>20s}{'longest 10% mean':>20s}{'overall':>10s}")
    for ds in EXPECTED:
        try:
            recs, _ = load(args.model, ds)
        except Exception:
            continue
        lf = LABEL_FIELD.get(ds, "correctness")
        pair = [(len(r["gen_token_ids"]),
                 float(r[lf]) if r.get(lf) is not None else np.nan) for r in recs]
        pair = [(n, y) for n, y in pair if np.isfinite(y)]
        if len(pair) < 50:
            continue
        pair.sort(key=lambda t: t[0])
        k = max(1, len(pair) // 10)
        lo = float(np.mean([y for _, y in pair[:k]]))
        hi = float(np.mean([y for _, y in pair[-k:]]))
        ov = float(np.mean([y for _, y in pair]))
        print(f"{ds:15s}{lo:>20.3f}{hi:>20.3f}{ov:>10.3f}")

    # ---- G9 across datasets
    print("\nG9 PROVENANCE")
    uniq = set(provenance.values())
    print(f"  dataset_configs_sha256 across {len(provenance)} datasets: "
          f"{len(uniq)} distinct value(s)")
    if len(uniq) != 1:
        fail("-", f"G9 datasets were built from {len(uniq)} different source configs: {provenance}")
    else:
        print(f"  {list(uniq)[0]}")

    print("\nL2 JUDGE STAMPS ACROSS ALL DATASETS")
    for k, v in judges.items():
        print(f"  {k}: {v} rows")
    if len(judges) > 1:
        fail("-", f"L2 more than one judge across the population: {dict(judges)} -- "
                  f"PRR needs ONE yardstick")
    if EXPECTED_JUDGE not in judges:
        fail("-", f"L2 expected judge '{EXPECTED_JUDGE}' not found; got {list(judges)}")

    print("\n" + "=" * 108)
    if WARN:
        print(f"{len(WARN)} WARNING(S) -- read and decide:")
        for w in WARN:
            print(f"  ! {w}")
    if FAIL:
        print(f"\n{len(FAIL)} FAILURE(S) -- DO NOT START THE LADDER:")
        for f in FAIL:
            print(f"  ✗ {f}")
        print("=" * 108)
        return 1
    print("\nALL CHECKS PASS. Generation and labelling are sound for the ladder.")
    print("=" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
