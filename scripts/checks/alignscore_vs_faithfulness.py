"""AlignScore-vs-judge read on the labelled ExpertQA set (the 4th gate read, at full scale).

Computes AlignScore(gen, gold) per record and correlates it with the stored three-state judge
`factuality`. The point (Gaia's ask): if a similarity-flavoured metric (AlignScore) and the
factuality judge disagree a lot, that is EVIDENCE the factuality projection differs from
gold-similarity — i.e. why we did not just use AlignScore. Low correlation is a RESULT, not noise.
Also correlates AlignScore with `uncovered` (an answer that diverges from the gold should be both
low-AlignScore and high-uncovered).

Runs on GPU (AlignScore = roberta-large). Reads the labels already in the records; does NOT re-judge.

    python scripts/checks/alignscore_vs_faithfulness.py --prompt-regime expertqa_rp12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import nltk

# AlignScore's sent_tokenize needs punkt_tab; it lives in the project nltk_data (home is over-quota),
# which isn't on NLTK's default search path on a compute node — add it explicitly.
nltk.data.path.insert(0, "/vol/gpudata/gs925-msc_project/nltk_data")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache                        # noqa: E402
from luq.config import Config                # noqa: E402
from luq.labels import alignscore            # noqa: E402


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and a.std() and b.std() else float("nan")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or not a.std() or not b.std():
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--prompt-regime", default="expertqa_rp12")
    ap.add_argument("--out", default="results/expertqa/alignscore_vs_factuality.txt")
    args = ap.parse_args()

    cfg = Config(dataset="expertqa", ood_setting="ID", model_name=args.model, prompt_regime=args.prompt_regime)
    key = cache.run_key(args.model, "expertqa", "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    print(f"scoring AlignScore on {len(recs)} records...", flush=True)

    for i, r in enumerate(recs):
        r["_align"] = alignscore.score(r)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(recs)}", flush=True)

    # factuality vs AlignScore: use records with a DEFINED factuality that were NOT quarantined
    # (quarantined=0.0 is a distrust label, not a judged factuality — including it would conflate).
    fj = [(r["_align"], r["factuality"]) for r in recs
          if r.get("factuality") is not None and not r.get("factuality_quarantined")
          and r.get("_align") is not None]
    au = [(r["_align"], r["uncovered"]) for r in recs
          if r.get("uncovered") is not None and r.get("_align") is not None]

    lines = [f"AlignScore vs judge on {args.prompt_regime}  (n_records={len(recs)})", ""]
    if fj:
        al, fa = zip(*fj)
        lines += [f"AlignScore vs FAITHFULNESS (judged, non-quarantined, n={len(fj)}):",
                  f"  Pearson {pearson(al,fa):.2f}  Spearman {spearman(al,fa):.2f}",
                  f"  AlignScore mean {np.mean(al):.2f} | factuality mean {np.mean(fa):.2f}",
                  "  -> LOW corr = the factuality projection differs from gold-similarity (why not just AlignScore).", ""]
    if au:
        al, uu = zip(*au)
        lines += [f"AlignScore vs UNCOVERED (n={len(au)}): Pearson {pearson(al,uu):.2f}  Spearman {spearman(al,uu):.2f}",
                  "  -> negative expected: more uncovered (diverges from gold) => lower AlignScore."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
