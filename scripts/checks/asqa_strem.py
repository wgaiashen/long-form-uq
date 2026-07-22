"""ASQA Str-EM coverage cross-check (FREE, no API). For each judge-labelled record, compute Str-EM =
fraction of the question's disambiguated qa_pairs whose ANY gold short answer appears (normalised
substring) in the generation, then report the Str-EM distribution and its correlation with the primary
gpt-5-mini judge `correctness`. Str-EM is a coverage/recall signal; the judge is the primary label —
this is the free cross-check the W8 plan asks for (they measure related-but-different things, so a
moderate, not perfect, correlation is expected and healthy).

    source pbs/_env.sh  # (RCS) or just activate luq on DoC
    python scripts/checks/asqa_strem.py --prompt-regime asqa_rp12
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import asqa, cache            # noqa: E402
from luq.config import Config          # noqa: E402


def _norm(s):
    s = (s or "").lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def str_em(gen, qa_pairs):
    g = _norm(gen)
    hits = n = 0
    for qa in qa_pairs:
        sa = qa.get("short_answers") or []
        if not sa:
            continue
        n += 1
        if any(_norm(a) and _norm(a) in g for a in sa):
            hits += 1
    return hits / n if n else float("nan")


def _spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--prompt-regime", default="asqa_rp12")
    ap.add_argument("--label-field", default="correctness", help="the primary judge label field")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset="asqa", ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, "asqa", args.ood)
    records = cache.load_records(cfg.cache_dir, key)
    src = asqa.load_records()                                  # idx-aligned to the records

    ems, judge, both = [], [], []
    for r in records:
        qa = src[r["idx"]]["qa_pairs"]
        em = str_em(r["gen_text"], qa)
        ems.append(em)
        j = r.get(args.label_field)
        judge.append(j)
        if np.isfinite(em) and isinstance(j, (int, float)):
            both.append((em, j))

    arr = np.array([e for e in ems if np.isfinite(e)])
    print(f"=== ASQA Str-EM (coverage) over {len(arr)} records ===")
    print(f"  mean={arr.mean():.3f} std={arr.std():.3f} frac@0={np.mean(arr==0):.2f} "
          f"frac@1={np.mean(arr==1):.2f} median={np.median(arr):.3f}")
    hist, _ = np.histogram(arr, bins=[0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.001])
    labels = ["=0", "(0,.25)", "[.25,.5)", "[.5,.75)", "[.75,1)", "=1"]
    print("  histogram: " + "  ".join(f"{l}:{c}" for l, c in zip(labels, hist)))

    jj = np.array([j for j in judge if isinstance(j, (int, float))])
    if len(jj):
        print(f"\n=== primary judge `{args.label_field}` over {len(jj)} labelled records ===")
        print(f"  mean={jj.mean():.3f} std={jj.std():.3f}")
    if both:
        e, j = np.array(both).T
        print(f"\n=== Str-EM vs judge correlation (n={len(both)}) ===")
        print(f"  Pearson r = {np.corrcoef(e, j)[0,1]:.3f}   Spearman = {_spearman(e, j):.3f}")
        print("  (moderate positive expected: coverage-recall vs graded-correctness measure related "
              "but not identical things; the judge is the primary label.)")


if __name__ == "__main__":
    main()
