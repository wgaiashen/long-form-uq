#!/usr/bin/env python
"""Which datasets have a scored-span contract that its quality label and its features agree on.

The corrected-span work fixed one dataset, MedQuAD, where the two disagreed: a re-judging pass had
written a clean-answer-span label and PROMOTED it over the canonical one, while every uncertainty
feature still read the full raw generation. Labels described one span, scores described a longer one.

The span module also carries never-applied rules for two other datasets, which raises the obvious
question of whether the same defect is latent there. That question is answered by evidence, not by
the existence of a rule: a rule that was never applied to EITHER the labels or the features leaves
the two describing the same text, which is consistent, merely on a different span contract from the
corrected dataset's.

Per dataset, this reports:
  the quality-label field the ladder reads
  whether a separate clean-span label was ever produced (`correctness_clean`)
  whether the canonical label equals it, which is the promotion signature
  whether the retained text differs from the saved generation
  the cut rate the span rule WOULD produce if applied, as a measure of what is at stake

    python scripts/checks/ch6_span_contract_audit.py
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import answer_span as A, cache                        # noqa: E402
from luq.config import Config                                  # noqa: E402
from attn_pool import PROMPT_REGIME                            # noqa: E402
from xl_rungs import label_of                                  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = "meta-llama_Meta-Llama-3.1-8B"
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
OUT = ROOT / "results" / "analysis" / f"ch6_span_contract_audit__{SLUG}.csv"


def audit(dataset, span_version):
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID",
                 prompt_regime=PROMPT_REGIME.get(dataset, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, dataset, "ID"))
    lf = label_of(dataset)

    has_clean = sum(1 for r in recs if isinstance(r.get("correctness_clean"), (int, float)))
    promoted = sum(1 for r in recs
                   if isinstance(r.get("correctness_clean"), (int, float))
                   and r.get(lf) == r.get("correctness_clean"))
    already_cut = sum(1 for r in recs if r.get("trunc_reason"))

    # answer_span returns (clean_text, cut_char, reason). A row counts as cut only when the retained
    # text is genuinely shorter than the generation -- a trailing-whitespace rstrip is not a cut, and
    # the pubmed echo marker is a FLAG the module explicitly says is never a cut.
    would_cut, flag_only = 0, 0
    for r in recs:
        text = r.get("gen_text", "")
        clean, _cut, reason = A.answer_span(text, dataset, context=r.get("prompt"),
                                            version=span_version)
        if len(clean) < len(text.rstrip()):
            would_cut += 1
        elif "ECHO-FLAG" in (reason or ""):
            flag_only += 1

    return {
        "dataset": dataset,
        "n_rows": len(recs),
        "label_field": lf,
        "clean_span_label_exists": has_clean > 0,
        "n_clean_span_label": has_clean,
        "n_label_equals_clean": promoted,
        "label_promoted_to_clean_span": has_clean > 0 and promoted == len(recs),
        "n_rows_already_cut_in_this_cache": already_cut,
        "n_rows_the_span_rule_would_cut": would_cut,
        "pct_the_span_rule_would_cut": round(100.0 * would_cut / max(len(recs), 1), 2),
        "n_rows_flagged_not_cut": flag_only,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--span-version", type=int, default=2)
    args = ap.parse_args()

    print("=" * 108)
    print(f"SCORED-SPAN CONTRACT AUDIT  model={MODEL}  span rule version={args.span_version}")
    print("A dataset is INCONSISTENT only if a clean-span label was promoted over the canonical one")
    print("while the features kept reading the full generation. An unapplied rule is not a defect.")
    print("=" * 108)
    print(f"{'dataset':14s}{'rows':>7s}{'label':>13s}{'clean lbl':>11s}{'promoted':>10s}"
          f"{'already cut':>13s}{'rule would cut':>16s}{'  verdict'}")

    rows = []
    for d in LONG:
        r = audit(d, args.span_version)
        if r["label_promoted_to_clean_span"] and r["n_rows_already_cut_in_this_cache"] == 0:
            verdict = "INCONSISTENT: clean label, raw features"
        elif r["label_promoted_to_clean_span"]:
            verdict = "consistent: clean label, clean features"
        elif r["clean_span_label_exists"]:
            verdict = "clean label exists but was not promoted"
        else:
            verdict = "consistent: raw label, raw features"
        r["verdict"] = verdict
        rows.append(r)
        print(f"{d:14s}{r['n_rows']:>7d}{r['label_field']:>13s}"
              f"{str(r['clean_span_label_exists']):>11s}"
              f"{str(r['label_promoted_to_clean_span']):>10s}"
              f"{r['n_rows_already_cut_in_this_cache']:>13d}"
              f"{r['n_rows_the_span_rule_would_cut']:>9d} ({r['pct_the_span_rule_would_cut']:>4.1f}%)"
              f"  {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    bad = [r["dataset"] for r in rows if r["verdict"].startswith("INCONSISTENT")]
    print("\nDatasets whose label and features describe different text: "
          + (", ".join(bad) if bad else "none"))


if __name__ == "__main__":
    main()
