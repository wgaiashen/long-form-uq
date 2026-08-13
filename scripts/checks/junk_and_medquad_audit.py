"""Recompute the continuation-junk rates (TASK G) and the MedQuAD label-protocol facts
(TASK F) directly from the record caches.

Everything here is read out of `cache/records/*.jsonl` and the committed cut rules
(`luq.template_restart`, `luq.answer_span`) — never from a markdown summary.  The two models
are separate populations and are reported side by side, never pooled.

Two honest limits, both reported rather than worked around:
  * only five datasets have a record cache on RCS for BOTH models (cnn_dailymail, med_quad,
    pubmed_qa, samsum, xsum).  `asqa`, `expertqa` and `factscore` have none on this cluster,
    so their rates cannot be recomputed here and are reported as NOT FOUND.
  * `luq.template_restart` distinguishes TEMPLATE_RESTART (the boundary is derivable from the
    prompt) from PRETRAINING_ARTEFACT (it is not).  The standing is carried through so a table
    can never silently merge the two.

Usage:
    python scripts/checks/junk_and_medquad_audit.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "src"))

from luq import answer_span, template_restart  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORDS = f"{REPO}/cache/records"
MODELS = ["meta-llama_Meta-Llama-3.1-8B", "Qwen_Qwen2.5-14B"]
# Every dataset the report's junk table mentions, so the absent ones are named rather than dropped.
DATASETS = ["cnn_dailymail", "med_quad", "pubmed_qa", "samsum", "xsum",
            "asqa", "expertqa", "factscore"]


def load(model, dataset):
    path = f"{RECORDS}/{model}__{dataset}__ID.jsonl"
    if not os.path.exists(path):
        return None, path
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows, path


def junk_table():
    print(f"\n{'=' * 110}\nTASK G — continuation-junk rates recomputed from the record caches\n{'=' * 110}")
    print(f"{'model':10s} {'dataset':14s} {'n':>5s} {'junk%':>7s} {'standing':22s} "
          f"{'medlen junk':>11s} {'medlen clean':>12s} {'label junk':>10s} {'label clean':>11s} {'span?':>6s}")
    missing = []
    for dataset in DATASETS:
        for model in MODELS:
            rows, path = load(model, dataset)
            if rows is None:
                missing.append(path)
                print(f"{model.split('_')[0][:9]:10s} {dataset:14s} {'—':>5s} "
                      f"{'NOT FOUND ON DISK':>7s}")
                continue
            flags, standings = [], set()
            for r in rows:
                _c, cut, _reason, standing = template_restart.restart_cut(r["gen_text"], dataset)
                flags.append(cut is not None)
                if standing:
                    standings.add(standing)
            flags = np.array(flags)
            # generated length in TOKENS, from the cached ids (not a re-tokenisation)
            lens = np.array([len(r["gen_token_ids"]) for r in rows], float)
            # the label the ladder actually reads
            labs = np.array([r.get("correctness", np.nan) for r in rows], float)
            has_span = dataset in answer_span.DATASETS_WITH_RULES
            j, c = flags, ~flags

            def med(a, m):
                return np.median(a[m]) if m.any() else np.nan

            def mean(a, m):
                v = a[m]
                v = v[~np.isnan(v)]
                return v.mean() if len(v) else np.nan

            print(f"{model.split('_')[0][:9]:10s} {dataset:14s} {len(rows):5d} "
                  f"{100 * flags.mean():6.1f}% {'/'.join(sorted(standings)) or '—':22s} "
                  f"{med(lens, j):11.1f} {med(lens, c):12.1f} "
                  f"{mean(labs, j):10.3f} {mean(labs, c):11.3f} {str(has_span):>6s}")
    if missing:
        print("\nrecord caches absent on RCS (cannot be recomputed here):")
        for p in sorted(set(missing)):
            print("   ", p)


def medquad_protocol():
    print(f"\n{'=' * 110}\nTASK F — MedQuAD label-protocol asymmetry, from the record caches\n{'=' * 110}")
    for model in MODELS:
        rows, path = load(model, "med_quad")
        if rows is None:
            print(f"{model}: NOT FOUND ON DISK ({path})")
            continue
        keys = set(rows[0])
        for r in rows:
            keys &= set(r)
        print(f"\n--- {model}  n={len(rows)}  {os.path.basename(path)} ---")
        print("  correctness-ish fields on every row:",
              sorted(k for k in keys if k.startswith("correctness") or k.startswith("ptrue")))
        print("  judge models:", {k: sorted({r.get(k) for r in rows})
                                  for k in sorted(keys) if k.endswith("_model")})

        # How many rows would the answer_span cut actually change?
        changed, cuts = 0, []
        for r in rows:
            clean, cut_char, _reason = answer_span.answer_span(r["gen_text"], "med_quad")
            if clean != r["gen_text"]:
                changed += 1
                cuts.append(len(r["gen_text"]) - cut_char)
        print(f"  rows the answer_span cut would ALTER: {changed}/{len(rows)} = "
              f"{100 * changed / len(rows):.1f}%   median chars dropped = "
              f"{np.median(cuts) if cuts else float('nan'):.0f}")

        # Is the canonical `correctness` the clean label or the raw one?
        for alt in ("correctness_clean", "correctness_raw"):
            if alt in keys:
                same = sum(1 for r in rows if r["correctness"] == r[alt])
                print(f"  correctness == {alt} on {same}/{len(rows)} rows")
        if {"correctness_raw", "correctness_clean"} <= keys:
            raw = np.array([r["correctness_raw"] for r in rows], float)
            cln = np.array([r["correctness_clean"] for r in rows], float)
            m = ~(np.isnan(raw) | np.isnan(cln))
            print(f"  mean label raw   = {raw[m].mean():.4f}")
            print(f"  mean label clean = {cln[m].mean():.4f}")
            print(f"  mean shift clean - raw = {(cln[m] - raw[m]).mean():+.4f}")
            moved = (raw[m] != cln[m]).sum()
            print(f"  rows whose label actually moved: {moved}/{m.sum()} = "
                  f"{100 * moved / m.sum():.1f}%")


def main():
    junk_table()
    medquad_protocol()


if __name__ == "__main__":
    main()
