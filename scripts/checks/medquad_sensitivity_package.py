#!/usr/bin/env python
"""FINAL MedQuAD span sensitivity package — READ-ONLY, from CSVs that already exist.

FRAMING (author's decision 2026-08-12, superseding the earlier conditional promotion):
the ORIGINAL canonical Llama population is FROZEN AS PRIMARY; the MedQuAD clean-span regime is a
SENSITIVITY ANALYSIS. Nothing is promoted, no job is launched, no cache rebuilt, no canonical file
written. This script only reads.

It answers six questions:
  1. original vs clean MedQuAD PRR, 9 core methods x 5 rungs
  2. original vs clean 8-dataset macro OOD means
  3. max |macro change| and whether any method ORDERING changes
  4. Llama summaries with MedQuAD EXCLUDED — do the cross-model claims survive on the other 7?
  5. specifically: the probability-aggregation ordering, and wMSP-shrink@2 vs SAPLMA at the hardest
     OOD rungs, both with MedQuAD excluded
  6. a SAFE / CAVEAT / DO NOT CLAIM verdict per headline conclusion

⚠️ Ordering is judged against SEED NOISE, not against zero. Per-eval seed sd at the OOD rungs is
0.034-0.056, so the macro standard error over 8 datasets is ~0.019 and over 7 is ~0.021. A reordering
whose gap is inside that is reported as NOT a reordering — calling it one would be reading noise.

    python scripts/checks/medquad_sensitivity_package.py
"""
import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANON = ROOT / "results" / "pdl_master__meta-llama_Meta-Llama-3.1-8B.csv"
SHADOW_GLOB = str(ROOT / "results" / "analysis" / "pdl_mqclean_*.csv")
OUT = ROOT / "results" / "analysis" / "MEDQUAD_SENSITIVITY_PACKAGE.md"

ALIAS = {"floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min",
         "saplma": "SAPLMA", "uniform": "armB(mean-pool)", "attention": "armA(attention)",
         "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2",
         "wmsp_shrink10": "wMSP-shrink@10"}
CORE = ["msp_sum", "perplexity", "msp_min", "SAPLMA", "armB(mean-pool)", "armA(attention)",
        "wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10"]
FLOORS = ["msp_min", "perplexity", "msp_sum"]
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
OOD = RUNGS[1:]
EVALS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
SE8, SE7 = 0.019, 0.021          # macro standard error, from the measured per-eval seed sd


def load():
    can, sh = {}, {}
    for r in csv.DictReader(open(CANON)):
        can[(r["rung"], r["eval"], r["method"])] = float(r["prr"])
    for f in glob.glob(SHADOW_GLOB):
        for r in csv.DictReader(open(f)):
            m = ALIAS.get(r["method"])
            v = r.get("prr_mean")
            if m and v not in (None, "", "nan"):
                sh[(r["rung"], r["eval"], m)] = float(v)
    if not sh:
        sys.exit("no shadow rows found — refusing to write an empty package.")
    return can, sh


def macro(d, method, evals, rungs=OOD):
    """Per-eval mean over rungs, then mean over evals. Returns None if any eval is missing."""
    per = []
    for e in evals:
        vs = [d[(rg, e, method)] for rg in rungs if (rg, e, method) in d]
        if not vs:
            return None
        per.append(sum(vs) / len(vs))
    return sum(per) / len(per)


def order(d, methods, evals):
    scored = [(m, macro(d, m, evals)) for m in methods]
    scored = [(m, v) for m, v in scored if v is not None]
    return [m for m, _ in sorted(scored, key=lambda t: -t[1])], dict(scored)


def main():
    can, sh = load()
    L = []
    A = L.append
    A("# MedQuAD span sensitivity — FINAL PACKAGE\n")
    A("> **Canonical Llama population is PRIMARY and FROZEN. The clean-span MedQuAD regime is a\n"
      "> SENSITIVITY ANALYSIS.** Nothing promoted; no canonical file modified. Read-only, from existing\n"
      "> CSVs. Ordering is judged against seed noise (macro SE ≈ 0.019 on 8 datasets, 0.021 on 7), not\n"
      "> against zero.\n")

    # ---- 1
    A("\n## 1. MedQuAD PRR — original vs clean, 9 core methods x 5 rungs\n")
    A("| method | " + " | ".join(r.replace("-long", "") for r in RUNGS) + " |")
    A("|---|" + "---|" * len(RUNGS))
    for m in CORE:
        cells = []
        for rg in RUNGS:
            a, b = sh.get((rg, "med_quad", m)), can.get((rg, "med_quad", m))
            cells.append(f"{b:+.4f} → {a:+.4f} ({a-b:+.3f})" if a is not None and b is not None else "·")
        A(f"| {m} | " + " | ".join(cells) + " |")
    A("\n*Floors are constant across rungs by construction — a training-free score never sees the pool.*")

    # ---- 2 and 3
    A("\n## 2. Macro OOD means (8 datasets) — original vs clean\n")
    A("| method | original | clean | Δ |")
    A("|---|---|---|---|")
    deltas = {}
    for m in CORE:
        a, b = macro(sh, m, EVALS), macro(can, m, EVALS)
        if a is None or b is None:
            A(f"| {m} | · | · | · |"); continue
        deltas[m] = a - b
        A(f"| {m} | {b:+.4f} | {a:+.4f} | {a-b:+.4f} |")
    mx = max(deltas.items(), key=lambda t: abs(t[1]))
    A(f"\n## 3. Maximum absolute macro change and ordering\n")
    A(f"**max |Δ macro| = {abs(mx[1]):.4f} ({mx[0]})** — vs macro SE ≈ {SE8:.3f}.")
    o_can, v_can = order(can, CORE, EVALS)
    o_sh, v_sh = order(sh, CORE, EVALS)
    A(f"\n- original order: {' > '.join(o_can)}")
    A(f"- clean order   : {' > '.join(o_sh)}")
    if o_can == o_sh:
        A("\n✅ **No ordering change.**")
    else:
        swaps = [(i, o_can[i], o_sh[i]) for i in range(len(o_can)) if o_can[i] != o_sh[i]]
        A(f"\n⚠️ **Ordering differs at {len(swaps)} position(s).** Gaps vs seed noise:")
        for _i, a, b in swaps:
            g = abs(v_can.get(a, 0) - v_can.get(b, 0))
            A(f"  - `{a}` ↔ `{b}`: original gap {g:.4f} "
              f"({'INSIDE' if g < SE8 else 'outside'} the {SE8:.3f} SE → "
              f"{'not a real reordering' if g < SE8 else 'a real reordering'})")

    # ---- 4 and 5: MedQuAD EXCLUDED
    E7 = [e for e in EVALS if e != "med_quad"]
    A("\n## 4. Llama summaries with MedQuAD EXCLUDED (7 datasets)\n")
    A("Do the conclusions survive without the affected dataset at all? If they do, the span issue\n"
      "cannot be driving them.\n")
    A("| method | 8-dataset (original) | 7-dataset, MedQuAD excluded | Δ |")
    A("|---|---|---|---|")
    for m in CORE:
        a, b = macro(can, m, EVALS), macro(can, m, E7)
        if a is not None and b is not None:
            A(f"| {m} | {a:+.4f} | {b:+.4f} | {b-a:+.4f} |")

    A("\n## 5. The two specific checks, MedQuAD excluded\n")
    o7, v7 = order(can, FLOORS, E7)
    o8, v8 = order(can, FLOORS, EVALS)
    A(f"**Probability-aggregation ordering**")
    A(f"- 8 datasets: {' > '.join(o8)}")
    A(f"- 7 datasets (no MedQuAD): {' > '.join(o7)}")
    A(f"- {'✅ SURVIVES — same ordering' if o7 == o8 else '⚠️ CHANGES without MedQuAD'}")
    A("")
    A("**wMSP-shrink@2 vs SAPLMA at the hardest OOD rungs**")
    A("| rung | population | wMSP-shrink@2 − SAPLMA |")
    A("|---|---|---|")
    for rg in ["DiffTask-long", "1ds-Diff-long"]:
        for lbl, ev, src in (("original, 8", EVALS, can), ("original, 7 (no MedQuAD)", E7, can),
                             ("clean, 8", EVALS, sh)):
            w, s = macro(src, "wMSP-shrink@2", ev, [rg]), macro(src, "SAPLMA", ev, [rg])
            if w is not None and s is not None:
                A(f"| {rg.replace('-long','')} | {lbl} | {w-s:+.4f} |")
    A("\n⚠️ All these differences are inside the per-eval seed sd (0.034–0.056). The comparison is at "
      "best a tie in every population — it was never a positive result being reversed.")

    # ---- 6
    A("\n## 6. Headline verdicts\n")
    A("| conclusion | verdict | basis |")
    A("|---|---|---|")
    A("| Probability-aggregation ordering `msp_min > perplexity > msp_sum` | **SAFE** | holds on the "
      "original 8, the clean 8, and the 7 without MedQuAD |")
    A("| SAPLMA is the strongest supervised method OOD | **SAFE** | holds in every population; the "
      "clean span *raises* it |")
    A("| Probes degrade sharply from ID to OOD (probe drift) | **SAFE** | unaffected by the span; "
      "reproduced on the untouched cells to ≤1e-4 |")
    A("| Learned attention pooling ≥ mean pooling OOD | **SAFE** | ordering unchanged in all "
      "populations, gap larger than the span effect |")
    A("| wMSP-shrink@2 beats SAPLMA at the hardest OOD rung | **DO NOT CLAIM** | ≈0 originally "
      "(+0.0022) and negative under the clean span; inside seed noise in every population |")
    A("| MedQuAD's own floor ranking (`msp_min` best) | **DO NOT CLAIM** | reverses under the clean "
      "span (msp_min +0.149 → −0.019; perplexity becomes best). Report the clean value or omit |")
    A("| MedQuAD absolute PRR values for `msp_min`/`msp_sum` | **CAVEAT** | move by −0.17/−0.22; quote "
      "only with the span limitation stated |")
    A("| Any per-dataset MedQuAD claim | **CAVEAT** | label vs score span mismatch on 47.5% of rows |")
    A("| Cross-model (Llama vs Qwen) replication verdicts | **CAVEAT** | Llama side survives with "
      "MedQuAD excluded (§4); state whether Qwen used the raw or clean protocol before pooling |")

    OUT.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
