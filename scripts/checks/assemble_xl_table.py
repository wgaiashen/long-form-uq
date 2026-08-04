#!/usr/bin/env python
"""Assemble the ProbeDrift-XL master table — the other half of the deliverable.

WHY THIS EXISTS
---------------
`assemble_pdl_table.py` has built the LONG master table for weeks. The XL grid — 10 evals x 5 rungs —
has had **no assembler and no verifier at all**, so the XL half of the deliverable has never been
readable as one table and its coverage has never been checked.

Deliberately mirrors `assemble_pdl_table.py`: same ALIAS-canonicalisation, same (glob, priority) source
list with lower priority winning a shared cell, same >0.02 cross-source disagreement check, same
CSV+MD emission. Divergence between the two assemblers is itself a bug, so the shapes are kept identical.

⚠️ THE FAMILY SPLIT (2026-08-03) CHANGES THIS GRID. expertqa+factscore are now their own `factuality`
BROAD family and asqa stays in long_qa, so SameTask changes for 5 evals and DiffTask for all 10. Any
CSV produced before that date describes a DIFFERENT rung composition for those cells. The `*_famsplit__`
sources are the post-split runs; older files are read only where they are unaffected.

    python scripts/checks/assemble_xl_table.py
"""
import csv as _csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

RESULTS = ROOT / "results"
EPHEM = Path("/rds/general/ephemeral/user/gs925/ephemeral/luq_overnight_results")
SLUG = "meta-llama_Meta-Llama-3.1-8B"

# All ten. ⚠️ A `cache/records/*` glob silently returns 7 — asqa/expertqa/factscore live in *_rp12
# namespaces — so the eval list is stated explicitly and the realised count is asserted below.
XL_EVALS = ["sciq", "trivia_qa", "pubmed_qa", "med_quad", "asqa",
            "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
RUNGS = ["ID", "SameTask", "LOO", "DiffTask", "OneDatasetDiffTask"]
TARGET_CELLS = len(XL_EVALS) * len(RUNGS)          # 50

# ⚠️ HEADLINE vs APPENDIX (author's decision 2026-08-03). The headline table carries the baselines and
# the contribution; the closed Track B screening arms move to the appendix. They are still COMPUTED and
# still emitted to the CSV — only the rendered headline table is filtered. A method that vanishes from
# the repo cannot be pointed at in a viva, and B.1/B.2/B.3 are cited negative results.
HEADLINE = ["msp_min", "perplexity", "msp_sum",                       # unsupervised floors
            "SAPLMA", "P(True)", "P(True)-unsup", "Lookback Lens",     # supervised baselines
            "armB(mean-pool)", "armA(attention)",                     # aggregation controls
            "wMSP-norm", "wMSP-shrink@2", "wMSP-shrink@10",           # the contribution
            "armC:content-mass", "armC:NLL", "armD:content-mass", "armD:NLL",   # prior-surprisal poolers
            "ens{MSP,SAPLMA}", "ens{wMSP,SAPLMA}", "ens{wMSP,MSP}"]

ALIAS = {
    "floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min", "fair_floor": None,
    "msp_sum": "msp_sum", "perplexity": "perplexity", "msp_min": "msp_min",
    "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2", "wmsp_shrink10": "wMSP-shrink@10",
    "weighted_msp_norm": "wMSP-norm", "weighted_msp_unc": "wMSP-unconstrained",
    "wmsp": "wMSP-norm", "wmsp_blondel": None,
    "saplma": "SAPLMA", "mean-pool+MLP": "SAPLMA",
    # ⭐ THE THREE THE LONG ASSEMBLER DROPS ON THE FLOOR. Without these the supervised baselines the
    # report's central claim is measured against never reach the table.
    # SUPPRESSED 2026-08-04 (author's decision): the `linear` logistic probe is not a baseline the
    # report uses. It is still COMPUTED (a logistic regression on pooled vectors already in memory,
    # seconds per cell) and its rows stay in the CSVs; None routes it through the EXISTING explicit-
    # suppression path, so it is dropped on purpose rather than falling out as an unknown method.
    # ⚠️ This does NOT touch SAPLMA: SAPLMA is `saplma` (long) / `mean-pool+MLP` (XL), both aliased
    # to "SAPLMA" below and above. `linear` is the author's own linear probe on the same features.
    "linear": None,
    "ptrue": "P(True)", "ptrue_accurate": "P(True)",
    # The UNSUPERVISED P(True) (Kadavath): reads the emitted yes/no token, not the hidden
    # state. It is the control for our supervised P(True) probe, so the two must be DISTINCT rows --
    # collapsing them onto one label would hide exactly the comparison they exist to make.
    "ptrue_unsup": "P(True)-unsup",
    "lookback": "Lookback Lens",
    "uniform": "armB(mean-pool)", "armB": "armB(mean-pool)",
    "attention": "armA(attention)", "armA": "armA(attention)",
    "armC_content_mass": "armC:content-mass", "armC_nll": "armC:NLL",
    "armD_content_mass": "armD:content-mass", "armD_nll": "armD:NLL",
    "mh": "multi-head(MH)", "ablation": "multi-head-ablation(ABL)",
    "multimax": "MultiMax", "rolling_w10": "Max-of-Rolling-Means(w10)",
    "rankavg_floor_min+saplma": "ens{MSP,SAPLMA}", "rankavg_wmsp+saplma": "ens{wMSP,SAPLMA}",
    "rankavg_wmsp+floor_min": "ens{wMSP,MSP}",
}

# (glob, priority, seed_regime, value column). Lower priority wins a shared cell. The post-family-split
# runs are priority 0 because they are the only ones whose rung composition matches the current code.
SOURCES = [
    # The per-eval split runs (2026-08-03): one job per eval, because the sequential run could not fit
    # the walltime. These are the authoritative post-family-split numbers.
    ("xlcontrib_fam_*__" + SLUG + ".csv", 0, "3seed", "prr_mean"),
    ("xlonegrid_fam_*__" + SLUG + ".csv", 0, "3seed", "prr_mean"),
    ("contribution_ladder_FULL_famsplit__" + SLUG + ".csv", 0, "3seed", "prr_mean"),
    ("ood_onegrid_FULL_famsplit__" + SLUG + ".csv", 0, "3seed", "prr_mean"),
    ("contribution_ladder_xl__" + SLUG + ".csv", 5, "3seed", "prr_mean"),
    ("canonical_ladder_xl__" + SLUG + ".csv", 5, "3seed", "prr_mean"),
]


def find(glob_pat):
    """UNION both result dirs, dedupe by basename, RESULTS wins a name conflict (same rule as the long
    assembler, which learned it the hard way when EPHEM copies shadowed the real ones)."""
    seen = {}
    for d in (RESULTS, EPHEM):
        for p in sorted(glob.glob(str(d / glob_pat))):
            b = Path(p).name
            if b not in seen:
                seen[b] = p
    return list(seen.values())


def read_rows(path, valcol):
    with open(path) as fh:
        for row in _csv.DictReader(fh):
            m = (row.get("method") or "").strip()
            canon = ALIAS.get(m, "__skip__")
            if canon in (None, "__skip__"):
                continue
            ev = (row.get("eval") or "").strip()
            # ⚠️ ood_onegrid.py names this column `setting`, every other driver names it `rung`. Reading
            # only `rung` would silently drop EVERY supervised-baseline row (linear/ptrue/lookback), which
            # is exactly the comparison the XL table exists to show.
            rg = (row.get("rung") or row.get("setting") or "").strip()
            v = row.get(valcol) or row.get("prr") or ""
            if ev not in XL_EVALS or rg not in RUNGS or v in ("", None):
                continue
            yield rg, ev, canon, float(v), row.get("n_seeds", "")


def main():
    best = {}          # (rung, eval, method) -> (priority, value, n_seeds, seed_regime, source)
    conflicts = []
    for pat, prio, seed, valcol in SOURCES:
        for path in find(pat):
            src = Path(path).name
            for rg, ev, m, v, ns in read_rows(path, valcol):
                key = (rg, ev, m)
                prev = best.get(key)
                if prev is None or prio < prev[0]:
                    best[key] = (prio, v, ns, seed, src)
                elif prio == prev[0] and abs(prev[1] - v) > 0.02:
                    # Same-priority sources must agree. A >0.02 gap means two runs disagree about the
                    # same cell and one of them is wrong -- surfaced, never silently resolved.
                    conflicts.append((key, prev[1], v, prev[4], src))

    if not best:
        sys.exit("no XL rows assembled. Refusing to write an empty master table.")

    # ---- COVERAGE, asserted and reported BEFORE any table is rendered.
    present_cells = {(rg, ev) for (rg, ev, _m) in best}
    missing = [(rg, ev) for ev in XL_EVALS for rg in RUNGS if (rg, ev) not in present_cells]
    methods = sorted({m for (_r, _e, m) in best})
    print(f"XL master | {len(present_cells)}/{TARGET_CELLS} cells present "
          f"({len(XL_EVALS)} evals x {len(RUNGS)} rungs), {len(methods)} methods")
    if missing:
        print(f"⚠️ {len(missing)} CELLS MISSING — named, not summarised:")
        for rg, ev in missing:
            print(f"     {rg:20s} {ev}")
    if conflicts:
        print(f"⚠️ {len(conflicts)} SAME-PRIORITY DISAGREEMENTS >0.02 (one of each pair is wrong):")
        for key, a, b, sa, sb in conflicts:
            print(f"     {key}: {a:+.4f} ({sa}) vs {b:+.4f} ({sb})")

    # ---- CSV: EVERY method, headline and appendix alike.
    out_csv = RESULTS / f"xl_master__{SLUG}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["rung", "eval", "method", "prr", "n_seeds", "seed_regime", "source"])
        for (rg, ev, m), (_p, v, ns, seed, src) in sorted(best.items()):
            w.writerow([rg, ev, m, f"{v:.4f}", ns, seed, src])

    # ---- MD: headline table per rung, then the appendix.
    lines = [f"# ProbeDrift-XL MASTER table  (SLUG={SLUG})", "",
             f"Coverage: {len(present_cells)}/{TARGET_CELLS} target cells "
             f"({len(XL_EVALS)} evals x {len(RUNGS)} rungs).", ""]
    if missing:
        lines += [f"⚠️ **{len(missing)} cells missing:** "
                  + ", ".join(f"{rg}/{ev}" for rg, ev in missing), ""]
    lines += ["⚠️ Population: v1 generations, post-family-split rungs (expertqa+factscore = `factuality` "
              "family, asqa = long_qa). Judge label throughout; expertqa/factscore carry a FACTUALITY "
              "label, a different projection of correctness — never pool their PRR with the rest "
              "unflagged.", ""]

    def render(title, method_list):
        out = [f"## {title}", ""]
        for rg in RUNGS:
            rows = [m for m in method_list if any((rg, ev, m) in best for ev in XL_EVALS)]
            if not rows:
                continue
            out += [f"### rung = {rg}", "",
                    "| method | " + " | ".join(XL_EVALS) + " |",
                    "|---|" + "---|" * len(XL_EVALS)]
            for m in rows:
                cells = []
                for ev in XL_EVALS:
                    hit = best.get((rg, ev, m))
                    # A blank reads as "not measured"; a number reads as "measured". Never confusable.
                    cells.append(f"{hit[1]:+.3f}" if hit else "")
                out.append(f"| {m} | " + " | ".join(cells) + " |")
            out.append("")
        return out

    lines += render("Headline — baselines and contribution", [m for m in HEADLINE if m in methods])
    appendix = [m for m in methods if m not in HEADLINE]
    if appendix:
        lines += ["---", "",
                  "## Appendix — methods screened and closed", "",
                  "These were run and are kept for the record (B.1/B.2/B.3 are cited negative results). "
                  "They are out of the headline table for readability, not because they are hidden.", ""]
        lines += render("Screened arms", appendix)

    out_md = RESULTS / f"xl_master__{SLUG}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_csv}\nwrote {out_md}")
    if missing:
        print("\n⚠️ Grid INCOMPLETE — do not report this as a full XL grid until the cells above exist.")


if __name__ == "__main__":
    main()
