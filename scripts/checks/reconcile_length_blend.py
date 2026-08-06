"""Reconcile the length-blend "+0.034", which an earlier plan flagged as an unreproducible number.

THE SHORT ANSWER: it is not a bug, it is a COMPARATOR MISMATCH. Two correct numbers were being compared
as if they answered the same question.

  worklog.md:913   "+0.034"  = blend vs the ROUTER'S OWN pooler column (`always_pooler`)
  master table     "+0.007"  = blend vs SAPLMA

Both reproduce here. The rule this violates is the project's own: a margin quoted without naming what it
is against is not a result. Every margin printed below carries its comparator.

⚠️ AND A SECOND FINDING, which is why "+0.034 vs the attention pooler" still may not be quoted. The
router's `always_pooler` sits systematically BELOW the canonical 3-seed attention pooler on the same
cells -- not by seed noise, but on 24 of 31 cells, which is a one-sided pattern a coin flip does not
produce. So the +0.034 is measured against a weaker pooler than the one the STOCKTAKE reports, and the
honest margin against the canonical pooler is smaller. The cause is narrowed but NOT closed here; see
the two surviving hypotheses printed at the end.

Reads only saved CSVs. No compute node, no training.
"""
import csv
import glob
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# `results/` is NOT duplicated into a git worktree -- it is gitignored and lives in the main checkout
# only. Resolving it relative to __file__ alone silently finds nothing when this runs from a worktree,
# so fall back to the canonical path. Same trap as the attention sidecars, which are also base-only.
BASE = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925")


def results_dir():
    local = ROOT / "results"
    if (local / f"router_pdl__meta-llama_Meta-Llama-3.1-8B.csv").exists():
        return local
    return BASE / "results"


SLUG = "meta-llama_Meta-Llama-3.1-8B"
OOD = ["SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
DATASETS = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]

# The STOCKTAKE's 32-cell OOD column, transcribed with its source so the comparison is auditable.
STOCKTAKE_OOD = {"SAPLMA": 0.2412, "armA_attention": 0.2225, "msp_min": 0.1855}
STOCKTAKE_SRC = "STOCKTAKE_post31July.md, table 'OOD mean (32)'"


def load_router():
    f = results_dir() / f"router_pdl__{SLUG}.csv"
    rows = list(csv.DictReader(open(f)))
    return {(r["eval"], r["rung"]): r for r in rows}


def load_canonical_attention():
    """Per-cell 3-seed attention PRR from the pdl_fam family files -- the canonical arm A."""
    out = {}
    for ds in DATASETS:
        f = results_dir() / f"pdl_fam_{ds}__{SLUG}.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            if r["rung"] in OOD and r["method"] == "attention" and r.get("prr_mean"):
                out[(ds, r["rung"])] = float(r["prr_mean"])
    return out


def sign_test(n_low, n):
    """Exact two-sided sign test. n=31 is small enough to enumerate, so no normal approximation.

    Binomial coefficients built from factorials rather than math.comb: the system python here predates
    comb, and this must run wherever the CSVs are, not only on a new interpreter.
    """
    from math import factorial as fac
    tail = sum(fac(n) // (fac(k) * fac(n - k)) for k in range(n_low, n + 1)) / float(2 ** n)
    return min(1.0, 2 * tail)


def main():
    rt = load_router()
    can = load_canonical_attention()
    if not rt:
        raise SystemExit("router_pdl CSV not found")

    print("=" * 78)
    print("POPULATION CHECK — the two tables must be on the same cells before any margin is quoted")
    print("=" * 78)
    floor_rt = st.mean([float(r["always_floor"]) for r in rt.values()])
    print(f"  router `always_floor`               {floor_rt:+.4f}   ({len(rt)} OOD cells)")
    print(f"  STOCKTAKE `msp_min` OOD mean (32)   {STOCKTAKE_OOD['msp_min']:+.4f}   [{STOCKTAKE_SRC}]")
    agree = abs(floor_rt - STOCKTAKE_OOD["msp_min"]) < 5e-4
    print(f"  -> {'SAME POPULATION' if agree else 'MISMATCH — refuse to compare'} "
          f"(the floor is training-free, so it can only differ if the CELLS differ)")
    if not agree:
        raise SystemExit("population mismatch: margins against the STOCKTAKE would be cross-population")

    blend = st.mean([float(r["blend"]) for r in rt.values()])
    pooler = st.mean([float(r["always_pooler"]) for r in rt.values()])
    print("\n" + "=" * 78)
    print(f"THE THREE MARGINS — length blend, {len(rt)} OOD cells")
    print("=" * 78)
    print(f"  blend                                    {blend:+.4f}")
    for name, ref in [("router's own pooler (always_pooler)", pooler),
                      ("canonical attention pooler (armA)", STOCKTAKE_OOD["armA_attention"]),
                      ("SAPLMA  <- the bar that matters", STOCKTAKE_OOD["SAPLMA"])]:
        print(f"    vs {name:38s} {ref:+.4f}   margin {blend - ref:+.4f}")
    print("\n  worklog.md:913 quoted +0.034 against the FIRST of these, not SAPLMA. Both numbers were")
    print("  right; they were being compared as if they answered the same question. It is a comparator")
    print("  mismatch, not a provenance bug. Note the worklog's own caveat at the time: the significance")
    print("  rested on 'an optimistic within-cell bootstrap'.")

    common = sorted(set(rt) & set(can))
    if not common:
        print("\n(no canonical per-cell attention rows found — skipping the pooler comparison)")
        return 0
    d = [float(rt[k]["always_pooler"]) - can[k] for k in common]
    n_low = sum(1 for x in d if x < 0)
    p = sign_test(n_low, len(d))
    print("\n" + "=" * 78)
    print("⚠️ WHY '+0.034 vs the attention pooler' STILL MAY NOT BE QUOTED")
    print("=" * 78)
    print(f"  matched cells                     {len(common)}")
    print(f"  canonical attention (3-seed mean)  {st.mean([can[k] for k in common]):+.4f}")
    print(f"  router `always_pooler`             {st.mean([float(rt[k]['always_pooler']) for k in common]):+.4f}")
    print(f"  mean difference                    {st.mean(d):+.4f}   (sd {st.pstdev(d):.4f})")
    print(f"  router LOWER on {n_low}/{len(d)} cells — exact two-sided sign test p = {p:.4f}")
    print()
    if p < 0.05:
        print("  This is ONE-SIDED, so it is not seed-to-seed noise: averaging three seeds instead of")
        print("  one changes the variance, not the expectation, and would land ~50/50. The router is")
        print("  scoring a systematically WEAKER pooler than the STOCKTAKE reports.")
    print("\n  Two surviving hypotheses, neither yet ruled out:")
    print("   (a) TOKEN WINDOW. router_pdl.cell_vectors pools over `states_full[rp]`, the full window,")
    print("       while the trained pooler pools under a MASK. If the two token sets differ the attention")
    print("       is a different distribution -- the same G vs G+1 alignment family that has bitten this")
    print("       project before.")
    print("   (b) SAVED-POOLER CONFIG. The router reads one pickled seed-1 pooler; if it was trained")
    print("       under different settings from probedriftlong's `attention` (which does")
    print("       select_temperature then train_attn(temperature=best_T)), the gap is a config gap.")
    print("       Temperature selection can only help, which matches the one-sided direction.")
    print("\n  The reconstruction ARITHMETIC is not the problem: (a*s).sum()+b is exactly W·(Σ a_i x_i)+b")
    print("  for a linear head, which is verified algebraically.")
    print("\n  Until this is settled, quote the blend against SAPLMA (+%.4f) and against the canonical"
          % (blend - STOCKTAKE_OOD["SAPLMA"]))
    print("  pooler (%+.4f), NOT the +0.034." % (blend - STOCKTAKE_OOD["armA_attention"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
