"""The Long->Short rung — the 2 cells per method that `pdl_master` silently drops.

THE GAP THIS CLOSES. The ProbeDriftLong grid is **42 cells per method**: 8 long evals x 5 rungs (40) PLUS
`sciq` and `trivia_qa` at the `Long->Short` rung (2). `assemble_pdl_table.py` hardcodes
`LONG_EVALS` (8 long sets) and `RUNGS` (5 long rungs), so those 2 cells are excluded from BOTH the tables
and the coverage denominator -- the master reports "1664/1680" and reads as near-complete while a whole
rung is missing. Recorded as a known gap in the stocktake on 2026-08-05 and not closed until now.

WHY IT MATTERS: transferring a probe trained on long-form generation to a SHORT-form test set is the
setting Joe singled out, and it is the rationale offered for weighted-MSP as the base component of the
proposed system. It cannot be checked from either master table.

⚠️ SEPARATE POPULATION, NEVER POOLED. These are short-form evals with a different label regime and a very
different floor (msp_min is +0.75 here against +0.19 on the long OOD rungs). Averaging them into the
32-cell long OOD mean would be exactly the cross-population comparison the standing rule forbids. They get
their own table and their own caption.

Reads the same `pdl_fam_*` sources as the master assembler. No compute node.
"""
import csv
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925")
SLUG = "meta-llama_Meta-Llama-3.1-8B"
RUNG = "Long->Short"
# All the UNSUPERVISED rows. msp_sum and perplexity are floors too -- an earlier version of this
# script excluded only msp_min and so reported a floor variant as the best "supervised" method.
FLOORS = {"floor_min", "floor_sum", "floor_ppl", "fair_floor"}

# Same spelled-out names the stocktake uses; a code name in a table is unreadable.
PRETTY = {
    "floor_min": "msp_min (floor)", "floor_sum": "msp_sum (floor)", "floor_ppl": "perplexity (floor)",
    "fair_floor": "max-of-three (footnote only)",
    "saplma": "SAPLMA", "attention": "learned attention pooling", "uniform": "mean-pool",
    "wmsp_norm": "weighted MSP (norm)", "wmsp_shrink2": "weighted MSP, shrink-2",
    "wmsp_shrink10": "weighted MSP, shrink-10", "wmsp_blondel": "weighted MSP (Blondel)",
    "wmsp_shrink2_blondel": "weighted MSP, shrink-2 (Blondel)",
    "wmsp_shrink10_blondel": "weighted MSP, shrink-10 (Blondel)",
    "wmsp_seg_flat": "weighted MSP, segment-flat", "wmsp_seg_softmax": "weighted MSP, segment-softmax",
    "ptrue": "P(True) probe", "ptrue_unsup": "P(True)-unsup (Kadavath)", "lookback": "Lookback Lens",
}


def results_dir():
    local = ROOT / "results"
    return local if list(local.glob("pdl_fam_*.csv")) else BASE / "results"


def main():
    rd = results_dir()
    by = {}
    for f in sorted(glob.glob(str(rd / f"pdl_fam_*__{SLUG}.csv"))):
        for r in csv.DictReader(open(f)):
            if r.get("rung") != RUNG or not r.get("prr_mean"):
                continue
            if r["method"].startswith("VERDICT:"):      # margins, not PRRs -- never mixed in
                continue
            try:
                by.setdefault(r["method"], {})[r["eval"]] = float(r["prr_mean"])
            except ValueError:
                continue                                # a NaN cell stays ABSENT, never zero-filled

    evals = sorted({e for d in by.values() for e in d})
    print("=" * 74)
    print(f"THE {RUNG} RUNG — train on the long-form pool, test on a SHORT-form set")
    print(f"population: {len(evals)} short evals x 1 rung, 3 seeds. NOT part of the 32-cell long OOD mean.")
    print("=" * 74)
    print(f"\n{'method':38s}" + "".join(f"{e:>12s}" for e in evals) + f"{'mean':>10s}")
    print("-" * 74)

    def mean_of(d):
        v = [d[e] for e in evals if e in d]
        return sum(v) / len(v) if len(v) == len(evals) else None

    rows = sorted(by.items(), key=lambda kv: -(mean_of(kv[1]) if mean_of(kv[1]) is not None else -9))
    floor = mean_of(by.get("floor_min", {}))
    for m, d in rows:
        mu = mean_of(d)
        cells = "".join(f"{d[e]:+12.4f}" if e in d else f"{'—':>12s}" for e in evals)
        mus = f"{mu:+10.4f}" if mu is not None else f"{'(partial)':>10s}"
        print(f"{PRETTY.get(m, m):38s}{cells}{mus}")

    if floor is None:
        print("\n(floor absent — cannot compute margins)")
        return 0
    print("\n" + "=" * 74)
    print("MARGIN AGAINST THE UNSUPERVISED FLOOR (msp_min) — the question that matters here")
    print("=" * 74)
    beat = []
    for m, d in rows:
        mu = mean_of(d)
        if mu is None or m in FLOORS:
            continue
        if mu > floor:
            beat.append((m, mu - floor))
    print(f"  floor (msp_min)                     {floor:+.4f}")
    if beat:
        for m, g in beat:
            print(f"  BEATS THE FLOOR: {PRETTY.get(m, m):30s} {g:+.4f}")
    else:
        best = max((mean_of(d), m) for m, d in rows
                   if mean_of(d) is not None and m not in FLOORS)
        print(f"  ⭐ NO supervised method beats the floor on this rung.")
        print(f"     Best is {PRETTY.get(best[1], best[1])} at {best[0]:+.4f}, i.e. {best[0]-floor:+.4f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
