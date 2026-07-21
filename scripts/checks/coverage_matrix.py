"""Coverage audit (Part B): which (family × dataset × rung) result cells exist in the committed CSVs.

Read-only. Scans results/*.csv for each method family and prints, per dataset, which of the 5 ladder rungs
(ID, SameTask, LOO, OneDatasetDiffTask, DiffTask) are present. Used to (1) find gaps before the fill-in runs
and (2) confirm they are filled afterwards. Pass --backup to scan results/_prehinge_backup instead (the
pre-run snapshot), so the before/after can be diffed.

    python scripts/checks/coverage_matrix.py           # current results/
    python scripts/checks/coverage_matrix.py --backup  # the pre-run snapshot
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "meta-llama_Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum", "expertqa"]
RUNGS = ["ID", "SameTask", "LOO", "OneDatasetDiffTask", "DiffTask"]

# family -> (csv basename, eval-col, rung-col, optional variant-col to list)
FAMILIES = [
    ("all_variants(core)", "weighted_msp_all_variants__{s}.csv", "eval", "rung", "variant"),
    ("all_variants(XL)", "weighted_msp_all_variants_organic_XL__{s}.csv", "eval", "rung", "variant"),
    ("keep_variants", "weighted_msp_keep_variants_organic__{s}.csv", "eval", "rung", "mode"),
    ("orgad_broad_tau0.3", "weighted_msp_orgad_ladder_organic__{s}.csv", "eval", "rung", "method"),
    ("idea2_broad_tau0.3", "idea2_weighted_probe_organic__{s}.csv", "eval", "rung", None),
    ("sar", "weighted_msp_sar__{s}.csv", "eval", "rung", None),
    ("msp_ablations", "msp_ablations__{s}.csv", "eval", "rung", "subset"),
]


def scan(base):
    for name, tmpl, ecol, rcol, vcol in FAMILIES:
        p = base / tmpl.format(s=SLUG)
        print(f"\n=== {name}  [{p.name}] ===")
        if not p.exists():
            print("  (missing)"); continue
        rows = list(csv.DictReader(open(p)))
        variants = sorted({r.get(vcol, "") for r in rows if vcol and r.get(vcol)}) if vcol else []
        if variants:
            print(f"  variants ({vcol}): {variants}")
        present = {d: set() for d in DATASETS}
        for r in rows:
            e = r.get(ecol, ""); rg = r.get(rcol, "")
            if e in present and rg in RUNGS:
                present[e].add(rg)
        hdr = "  dataset".ljust(18) + "".join(f"{rg[:9]:>11}" for rg in RUNGS)
        print(hdr)
        for d in DATASETS:
            if not present[d]:
                continue
            cells = "".join(("  ✓".ljust(11) if rg in present[d] else "  ·".ljust(11)) for rg in RUNGS)
            print(f"  {d:16s}{cells}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()
    base = ROOT / "results" / ("_prehinge_backup" if args.backup else ".")
    print(f"scanning {base}")
    scan(base)
