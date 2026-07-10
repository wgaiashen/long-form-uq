"""Assemble the tidy canonical-ladder CSV for slides 8/9 from the already-verified result CSVs.

Deliverable (b) of the 10-July slide task: one long CSV with columns
    eval, rung, method, prr_mean, prr_std, beats_floor
covering the samsum-updated 5-rung ladder. No compute -- it only reads and reshapes existing CSVs:
  * contribution_ladder__*.csv   -> uniform, attention, weighted-MSP norm, MSP floor (msp_sum)
  * weighted_msp_blondel__*.csv  -> weighted-MSP Blondel
  * aggregation_table__*.csv     -> SAPLMA mean-pool/last-token/per-sentence/per-token (ID only)
  * canonical_ladder_regen__*.csv (OPTIONAL) -> SAPLMA aggregators at OOD on the samsum pool, folded in
                                                when that GPU regen has produced it.
beats_floor is computed per (eval, rung) against that cell's msp_sum PRR.

    python scripts/checks/assemble_canonical_csv.py
"""
import csv as _csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Pure stdlib (only reshapes small CSVs), so it runs anywhere without the env. Slug is fixed for the
# one model this project uses (cache._slug turns '/' into '_').
SLUG = "meta-llama_Meta-Llama-3.1-8B"
RES = ROOT / "results"
EVALS = ["sciq", "trivia_qa", "pubmed_qa"]
RUNGS = ["ID", "SameTask", "LOO", "OneDatasetDiffTask", "DiffTask"]
# Canonical display label per source-CSV method name.
LABEL = {"msp_sum": "MSP floor", "weighted_msp_msp_sum": "MSP floor",
         "uniform": "uniform(frozen-q)", "attention": "attention",
         "weighted_msp_norm": "weighted-MSP norm", "weighted_msp_pairwise": "weighted-MSP norm",
         "weighted_msp_blondel": "weighted-MSP Blondel",
         "mean-pool+MLP": "SAPLMA mean-pool", "last-token": "SAPLMA last-token",
         "per-sentence(mean)": "SAPLMA per-sentence", "per-token(mean)": "SAPLMA per-token"}
ROW_ORDER = ["MSP floor", "SAPLMA mean-pool", "SAPLMA last-token", "SAPLMA per-sentence",
             "SAPLMA per-token", "uniform(frozen-q)", "attention",
             "weighted-MSP norm", "weighted-MSP Blondel"]


def read_csv(name):
    p = RES / name
    if not p.exists():
        return []
    with open(p) as f:
        return list(_csv.DictReader(f))


def main():
    # cell[(eval, rung, label)] = (prr_mean, prr_std)
    cell = {}
    floor = {}  # (eval, rung) -> msp_sum prr

    def take(rows, method_field="method", want=None):
        for r in rows:
            m = r.get(method_field, "")
            if m.startswith("VERDICT") or m.startswith("DIAG"):
                continue
            if want is not None and m not in want:
                continue
            ev, rung, lab = r.get("eval"), r.get("rung"), LABEL.get(m)
            if lab is None or ev not in EVALS:
                continue
            mean = float(r["prr_mean"]); std = float(r.get("prr_std") or 0.0)
            cell[(ev, rung, lab)] = (mean, std)
            if m in ("msp_sum", "weighted_msp_msp_sum"):
                floor[(ev, rung)] = mean

    take(read_csv(f"contribution_ladder__{SLUG}.csv"))
    take(read_csv(f"weighted_msp_blondel__{SLUG}.csv"))
    # SAPLMA aggregators: ID from the aggregation table (its column is 'dataset', rung is implicitly ID).
    for r in read_csv(f"aggregation_table__{SLUG}.csv"):
        if r.get("label_field") != "correctness":
            continue
        lab = LABEL.get(r.get("aggregator", ""))
        ev = r.get("dataset")
        if lab is None or ev not in EVALS:
            continue
        cell[(ev, "ID", lab)] = (float(r["prr_mean"]), float(r.get("prr_std") or 0.0))
    # OPTIONAL: SAPLMA aggregator OOD from the samsum regen, folded in if present (overrides nothing else).
    take(read_csv(f"canonical_ladder_regen__{SLUG}.csv"))
    take(read_csv(f"canonical_ladder__{SLUG}.csv"))

    out = RES / f"canonical_ladder_slides__{SLUG}.csv"
    n = 0
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["eval", "rung", "method", "prr_mean", "prr_std", "beats_floor"])
        for ev in EVALS:
            for rung in RUNGS:
                fl = floor.get((ev, rung))
                for lab in ROW_ORDER:
                    if (ev, rung, lab) in cell:
                        mean, std = cell[(ev, rung, lab)]
                        bf = "" if fl is None else str(mean > fl)
                        w.writerow([ev, rung, lab, f"{mean:.4f}", f"{std:.4f}", bf])
                        n += 1
    print(f"wrote {out}  ({n} rows)")
    # quick coverage report so gaps are visible
    for ev in EVALS:
        have = {lab: sum((ev, rg, lab) in cell for rg in RUNGS) for lab in ROW_ORDER}
        print(f"  {ev}: " + ", ".join(f"{lab}={have[lab]}/5" for lab in ROW_ORDER))


if __name__ == "__main__":
    main()
