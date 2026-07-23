"""Post-hoc FAIR-FLOOR corrector for any results CSV — no job re-run required.

WHY THIS WORKS (the point that decides "cancel now vs correct later"): the unsupervised floors are
DETERMINISTIC functions of the cached records. They do not depend on the training pool, the seed, or the
probe, so a floor can always be recomputed after the fact. What a ladder job spends hours on -- training the
probe/pooler/weighter per cell -- is completely unaffected by which floor it was compared against. So a
weak-floor CSV does NOT need re-running: it needs its floor column recomputed, which takes seconds.

The ONE thing this cannot repair is a paired-bootstrap VERDICT against the floor, because that needs the
method's per-example uncertainty vectors and the drivers do not persist them. Method-vs-method verdicts
(blondel vs pairwise, hier vs attention, orgad vs unmasked) are untouched by the floor bug, so in practice
only a floor-vs-method verdict that is actually load-bearing justifies a re-run.

Reports, per (csv, eval): the floor each driver USED, the honest fair floor, and how much every method's
margin was overstated.

    python scripts/checks/recompute_fair_floors.py results/weighted_msp_keep_variants*.csv
    python scripts/checks/recompute_fair_floors.py --all
"""
import argparse
import csv as _csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import msp, results  # noqa: E402

MODEL_SLUG = "meta-llama_Meta-Llama-3.1-8B"
# regime-namespaced sets live under their own cache root
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12"}


def records_for(dataset):
    root = ROOT / "cache" / REGIME.get(dataset, "") if dataset in REGIME else ROOT / "cache"
    hits = list((root / "records").glob(f"*__{dataset}__ID.jsonl"))
    if not hits:
        return None
    return [json.loads(l) for l in open(hits[0])]


def floors_for(dataset, label="correctness"):
    """(floor_prr_by_aggregate, fair_name) on the dataset's labelled test rows, or None."""
    recs = records_for(dataset)
    if recs is None:
        return None
    lab = "faithfulness" if dataset == "expertqa" else label
    te = [r for r in recs if r.get("split") == "test" and isinstance(r.get(lab), (int, float))]
    if len(te) < 50:                      # split-less XL sets: fall back to all labelled rows
        te = [r for r in recs if isinstance(r.get(lab), (int, float))]
    if len(te) < 20:
        return None
    y = np.array([r[lab] for r in te], float)
    prr = {a: results.prr(y, np.array([msp.msp_uncertainty(r["token_logprobs"], a) for r in te]))
           for a in msp.FLOOR_AGGREGATES}
    return prr, max(prr, key=prr.get)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", help="results CSVs to audit")
    ap.add_argument("--all", action="store_true", help="audit every csv in results/")
    args = ap.parse_args()
    files = args.csvs or (sorted(glob.glob(str(ROOT / "results" / "*.csv"))) if args.all else [])
    if not files:
        ap.error("give CSV paths or --all")

    cache_floors = {}
    print(f"{'dataset':14s} {'sum':>8s} {'perplex':>8s} {'min':>8s}   fair floor")
    print("-" * 60)
    for d in ["sciq", "trivia_qa", "pubmed_qa", "xsum", "cnn_dailymail", "med_quad", "samsum",
              "expertqa", "asqa"]:
        f = floors_for(d)
        if f is None:
            continue
        cache_floors[d] = f
        prr, best = f
        star = "" if best == "sum" else "   <- sum was NOT the floor"
        print(f"{d:14s} {prr['sum']:+8.3f} {prr['perplexity']:+8.3f} {prr['min']:+8.3f}   "
              f"{best} {prr[best]:+.3f}{star}")

    print("\n=== per-CSV audit: is its floor column the honest one? ===")
    for path in files:
        rows = list(_csv.DictReader(open(path)))
        if not rows:
            continue
        name = Path(path).name
        evals = sorted({r.get("eval", "") for r in rows if r.get("eval")})
        hits = [e for e in evals if e in cache_floors]
        if not hits:
            continue
        # a driver's reported floor appears either as a `floor` column or as an msp_sum/fair_floor row
        col = "floor" in rows[0]
        print(f"\n{name}")
        for e in hits:
            prr, best = cache_floors[e]
            used = None
            if col:
                vals = {float(r["floor"]) for r in rows if r.get("eval") == e and r.get("floor")}
                used = next(iter(vals)) if len(vals) == 1 else None
            else:
                fr = [r for r in rows if r.get("eval") == e
                      and r.get("method", "").replace("weighted_msp_", "") in ("msp_sum", "fair_floor")]
                if fr:
                    used = float(fr[0]["prr_mean"])
            if used is None:
                print(f"   {e:14s} (no single floor value found)")
                continue
            gap = prr[best] - used
            flag = "OK" if abs(gap) < 1e-3 else f"OVERSTATED by {gap:+.3f}  (used {used:+.3f}, honest {prr[best]:+.3f} = {best})"
            print(f"   {e:14s} {flag}")


if __name__ == "__main__":
    main()
