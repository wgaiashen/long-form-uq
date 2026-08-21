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
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}


def records_for(dataset):
    root = ROOT / "cache" / REGIME.get(dataset, "") if dataset in REGIME else ROOT / "cache"
    hits = list((root / "records").glob(f"*__{dataset}__ID.jsonl"))
    if not hits:
        return None
    # GUARD (V1 fix, 2026-07-27): meta-llama/Llama-3.1-8B is the ONLY model. A model-agnostic glob once
    # silently picked the dropped Qwen cache for PART A. Pin to Llama and FAIL LOUD on any ambiguity.
    hits = [h for h in hits if "Meta-Llama-3.1-8B" in h.name]
    if len(hits) != 1:
        raise SystemExit(f"{dataset}: expected exactly ONE Llama record file, got {[h.name for h in hits]}")
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
    ap.add_argument("--i-understand-wrong-population", action="store_true",
                    help="required acknowledgement: this tool reads the standalone RECORDS file, which is a "
                         "DIFFERENT test population from the one the drivers score on (load_per_token + "
                         "eval_split). Its floors DO NOT match the drivers (pubmed: this tool −0.210 vs driver "
                         "+0.371 -- a sign flip). See the project record.1 / XXIV. Do NOT use these numbers to "
                         "judge any method. Only for a rough sanity sweep, and only with this flag.")
    args = ap.parse_args()
    if not args.i_understand_wrong_population:
        raise SystemExit(
            "REFUSING TO RUN. recompute_fair_floors.py computes floors on the RECORDS-file population, NOT the\n"
            "eval_split/pertok population the drivers actually score methods on. The two disagree (pubmed:\n"
            "−0.210 here vs +0.371 in every driver -- a sign flip), so these numbers must NEVER be used to\n"
            "judge a method (that was the PART XX bug). The authoritative floors are `luq.msp.fair_floor` as\n"
            "called inside the drivers (they pass it the eval_split records). If you only want a rough sweep\n"
            "and understand the caveat, pass --i-understand-wrong-population. See the project record.1 / XXIV.")
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
