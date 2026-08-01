"""Where does the error signal live: in the token probabilities, or not?

This is the committed version of the analysis behind the cnn finding (plan item 0.4). It was originally
run as an inline one-off, which meant the numbers were quoted in the stocktake with no way to reproduce
them. ⚠️ **It must be re-run on v2**: cnn and xsum, the two datasets that anchor both ends of the
taxonomy, are both being regenerated.

THE ARGUMENT. The MSP floors are UNSUPERVISED -- they read nothing but the model's own token
probabilities. So the best free floor a dataset admits *is* a measurement of how much of its error
signal is recoverable from probabilities alone. A high floor does not mean the dataset is easy; it means
the signal is in the probabilities, which is precisely where a supervised probe adds least.

The k-sweep then says WHICH probability statistic works, and that distinguishes two ways of having
signal:
  - best k = 1        CONCENTRATED: one bad token carries the error -> an extreme-value statistic
                      (msp_min) wins.
  - best k = all      SPREAD: every token contributes a little -> an averaging statistic (perplexity)
                      wins, and both msp_min and the unnormalised sum fail.
  - floor <= 0 at every k   NOT IN THE PROBABILITIES at all -> this is where the probe earns its keep.

Reads only cached scores. No GPU, no API, no £.

    python scripts/checks/floor_taxonomy.py
    python scripts/checks/floor_taxonomy.py --rung ID --out results/floor_taxonomy_v2.csv
"""
import argparse
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

SLUG = "meta-llama_Meta-Llama-3.1-8B"
FLOORS = ["msp_min", "perplexity", "msp_sum"]


def read_master(path):
    d = {}
    for r in _csv.DictReader(open(path)):
        try:
            d[(r["method"], r["eval"], r["rung"])] = float(r["prr"])
        except (ValueError, KeyError):
            pass
    return d


def read_ksweep(path):
    """best k per dataset from the top-k floor sweep, or {} if the file is absent."""
    if not Path(path).exists():
        return {}
    best = {}
    for r in _csv.DictReader(open(path)):
        ds = r.get("dataset") or r.get("eval")
        try:
            v = float(r.get("prr") or r.get("prr_mean"))
        except (TypeError, ValueError):
            continue
        if ds not in best or v > best[ds][1]:
            best[ds] = (r.get("k"), v)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(ROOT / "results" / f"pdl_master__{SLUG}.csv"))
    ap.add_argument("--ksweep", default=str(ROOT / "results" / f"topk_floor_sweep__{SLUG}.csv"))
    ap.add_argument("--probe", default="SAPLMA", help="supervised method to show alongside")
    ap.add_argument("--rung", default="ID")
    ap.add_argument("--out", default=str(ROOT / "results" / f"floor_taxonomy__{SLUG}.csv"))
    args = ap.parse_args()

    d = read_master(args.master)
    ks = read_ksweep(args.ksweep)
    evals = sorted({e for (_m, e, _g) in d})

    rows = []
    for e in evals:
        fl = {m: d.get((m, e, args.rung)) for m in FLOORS}
        if any(v is None for v in fl.values()):
            print(f"  {e}: missing a floor at rung {args.rung} -> SKIPPED (left out, not zero-filled)")
            continue
        best_m = max(fl, key=lambda k: fl[k])
        spread = max(fl.values()) - min(fl.values())
        bk, _bv = ks.get(e, ("", None))
        # the reading is driven by the best floor and which statistic wins, not by the probe
        if fl[best_m] <= 0.02:
            reading = "NOT-IN-PROBABILITIES"
        elif best_m == "perplexity":
            reading = "SPREAD"
        elif best_m == "msp_min":
            reading = "CONCENTRATED"
        else:
            reading = "mixed"
        rows.append({"eval": e, "rung": args.rung, **{f"floor_{m}": round(fl[m], 4) for m in FLOORS},
                     "best_floor": best_m, "best_floor_prr": round(fl[best_m], 4),
                     "floor_spread": round(spread, 4), "best_k": bk,
                     "probe": round(d.get((args.probe, e, args.rung), float("nan")), 4),
                     "reading": reading})

    rows.sort(key=lambda r: -r["best_floor_prr"])
    hdr = f"{'eval':<15} {'msp_min':>8} {'perplex':>8} {'msp_sum':>8} {'BEST':>8} {'stat':>11} " \
          f"{'spread':>7} {'k':>5} {'probe':>7}  reading"
    print(f"\nrung = {args.rung}   (floors are UNSUPERVISED: a high floor = signal IS in the probabilities)")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['eval']:<15} {r['floor_msp_min']:>8.3f} {r['floor_perplexity']:>8.3f} "
              f"{r['floor_msp_sum']:>8.3f} {r['best_floor_prr']:>8.3f} {r['best_floor']:>11} "
              f"{r['floor_spread']:>7.3f} {str(r['best_k']):>5} {r['probe']:>7.3f}  {r['reading']}")

    with open(args.out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}")
    print("Read: the widest floor_spread with best_floor=perplexity is the clearest SPREAD case; a "
          "best_floor at or below 0 is the clearest case for the probe.")


if __name__ == "__main__":
    main()
