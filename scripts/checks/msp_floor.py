"""The unsupervised OOD floor: is plain MSP actually "still king" under shift?

WHY THIS EXISTS
---------------
The whole weighted-MSP direction rests on a premise from the North star -- "MSP is the strong OOD
baseline, still king for long-form under shift". But MSP/entropy appear in NONE of the OOD grids
(ood_onegrid / transfer_matrix / pooled_loo list only trained probes). This script closes that gap.

THE KEY PROPERTY (why this is CPU-only and shift-invariant)
-----------------------------------------------------------
An unsupervised score is computed only from the model's own per-token logprobs -- it does NOT train,
so it does not change with the training pool. Its PRR on a dataset's TEST split is therefore the SAME
number in every OOD rung (ID == LOO == DiffTask) for that eval dataset. So one number per (dataset,
variant) is the floor that every trained probe is measured against at every rung. No GPU, no seeds.

LABEL (the bug this file was rewritten to fix)
----------------------------------------------
We MUST score against the SAME label the OOD ladder uses, or the comparison is apples-to-oranges.
The ladder (ood_onegrid.py / aggregation_table.py) reads the bare `correctness` field from the
cached RECORDS -- the judge label (gpt-5 on sciq/trivia, gpt-5-mini on pubmed/xsum). The per-example
results CSVs on disk carry the AlignScore label in their `correctness` column, NOT the judge (a known
gotcha), so an earlier version of this script that read the CSV scored MSP against the wrong label and
got a different, non-comparable floor. We now load the records and use `correctness`, exactly like the
ladder, and recompute MSP from the record's `token_logprobs` so the file is self-contained.

    python scripts/checks/msp_floor.py

Writes results/ood_floor__<model>.csv and prints the floor-vs-supervised verdict per rung.
Point estimates only -- close cells (flagged) need the paired test-set bootstrap once the supervised
per-example OOD scores are persisted.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from luq import cache, msp, results  # noqa: E402
from luq.config import Config  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASETS = ["sciq", "trivia_qa", "pubmed_qa", "xsum"]
# The judge model behind the bare `correctness` field, per dataset (for the stamp), matching the
# ladder / aggregation_table label_model column.
LABEL_MODEL = {"sciq": "gpt-5", "trivia_qa": "gpt-5", "pubmed_qa": "gpt-5-mini", "xsum": "gpt-5-mini"}
# The MSP-family variants that define the "MSP floor" a trained probe must beat.
FLOOR_OF = ["msp_sum", "perplexity"]


def test_records(d):
    """Test-split records for dataset d (same records the ladder trains/evaluates on)."""
    cfg = Config(model_name=MODEL, dataset=d, ood_setting="ID")
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, d, "ID"))
    return [r for r in recs if r["split"] == "test"]


def unsup_prr():
    """PRR of every unsupervised score on each dataset's test split (== the OOD floor), scored
    against the JUDGE label from the records -- apples-to-apples with the ladder."""
    floor = {}
    print("=== Unsupervised PRR on the TEST split (shift-invariant: same for ID/LOO/DiffTask) ===")
    print("=== label = judge `correctness` from records (gpt-5 short / gpt-5-mini long) ===")
    for d in DATASETS:
        recs = test_records(d)
        y = [float(r["correctness"]) for r in recs]
        floor[d] = {}
        print(f"\n{d}  (n_test={len(recs)}, label={LABEL_MODEL[d]}, mean_correct={np.mean(y):.3f})")
        # MSP family, recomputed from the record logprobs (self-contained).
        for name, agg in (("msp_sum", "sum"), ("perplexity", "perplexity"),
                          ("msp_mean", "mean"), ("msp_min", "min")):
            unc = [msp.msp_uncertainty(r["token_logprobs"], agg) for r in recs]
            floor[d][name] = results.prr(y, unc)
            print(f"  {name:12s}  PRR {floor[d][name]:+.4f}")
        # P(True) unsupervised (verdict-token mass), if present on the record.
        if "ptrue_unsup" in recs[0]:
            unc = [float(r["ptrue_unsup"]) for r in recs]
            floor[d]["ptrue_unsup"] = results.prr(y, unc)
            print(f"  {'ptrue_unsup':12s}  PRR {floor[d]['ptrue_unsup']:+.4f}")
    return floor


def ladder():
    """Supervised PRR per (setting, eval, method) from the one-harness OOD grid (also judge-labelled)."""
    grid = {}
    og = ROOT / "results" / f"ood_onegrid__{cache._slug(MODEL)}.csv"
    for r in csv.DictReader(open(og)):
        grid.setdefault((r["setting"], r["eval"]), {})[r["method"]] = float(r["prr_mean"])
    return grid


def main():
    floor = unsup_prr()
    grid = ladder()

    verdicts = []
    print("\n\n=== Supervised best vs the MSP floor, per rung (floor = best of msp_sum/perplexity) ===")
    for (setting, ev) in sorted(k for k in grid if k[1] in DATASETS):
        fl = floor.get(ev, {})
        best_floor = max(fl.get(m, -9.0) for m in FLOOR_OF)
        methods = grid[(setting, ev)]
        best_sup_name, best_sup = max(methods.items(), key=lambda kv: kv[1])
        margin = best_sup - best_floor
        close = abs(margin) < 0.05  # flag: within plausible sampling noise, needs bootstrap
        tag = "sup>floor" if margin > 0 else "FLOOR WINS"
        flag = "  <-- close (needs bootstrap)" if close else ""
        print(f"[{setting:8s}] {ev:10s} floor={best_floor:+.3f} | "
              f"best_sup={best_sup_name}({best_sup:+.3f}) | {tag}{flag}")
        verdicts.append((setting, ev, best_floor, best_sup_name, best_sup, tag, close))

    fcsv = ROOT / "results" / f"ood_floor__{cache._slug(MODEL)}.csv"
    with open(fcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kind", "dataset", "variant_or_setting", "prr_or_bestsup", "method", "verdict", "close"])
        for d in DATASETS:
            for m, prr in floor.get(d, {}).items():
                w.writerow(["floor", d, m, round(prr, 4), "", "", ""])
        for setting, ev, bf, sup_name, sup, tag, close in verdicts:
            w.writerow(["verdict", ev, setting, round(sup, 4), sup_name, tag, close])
            w.writerow(["verdict_floor", ev, setting, round(bf, 4), "msp_family", tag, close])
    print(f"\nwrote {fcsv}")


if __name__ == "__main__":
    main()
