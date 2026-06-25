"""PRR-impact check: does swapping GPT-5 -> gpt-5-mini change the PRR conclusions?

Label agreement (validate_mini_judge) shows the two judges give similar SCORES. This
goes one step further to the thing we actually report: it recomputes each method's PRR
using GPT-5 labels vs gpt-5-mini labels on the SAME records, so we can see whether the
judge swap moves any method's PRR or the overall method RANKING. If the ranking is
unchanged and the PRR deltas are small, adopting mini is safe for our conclusions, not
just for the raw labels.

Inputs (both already on disk, so NO new API calls):
  - the pair-level CSV that `validate_mini_judge.py --save` wrote: idx, ref_score,
    cand_score (ref = GPT-5, cand = gpt-5-mini), for the judged subset.
  - results/<key>.csv: the per-record method uncertainties (one column per method).
Records line up by idx: in 01_extract the test split is enumerated from 0, so a test
record's `idx` is its position among the test rows of the results CSV.

    python scripts/checks/judge_prr_impact.py --dataset xsum --ood ID \
        --model google/gemma-2-9b-it
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.results import prr  # noqa: E402

NON_METHOD = {"dataset", "task", "split", "correctness"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="xsum")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--val-csv", default=None,
                    help="pair CSV from validate_mini_judge --save "
                         "(default: logs/judge_val_<dataset>.csv)")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    val_csv = args.val_csv or f"logs/judge_val_{args.dataset}.csv"
    res_csv = cfg.results_dir / f"{key}.csv"

    val = list(csv.DictReader(open(val_csv)))
    res = list(csv.DictReader(open(res_csv)))
    test = [r for r in res if r["split"] == "test"]  # in idx order (enumerate from 0)
    methods = [c for c in res[0] if c not in NON_METHOD]

    refs, cands = [], []
    unc = {m: [] for m in methods}  # method -> uncertainty for the judged subset
    skipped = 0
    for row in val:
        k = int(float(row["idx"]))
        if k >= len(test):
            skipped += 1
            continue
        trow = test[k]
        vals, ok = {}, True
        for m in methods:
            v = trow[m]
            if v == "" or v is None:  # method not run for this dataset (e.g. ptrue_accurate)
                continue
            vals[m] = float(v)
        if not vals:
            ok = False
        if not ok:
            skipped += 1
            continue
        refs.append(float(row["ref_score"]))
        cands.append(float(row["cand_score"]))
        for m, v in vals.items():
            unc[m].append(v)

    n = len(refs)
    extra = f" ({skipped} skipped)" if skipped else ""
    print(f"{args.dataset}: PRR under GPT-5 vs gpt-5-mini labels on the SAME {n} records{extra}")
    print(f"{'method':16s} {'PRR(GPT-5)':>11s} {'PRR(mini)':>10s} {'delta':>8s}")
    results = []
    for m in methods:
        if len(unc[m]) != n:  # method missing for some judged records -> not comparable
            continue
        p_ref = prr(refs, unc[m])
        p_cand = prr(cands, unc[m])
        results.append((m, p_ref, p_cand))
        print(f"{m:16s} {p_ref:11.3f} {p_cand:10.3f} {p_cand - p_ref:+8.3f}")

    if not results:
        print("no comparable methods found"); return

    rank_ref = [m for m, _, _ in sorted(results, key=lambda x: -x[1])]
    rank_cand = [m for m, _, _ in sorted(results, key=lambda x: -x[2])]
    print("\nranking under GPT-5 :", " > ".join(rank_ref))
    print("ranking under mini  :", " > ".join(rank_cand))
    print("ranking identical   :", rank_ref == rank_cand)
    print(f"max |delta PRR|     : {max(abs(c - r) for _, r, c in results):.3f}")


if __name__ == "__main__":
    main()
