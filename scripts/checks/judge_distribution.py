"""G2 diagnostic: does our judge's graded score distribution match the Hidden Failures
paper (Table 19)? the judge deliberately emits 0 / a partial score / 1 (Appendix M), and
Table 19 gives the per-dataset split. After labelling, ours should line up in ballpark — a
SciQ split near 50/50 instead of ~91% correct means the judge scheme (graded vs binary) or
the generations differ, and we catch it before it propagates into PRR.

Caveat: Table 19 is for the paper's own Llama generations, so exact agreement is not
expected — treat it as a ballpark, with the % fully-correct (the `1` bin) the most telling.

  python scripts/checks/judge_distribution.py --model meta-llama/Meta-Llama-3.1-8B --dataset sciq --ood ID
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

# Table 19 (Hidden Failures, updated paper): % of records at 0 / partial (0<x<1) / 1.
TABLE19 = {
    "sciq":      (4.2, 4.5, 91.3),
    "pubmed_qa": (5.2, 17.4, 77.4),
    "trivia_qa": (35.2, 3.9, 61.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--field", default="correctness",
                    help="label field to inspect (correctness | correctness_judge)")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)
    vals = [r[args.field] for r in records if isinstance(r.get(args.field), (int, float))]
    n = len(vals)
    if not n:
        sys.exit(f"no numeric `{args.field}` labels for {key} — run 02_label.py first")

    zero = 100 * sum(1 for v in vals if v == 0) / n
    one = 100 * sum(1 for v in vals if v == 1) / n
    partial = 100 - zero - one
    print(f"{key}  field={args.field}  n={n}  mean={sum(vals) / n:.3f}")
    print(f"  ours      0: {zero:5.1f}%   partial: {partial:5.1f}%   1: {one:5.1f}%")

    ref = TABLE19.get(args.dataset)
    if not ref:
        print(f"  (no Table 19 reference for {args.dataset})")
        return
    print(f"  Table 19  0: {ref[0]:5.1f}%   partial: {ref[1]:5.1f}%   1: {ref[2]:5.1f}%")
    d0, dp, d1 = zero - ref[0], partial - ref[1], one - ref[2]
    print(f"  delta     0: {d0:+5.1f}    partial: {dp:+5.1f}    1: {d1:+5.1f}")
    if partial < 0.5:
        print("  WARNING: ~no partial-credit scores — judge may be returning BINARY, not graded "
              "(Table 14 needs the graded judge; binary changes PRR).")
    if abs(d1) > 15 or abs(d0) > 15:
        print("  WARNING: distribution far from Table 19 — check the judge scheme or the "
              "generations before trusting downstream PRR.")


if __name__ == "__main__":
    main()
