"""Eyeball where string-match and the LLM judge disagree on a short-form dataset.

Read-only, no API calls. After you have run, e.g.:

    python scripts/02_label.py --dataset sciq --ood ID --judge-short-form --judge gpt-5-mini

each record carries BOTH `correctness` (string match, 0/1) and `correctness_judge`
(the graded 0-1 judge score). This script joins them and shows where they part ways.

Why look at disagreements rather than at PRR: PRR is measured AGAINST the correctness
label, so "the labelling that gives higher PRR is better" is circular — a higher PRR
can just mean the label is easier for the probe to predict. The valid comparison is
label QUALITY, decided by hand on the cases where the two labels disagree.

sciq is short-form QA where string match (normalised, with alias handling) is reliable,
so it doubles as a check on the judge: if the judge agrees with deterministic string
match where string match is trustworthy, that is evidence the judge is trustworthy.

    python scripts/checks/inspect_label_disagreement.py --dataset sciq --ood ID
    python scripts/checks/inspect_label_disagreement.py --dataset sciq --top 15
    python scripts/checks/inspect_label_disagreement.py --dataset sciq --bucket false_neg
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.data import JUDGE_NAME_MAP, SHORT_FORM  # noqa: E402
from luq.labels.llm_judge import _extract_question  # noqa: E402


def _wrap(text, width=100, limit=900):
    """Wrap text for the terminal and clip very long context so the screen stays readable."""
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit] + " …[clipped]"
    return textwrap.fill(text, width=width)


# Each bucket sorts a disagreement by what it tells us about the labels. `t` is the
# threshold at which we read the graded judge score as "the judge thinks it is right".
def _bucket(sm, jg, t):
    if sm == 0.0 and jg >= t:
        return "false_neg"   # string match missed it (paraphrase / semantic match)
    if sm == 1.0 and jg < t:
        return "false_pos"   # substring matched but the judge says the answer is wrong
    if 0.0 < jg < 1.0:
        return "partial"     # graded credit the binary label cannot express
    return "agree"           # both call it the same way at the threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sciq", help="must be a short-form dataset")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default=Config.model_name)
    ap.add_argument("--top", type=int, default=12, help="how many worst disagreements to show")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="judge score at/above which the answer counts as 'right' for bucketing")
    ap.add_argument("--bucket", choices=["false_neg", "false_pos", "partial", "all"],
                    default="all", help="show only disagreements of this kind")
    args = ap.parse_args()

    if args.dataset not in SHORT_FORM:
        sys.exit(f"{args.dataset} is not short-form — it has no string-match label to "
                 "compare the judge against. This check is for sciq/trivia_qa/qa.")

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    judge_name = JUDGE_NAME_MAP[args.dataset]
    records = cache.load_records(cfg.cache_dir, key)

    # Keep only records that carry BOTH labels as numbers (the judge run may be partial
    # — that is fine, we inspect whatever has been judged so far).
    paired = [r for r in records
              if isinstance(r.get("correctness"), (int, float))
              and isinstance(r.get("correctness_judge"), (int, float))]
    if not paired:
        sys.exit("no records carry both `correctness` and `correctness_judge` — run "
                 "scripts/02_label.py with --judge-short-form first (it is resumable, "
                 "so even a partial run gives you records to inspect).")

    t = args.threshold
    buckets = {"false_neg": [], "false_pos": [], "partial": [], "agree": []}
    for r in paired:
        buckets[_bucket(r["correctness"], r["correctness_judge"], t)].append(r)

    n = len(paired)
    judge_model = next((r.get("correctness_judge_model") for r in paired
                        if r.get("correctness_judge_model")), "unknown")
    print(f"=== {args.dataset} ({args.ood}), judge={judge_model}, threshold={t} ===")
    print(f"{n} records carry both labels.")
    print(f"  string-match FALSE NEGATIVES (sm=0, judge≥{t}): {len(buckets['false_neg'])}"
          "   ← answers string match wrongly marked wrong")
    print(f"  string-match FALSE POSITIVES (sm=1, judge<{t}): {len(buckets['false_pos'])}"
          "   ← answers string match wrongly marked right")
    print(f"  partial credit (0<judge<1, agree at threshold): {len(buckets['partial'])}"
          "   ← gradation the 0/1 label loses")
    print(f"  agree at the threshold: {len(buckets['agree'])}")

    # Choose what to show, ranked by how far apart the two labels are.
    if args.bucket == "all":
        shown = buckets["false_neg"] + buckets["false_pos"] + buckets["partial"]
    else:
        shown = buckets[args.bucket]
    shown.sort(key=lambda r: abs(r["correctness"] - r["correctness_judge"]), reverse=True)
    shown = shown[: args.top]

    print(f"\n--- showing {len(shown)} disagreement(s)"
          f"{'' if args.bucket == 'all' else f' (bucket={args.bucket})'}, "
          "biggest gap first ---")
    for rank, r in enumerate(shown, 1):
        question, _ = _extract_question(r["prompt"], judge_name)
        print(f"\n[{rank}]  string-match={r['correctness']:.0f}   "
              f"judge={r['correctness_judge']:.2f}   "
              f"|Δ|={abs(r['correctness'] - r['correctness_judge']):.2f}")
        print("--- question / context ---")
        print(_wrap(question))
        print("--- gold answer ---")
        print(_wrap(str(r["target"])))
        print("--- model answer (the text being labelled) ---")
        print(_wrap(r["gen_text"]))


if __name__ == "__main__":
    main()
