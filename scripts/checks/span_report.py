"""STEP 1 of the answer-span truncation plan: eyeball the cut points.

Applies `answer_span` (src/luq/answer_span.py) to the cached generations and prints, per dataset,
the aggregate cut rate plus 10 before/after examples -- prioritising examples where a cut actually
fired, so the cut point is visible. NO model, no regeneration; reads cache/records/*.jsonl only.

    python scripts/checks/span_report.py                       # all five datasets
    python scripts/checks/span_report.py --datasets med_quad   # one dataset
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luq import answer_span as A  # noqa: E402
from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402

DEFAULT = ["med_quad", "xsum", "pubmed_qa", "sciq", "trivia_qa"]
MODEL = "meta-llama/Meta-Llama-3.1-8B"


def show(raw, clean, cut):
    """One-line-ish before/after: the kept text, a ⟪CUT⟫ marker, then the discarded tail."""
    tail = raw[cut:]
    kept = clean if len(clean) <= 240 else clean[:240] + "…"
    disc = tail if len(tail) <= 160 else tail[:160] + "…"
    return kept, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT)
    ap.add_argument("--n", type=int, default=10, help="examples to show per dataset")
    args = ap.parse_args()

    for ds in args.datasets:
        cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID")
        key = cache.run_key(MODEL, ds, "ID")
        recs = cache.load_records(cfg.cache_dir, key)

        rows = []
        for r in recs:
            clean, cut, reason = A.answer_span(r["gen_text"], ds, context=r.get("prompt"))
            rows.append({"idx": r["idx"], "raw": r["gen_text"], "clean": clean,
                         "cut": cut, "reason": reason})

        n = len(rows)
        n_cut = sum(1 for x in rows if not x["reason"].startswith("no-cut") and x["reason"] != "empty")
        n_echo = sum(1 for x in rows if "ECHO-FLAG" in x["reason"])
        # mean chars removed (only where cut)
        removed = [len(x["raw"]) - len(x["clean"]) for x in rows if x["cut"] < len(x["raw"])]
        avg_rm = sum(removed) / len(removed) if removed else 0

        print("=" * 100)
        print(f"[{ds}]  n={n}  cut={n_cut} ({100*n_cut/n:.1f}%)  "
              f"echo-flagged={n_echo}  mean_chars_removed_when_cut={avg_rm:.0f}")
        print("=" * 100)

        # prioritise examples where a cut fired; fill with no-cut ones if fewer than n
        cut_rows = [x for x in rows if x["cut"] < len(x["raw"])]
        nocut_rows = [x for x in rows if x["cut"] >= len(x["raw"])]
        picked = cut_rows[:args.n] + nocut_rows[:max(0, args.n - len(cut_rows))]

        for x in picked:
            kept, disc = show(x["raw"], x["clean"], x["cut"])
            print(f"\n  idx={x['idx']}  reason={x['reason']}")
            print(f"    KEPT : {kept!r}")
            print(f"    ⟪CUT⟫ {disc!r}" if disc else "    ⟪CUT⟫ (nothing removed)")
        print()


if __name__ == "__main__":
    main()
