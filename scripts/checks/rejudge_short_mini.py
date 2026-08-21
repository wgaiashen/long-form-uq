"""Judge-consistency SENSITIVITY: re-label a short-form set with the LONG-form judge.

WHY. The short-form sets (sciq, trivia_qa) carry `gpt-5` judge labels; every long-form set carries
`gpt-5-mini`. Both are graded 0-1 reference-agreement judges from the same prompt family, so the
cross-length arms differ in SOURCE REGIME as intended -- but a LONG+SHORT pool does mix two judge
models, against the standing "never mix judges within one comparison" rule. This quantifies how much
that matters instead of leaving it as an unquantified caveat.

IT NEVER TOUCHES THE CANONICAL RECORDS. `02_label.py --judge-short-form` would: it rewrites
`correctness` with `string_match` before judging (02_label.py:236-239), which on these caches would
DESTROY the promoted gpt-5 judge label that the canonical Long->Short cells were scored against.
This script instead writes a SHADOW REGIME under `cache/<regime>/`, exactly the mechanism the
MedQuAD clean-span sensitivity used, so the canonical population is untouched and the primary
numbers stay comparable.

In the shadow records:
    correctness        <- the NEW gpt-5-mini score        (what the ladder will read)
    correctness_gpt5   <- the canonical gpt-5 score       (kept for the agreement report)
    correctness_model  <- "gpt-5-mini"                    (stamped, so nothing is ambiguous)
The per-token cache is SYMLINKED, not copied: the representation is identical, only the label moves.

Resumable and checkpointed -- a rate-limit or a crash never re-spends on a row already judged.

    python scripts/checks/rejudge_short_mini.py --dataset trivia_qa --regime trivia_mini
    # then score it with:
    #   LUQ_REGIME="trivia_qa=trivia_mini" python scripts/checks/probedriftlong.py ...
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

MINI = "gpt-5-mini"
FIELD = "correctness_judge_mini"
CHECKPOINT = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="trivia_qa")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B")
    ap.add_argument("--judge-model", default=MINI)
    ap.add_argument("--regime", default=None, help="shadow cache root, default <dataset>_mini")
    ap.add_argument("--limit", type=int, default=None,
                    help="SMOKE TEST: judge only the first N rows. Output is labelled SMOKE and must "
                         "never enter a results table.")
    ap.add_argument("--report-only", action="store_true",
                    help="skip judging; just re-print the agreement report from what is on disk")
    args = ap.parse_args()

    regime = args.regime or f"{args.dataset}_mini"
    key = cache.run_key(args.model, args.dataset, "ID")

    canon_dir = Config(model_name=args.model, dataset=args.dataset, ood_setting="ID").cache_dir
    # A regime root is cache/<regime> -- the same layout Config produces for `prompt_regime`
    # (cache/expertqa_rp12 etc.), which is what LUQ_REGIME resolves to. Verified against
    # Config(prompt_regime=...).cache_dir rather than assumed.
    shadow_dir = Path(canon_dir) / regime
    (shadow_dir / "records").mkdir(parents=True, exist_ok=True)
    (shadow_dir / "pertok").mkdir(parents=True, exist_ok=True)

    canon = cache.load_records(canon_dir, key)
    print(f"canonical records: {len(canon)} from {canon_dir}/records/{key}.jsonl")

    shadow_path = shadow_dir / "records" / f"{key}.jsonl"
    if shadow_path.exists():
        recs = cache.load_records(shadow_dir, key)
        if len(recs) != len(canon):
            raise SystemExit(f"shadow has {len(recs)} rows, canonical {len(canon)} -- refusing to "
                             "resume onto a mismatched file")
        print(f"resuming from {shadow_path}")
    else:
        # Seed the shadow from the canonical rows, preserving the gpt-5 label under its own name.
        recs = []
        for r in canon:
            r2 = dict(r)
            r2["correctness_gpt5"] = r.get("correctness")
            r2[FIELD] = None
            recs.append(r2)
        print(f"seeded a fresh shadow at {shadow_path}")

    todo = [i for i, r in enumerate(recs) if r.get(FIELD) is None]
    if args.limit:
        todo = todo[:args.limit]
    print(f"to judge: {len(todo)} rows with {args.judge_model}"
          + ("  [SMOKE TEST -- not for any results table]" if args.limit else ""))

    if not args.report_only and todo:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set. `source ../.openai_key` first.")
        done = 0
        for n, i in enumerate(todo, 1):
            score = llm_judge.judge(recs[i], args.dataset, model=args.judge_model)
            # A None here means the judge never returned a valid number after its retries. Leave the
            # field None -- NOT 0.0. A not-measured value must never be recorded as a measurement.
            recs[i][FIELD] = score
            done += score is not None
            if n % CHECKPOINT == 0 or n == len(todo):
                for r in recs:
                    if r.get(FIELD) is not None:
                        r["correctness"] = r[FIELD]
                        r["correctness_model"] = args.judge_model
                cache.save_records(recs, shadow_dir, key)
                print(f"  [{n}/{len(todo)}] checkpointed ({done} valid)", flush=True)

    # Promote and stamp, then symlink the representation.
    for r in recs:
        if r.get(FIELD) is not None:
            r["correctness"] = r[FIELD]
            r["correctness_model"] = args.judge_model
    cache.save_records(recs, shadow_dir, key)

    for layer in (15,):
        src = Path(canon_dir) / "pertok" / f"{cache._slug(args.model)}__{args.dataset}__ID__L{layer}.npz"
        dst = shadow_dir / "pertok" / src.name
        if src.exists() and not dst.exists():
            os.symlink(os.path.relpath(src, dst.parent), dst)
            print(f"symlinked pertok L{layer} (representation unchanged, only the label moves)")

    # ---- the agreement report: this IS the sensitivity ----------------------------------------
    a = np.array([r.get("correctness_gpt5") if r.get("correctness_gpt5") is not None else np.nan
                  for r in recs], float)
    b = np.array([r.get(FIELD) if r.get(FIELD) is not None else np.nan for r in recs], float)
    ok = np.isfinite(a) & np.isfinite(b)
    print("\n" + "=" * 70)
    print(f"JUDGE AGREEMENT — gpt-5 (canonical) vs {args.judge_model} (shadow)")
    print("=" * 70)
    print(f"  rows compared      : {ok.sum()} / {len(recs)}")
    if ok.sum():
        unjudged = int((~np.isfinite(b)).sum())
        print(f"  unjudged (None)    : {unjudged}  (left None, never 0.0)")
        print(f"  mean  gpt-5        : {a[ok].mean():.4f}")
        print(f"  mean  {args.judge_model:<13}: {b[ok].mean():.4f}")
        print(f"  mean  difference   : {(b[ok]-a[ok]).mean():+.4f}")
        print(f"  mean |difference|  : {np.abs(b[ok]-a[ok]).mean():.4f}")
        print(f"  Pearson r          : {np.corrcoef(a[ok], b[ok])[0,1]:.4f}")
        exact = float((np.abs(a[ok]-b[ok]) < 1e-9).mean())
        print(f"  exact agreement    : {exact:.1%}")
        binagree = float(((a[ok] >= .5) == (b[ok] >= .5)).mean())
        print(f"  binarised@0.5 agree: {binagree:.1%}")
        pos_a = float((a[ok] >= .5).mean()); pos_b = float((b[ok] >= .5).mean())
        print(f"  positive rate      : gpt-5 {pos_a:.1%} vs {args.judge_model} {pos_b:.1%}")
        print(f"  -> error mass      : gpt-5 {1-pos_a:.1%} vs {args.judge_model} {1-pos_b:.1%}")
    print(f"\nshadow records: {shadow_path}")
    print(f"score it with: LUQ_REGIME=\"{args.dataset}={regime}\" python scripts/checks/probedriftlong.py ...")


if __name__ == "__main__":
    main()
