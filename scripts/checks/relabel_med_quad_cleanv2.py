#!/usr/bin/env python
"""Judge the MedQuAD clean-v2 rows whose retained span changed under answer_span v2.

WHAT THIS IS FOR. The clean-v2 population truncates each MedQuAD generation at the v2 answer-span
boundary. Most rows keep the label they already had, because their retained span is unchanged: of
1800 rows, 938 are uncut under both rules and 852 are cut at an identical boundary, so 1790 labels
carry over untouched. Only the rows whose retained span actually MOVED need a judge call, and
build_truncated_records.py has already dropped every judge-written field on exactly those rows.

WHY THE COUNT IS SMALL AND WHY IT IS NOT ZERO. v2 differs from v1 only in tolerating whitespace
around the `Question:`/`Answer:` colon. On Llama that adds 7 newly cut rows and moves the boundary
EARLIER on 3 more -- those 3 are v1 errors, where v1 cut at the `\\nAnswer:` inside the fabricated
block and so retained the invented question stem. Both kinds need re-judging: their existing label
describes more text than the clean-v2 features cover, which is the precise mismatch this whole
correction exists to remove.

WHAT IS JUDGED. The record's own `gen_text`, verbatim. In the clean-v2 namespace that field is
already `tok.decode(gen_token_ids)` for the retained prefix, i.e. exactly the text the token
logprobs and the layer-15 states describe. It can be up to a character shorter than the regex slice
answer_span returns, because the cut is made in token space; judging the token-space text is
deliberate, since that is what the features cover.

NON-DESTRUCTIVE. Only the clean-v2 records are opened for writing. Canonical `correctness`,
`correctness_raw` and `correctness_clean` live in a different file and are never touched. The
inherited fields carried into this namespace are left exactly as they arrived.

SPEND GUARD. The number of rows to judge is asserted against --expect before a single call is made.
A relabel that suddenly wants to judge hundreds of rows is a bug in the span rule, not a licence to
spend; it aborts instead.

    export OPENAI_API_KEY=...        # source ../.openai_key
    python scripts/checks/relabel_med_quad_cleanv2.py --expect 10
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from luq import cache                                    # noqa: E402
from luq.config import Config                            # noqa: E402
from luq.labels import llm_judge                         # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "med_quad"
REGIME = "cleanv2"
JUDGE = "gpt-5-mini"          # the judge the inherited labels were produced with; never mix judges
SAVE_EVERY = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect", type=int, default=10,
                    help="rows expected to need judging; a mismatch aborts before any spend")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = Config(model_name=MODEL, dataset=DATASET, ood_setting="ID", prompt_regime=REGIME)
    key = cache.run_key(MODEL, DATASET, "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    print(f"loaded {len(recs)} clean-v2 records from {cfg.cache_dir}")

    # never mix judges within one comparison
    models = {r.get("correctness_model") for r in recs if r.get("correctness") is not None}
    if models - {JUDGE}:
        sys.exit(f"inherited labels carry {models}, expected only {{{JUDGE!r}}}; refusing to mix judges")

    todo = [r for r in recs if r.get("correctness") is None]
    print(f"rows already labelled (inherited): {len(recs) - len(todo)}")
    print(f"rows needing a judge call        : {len(todo)}  idx={[r['idx'] for r in todo]}")
    if len(todo) != args.expect:
        sys.exit(f"!!! expected {args.expect} rows to judge, found {len(todo)}. Refusing to spend: "
                 f"a changed count means the span rule moved, which is a decision, not a relabel.")
    if args.dry_run:
        print("DRY RUN -- no judge call made."); return

    done = failed = 0
    for r in todo:
        score = llm_judge.judge(r, DATASET, model=JUDGE)   # judges r["gen_text"] as stored
        if score is None:
            failed += 1
            print(f"  idx {r['idx']}: judge returned nothing, left unset for a rerun")
            continue
        r["correctness"] = score
        r["correctness_model"] = JUDGE
        r["label_span"] = "cleanv2"                        # provenance: which span this describes
        done += 1
        print(f"  idx {r['idx']}: correctness={score}")
        if done % SAVE_EVERY == 0:
            cache.save_records(recs, cfg.cache_dir, key)

    # stamp provenance on every row so the namespace is self-documenting
    for r in recs:
        r.setdefault("label_span", "cleanv2-inherited")
    cache.save_records(recs, cfg.cache_dir, key)

    n_lab = sum(1 for r in recs if r.get("correctness") is not None)
    print(f"\njudged {done}, failed {failed}. Labelled rows now {n_lab}/{len(recs)}.")
    if n_lab != len(recs):
        print("!!! not every row is labelled -- the ladder drops unlabelled rows, so rerun before use.")


if __name__ == "__main__":
    main()
