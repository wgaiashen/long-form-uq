"""Re-label med_quad on the ANSWER SPAN (clean_text), not the raw over-generation.

WHY: step-4 showed med_quad's raw labels are noisy -- base-Llama tacks fake Q/A and rambles onto a
good answer, and the judge marks the whole thing down, so cheap-vs-GPT-5 agreement on the RAW text was
only spearman 0.26, rising to 0.81 on the truncated text. So the raw `correctness` we'd train the
SameTask rung on is a noisy target. This re-judges the CLEAN answer span for the whole set.

WHAT (non-destructive, full provenance):
  * writes a NEW field `correctness_clean` (+ `correctness_clean_model`); the raw `correctness` is
    left untouched, so nothing is overwritten and the two can be compared.
  * SAME judge model as the raw label (gpt-5-mini) -- never mix judges within one comparison.
  * COST-SAVING + correctness: a no-cut row has clean_text == gen_text, so its clean label is just its
    raw label (identical judge input) -- we COPY it, no API call. Only the ~48% cut rows are re-judged.
  * resumable: skips rows that already have `correctness_clean`, checkpoints every 25 new judgements,
    so a crash/rate-limit never re-spends.

Promotion to the canonical `correctness` (what the ladder reads) is a SEPARATE, flagged step -- this
script only produces the field and reports the shift. Run on the LOGIN NODE (API, network-bound).

    export OPENAI_API_KEY=...   # (source .openai_key)
    python scripts/checks/relabel_med_quad_clean.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq import answer_span as A, cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
DATASET = "med_quad"
JUDGE = "gpt-5-mini"          # MUST match the raw med_quad judge (no judge mixing)
SAVE_EVERY = 25


def main():
    cfg = Config(model_name=MODEL, dataset=DATASET, ood_setting="ID")
    key = cache.run_key(MODEL, DATASET, "ID")
    recs = cache.load_records(cfg.cache_dir, key)

    # sanity: the raw label must be gpt-5-mini, else re-judging clean with gpt-5-mini WOULD mix judges
    raw_judges = {r.get("correctness_model") for r in recs}
    if raw_judges != {JUDGE}:
        sys.exit(f"raw correctness_model = {raw_judges}, expected {{{JUDGE!r}}}; refusing to mix judges")

    n_judged = n_copied = n_failed = 0
    for i, r in enumerate(recs):
        if isinstance(r.get("correctness_clean"), (int, float)):
            continue  # resume: already done on a previous run
        clean, _, rsn = A.answer_span(r["gen_text"], DATASET, context=r.get("prompt"))
        if rsn.startswith("no-cut") or clean.strip() == r["gen_text"].strip():
            # clean == raw text -> the clean label IS the raw label (same judge, same input). Copy it,
            # no API call. This keeps the field complete over all 1800 without paying for unchanged rows.
            r["correctness_clean"] = r.get("correctness")
            r["correctness_clean_model"] = f"{JUDGE} (copied: no cut)"
            n_copied += 1
            continue
        # cut row: re-judge the CLEAN answer span with the same judge
        trec = dict(r); trec["gen_text"] = clean
        score = llm_judge.judge(trec, DATASET, model=JUDGE)
        if score is None:
            n_failed += 1
            continue  # leave unset so a rerun retries it
        r["correctness_clean"] = score
        r["correctness_clean_model"] = JUDGE
        n_judged += 1
        if n_judged % SAVE_EVERY == 0:
            cache.save_records(recs, cfg.cache_dir, key)
            print(f"judged {n_judged} clean (at row {i+1}/{len(recs)})", flush=True)

    cache.save_records(recs, cfg.cache_dir, key)  # final checkpoint (captures the tail)

    # report the shift raw -> clean over the rows that actually moved (the cut rows)
    have = [(r.get("correctness"), r.get("correctness_clean")) for r in recs
            if isinstance(r.get("correctness_clean"), (int, float))
            and isinstance(r.get("correctness"), (int, float))]
    raw = np.array([a for a, _ in have]); cl = np.array([b for _, b in have])
    cut_mask = np.array([not A.answer_span(r["gen_text"], DATASET, context=r.get("prompt"))[2]
                         .startswith("no-cut") for r in recs
                         if isinstance(r.get("correctness_clean"), (int, float))
                         and isinstance(r.get("correctness"), (int, float))])
    print(f"\n=== med_quad clean re-label ({JUDGE}) ===")
    print(f"  re-judged (cut rows): {n_judged}   copied (no-cut): {n_copied}   failed: {n_failed}")
    print(f"  ALL rows   mean raw={raw.mean():.3f}  clean={cl.mean():.3f}  delta={ (cl-raw).mean():+.3f}")
    if cut_mask.any():
        rc, cc = raw[cut_mask], cl[cut_mask]
        d = cc - rc
        print(f"  CUT rows   mean raw={rc.mean():.3f}  clean={cc.mean():.3f}  delta={d.mean():+.3f}")
        print(f"             |delta|>0.2: {100*np.mean(np.abs(d)>0.2):.1f}% of cut rows")
    print("  wrote field `correctness_clean` (+`correctness_clean_model`); raw `correctness` untouched.")
    print("  PROMOTION (flagged, not done here): copy correctness_clean -> correctness to feed the ladder.")


if __name__ == "__main__":
    main()
