"""Validate gpt-5-mini against the pinned GPT-5 judge on the SCALED model's outputs.

This is the "make sure it makes sense" gate before mass-labelling long-form with the
cheaper judge. It scores the SAME records with BOTH judges in memory, reports agreement,
and exits with a code the overnight orchestrator reads:

    exit 0  -> ADOPT  (agreement healthy; safe to mass-label with gpt-5-mini)
    exit 2  -> REJECT (judges disagree too much; leave long-form for manual review)
    exit 1  -> ERROR / inconclusive (too few valid pairs, no key, etc.)

It does NOT save anything: the canonical records stay unlabelled, so the later mass
pass labels every record with ONE judge (never mixing judges within a comparison).

    python scripts/checks/validate_mini_judge.py --dataset pubmed_qa --ood ID \
        --model google/gemma-2-9b-it --n 100

Cost: ~2*n judge calls (n GPT-5 + n gpt-5-mini).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from luq import cache  # noqa: E402
from luq.config import Config  # noqa: E402
from luq.labels import llm_judge  # noqa: E402

# Adoption thresholds. The earlier 1.5B validation put gpt-5-mini at ~0.85 agreement;
# we ask for a clear positive correlation and small average gap on the scaled model.
MIN_PEARSON = 0.70
MAX_MEAN_ABS_DIFF = 0.15
MIN_PAIRS = 30  # below this the comparison is too noisy to trust either way


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pubmed_qa")
    ap.add_argument("--ood", default="ID")
    ap.add_argument("--model", default="google/gemma-2-9b-it")
    ap.add_argument("--n", type=int, default=100, help="how many test records to compare")
    ap.add_argument("--reference", default=llm_judge.MODEL, help="reference judge (GPT-5)")
    ap.add_argument("--candidate", default="gpt-5-mini", help="cheaper judge under test")
    ap.add_argument("--show", type=int, default=8,
                    help="print this many largest judge disagreements")
    ap.add_argument("--save", default=None,
                    help="optional CSV path to dump every (ref, cand) pair for inspection")
    ap.add_argument("--prompt-regime", default="",
                    help="cache namespace tag (must match 01_extract; e.g. expertqa). "
                         "Without it the default cache/ is read, not cache/<regime>/.")
    ap.add_argument("--ref-from-cache", action="store_true",
                    help="use the cached gpt-5 `correctness` label as the reference instead of "
                         "re-calling GPT-5 live: mini-only (halves cost, skips the pricey judge) "
                         "and uses the EXACT paid-for labels the probe was trained/evaluated on.")
    args = ap.parse_args()

    cfg = Config(model_name=args.model, dataset=args.dataset, ood_setting=args.ood,
                 prompt_regime=args.prompt_regime)
    key = cache.run_key(cfg.model_name, cfg.dataset, cfg.ood_setting)
    records = cache.load_records(cfg.cache_dir, key)

    # Compare on test records that actually produced text (a blank generation is not a
    # meaningful judge target).
    pool = [r for r in records if r.get("split") == "test" and r.get("gen_text", "").strip()]
    pool = pool[: args.n]
    ref_src = "cached gpt-5 correctness" if args.ref_from_cache else f"live {args.reference}"
    print(f"validating {args.candidate} vs {ref_src} on {len(pool)} "
          f"{args.dataset} records", flush=True)

    # Keep the record alongside both scores so we can show WHERE they disagree, not
    # just an aggregate (a small mean-abs-diff can still hide a few bad cases).
    pairs = []  # each: (idx, ref_score, cand_score, gen_snippet)
    for i, r in enumerate(pool):
        if args.ref_from_cache:
            a = r.get("correctness")
            a = float(a) if isinstance(a, (int, float)) else None
        else:
            a = llm_judge.judge(r, cfg.dataset, model=args.reference)
        b = llm_judge.judge(r, cfg.dataset, model=args.candidate)
        if a is not None and b is not None:
            snippet = " ".join(r.get("gen_text", "").split())[:160]
            pairs.append((r.get("idx", i), a, b, snippet))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(pool)} scored ({len(pairs)} valid pairs)", flush=True)

    n = len(pairs)
    if n < MIN_PAIRS:
        print(f"VERDICT: ERROR — only {n} valid pairs (< {MIN_PAIRS}); inconclusive")
        sys.exit(1)

    ref = np.array([p[1] for p in pairs])
    cand = np.array([p[2] for p in pairs])
    diffs = np.abs(ref - cand)
    mad = float(diffs.mean())
    # corrcoef / Spearman are undefined if either side is constant (e.g. all 1.0); guard.
    constant = ref.std() < 1e-9 or cand.std() < 1e-9
    pearson = float("nan") if constant else float(np.corrcoef(ref, cand)[0, 1])
    # Spearman = rank agreement. PRR is computed by RANKING outputs by uncertainty, so
    # the judges agreeing on the *order* matters more than on the absolute score.
    if constant:
        spearman = float("nan")
    else:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(ref, cand).statistic)

    print(f"\n--- agreement on {n} pairs ---")
    print(f"  reference ({args.reference}) mean: {ref.mean():.3f}")
    print(f"  candidate ({args.candidate}) mean: {cand.mean():.3f}")
    print(f"  Pearson r        : {pearson:.3f}  (adopt if >= {MIN_PEARSON})")
    print(f"  Spearman rho     : {spearman:.3f}  (rank agreement; most relevant to PRR)")
    print(f"  mean abs diff    : {mad:.3f}  (adopt if <= {MAX_MEAN_ABS_DIFF})")
    print(f"  |diff| > 0.25    : {int((diffs > 0.25).sum())}/{n} pairs")

    # Show the worst disagreements so we can eyeball which judge is right.
    if args.show:
        print(f"\n--- {min(args.show, n)} largest disagreements (ref vs mini) ---")
        for j in np.argsort(-diffs)[: args.show]:
            idx, a, b, snip = pairs[j]
            print(f"  Δ{abs(a - b):.2f}  ref={a:.2f} mini={b:.2f}  #{idx}  {snip}")

    if args.save:
        import csv
        with open(args.save, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "ref_score", "cand_score", "abs_diff", "gen_snippet"])
            for idx, a, b, snip in pairs:
                w.writerow([idx, a, b, round(abs(a - b), 4), snip])
        print(f"\nwrote pair-level CSV -> {args.save}")

    adopt = (not np.isnan(pearson)) and pearson >= MIN_PEARSON and mad <= MAX_MEAN_ABS_DIFF
    if adopt:
        print("VERDICT: ADOPT — gpt-5-mini agrees closely enough; safe to mass-label.")
        sys.exit(0)
    print("VERDICT: REJECT — disagreement too large; do NOT mass-label with mini.")
    sys.exit(2)


if __name__ == "__main__":
    main()
