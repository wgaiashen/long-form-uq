"""Severe-degeneracy rate split by generation length -- the test of the "hidden loops" prediction.

WHY THIS EXISTS. `prereg/A2_med_quad_degeneracy_gate.md` registers that med_quad's v1 severe rate of
12.33% is a LOWER BOUND, because a cap at 128 tokens truncates a loop before `max_content_run` can
reach its >=25 threshold. Raising the cap to 768 removes that truncator, so the severe rate is expected
to RISE, and the prereg says that rise is not a failure.

But "the rate rose" is compatible with two different stories, and the whole-dataset rate cannot tell
them apart:

  A. THE CAP WAS HIDING LOOPS. The loops live in the generations that are now allowed to run long. Then
     the severe rate in the NEWLY-LONG band (>=128 tokens) should be markedly HIGHER than in the short
     band, and the short band should look like v1 did -- because short generations were never truncated
     and nothing about them changed.
  B. SOMETHING ELSE CHANGED. If the rate rises roughly UNIFORMLY across both bands, the cap was not
     masking anything; the extra degeneracy is not "previously hidden" and the explanation lies
     elsewhere (the budget change altering decoding behaviour generally, say).

Those imply different follow-ups, which is the point: story A says keep the budget and add drift
control (the prepared rep-pen arm); story B says the rise is unexplained and the budget change itself
needs re-examining before any judge spend.

⚠️ 128 IS THE BAND BOUNDARY BECAUSE IT WAS v1's CAP, not because it is a round number. Rows under it
are generations that would have finished on their own under v1; rows at or above it are the ones v1
could not have produced. The comparison is only meaningful at that specific split point.

Uses `luq.degeneracy.is_severe` -- the SAME detector `generation_quality.py` gates on, imported rather
than reimplemented, so the numbers here and the numbers in the gate table cannot drift apart.

    python scripts/checks/degeneracy_by_length_band.py --dataset med_quad --regime v2_med_quad
    python scripts/checks/degeneracy_by_length_band.py --dataset med_quad --regime v2_med_quad \
        --compare-v1                       # also band the v1 records, for the story-A/B read

Reads records only. CPU, no model, no GPU, no judge calls.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import cache, degeneracy  # noqa: E402
from luq.config import Config  # noqa: E402
from attn_pool import PROMPT_REGIME  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
BAND = 128          # v1's med_quad cap -- see the docstring; not an arbitrary round number


def load(dataset: str, regime: str | None):
    """Return (gen_lengths, severe_flags, texts) for one dataset/regime, or None if absent."""
    rg = regime if regime is not None else PROMPT_REGIME.get(dataset, "")
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID", prompt_regime=rg)
    key = cache.run_key(MODEL, dataset, "ID")
    path = Path(cfg.cache_dir) / "records" / f"{key}.jsonl"
    if not path.exists():
        return None, str(path)
    recs = cache.load_records(cfg.cache_dir, key)
    glen = np.array([len(r["gen_token_ids"]) for r in recs])
    sev = np.array([degeneracy.is_severe(r["gen_text"]) for r in recs])
    return (glen, sev, [r["gen_text"] for r in recs]), str(path)


def band_table(label: str, glen, sev):
    """Print the two bands plus the whole-set rate. Empty bands print n=0, never a rate."""
    print(f"\n{label}")
    print(f"  {'band':<22} {'n':>6} {'severe':>8} {'rate':>8}")
    rows = {}
    for name, mask in [(f"short  (<{BAND} tok)", glen < BAND),
                       (f"long   (>={BAND} tok)", glen >= BAND)]:
        n = int(mask.sum())
        if n == 0:
            # A blank rate, not 0.0 -- an empty band is "not measured", not "measured and clean".
            print(f"  {name:<22} {n:>6} {'--':>8} {'--':>8}")
            rows[name] = None
            continue
        s = int(sev[mask].sum())
        r = 100 * s / n
        print(f"  {name:<22} {n:>6} {s:>8} {r:>7.2f}%")
        rows[name] = r
    n, s = len(glen), int(sev.sum())
    print(f"  {'ALL':<22} {n:>6} {s:>8} {100*s/n:>7.2f}%")
    return rows


def truncation_test(dataset: str, regime: str, v1_rate: float | None):
    """The DIRECT test of "the cap was hiding loops", and a better-powered one than the bands.

    The band split has a weakness that only shows up once the data exists: at budget 768 almost
    nothing stops before 128 tokens (n=5 on the first med_quad checkpoint), so the short band has no
    power and the long-vs-short contrast cannot discriminate.

    This does not have that problem. Take the SAME v2 generations, cut them at 128 tokens exactly as
    v1's cap did, and re-run the detector. Same examples, same decoding, same everything -- ONLY the
    observation window differs. So the gap between the two rates is caused by the window alone, with
    no population difference and nothing to control for.

    The cross-check is what makes it conclusive: if the cut-at-128 rate lands on v1's independently
    measured rate, the window fully explains v1's number and the rise is masked loops. If it does
    NOT, then something beyond the cap changed and the "hidden loops" story is incomplete.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    cfg = Config(model_name=MODEL, dataset=dataset, ood_setting="ID", prompt_regime=regime)
    key = cache.run_key(MODEL, dataset, "ID")
    recs = cache.load_records(cfg.cache_dir, key)
    full = np.array([degeneracy.is_severe(r["gen_text"]) for r in recs])
    cut = np.array([degeneracy.is_severe(tok.decode(r["gen_token_ids"][:BAND],
                                                    skip_special_tokens=True)) for r in recs])
    print("\n" + "=" * 78)
    print(f"TRUNCATION TEST -- the same {len(recs)} generations, seen through both windows")
    print("=" * 78)
    print(f"  severe, seen IN FULL                    : {100*full.mean():>6.2f}%")
    print(f"  severe, the SAME rows cut at {BAND} tokens : {100*cut.mean():>6.2f}%   <- all v1 could see")
    if v1_rate is not None:
        print(f"  v1's independently measured rate        : {v1_rate:>6.2f}%")
        gap = 100 * cut.mean() - v1_rate
        print(f"  cut-vs-v1 agreement: {gap:+.2f} pp", end="  ")
        print("-> the window explains v1's number" if abs(gap) < 2
              else "-> DOES NOT agree; something beyond the cap differs, the story is incomplete")
    hidden = int((full & ~cut).sum())
    tot = int(full.sum())
    print(f"  masked by the cap: {100*(full.mean()-cut.mean()):+.2f} pp; {hidden} of {tot} severe rows "
          f"({100*hidden/max(tot,1):.0f}%) are undetectable at {BAND} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncation-test", action="store_true",
                    help="re-score the v2 generations through v1's 128-token window (the direct test)")
    ap.add_argument("--v1-rate", type=float, default=None,
                    help="v1's measured severe %%, for the agreement cross-check (med_quad: 12.33)")
    ap.add_argument("--dataset", default="med_quad")
    ap.add_argument("--regime", default=None,
                    help="cache namespace, e.g. v2_med_quad. Omit for the dataset's default root.")
    ap.add_argument("--compare-v1", action="store_true",
                    help="also band the v1 (default-root) records, for the story-A vs story-B read")
    args = ap.parse_args()

    got, path = load(args.dataset, args.regime)
    if got is None:
        sys.exit(f"no records at {path}")
    glen, sev, _ = got
    print(f"dataset {args.dataset}   regime {args.regime or '<default>'}   n={len(glen)}")
    print(f"records {path}")
    print(f"gen tokens: p50 {np.percentile(glen,50):.0f}  p90 {np.percentile(glen,90):.0f}  "
          f"max {glen.max()}")
    v2 = band_table(f"v2  ({args.regime or 'default root'})", glen, sev)

    if args.truncation_test:
        truncation_test(args.dataset, args.regime or PROMPT_REGIME.get(args.dataset, ""),
                        args.v1_rate)

    if not args.compare_v1:
        return
    got1, path1 = load(args.dataset, "")
    if got1 is None:
        print(f"\n(no v1 records at {path1} -- skipping the comparison)")
        return
    g1, s1, _ = got1
    v1 = band_table("v1  (default root)", g1, s1)

    # THE READ. Deliberately stated as the two pre-registered stories rather than a single verdict,
    # and it reports the numbers that separate them rather than asserting which holds.
    print("\n" + "=" * 78)
    print("THE PRE-REGISTERED READ (prereg/A2_med_quad_degeneracy_gate.md)")
    print("=" * 78)
    short_v1 = v1.get(f"short  (<{BAND} tok)")
    short_v2 = v2.get(f"short  (<{BAND} tok)")
    long_v2 = v2.get(f"long   (>={BAND} tok)")
    if short_v2 is None or long_v2 is None:
        print("One v2 band is empty, so the bands cannot be compared. Reporting the raw table only.")
        return
    gap = long_v2 - short_v2
    print(f"  v2 long band  {long_v2:.2f}%   vs   v2 short band {short_v2:.2f}%   "
          f"difference {gap:+.2f} pp")
    if short_v1 is not None:
        n_short_v1 = int((g1 < BAND).sum())
        print(f"  v1 short band {short_v1:.2f}%   vs   v2 short band {short_v2:.2f}%   "
              f"difference {short_v2 - short_v1:+.2f} pp   (v1 n={n_short_v1})")
        print("  (the short band is the CONTROL: those generations were never truncated, so if the")
        print("   cap was the only thing that changed, this pair should be roughly equal)")
        if n_short_v1 < 100:
            # v1 med_quad is 97.7% capped, so its short band is a few dozen rows. A rate on n=44
            # carries a standard error of several points, which is wide enough to swallow the
            # difference being read -- say so rather than letting the number look decisive.
            se = 100 * (short_v1 / 100 * (1 - short_v1 / 100) / n_short_v1) ** 0.5
            print(f"  ⚠️ the v1 control band is only n={n_short_v1} (v1 was 97.7% capped), so its "
                  f"{short_v1:.2f}% carries a standard error of ~{se:.1f} pp.")
            print("     Treat the v1-vs-v2 short-band comparison as weak evidence; the WITHIN-v2")
            print("     long-vs-short contrast above is the better-powered half of this test.")
    print("\n  Story A -- the cap was hiding loops: long band markedly ABOVE short, and the short")
    print("            band roughly unchanged from v1. Follow-up = keep the budget, add drift control.")
    print("  Story B -- something else changed: both bands rise together. Follow-up = the rise is")
    print("            unexplained and the budget change itself needs re-examining before judge spend.")
    print("\n  These numbers are the evidence; the call between A and B is a judgement to be made on")
    print("  them, and it belongs in the writeup rather than being asserted here.")


if __name__ == "__main__":
    main()
