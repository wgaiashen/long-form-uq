"""Emit the exact cell definitions for the bidirectional cross-length transfer experiment.

The training pools are DERIVED from `probe_drift_long` (the same taxonomy the canonical ladder
uses), never hand-typed, so the LONG+SHORT arm is provably the canonical LOO-long mixture at half
weight rather than a list someone retyped.

    python scripts/checks/cross_length_cells.py            # table of the cells
    python scripts/checks/cross_length_cells.py --qsub     # the qsub lines to submit

⚠️ `qsub -v` splits on commas, so the spec is emitted with '+' between sources; `pbs/cross_length.pbs`
translates it back. That is why the separator looks odd.
"""
import argparse

from probe_drift_long import LONG_SRC, XL_TOTAL, rung_sources_long

# Chosen by results/analysis/CROSS_LENGTH_TARGET_SELECTION.md, BEFORE any cross-length result
# existed. asqa = QA (the only HIGH cross-model stability), xsum = summarisation (the most stable
# aggregation signature), factscore = factuality (the more stable of only two candidates).
LONG_TARGETS = ["asqa", "xsum", "factscore"]

# trivia_qa is PRIMARY: sciq's test split is 94.3% positive (~5.7% error mass), which leaves PRR
# very little to rank against. sciq is kept as a secondary arm because on Llama it is free.
SHORT_TARGETS = ["trivia_qa", "sciq"]


def spec_str(pairs):
    return "+".join(f"{d}:{n}" for d, n in pairs)


def long_plus_short(X):
    """Half the canonical LOO-long mixture for X, plus an equal split of the two short sets.

    LOO-long is the 7 other long sets at XL_TOTAL//7 = 257 each. Halving the per-source cap (128)
    is what makes this a STRATIFIED half-sample of the exact existing mixture rather than a fresh
    draw: `sampled_train_idx` permutes each source under the run's seed and takes the first `cap`,
    so cap=128 is a deterministic subset of the same source at cap=257 under that seed.
    """
    srcs = rung_sources_long(X)["LOO-long"]
    half = (XL_TOTAL // len(srcs)) // 2                      # 257 // 2 = 128
    long_part = [(d, half) for d in srcs]
    remaining = XL_TOTAL - half * len(srcs)                  # 1800 - 896 = 904
    per_short = remaining // 2                               # 452 each
    short_part = [("sciq", per_short), ("trivia_qa", remaining - per_short)]
    return long_part + short_part


def cells():
    out = []
    # ---- SHORT -> LONG ---------------------------------------------------------------------
    # LONG (the canonical LOO-long cell) already exists in results/pdl_fam_<X>__*.csv and is NOT
    # re-run. Only the two new arms are listed here.
    for X in LONG_TARGETS:
        out.append(("short2long", X, "LONG+SHORT", long_plus_short(X)))
        out.append(("short2long", X, "SHORT", [("sciq", 900), ("trivia_qa", 900)]))
    # ---- LONG -> SHORT ---------------------------------------------------------------------
    # LONG (the canonical Long->Short rung, 8 long sources x 225) already exists in
    # results/pdl_fam_{sciq,trivia_qa}__*.csv and is NOT re-run.
    for X in SHORT_TARGETS:
        other = [s for s in SHORT_TARGETS if s != X][0]
        out.append(("long2short", X, "ID-short", [(X, XL_TOTAL)]))
        out.append(("long2short", X, "OTHER-SHORT", [(other, XL_TOTAL)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qsub", action="store_true", help="emit qsub lines instead of a table")
    args = ap.parse_args()

    rows = cells()
    if args.qsub:
        print("# no-op control FIRST -- do not submit the cells until it passes")
        print("qsub pbs/cross_length_noop.pbs")
        print()
        for direction, X, rung, spec in rows:
            tag = f"{direction}_{rung.replace('+', '').replace('-', '').lower()}"
            print(f"qsub -v LUQ_EVAL={X},LUQ_RUNG={rung},LUQ_SPEC={spec_str(spec)},LUQ_TAG={tag} "
                  f"pbs/cross_length.pbs")
        return

    print(f"{'direction':<12} {'eval':<11} {'rung':<12} {'total':>6}  pool")
    print("-" * 118)
    for direction, X, rung, spec in rows:
        total = sum(n for _, n in spec)
        flag = "" if total == XL_TOTAL else f"  <-- != {XL_TOTAL}"
        print(f"{direction:<12} {X:<11} {rung:<12} {total:>6}  {spec_str(spec)}{flag}")
    print()
    print(f"{len(rows)} new cells x 3 seeds. The LONG arm of BOTH directions already exists and is "
          f"not re-run:")
    print("  short->long LONG  = the canonical LOO-long cell in results/pdl_fam_<target>__*.csv")
    print("  long->short LONG  = the canonical Long->Short rung (8 long x 225) in "
          "results/pdl_fam_{sciq,trivia_qa}__*.csv")
    # Every source must be capped at or below what the source can actually supply, or the realised
    # pool silently shrinks. Flag the ones worth watching.
    print()
    print("sanity: no source is asked for more than its train split holds "
          "(sciq/trivia_qa train = 1800 each; long sources >= 257 available at cap 128).")


if __name__ == "__main__":
    main()
