"""R1 — is the regime taxonomy decidable from UNLABELLED data?

WHY THIS EXISTS
---------------
The regime label in `results/floor_taxonomy__*.csv` is currently **label-derived**: it is `best_floor`,
i.e. whichever floor achieves the highest PRR, and PRR is computed against the judge labels. If the
regime a dataset belongs to cannot be decided WITHOUT those labels, the routing story is fitted to its
own evaluation and every result downstream of it is circular. This script is the gate.

Pre-registered at `prereg/R1_taxonomy_label_free.md`, with the statistic, the population, the null
distribution and the success threshold all fixed BEFORE the first run.

WHAT IT DOES NOT DO — and why that is the point
-----------------------------------------------
It fits NOTHING. With 8 datasets (and rungs that contribute no independent points, because a floor's PRR
depends only on the eval set, which is identical across rungs), a leave-one-out classifier would separate
the classes almost regardless of whether the underlying claim is true — it would be a false-positive
generator. So R1 registers ONE parameter-free statistic with its direction fixed in advance and reports
how many datasets it gets right.

THE STATISTIC
-------------
The mechanism the taxonomy claims is about WHERE within a generation the uncertainty sits:

  CONCENTRATED — the failure localises to one or a few tokens, so an example's worst token is a dramatic
                 outlier against the rest of its OWN generation. `msp_min` wins.
  SPREAD       — uncertainty is distributed, the whole curve is depressed together, the worst token is
                 unremarkable. `perplexity` wins.

Expressed WITHIN each example, so no cross-example scale choice is needed:

    zgap[i] = ( mean(lp_i) - min(lp_i) ) / std(lp_i)        # dimensionless, per example
    ZGAP(dataset) = mean over examples of zgap[i]

⚠️ Dimensionless and invariant to affine rescaling of the logprobs — which is what killed the rejected
first draft (see prereg §3.0: comparing the spread of min-logprob against mean-logprob smuggled in a
choice of measurement space, because PRR is rank-based and invariant to monotone transforms but standard
deviation is not).

    python scripts/checks/taxonomy_label_free.py
"""
import csv as _csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from luq.config import Config  # noqa: E402
from luq import cache  # noqa: E402

MODEL = "meta-llama/Meta-Llama-3.1-8B"
SLUG = cache._slug(MODEL)

# The namespaced datasets are invisible to a `cache/records/*` glob. Same map the existing drivers use
# (audit_realised_pools.py:37, attn_pool.py:60) rather than a second copy that can drift.
REGIME = {"expertqa": "expertqa_rp12", "asqa": "asqa_rp12", "factscore": "factscore_rp12"}

TAXONOMY_CSV = ROOT / "results" / f"floor_taxonomy__{SLUG}.csv"

# Registered null distribution (prereg §4). 4 CONCENTRATED x 3 SPREAD = 12 cross-group pairs;
# C(7,4) = 35 orderings. Fixed here so the result cannot be read against a moving bar.
NULL_TABLE = {12: (1.000, 0.029, "P1 SUPPORTED"),
              11: (0.917, 0.057, "suggestive only, NOT support"),
              10: (0.833, 0.114, "inconclusive")}
MIN_TOKENS = 2   # std() of a 1-token generation is undefined; such rows are excluded and COUNTED


def load_regime_labels():
    """The regime label per dataset, read from the taxonomy CSV rather than hard-coded.

    ⚠️ Reads `reading` (the regime) ONLY. The PRR columns in this file are label-derived and are exactly
    what R1 exists to avoid depending on — they are never used to compute ZGAP.
    """
    if not TAXONOMY_CSV.exists():
        sys.exit(f"missing {TAXONOMY_CSV} -- the regime labels come from there. Refusing to invent them.")
    labels = {}
    with open(TAXONOMY_CSV) as fh:
        for row in _csv.DictReader(fh):
            if row["rung"] != "ID":
                continue
            labels[row["eval"]] = row["reading"].strip()
    return labels


def zgap_for_dataset(ds):
    """Per-example zgap plus the generation lengths, computed from the CACHED LOGPROBS ALONE.

    No label is read here. That is the whole claim being tested, so it is worth stating in code: the only
    fields touched are `token_logprobs`, which is model behaviour, not correctness.
    """
    cfg = Config(model_name=MODEL, dataset=ds, ood_setting="ID", prompt_regime=REGIME.get(ds, ""))
    recs = cache.load_records(cfg.cache_dir, cache.run_key(MODEL, ds, "ID"))
    zg, lens, skipped = [], [], 0
    for r in recs:
        lp = np.asarray(r.get("token_logprobs", []), dtype=float)
        lp = lp[np.isfinite(lp)]
        if len(lp) < MIN_TOKENS:
            skipped += 1
            continue
        sd = float(lp.std())
        if sd <= 0:                      # every token identical: the ratio is undefined, not zero.
            skipped += 1                 # NEVER coerce an undefined value to a number -- it would read
            continue                     # as "measured, and perfectly flat".
        zg.append((float(lp.mean()) - float(lp.min())) / sd)
        lens.append(len(lp))
    if not zg:
        sys.exit(f"{ds}: no usable rows. Refusing to report a dataset with nothing measured.")
    return np.array(zg), np.array(lens), skipped, len(recs)


def cross_group_pairs_correct(values, labels, hi_label, lo_label):
    """How many (hi, lo) pairs are ordered as predicted: hi_label should score ABOVE lo_label.

    This is the Mann-Whitney U statistic counted directly, which is clearer here than a p-value from a
    library and makes the tie handling explicit: a tie counts as half, so it can never flatter the result.
    """
    hi = [v for v, l in zip(values, labels) if l == hi_label]
    lo = [v for v, l in zip(values, labels) if l == lo_label]
    correct = 0.0
    for a in hi:
        for b in lo:
            correct += 1.0 if a > b else (0.5 if a == b else 0.0)
    return correct, len(hi) * len(lo)


def spearman(a, b):
    """Rank correlation, implemented directly to avoid a scipy dependency in this script."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main():
    labels = load_regime_labels()
    datasets = sorted(labels)
    print(f"R1 taxonomy label-free test | {len(datasets)} datasets from {TAXONOMY_CSV.name}", flush=True)

    # ⚠️ Assert the realised count and FAIL LOUD. A glob over cache/records returns 7 while claiming 10;
    # this test's whole population is 8, and a short load would silently change the null distribution.
    if len(datasets) != 8:
        sys.exit(f"expected 8 long evals in the taxonomy, found {len(datasets)}: {datasets}. "
                 "The registered null distribution assumes 8. Refusing to run against a different pool.")

    rows = []
    for ds in datasets:
        zg, lens, skipped, n_tot = zgap_for_dataset(ds)
        rows.append({"dataset": ds, "regime": labels[ds], "zgap": round(float(zg.mean()), 4),
                     "zgap_median": round(float(np.median(zg)), 4),
                     "mean_gen_len": round(float(lens.mean()), 2),
                     "n_used": len(zg), "n_total": n_tot, "n_skipped": skipped})
        print(f"  {ds:14s} {labels[ds]:22s} ZGAP {zg.mean():6.3f}  len {lens.mean():7.1f}  "
              f"n={len(zg)}/{n_tot}" + (f"  ({skipped} skipped)" if skipped else ""), flush=True)

    # ---- PRIMARY TEST (prereg §4): the two-class population only. xsum is scored in §5, not dropped.
    two = [r for r in rows if r["regime"] in ("CONCENTRATED", "SPREAD")]
    if len(two) != 7:
        sys.exit(f"expected 7 two-class datasets, got {len(two)}. Registered null assumes 4x3.")

    vals = [r["zgap"] for r in two]
    labs = [r["regime"] for r in two]
    correct, total = cross_group_pairs_correct(vals, labs, "CONCENTRATED", "SPREAD")
    auc = correct / total

    print("\nRANKING BY ZGAP (descending). Registered prediction: all CONCENTRATED above all SPREAD.",
          flush=True)
    for r in sorted(rows, key=lambda r: -r["zgap"]):
        mark = "  <- NOT-IN-PROBABILITIES (reported, not scored)" if r["regime"].startswith("NOT") else ""
        print(f"  {r['zgap']:7.3f}  {r['dataset']:14s} {r['regime']}{mark}", flush=True)

    auc_r, p, reading = NULL_TABLE.get(int(correct), (auc, 0.20, "P1 FALSIFIED"))
    if int(correct) != correct:                       # a tie landed us between registered rows
        reading, p = "TIE PRESENT -- read as the lower row", 0.20
    print(f"\nPRIMARY: {correct:g}/{total} cross-group pairs correct, AUC {auc:.3f}, "
          f"one-sided p {p:.3f}  ->  {reading}", flush=True)
    if correct < total:
        print("  ⚠️ Anything short of perfect separation is a near miss, not support (prereg §4): with "
              "n=7 nothing weaker is worth acting on.", flush=True)

    # ---- P2: the competing simple explanation. Reported ALWAYS, pass or fail.
    # The max-of-L statistic grows ~sqrt(2 ln L), so a longer generation has a more extreme worst token
    # for purely statistical reasons, and these datasets differ in length by an order of magnitude.
    all_z = [r["zgap"] for r in rows]
    all_len = [r["mean_gen_len"] for r in rows]
    rho = spearman(np.array(all_z), np.array(all_len))
    len_correct, len_total = cross_group_pairs_correct([r["mean_gen_len"] for r in two], labs,
                                                       "CONCENTRATED", "SPREAD")
    print(f"\nP2 (length confound, always reported): Spearman(ZGAP, mean_gen_len) over 8 datasets "
          f"= {rho:+.3f}", flush=True)
    print(f"  Does LENGTH ALONE separate the groups? {len_correct:g}/{len_total} pairs "
          f"(AUC {len_correct/len_total:.3f}) vs ZGAP's {correct:g}/{total} (AUC {auc:.3f}).", flush=True)
    if len_correct >= correct:
        print("  ⚠️ LENGTH SEPARATES AT LEAST AS WELL AS ZGAP. Per the pre-registration the mechanism "
              "claim is NOT established even if P1 passed: the honest report is 'regime is predictable "
              "from length', which is weaker and much less interesting. It goes in the write-up as that.",
              flush=True)

    # ---- §5: the third class, scored not excluded.
    xs = [r for r in rows if r["regime"].startswith("NOT")]
    for r in xs:
        rank = sorted(rows, key=lambda q: -q["zgap"]).index(r) + 1
        print(f"\n§5 THIRD CLASS: {r['dataset']} (NOT-IN-PROBABILITIES) has ZGAP {r['zgap']:.3f}, "
              f"rank {rank}/{len(rows)}. No rule is registered for detecting this class -- with n=1 any "
              "rule would be fitted to a single point. Recorded as an observation, NOT a validated rule.",
              flush=True)

    for r in rows:
        r["primary_pairs_correct"] = correct
        r["primary_pairs_total"] = total
        r["primary_auc"] = round(auc, 4)
        r["primary_reading"] = reading
        r["length_pairs_correct"] = len_correct
        r["spearman_zgap_len"] = round(rho, 4)
    out = ROOT / "results" / f"regime_R1_label_free__{SLUG}.csv"
    with open(out, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
