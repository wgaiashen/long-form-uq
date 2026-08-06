"""S4 — summarise the top-k surprisal prior run (pre-reg: prereg/S4_topk_surprisal_prior.md).

Reads the per-eval CSVs written by fixed_prior_ladder.py and answers the three questions the
pre-registration asks, in this order:

  1. does FIXED top-k pooling beat the unsupervised floor (msp_min)?
  2. does it beat LEARNED attention pooling (the thing we are trying to improve)?
  3. does it beat its OWN shuffled control (same number of tokens, random positions)?

(3) is the one that matters. (1) and (2) can both be passed by a method that is really just
"pool over fewer tokens"; only beating the shuffled control shows that WHICH tokens were chosen
carries information.

Method names in the CSV are code names; they are translated on output because a code name in a
report is unreadable. The mapping is read off fixed_prior_ladder.py:143-195:
  floor_min          -> unsupervised floor, msp_min
  armA               -> learned attention pooling (query trained, temperature selected)
  armB               -> attention pooling, query FROZEN at init (does learning where to look help?)
  armC_<prior>       -> pooling weights FIXED to the recipe, nothing learned about where to look
  armD_<prior>       -> recipe used as INITIALISATION, query then free to learn

Coverage is printed first and never silently completed: a dataset missing rungs is stated, and the
missing rungs are named. Partial grids are not aggregated into a headline.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

ROOT = "/rds/general/user/gs925/home/gs925-msc_project/msc-project-gs925"
RESULTS = os.path.join(ROOT, "results")
ALL_RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]

PRETTY = {
    "floor_min": "unsupervised floor (msp_min)",
    "armA": "learned attention pooling",
    "armB": "attention pooling, query frozen",
}


def pretty(m):
    if m in PRETTY:
        return PRETTY[m]
    for code, kind in (("armC_", "fixed"), ("armD_", "init")):
        if m.startswith(code):
            p = m[len(code):]
            return f"{describe_prior(p)} [{kind}]"
    return m


def describe_prior(p):
    """Turn a prior spec into words. 'topk:5:shuf' -> 'top-5 surprisal, SHUFFLED control'."""
    if p == "nll":
        return "every-token surprisal"
    parts = p.split(":")
    if parts[0] != "topk":
        return p
    k = parts[1]
    label = f"top-{k} most surprising" if not k.startswith("0.") else f"top-{float(k):.0%} most surprising"
    for extra in parts[2:]:
        if extra == "shuf":
            label += ", SHUFFLED control"
        elif extra.startswith("floor="):
            label += f", floor {extra.split('=')[1]}"
    return label


def load():
    rows = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(RESULTS, "s4_topk_*.csv"))):
        for r in csv.DictReader(open(path)):
            if not r.get("prr_mean"):
                continue
            # The driver writes paired-margin records into the SAME method column, prefixed
            # "VERDICT:". They are differences, not PRRs, so averaging them alongside methods would
            # mix two different quantities in one column. Drop them; margins are recomputed here.
            if r["method"].startswith("VERDICT:"):
                continue
            key = (r["eval"], r["rung"], r["method"])
            if key in seen:
                # Recovery files deliberately repeat the ID rung. Identical value = fine (it is a free
                # reproducibility check); a DIFFERENT value means two populations, which must not merge.
                prev = next(x for x in rows if (x["eval"], x["rung"], x["method"]) == key)
                if abs(float(prev["prr_mean"]) - float(r["prr_mean"])) > 1e-9:
                    raise SystemExit(
                        f"CONFLICT {key}: {prev['prr_mean']} vs {r['prr_mean']} in {os.path.basename(path)} "
                        "— two different populations, refusing to aggregate")
                continue
            seen.add(key)
            r["prr_mean"] = float(r["prr_mean"])
            rows.append(r)
    return rows


def coverage(rows):
    have = defaultdict(set)
    for r in rows:
        have[r["eval"]].add(r["rung"])
    print("=" * 78)
    print("COVERAGE — ProbeDriftLong, 8 long evals x 5 rungs")
    print("=" * 78)
    complete, partial = [], []
    for ds in sorted(have):
        missing = [g for g in ALL_RUNGS if g not in have[ds]]
        if missing:
            partial.append(ds)
            print(f"  {ds:15s} {5 - len(missing)}/5  MISSING: {', '.join(missing)}")
        else:
            complete.append(ds)
            print(f"  {ds:15s} 5/5  complete")
    print(f"\n  complete: {len(complete)}/8 datasets -> {', '.join(complete) if complete else 'none'}")
    if partial:
        print(f"  PARTIAL (not aggregated into any headline): {', '.join(partial)}")
    return complete


def table(rows, evals, rungs, title):
    """Per-dataset margins of each top-k arm against the three references."""
    sel = [r for r in rows if r["eval"] in evals and r["rung"] in rungs]
    by = {(r["eval"], r["rung"], r["method"]): r["prr_mean"] for r in sel}
    methods = sorted({r["method"] for r in sel})
    topk = [m for m in methods if "topk" in m and "shuf" not in m]

    print("\n" + "=" * 78)
    print(title)
    print(f"population: {len(evals)} evals x {len(rungs)} rungs")
    print("=" * 78)

    agg = defaultdict(list)
    for m in methods:                       # EVERY method, not a hand-listed subset. An earlier version
        for ds in evals:                    # listed arms explicitly and silently dropped the `nll` prior,
                                            # which is the pre-registered comparator — so the decision rule
                                            # could not be evaluated at all.
            for g in rungs:
                if (ds, g, m) in by:
                    agg[m].append(by[(ds, g, m)])
    print(f"\n{'method':52s} {'mean PRR':>9s} {'n':>4s}")
    print("-" * 68)
    for m, vals in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"{pretty(m):52s} {sum(vals)/len(vals):+9.3f} {len(vals):4d}")

    # The decisive contrast: each top-k arm vs its own shuffled twin, paired per cell.
    print(f"\n{'DECISIVE — real vs its own shuffled control (paired)':68s}")
    print("-" * 68)
    for m in topk:
        shuf = m + ":shuf" if not m.endswith(":shuf") else m
        pairs = [(by[(ds, g, m)], by[(ds, g, shuf)])
                 for ds in evals for g in rungs
                 if (ds, g, m) in by and (ds, g, shuf) in by]
        if not pairs:
            continue
        d = [a - b for a, b in pairs]
        wins = sum(1 for x in d if x > 0)
        print(f"{pretty(m):52s} {sum(d)/len(d):+9.3f}  ({wins}/{len(d)} cells)")

    # The PRE-REGISTERED decision rule (prereg/S4 §"Decision rule"): carry the method forward only if a
    # top-k arm beats the PAIRED every-token-surprisal prior by > 0.02. The comparator is `nll`, not the
    # learned pooler and not the floor — both of those answer different questions.
    print(f"\n{'PRE-REGISTERED RULE — top-k vs paired every-token surprisal (>+0.02 to carry)':68s}")
    print("-" * 68)
    for kind in ("armC", "armD"):
        ref = f"{kind}_nll"
        for m in [x for x in topk if x.startswith(kind + "_")]:
            pairs = [(by[(ds, g, m)], by[(ds, g, ref)])
                     for ds in evals for g in rungs
                     if (ds, g, m) in by and (ds, g, ref) in by]
            if not pairs:
                continue
            d = [a - b for a, b in pairs]
            mu = sum(d) / len(d)
            print(f"{pretty(m):52s} {mu:+9.3f}  {'CARRY' if mu > 0.02 else 'drop'}")
    return by


def main():
    rows = load()
    if not rows:
        raise SystemExit("no rows found")
    complete = coverage(rows)
    if not complete:
        raise SystemExit("\nno dataset has a complete grid — nothing to aggregate")

    table(rows, complete, ["ID"], "IN-DISTRIBUTION (ID rung only)")
    ood = [g for g in ALL_RUNGS if g != "ID"]
    table(rows, complete, ood, "OUT-OF-DISTRIBUTION (all 4 OOD rungs)")

    # Per-dataset OOD detail: the taxonomy claim is that concentrated datasets favour small k.
    print("\n" + "=" * 78)
    print("PER-DATASET, OOD rungs pooled — best top-k arm vs the two references")
    print("=" * 78)
    by = {(r["eval"], r["rung"], r["method"]): r["prr_mean"] for r in rows}
    for ds in complete:
        cells = [g for g in ood if (ds, g, "armA") in by]
        def mean(m):
            v = [by[(ds, g, m)] for g in cells if (ds, g, m) in by]
            return sum(v) / len(v) if v else None
        topk = sorted({m for (e, g, m) in by if e == ds and "topk" in m and "shuf" not in m})
        scored = [(mean(m), m) for m in topk if mean(m) is not None]
        if not scored:
            continue
        best, bm = max(scored)
        f, a = mean("floor_min"), mean("armA")
        print(f"\n  {ds}  (n={len(cells)} OOD cells)")
        print(f"    unsupervised floor       {f:+.3f}")
        print(f"    learned attention        {a:+.3f}")
        print(f"    best fixed/init recipe   {best:+.3f}  = {pretty(bm)}")
        print(f"    -> vs floor {best - f:+.3f} | vs learned attention {best - a:+.3f}")


if __name__ == "__main__":
    sys.exit(main())
