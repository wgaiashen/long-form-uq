#!/usr/bin/env python
"""PREFLIGHT: refuse to run a grid job whose driver disagrees with the canonical cohort.

⭐ THIS IS THE FIX AT THE ORIGIN. Three defects on 2026-08-03/04 were all the same thing: a driver's
hard-coded dataset list was quietly smaller than the study's, and the result was a WRONG TABLE rather
than an error. Checking the lists by hand is what already failed — twice, including once while
explicitly "triple-checking". So this makes the disagreement a HARD FAILURE, and puts it at SUBMIT
TIME, before any compute is spent.

Called at the top of every grid job script. Exits 1 on any mismatch, which kills the job in seconds
instead of producing a plausible-looking table eight hours later.

⚠️ WHAT IT DOES NOT DO. It cannot know that a driver SHOULD cover all ten — some scripts are correctly
scoped to a subset. So a driver states its intent by being listed here with its expected cohort. That
is the point: the expectation becomes explicit and version-controlled instead of implicit in a literal.

    python scripts/checks/preflight_cohort.py --driver contribution_ladder,ood_onegrid
    python scripts/checks/preflight_cohort.py --driver probedriftlong
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from cohort import CANONICAL_10, LONG_8, SHORT_2, FINE, BROAD, XL_RUNGS  # noqa: E402

# driver -> [(attribute, expected set, what it is)]. A driver ABSENT from here is not checked, and that
# absence is itself a decision someone has to make deliberately.
EXPECT = {
    "contribution_ladder": [("CANDIDATE_SOURCES", set(CANONICAL_10), "XL training-source pool")],
    "ood_onegrid": [("AVAIL", set(CANONICAL_10), "XL training-source pool + datasets loaded")],
    "probedriftlong": [("LONG", set(LONG_8), "long evals"),
                       ("SHORT", set(SHORT_2), "Long->Short targets"),
                       ("LONG_SRC", set(LONG_8), "long training-source pool")],
}


def check_driver(name):
    fails = []
    try:
        mod = __import__(name)
    except Exception as e:                       # an unimportable driver cannot be verified -> refuse
        return [f"{name}: FAILED TO IMPORT ({type(e).__name__}: {e})"]
    for attr, want, what in EXPECT[name]:
        got = set(getattr(mod, attr, []) or [])
        if got != want:
            miss, extra = sorted(want - got), sorted(got - want)
            fails.append(f"{name}.{attr} ({what}): {len(got)} entries, expected {len(want)}"
                         + (f" | MISSING {miss}" if miss else "")
                         + (f" | UNEXPECTED {extra}" if extra else ""))
        else:
            print(f"  ✅ {name}.{attr:20s} {len(got):2d} entries  ({what})")
    return fails


def check_families():
    """The family map must agree with cohort.py wherever a driver keeps its own copy — that duplication
    is how the taxonomy drifted before the split was applied everywhere."""
    fails = []
    for name in ("xl_rungs", "probedriftlong"):
        try:
            mod = __import__(name)
        except Exception:
            continue
        f = getattr(mod, "FINE", None)
        if not f:
            continue
        for ds in ("expertqa", "factscore", "asqa"):
            if ds in f and ds in FINE:
                # probedriftlong names the QA family "correctness_qa", xl_rungs "long_qa" — different
                # labels, same grouping. What must match is WHICH DATASETS SHARE A FAMILY, not the name.
                same_here = {d for d in f if f[d] == f[ds]}
                same_canon = {d for d in FINE if FINE[d] == FINE[ds]} & set(f)
                if same_here != same_canon:
                    fails.append(f"{name}.FINE: {ds} groups with {sorted(same_here)}, "
                                 f"canonical says {sorted(same_canon)}")
        print(f"  ✅ {name}.FINE families agree with cohort.py")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="comma-separated driver module names")
    args = ap.parse_args()
    names = [d.strip() for d in args.driver.split(",") if d.strip()]
    unknown = [d for d in names if d not in EXPECT]
    if unknown:
        sys.exit(f"PREFLIGHT: no cohort expectation registered for {unknown}. Add one to "
                 f"preflight_cohort.EXPECT — an unchecked driver is how this class of bug got in.")
    print(f"PREFLIGHT cohort check | canonical = {len(CANONICAL_10)} datasets")
    fails = []
    for n in names:
        fails += check_driver(n)
    fails += check_families()
    if fails:
        print("\n❌ PREFLIGHT FAILED — refusing to start the job:")
        for f in fails:
            print(f"   {f}")
        print("\nA cohort smaller than canonical produces a WRONG TABLE, not an error. Fix the driver "
              "(or register a deliberate exception) before resubmitting.")
        sys.exit(1)
    print("\n✅ PREFLIGHT OK — cohorts and families match the canonical definition.")


if __name__ == "__main__":
    main()
