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


# Which POOLED feature files each driver actually opens. ⚠️ CORRECTED 2026-08-11 after this check
# false-positived and killed all 8 clean-span ladder jobs three minutes in. It demanded
# saplma/ptrue_accurate/lookback for EVERY driver, but:
#   * probedriftlong computes SAPLMA by mean-pooling the PERTOK states (`Xmean = np.stack([s.mean(0)
#     ...])`, :494) and only touches load_features inside `for bm in active_base` — i.e. solely when
#     --baselines is passed, and even then it degrades to a loud blank cell rather than failing.
#   * contribution_ladder never calls load_features at all (pertok + record logprobs only).
#   * ood_onegrid genuinely does read them: it is the driver that scores linear/ptrue/lookback.
# The shadow regime cache/med_quad_clean/ deliberately ships pertok+records WITHOUT pooled features,
# which is correct for this ladder — so demanding them blocked a valid run.
# ⚠️ A guard that fails a CORRECT run teaches people to bypass guards. Being accurate about what each
# driver reads matters as much as failing loud.
FEATURES_READ = {
    "probedriftlong": [],            # pertok only, for the no --baselines invocation
    "contribution_ladder": [],       # pertok + record logprobs only
    "ood_onegrid": ["saplma", "ptrue_accurate", "lookback"],
}


def check_caches(datasets, needed=None):
    """Every cache file the run will open must EXIST, checked per dataset under its own prompt regime.

    ADDED 2026-08-04, after the widened `ood_onegrid` crashed all six rerun jobs four hours in. It
    resolved the cache dir ONCE off dataset="sciq" and reused it for all ten, so asqa/expertqa/factscore
    -- whose caches live in a regime namespace (`cache/asqa_rp12/`) -- were looked up in the wrong
    directory. Same family as the cohort defects: a value that was right for the original four core sets
    and silently wrong once the cohort widened.

    This only STATS the files (no loading), so it is safe on a login node and costs seconds.
    """
    needed = [] if needed is None else needed
    from luq import cache as _cache
    from luq.config import Config
    from attn_pool import PROMPT_REGIME
    MODEL = "meta-llama/Meta-Llama-3.1-8B"
    fails = []
    for d in datasets:
        cd = Config(model_name=MODEL, dataset=d, ood_setting="ID",
                    prompt_regime=PROMPT_REGIME.get(d, "")).cache_dir
        want = [cd / "pertok" / f"{_cache._slug(MODEL)}__{d}__ID__L15.npz"]
        want += [cd / "features" / f"{_cache._slug(MODEL)}__{d}__ID__{fm}.npz" for fm in needed]
        miss = [p.name for p in want if not p.exists()]
        if miss:
            fails.append(f"{d}: {len(miss)} cache file(s) absent under {cd} -> {miss}")
    if not fails:
        print(f"  ✅ caches resolve for all {len(datasets)} datasets (pertok L15 + {len(needed)} pooled)")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True, help="comma-separated driver module names")
    ap.add_argument("--skip-caches", action="store_true",
                    help="skip the cache-existence check (for drivers that read neither cache)")
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
    if not args.skip_caches:
        # Check the cohort each named driver actually reads, not a fixed list.
        need = set()
        for n in names:
            for attr, want, _ in EXPECT[n]:
                need |= want
        feats = sorted({f for n in names for f in FEATURES_READ.get(n, [])})
        fails += check_caches(sorted(need), feats)
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
