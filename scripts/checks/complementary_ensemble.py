#!/usr/bin/env python
"""Does CAWSA supply a complementary signal to SAPLMA?

Pre-registration: prereg/cawsa_saplma_ensemble.md (committed before any ensemble PRR was read).

THE QUESTION (14 August 2026 supervision meeting, P0 #2)
--------------------------------------------------------
Does CAWSA provide a complementary uncertainty signal to SAPLMA, such that a simple combination can
retain SAPLMA's strong near-ID performance while benefiting from CAWSA under stronger shift?

    unconstrained = wmsp_norm    (unregularised activation-weighted token NLL)
    CAWSA λ=2     = wmsp_shrink2 (weights shrunk toward uniform; λ fixed at 2, never swept)

WHY THIS IS A PURE POST-HOC READ
--------------------------------
Every number M6 asks for is a function of the PER-EXAMPLE uncertainty vectors, not of per-cell PRRs.
pbs/pdl_perex_ens.pbs runs probedriftlong.py once with --perex-dir, which persists
unc__<method> at (n_seeds, n_te) float64 per cell under results/pdl_perex_ens/. So this script does
no training, touches no GPU, and can be re-run for free every time a follow-up question is asked.

READS results/pdl_perex_ens/, NOT results/pdl_perex/. The filename patterns are identical; the
latter is the floors+SAPLMA-only population read by orthogonality_map.py / pdl_significance.py /
length_blend_perinstance.py and must not be confused with this one.

WHAT IS PRE-REGISTERED, AND THEREFORE NOT DECIDED HERE
------------------------------------------------------
  * primary ensemble = rankavg{CAWSA λ=2, SAPLMA}, equal weight, one combiner;
  * unit of analysis = DATASET, n = 8 (the 32 OOD cells are NOT 32 independent observations: the
    floors do not depend on the training pool, so cell pooling is 4x pseudo-replication);
  * estimand A = per-dataset mean over its 4 OOD rungs, ensemble - SAPLMA;
  * estimand B = the same on the 8 ID cells;
  * ID reference scale 0.0044 = the measured seed-to-seed sd of SAPLMA's ID macro. An interpretive
    scale, NOT a pass/fail target and NOT an invented effect size;
  * control = rankavg{SAPLMA, attention} -- SECONDARY mechanism evidence only;
  * references = rankavg{msp_min, SAPLMA} and rankavg{unconstrained, SAPLMA}, recomputed from these same
    vectors so everything is internally comparable. They never enter a winner-selection sweep.

rankavg IS TEST-COHORT DEPENDENT and that is reported, not hidden. rankdata ranks within the
vector it is given, i.e. within the cell's test cohort, so the score of response i depends on which
other responses are in the cohort. Audited 2026-08-17 on xsum/DiffTask-long (n=2000): 2/400 random
pairs flip order between the full cohort and a 200-example cohort, and PRR on a fixed 1000 rows moves
+0.0006 between in-subset and inherited ranks. Small, but it makes this a RANK-ENSEMBLE DIAGNOSTIC
rather than a deployable per-response estimator. §5 prints the measurement so the caveat carries its
own evidence.

    python scripts/checks/complementary_ensemble.py
    python scripts/checks/complementary_ensemble.py --perex-dir results/pdl_perex_ens
"""
import argparse
import csv as _csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

from luq import results                                                    # noqa: E402
from orthogonality_map import self_agreement, cross_agreement, rho         # noqa: E402


# The two combiners. These are BYTE-EQUIVALENT to ensemble_ladder.rankavg / .zavg, and
# --verify-combiner imports that module and asserts it numerically rather than trusting this comment.
# They are restated here rather than imported because ensemble_ladder pulls in torch, transformers,
# attn_pool and probedriftlong at module level -- a ~2 minute import of the whole training stack for
# two three-line functions, in a script whose entire point is to be a cheap read on the login node.
def rankavg(*us):
    """rank-average of uncertainty vectors (higher = more uncertain).

    rankdata ranks WITHIN the vector it is given, so this is TEST-COHORT DEPENDENT -- see the
    module docstring and the measured §rankavg section below.
    """
    from scipy.stats import rankdata
    return np.mean([rankdata(u) for u in us], axis=0)


def zavg(*us):
    def z(u):
        s = np.std(u)
        return (u - np.mean(u)) / s if s > 0 else u * 0.0
    return np.mean([z(u) for u in us], axis=0)


def verify_combiner():
    """Prove the local combiners equal the shared ones instead of asserting it in a comment."""
    from ensemble_ladder import rankavg as ra_ref, zavg as za_ref
    rng = np.random.RandomState(0)
    for _ in range(20):
        a, b = rng.randn(500), rng.randn(500)
        assert np.allclose(rankavg(a, b), ra_ref(a, b), atol=0, rtol=0), "rankavg diverged"
        assert np.allclose(zavg(a, b), za_ref(a, b), atol=0, rtol=0), "zavg diverged"
    print("combiners are bit-identical to ensemble_ladder.rankavg / .zavg (20 random vectors)")

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B"
# n = 8 datasets. This is the unit of analysis.
LONG = ["pubmed_qa", "med_quad", "asqa", "xsum", "cnn_dailymail", "samsum", "expertqa", "factscore"]
# REPORT / Hidden Failures rung order. LOO comes before SameTask. Do not silently reorder.
RUNGS = ["ID", "LOO-long", "SameTask-long", "DiffTask-long", "1ds-Diff-long"]
OOD = RUNGS[1:]

# The components whose canonical values must be reproduced before any ensemble PRR is read.
GATE_COMPONENTS = ["floor_min", "floor_ppl", "floor_sum", "saplma", "uniform", "attention",
                   "wmsp_norm", "wmsp_shrink2"]

# PER-CELL GATE (strengthened 2026-08-20). The earlier version compared a computed MACRO against a
# hardcoded table of seven Llama values at 0.02 tolerance. That is far weaker than it looked: a macro
# can match while individual cells are wrong in cancelling directions, and the hardcoded numbers are
# model-specific so the gate simply could not run on a second population. It now reads the master CSV
# and compares EVERY (component, eval, rung) cell -- 8 x 8 x 5 = up to 320 comparisons per model.
GATE_CELL_TOL = 1e-3    # per-cell; the masters are stored to 4 dp so this is the resolution floor
SD_MULT = 2.0           # trained components: bar = 2 x the master's own per-cell seed sd, when recorded
DETERMINISTIC_COMPONENTS = {"floor_min", "floor_ppl", "floor_sum"}

# The two masters use DIFFERENT schemas, so the reader auto-detects rather than assuming:
#   Llama: rung,eval,method,prr,...            with DISPLAY names  ("wMSP-shrink@2", "SAPLMA")
#   Qwen : model,eval,rung,method,prr_mean,... with RAW keys       ("wmsp_shrink2",  "saplma")
# ALIAS is copied from scripts/checks/assemble_pdl_table.py, which is what BUILT the Llama master --
# not from luq.method_names, whose display strings differ ("wMSP-normalised" vs the master's "wMSP-norm").
# The ID reference scale, DERIVED FROM MEASURED VARIABILITY, not invented (the ensemble registration §5.1). SAPLMA's
# ID macro across the three Llama seed sets is +0.5889 / +0.5822 / +0.5905 -> sd 0.0044. An ID delta
# whose bootstrap CI lies entirely below -0.0044 exceeds run-to-run noise and counts as MATERIAL.
# An interpretive scale, NOT a pass/fail target.
ID_SCALE = 0.0044

ALIAS_TO_MASTER = {"floor_sum": "msp_sum", "floor_ppl": "perplexity", "floor_min": "msp_min",
                   "wmsp_norm": "wMSP-norm", "wmsp_shrink2": "wMSP-shrink@2",
                   "saplma": "SAPLMA", "uniform": "armB(mean-pool)", "attention": "armA(attention)"}

NAMES = {"wmsp_shrink2": "CAWSA λ=2", "wmsp_norm": "unconstrained", "saplma": "SAPLMA",
         "attention": "attention-pool", "uniform": "mean-pool", "floor_min": "msp_min",
         "floor_ppl": "perplexity", "floor_sum": "msp_sum"}

# (label, component_a, component_b) -- PRIMARY first, then the control, then the references.
ENSEMBLES = [
    ("PRIMARY  CAWSA λ=2 + SAPLMA", "wmsp_shrink2", "saplma"),
    ("CONTROL  SAPLMA + attention-pool", "saplma", "attention"),
    ("REF      msp_min + SAPLMA", "floor_min", "saplma"),
    ("REF      unconstrained + SAPLMA", "wmsp_norm", "saplma"),
]


def load_master(model_slug):
    """{(component_key, eval, rung): prr} from the canonical master, schema auto-detected.

    Returns raw-component-keyed entries so the caller never has to know which naming a master uses.
    Raises if the file is missing: a gate that silently finds nothing to compare against is not a gate.
    """
    path = ROOT / "results" / f"pdl_master__{model_slug}.csv"
    if not path.exists():
        raise SystemExit(f"GATE: canonical master not found at {path}. Refusing to run without the "
                         "reference the continuity gate exists to check against.")
    with open(path) as f:
        rdr = _csv.DictReader(f)
        cols = rdr.fieldnames or []
        prr_col = "prr" if "prr" in cols else ("prr_mean" if "prr_mean" in cols else None)
        std_col = "prr_std" if "prr_std" in cols else None
        if prr_col is None:
            raise SystemExit(f"GATE: {path.name} has neither a 'prr' nor a 'prr_mean' column ({cols}).")
        rows = [r for r in rdr if r.get("rung") != "rung"]
    # Detect naming: if any master method equals a DISPLAY name we know, it is display-keyed.
    methods = {r["method"] for r in rows}
    display_keyed = bool(methods & set(ALIAS_TO_MASTER.values()))
    out = {}
    for r in rows:
        try:
            v = float(r[prr_col])
        except (ValueError, KeyError, TypeError):
            continue
        sd = None
        if std_col:
            try:
                sd = float(r[std_col])
            except (ValueError, TypeError, KeyError):
                sd = None
        for raw, disp in ALIAS_TO_MASTER.items():
            if r["method"] == (disp if display_keyed else raw):
                out[(raw, r["eval"], r["rung"])] = (v, sd)
    print(f"  master {path.name}: {prr_col!r} column, "
          f"{'display' if display_keyed else 'raw'}-keyed methods, {len(out)} component cells"
          + (f", {std_col!r} available" if std_col else ", no per-cell sd (strict bar only)"))
    return out


def load_cells(perex_dir, slug):
    """{(eval, rung): (methods dict of (n_seeds, n_te), y)} plus a coverage report."""
    cells, missing = {}, []
    for d in LONG:
        for rg in RUNGS:
            p = Path(perex_dir) / f"{d}__{rg}__{slug}.npz"
            if not p.exists():
                missing.append(f"{d}/{rg}")
                continue
            z = np.load(p, allow_pickle=True)
            meth = {k[len("unc__"):]: z[k] for k in z.files
                    if k.startswith("unc__") and k != "unc__fair_floor"}
            cells[(d, rg)] = (meth, np.asarray(z["y"], float))
    return cells, missing


# SEED HANDLING -- THE ONE THING THAT MUST NOT BE GOT WRONG HERE.
#
# There are two ways to turn 3 seeds into one PRR and they are NOT the same number:
#   (a) mean over seeds of PRR(y, unc[s])            <- what pdl_master and the ladder report
#   (b) PRR(y, mean over seeds of unc[s])            <- scores a 3-member self-ensemble
# (b) is systematically HIGHER because averaging cancels seed noise. Measured on this grid: SAPLMA at
# LOO-long is +0.2369 under (a) and +0.287 under (b) -- a +0.050 inflation, larger than any effect M6
# is looking for.
#
# A first version of this script used (b) for components AND ensembles. That is doubly wrong: it does
# not compare like-with-like against the master, and it silently rewards whichever score gets averaged
# -- i.e. it would flatter the ensemble, in exactly the direction that would make the result look
# good. ensemble_ladder.py has always done it correctly: it forms the ensemble INSIDE the seed loop
# and averages the resulting PRRs.
#
# So: everything reported here is (a). Ensembles are formed PER SEED from that seed's own component
# vectors, scored, and the PRRs averaged. Seed-averaged vectors are used ONLY where the quantity is
# not a PRR (the correlation ceiling machinery, which needs per-seed vectors anyway).
def per_seed_prr(y, V):
    """PRR per seed. V is (n_seeds, n_te)."""
    V = np.asarray(V, float)
    return np.array([results.prr(y, V[s]) for s in range(V.shape[0])], float)


def component_prr(y, meth, m):
    """Convention (a): mean over seeds of the per-seed PRR."""
    return float(np.mean(per_seed_prr(y, meth[m])))


def ensemble_prr(y, meth, a, b, mode="rank"):
    """Convention (a) for an ensemble: combine WITHIN each seed, score, then average the PRRs."""
    A, B = np.asarray(meth[a], float), np.asarray(meth[b], float)
    ns = min(A.shape[0], B.shape[0])
    comb = rankavg if mode == "rank" else zavg
    return float(np.mean([results.prr(y, comb(A[s], B[s])) for s in range(ns)]))


def seed_mean(V):
    """Seed-averaged vector. NOT for PRR -- see the note above. Used only for cohort diagnostics."""
    return np.mean(np.asarray(V, float), axis=0)


def wilcoxon_exact(d):
    """Exact two-sided Wilcoxon signed-rank p over the 8 datasets.

    scipy's reference implementation, with method='exact' forced: at n = 8 the normal approximation
    is meaningless, and a hand-rolled enumeration is one convention choice away from being wrong (a
    first version here centred the positive-rank sum instead of using scipy's min-statistic and
    returned 0.0859 where the exact answer is 0.0781). Zeros are dropped, which is what
    zero_method='wilcox' does and what the meeting's protocol implies (a dataset with exactly no
    change carries no directional information).
    """
    d = np.asarray([x for x in d if x != 0.0], float)
    n = len(d)
    if n == 0:
        return float("nan"), 0
    from scipy.stats import wilcoxon
    return float(wilcoxon(d, alternative="two-sided", method="exact").pvalue), n


def boot_ci_mean(d, b=10000, seed=0):
    """Bootstrap 95% CI on the macro mean of the n=8 per-DATASET deltas (resamples datasets)."""
    d = np.asarray(d, float)
    rng = np.random.RandomState(seed)
    m = np.array([np.mean(d[rng.randint(0, len(d), len(d))]) for _ in range(b)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="population to analyse. Default is the Llama development population, so every "
                         "existing invocation stays byte-identical.")
    ap.add_argument("--perex-dir", default=None,
                    help="per-example sidecar dir. Default: results/pdl_perex_ens for Llama, "
                         "results/pdl_perex_ens_<slug> otherwise.")
    ap.add_argument("--exclude", default="",
                    help="comma-separated datasets to DROP, for a sensitivity arm (e.g. expertqa). "
                         "The excluded run is a SENSITIVITY and never replaces the full-grid primary.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify-combiner", action="store_true",
                    help="import ensemble_ladder (slow: pulls in torch) and assert the local rankavg/"
                         "zavg are bit-identical to the shared ones. Run once, not every time.")
    args = ap.parse_args()

    if args.verify_combiner:
        verify_combiner()

    # model-derived paths; nothing about the Llama invocation changes
    from luq import cache as _cache
    slug = _cache._slug(args.model)
    global LONG
    excluded = [d for d in args.exclude.split(",") if d]
    if excluded:
        missing = [d for d in excluded if d not in LONG]
        if missing:
            raise SystemExit(f"--exclude names datasets not in the grid: {missing}")
        LONG = [d for d in LONG if d not in excluded]
    perex_dir = args.perex_dir or str(
        ROOT / "results" / ("pdl_perex_ens" if args.model == DEFAULT_MODEL
                            else f"pdl_perex_ens_{slug}"))
    suffix = ("__excl-" + "-".join(excluded)) if excluded else ""
    out_path = args.out or str(ROOT / "results" / f"complementary_ensemble__{slug}{suffix}.csv")

    print("=" * 104)
    print("CAWSA + SAPLMA: is the combination better than SAPLMA far OOD without losing ID?")
    print(f"Population: ProbeDriftLong long grid, {len(LONG)} evals x 5 rungs, 3 seeds, {args.model}.")
    if excluded:
        print(f"SENSITIVITY ARM -- EXCLUDED: {excluded}. This never replaces the full-grid primary.")
    print(f"Source: {perex_dir}   (per-example sidecars; no training, no GPU)")
    print("=" * 104)

    cells, missing = load_cells(perex_dir, slug)
    print(f"\nCOVERAGE: {len(cells)}/40 cells")
    if missing:
        print(f"MISSING ({len(missing)}): {', '.join(missing)}")
        print("   Reported as INCOMPLETE. A partial grid is never presented as the whole one.")
    if not cells:
        raise SystemExit(f"no sidecars under {perex_dir} -- run the component pass first")

    rows = []

    # ---------------------------------------------------------------- GATE 1: component continuity
    print("\n" + "-" * 104)
    print("GATE 1 -- COMPONENT CONTINUITY vs the canonical pdl_master, PER CELL")
    print(f"         Every (component, eval, rung) compared at tol {GATE_CELL_TOL:g}. A macro-only check")
    print("         can pass while individual cells are wrong in cancelling directions.")
    print("-" * 104)
    master = load_master(slug)
    gate_fail, gate_warn, n_cmp, worst, worst_at = [], [], 0, 0.0, None
    signed = {}
    print(f"{'component':16s}{'cells':>7s}{'max|d|':>11s}{'macro ID':>10s}{'macro OOD':>11s}  status")
    for m in GATE_COMPONENTS:
        got_cells, diffs = {}, []
        for (d, rg), (meth, y) in cells.items():
            if m not in meth:
                continue
            got = component_prr(y, meth, m)
            got_cells[(d, rg)] = got
            ent = master.get((m, d, rg))
            if ent is None:
                continue
            exp, sd = ent
            n_cmp += 1
            dv = got - exp
            diffs.append(abs(dv))
            if abs(dv) > worst:
                worst, worst_at = abs(dv), f"{m}/{d}/{rg}"
            # TWO-TIER BAR. Deterministic components (no training) must match absolutely: a device
            # or a library version cannot excuse them. Stochastically TRAINED components are judged
            # against the master's own recorded seed spread where the schema provides it, because a
            # re-fit on a different device takes a different arithmetic path and is a RE-DRAW, not a
            # reproduction. Criterion pre-specified in scripts/checks/continuity_device_diagnosis.py,
            # written when only one dataset had landed. Run that script for the bias check, which is
            # the part that decides whether a re-draw is acceptable at all.
            trained = m not in DETERMINISTIC_COMPONENTS
            noise_aware = trained and sd is not None
            bar = max(GATE_CELL_TOL, SD_MULT * sd) if noise_aware else GATE_CELL_TOL
            if abs(dv) > bar:
                msg = (f"{m}/{d}/{rg}: got {got:+.6f} vs master {exp:+.6f} "
                       f"(d {dv:+.2e}, bar {bar:.2e})")
                # STOP vs WARN, per the rule pre-specified in continuity_device_diagnosis.py before
                # the full grid existed: a component that involves TRAINING may legitimately re-draw
                # when refitted on another device, so exceedance alone is reported, not fatal. What is
                # fatal is a deterministic component moving at all, or a trained component shifting
                # ONE-SIDEDLY -- that is a biased estimator rather than a re-draw, and is tested below.
                (gate_warn if noise_aware else gate_fail).append(msg)
            if trained and sd is not None:
                signed.setdefault(m, []).append(dv)
            rows.append({"section": "gate_continuity", "method": m, "eval": d, "rung": rg,
                         "value": round(got, 6), "canonical": exp, "delta": round(dv, 8),
                         "pass": abs(dv) <= GATE_CELL_TOL})
        if not got_cells:
            print(f"{NAMES.get(m, m):16s}{'absent':>7s}"); continue
        mid = [v for (d, rg), v in got_cells.items() if rg == "ID"]
        mood = [v for (d, rg), v in got_cells.items() if rg != "ID"]
        mx = max(diffs) if diffs else float("nan")
        n_bad = sum(1 for x in gate_fail + gate_warn if x.startswith(f"{m}/"))
        ok = n_bad == 0
        print(f"{NAMES.get(m, m):16s}{len(diffs):>7d}{mx:>11.2e}"
              f"{np.mean(mid) if mid else float('nan'):>+10.3f}"
              f"{np.mean(mood) if mood else float('nan'):>+11.3f}  "
              f"{'ok' if ok else f'{n_bad} cell(s) beyond bar'}")
    from scipy.stats import binomtest
    for m, dvs in signed.items():
        pos = sum(1 for x in dvs if x > 0)
        bp = binomtest(pos, len(dvs), 0.5).pvalue
        if bp < 0.01 and pos in (0, len(dvs)):
            gate_fail.append(f"{m}: signed deltas {pos}/{len(dvs)} ONE-SIDED (binom p={bp:.4f}) -- "
                             "a shifted estimator, not a re-draw")
        print(f"    bias check {NAMES.get(m, m):16s} signs {pos}/{len(dvs)-pos}  binom p={bp:.4f}")
    if gate_warn:
        print(f"\n  {len(gate_warn)} trained-component cell(s) exceed 2x the master's 3-seed sd "
              "(reported, not fatal -- see continuity_device_diagnosis.py):")
        for w in gate_warn[:8]:
            print(f"    ~ {w}")
    print(f"\n  GATE 1: {'PASS' if not gate_fail else 'FAIL'}  "
          f"({n_cmp} per-cell comparisons, max |d| = {worst:.2e}"
          + (f" at {worst_at}" if worst_at else "") + f", tol {GATE_CELL_TOL:g})")
    for f in gate_fail[:15]:
        print(f"    - {f}")

    # ---------------------------------------------------------------- GATE 2: sidecar integrity
    print("\n" + "-" * 104)
    print("GATE 2 -- SIDECAR INTEGRITY")
    print("-" * 104)
    bad = []
    for (d, rg), (meth, y) in sorted(cells.items()):
        for m, V in meth.items():
            V = np.asarray(V)
            if V.shape[1] != len(y):
                bad.append(f"{d}/{rg}/{m}: {V.shape[1]} values for {len(y)} labels")
            if not np.isfinite(V).all():
                bad.append(f"{d}/{rg}/{m}: non-finite values present")
    # the three floors are deterministic functions of the cached logprobs -> seed self-agreement == 1
    det_bad = []
    for (d, rg), (meth, y) in sorted(cells.items()):
        for m in ["floor_min", "floor_ppl", "floor_sum"]:
            if m in meth and np.asarray(meth[m]).shape[0] > 1:
                sa = self_agreement(np.asarray(meth[m], float))
                if not (np.isnan(sa) or abs(sa - 1.0) < 1e-9):
                    det_bad.append(f"{d}/{rg}/{m}: seed self-agreement {sa:.6f} != 1.0")
    nseeds = sorted({np.asarray(V).shape[0] for meth, _ in cells.values() for V in meth.values()})
    print(f"  row alignment / finiteness : {'PASS' if not bad else 'FAIL'}")
    for b in bad[:10]:
        print(f"    - {b}")
    print(f"  floors deterministic       : {'PASS' if not det_bad else 'FAIL'}")
    for b in det_bad[:10]:
        print(f"    - {b}")
    print(f"  seed counts present        : {nseeds}")
    print("  (the writer's own 1e-6 PRR self-gate already ran at write time and would have refused)")

    if gate_fail or bad or det_bad:
        print("\nA GATE FAILED. The primary result is not read or interpreted until the")
        print("   gates pass. Stopping here.")
        _write(out_path, rows)
        return

    # ---------------------------------------------------------------- the rung profile
    print("\n" + "-" * 104)
    print("RUNG PROFILE -- macro PRR over the 8 datasets, report/Hidden Failures rung order")
    print("-" * 104)
    prof = {}
    def macro(prrfn, rg, need):
        """Macro over the datasets present at this rung. Cells missing a required component are
        SKIPPED, never counted as zero -- absent must read as 'not measured'."""
        vals = [prrfn(y, meth) for (d, r), (meth, y) in cells.items()
                if r == rg and all(k in meth for k in need)]
        return float(np.mean(vals)) if vals else float("nan")

    def show(label, vals):
        prof[label] = vals
        cells_txt = "".join(f"{v:>+13.3f}" if np.isfinite(v) else f"{'absent':>13s}" for v in vals)
        ood = np.mean(vals[1:])
        print(f"{label:34s}" + cells_txt + (f"{ood:>+12.3f}" if np.isfinite(ood) else f"{'-':>12s}"))
        for rg, v in zip(RUNGS, vals):
            if np.isfinite(v):
                rows.append({"section": "rung_profile", "method": label, "rung": rg,
                             "value": round(v, 4)})

    header = f"{'method':34s}" + "".join(f"{r.replace('-long',''):>13s}" for r in RUNGS) + f"{'OODmacro':>12s}"
    print(header)
    for m in ["saplma", "wmsp_shrink2"]:
        show(NAMES[m], [macro(lambda y, mt, m=m: component_prr(y, mt, m), rg, [m]) for rg in RUNGS])
    for label, a, b in ENSEMBLES:
        show(label, [macro(lambda y, mt, a=a, b=b: ensemble_prr(y, mt, a, b), rg, [a, b])
                     for rg in RUNGS])

    # ---------------------------------------------------------------- primary estimands
    print("\n" + "=" * 104)
    print("PRIMARY ESTIMANDS -- UNIT OF ANALYSIS IS THE DATASET, n = 8")
    print("  A: per dataset, mean PRR over its 4 OOD rungs, ensemble - SAPLMA")
    print("  B: the same on the 8 ID cells")
    print("=" * 104)

    ood_stat = id_stat = None
    for label, a, b in ENSEMBLES:
        tag = "PRIMARY" if label.startswith("PRIMARY") else (
              "CONTROL" if label.startswith("CONTROL") else "REFERENCE")
        print(f"\n### {label}   [{tag}]")
        for est, rung_set in [("A (OOD)", OOD), ("B (ID)", ["ID"])]:
            per_ds = []
            for d in LONG:
                e_v, s_v = [], []
                for rg in rung_set:
                    if (d, rg) not in cells:
                        continue
                    meth, y = cells[(d, rg)]
                    if a not in meth or b not in meth or "saplma" not in meth:
                        continue
                    e_v.append(ensemble_prr(y, meth, a, b))
                    s_v.append(component_prr(y, meth, "saplma"))
                if e_v:
                    per_ds.append((d, float(np.mean(e_v)) - float(np.mean(s_v))))
            if not per_ds:
                print(f"  {est}: no data"); continue
            dv = np.array([x for _, x in per_ds])
            lo, hi = boot_ci_mean(dv)
            p, n_nz = wilcoxon_exact(dv)
            pos = int(np.sum(dv > 0))
            loo = [float(np.mean(np.delete(dv, i))) for i in range(len(dv))]
            print(f"  {est}  macro {dv.mean():+.4f}   median {np.median(dv):+.4f}   "
                  f"signs {pos}/{len(dv)}   bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
            print(f"          exact two-sided Wilcoxon p = {p:.4f} (n={n_nz} non-zero)   "
                  f"LODO macro range [{min(loo):+.4f}, {max(loo):+.4f}]")
            print("          per dataset: " + "  ".join(f"{d}:{v:+.3f}" for d, v in per_ds))
            rows.append({"section": f"estimand_{est.split()[0]}", "method": label,
                         "macro": round(float(dv.mean()), 4), "median": round(float(np.median(dv)), 4),
                         "signs_positive": pos, "n": len(dv),
                         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "wilcoxon_p": round(p, 4),
                         "lodo_lo": round(min(loo), 4), "lodo_hi": round(max(loo), 4)})
            for d, v in per_ds:
                rows.append({"section": f"estimand_{est.split()[0]}_perdataset",
                             "method": label, "eval": d, "value": round(v, 4)})
            if label.startswith("PRIMARY"):
                if est.startswith("A"):
                    ood_stat = (dv.mean(), lo, hi, p, pos, len(dv))
                else:
                    id_stat = (dv.mean(), lo, hi)

    # ---------------------------------------------------------------- the pre-registered verdict
    print("\n" + "=" * 104)
    print("PRE-REGISTERED INTERPRETATION (the ensemble registration §6) -- applied to the PRIMARY ensemble only")
    print("=" * 104)
    if ood_stat is None or id_stat is None:
        print("  PRIMARY ensemble not computable on this population (a component is ABSENT, not zero).")
        print("     No verdict is issued. Run pbs/pdl_perex_ens.pbs to produce CAWSA λ=2 + attention.")
        _write(out_path, rows)
        return
    om, olo, ohi, op, opos, on = ood_stat
    im, ilo, ihi = id_stat
    ood_improved = olo > 0                      # CI excludes 0 in the positive direction
    consistent = opos >= 6                      # improves on at least 6 of 8 datasets
    id_material_loss = ihi < -ID_SCALE          # entire CI below run-to-run reproducibility
    print(f"  OOD leg : macro {om:+.4f}, CI [{olo:+.4f},{ohi:+.4f}], {opos}/{on} datasets, "
          f"Wilcoxon p={op:.4f}  -> {'improvement' if ood_improved else 'NOT established'}"
          f"{'' if consistent else ' (and not consistent across datasets)'}")
    print(f"  ID leg  : macro {im:+.4f}, CI [{ilo:+.4f},{ihi:+.4f}]; reference scale ±{ID_SCALE} "
          f"(measured seed sd of SAPLMA's ID macro)")
    print(f"            -> {'MATERIAL ID loss' if id_material_loss else 'no material ID loss'}")
    if ood_improved and consistent and not id_material_loss:
        verdict = "BEST OF BOTH WORLDS -- both legs hold"
    elif ood_improved and id_material_loss:
        verdict = "TRADE-OFF -- OOD gain bought with a material ID loss (NOT 'best of both worlds')"
    elif ood_improved:
        verdict = "PARTIAL -- OOD improvement, but not consistent enough across datasets"
    else:
        verdict = "NULL -- no established OOD improvement over SAPLMA. A valid result."
    print(f"\n  VERDICT: {verdict}")
    rows.append({"section": "verdict", "method": "PRIMARY", "value": verdict})

    # ---------------------------------------------------------------- complementarity (SECONDARY)
    print("\n" + "-" * 104)
    print("COMPLEMENTARITY -- SECONDARY MECHANISM EVIDENCE, NOT part of the success criterion")
    print("  A cross-correlation cannot exceed sqrt(r_xx * r_yy), so the disattenuated value is the")
    print("  interpretable one. Lower correlation alone is NOT evidence of a better method:")
    print("  the project's working notes already recorded that distinctness != incremental value.")
    print("-" * 104)
    print(f"{'pair':40s}{'self A':>9s}{'self B':>9s}{'cross':>9s}{'disatt.':>10s}   population")
    for label, a, b in [("CAWSA λ=2 ~ SAPLMA", "wmsp_shrink2", "saplma"),
                        ("SAPLMA ~ attention-pool", "saplma", "attention"),
                        ("msp_min ~ SAPLMA", "floor_min", "saplma")]:
        for pop, rset in [("OOD only", OOD), ("ID", ["ID"])]:
            sa, sb, cr = [], [], []
            for (d, rg), (meth, y) in cells.items():
                if rg not in rset or a not in meth or b not in meth:
                    continue
                A, B = np.asarray(meth[a], float), np.asarray(meth[b], float)
                sa.append(self_agreement(A)); sb.append(self_agreement(B))
                cr.append(cross_agreement(A, B) if min(A.shape[0], B.shape[0]) > 1
                          else rho(A[0], B[0]))
            if not cr:
                continue
            ra, rb, rc = np.nanmean(sa), np.nanmean(sb), np.nanmean(cr)
            den = np.sqrt(max(ra, 0) * max(rb, 0))
            dis = rc / den if den > 1e-9 else float("nan")
            print(f"{label:40s}{ra:>9.3f}{rb:>9.3f}{rc:>9.3f}{dis:>10.3f}   {pop}")
            rows.append({"section": "complementarity", "method": label, "rung": pop,
                         "self_a": round(float(ra), 4), "self_b": round(float(rb), 4),
                         "cross": round(float(rc), 4), "disattenuated": round(float(dis), 4)})

    # ---------------------------------------------------------------- rankavg cohort dependence
    print("\n" + "-" * 104)
    print("rankavg COHORT DEPENDENCE -- measured, so the caveat carries its own evidence")
    print("-" * 104)
    rng = np.random.RandomState(0)
    flips_tot = trials_tot = 0
    dprr = []
    for (d, rg), (meth, y) in sorted(cells.items()):
        if "wmsp_shrink2" not in meth or "saplma" not in meth or len(y) < 400:
            continue
        # seed 0 only: this diagnoses the RANK OPERATION, so it must not mix in seed averaging
        ua, ub = np.asarray(meth["wmsp_shrink2"], float)[0], np.asarray(meth["saplma"], float)[0]
        full = rankavg(ua, ub)
        for _ in range(50):
            idx = rng.choice(len(y), 200, replace=False)
            sub = rankavg(ua[idx], ub[idx])
            oa = np.sign(full[idx[0]] - full[idx[1]]); ob = np.sign(sub[0] - sub[1])
            if oa != 0 and ob != 0:
                trials_tot += 1
                flips_tot += int(oa != ob)
        k = min(len(y) // 2, 1000)
        idx = np.sort(rng.choice(len(y), k, replace=False))
        dprr.append(results.prr(y[idx], rankavg(ua[idx], ub[idx])) - results.prr(y[idx], full[idx]))
    if trials_tot:
        print(f"  pairwise order flips, full cohort vs 200-example cohort: {flips_tot}/{trials_tot} "
              f"({100*flips_tot/trials_tot:.1f}%)")
        print(f"  PRR shift when ranks are recomputed in-subset vs inherited: "
              f"mean {np.mean(dprr):+.4f}, max |Δ| {np.max(np.abs(dprr)):.4f}  (n={len(dprr)} cells)")
        print("  -> reported as a RANK-ENSEMBLE DIAGNOSTIC, not a deployable per-response estimator.")
        rows.append({"section": "rankavg_cohort_dependence", "method": "PRIMARY",
                     "value": f"{flips_tot}/{trials_tot} flips",
                     "delta": round(float(np.mean(dprr)), 4)})

    # ---------------------------------------------------------------- zavg robustness footnote
    print("\n  ROBUSTNESS FOOTNOTE (not a headline, never an alternative to choose from):")
    for label, a, b in ENSEMBLES[:2]:
        vals = [macro(lambda y, mt, a=a, b=b: ensemble_prr(y, mt, a, b, "z"), rg, [a, b]) for rg in RUNGS]
        print(f"    zavg {label:34s} ID {vals[0]:+.3f}  OODmacro {np.mean(vals[1:]):+.3f}")
        rows.append({"section": "zavg_footnote", "method": label,
                     "value": round(float(np.mean(vals[1:])), 4), "rung": "OODmacro"})

    _write(out_path, rows)


def _write(out, rows):
    fields = ["section", "method", "eval", "rung", "value", "canonical", "delta", "pass",
              "macro", "median", "signs_positive", "n", "ci_lo", "ci_hi", "wilcoxon_p",
              "lodo_lo", "lodo_hi", "self_a", "self_b", "cross", "disattenuated"]
    with open(out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
