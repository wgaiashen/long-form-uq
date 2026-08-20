#!/usr/bin/env python
"""Integrity checks for the eight-dataset W-Models sensitivity grid (prereg M5 D6).

Three checks, in one pass over the per-example sidecars.

1. POOL SEMANTICS, all 40 cells. The training sources and row counts recorded on every cell are
   compared against what `cells_long` says the eight-dataset grid should be. The expectation is
   COMPUTED from the library, never transcribed: `probe_drift_long.ood_settings` is the one
   definition of the taxonomy, and a hand-copied table here would be a copy that could drift.

2. THE FREE CONTROL. Widening the training-source pool from six to eight cannot touch a cell whose
   source spec is unchanged. Those cells must reproduce the committed six-dataset numbers EXACTLY.
   Which cells those are is also computed, by diffing `cells_long(six)` against `cells_long(eight)`
   -- at the time of writing that is 15 of 40 (`ID` and `1ds-Diff-long` on all six existing evals,
   plus `SameTask-long` for xsum / cnn_dailymail / samsum), but the code does not assume the number.

3. THE NEGATIVE CONTROL (`--negative`). Pointed at the CHANGED cells, the comparison must FAIL and
   name med_quad / expertqa as the difference. A control that cannot fail is not a control -- the
   panel-6 version of this check earned its keep precisely by being shown to fail here.

WHY VECTORS AND NOT PRR. PRR is a rank statistic over ~2,000 examples and moves ~1e-4 on 1e-7 of
floating-point round-off, so "the PRRs agree to 3 dp" would pass runs that are not the same
computation. Everything below compares per-example uncertainty vectors, plus the label vector, the
seed list, and the training-spec string.

WHAT THIS CONTROL CAN SEE. Of the ten methods the sidecars carry, six are trained on the pool
(saplma, uniform, attention, and the three wMSP variants) and four are training-free floors that
index only the eval rows. The floors would match even if the pool had changed, so they are a weak
control on their own; the pool-dependent six are what actually carry it. A cell in which no
pool-dependent method was compared is REFUSED, not passed.

    python scripts/checks/wmodels_panel8_control.py --model google/gemma-2-9b
    python scripts/checks/wmodels_panel8_control.py --model google/gemma-2-9b --negative
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probe_drift_long import cells_long                                    # noqa: E402
from probe_drift_long.dataset_configs import LONG_DATASETS                 # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# The six-dataset primary panel (prereg M5 section 2). Used only to work out which cells the widening
# leaves alone; it is not a claim about what should be run.
PANEL6 = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]

# Trained on the pool, so an identical vector is real evidence the pool is identical. The floors
# (floor_sum / floor_ppl / floor_min / fair_floor) are deliberately absent: they never read the
# training pool, so they match whatever happened.
POOL_DEPENDENT = {"saplma", "uniform", "attention",
                  "wmsp_norm", "wmsp_shrink1_5", "wmsp_shrink2"}


def _slug(model: str) -> str:
    return model.replace("/", "_")


def _methods(z) -> set:
    return {k[len("unc__"):] for k in z.keys() if k.startswith("unc__")}


def _spec_str(spec) -> str:
    """Render a cells_long spec for display. `None` is the ID cell's "all of X's train rows"."""
    return "+".join(f"{d}:{'all' if n is None else n}" for d, n in spec)


def _parse_train(recorded: str):
    """meta__train 'a:360+b:257' -> [('a', 360), ('b', 257)], preserving order."""
    out = []
    for part in recorded.split("+"):
        d, _, n = part.rpartition(":")
        out.append((d, int(n)))
    return out


def _compare_spec(recorded: str, expected):
    """Check a recorded training spec against the cells_long expectation.

    Two things are compared, and they are NOT the same kind of claim:

      * The ORDERED LIST OF SOURCES is structural and must match exactly. This is what says the rung
        was built from the pool it claims.
      * The per-source COUNT is a CAP, not a target. `cells_long` splits a matched 1800 budget evenly,
        but a source with fewer available rows contributes all it has -- e.g. at six sources
        cnn_dailymail/DiffTask-long records `factscore:478` against a cap of 600, because factscore
        only has 478 train rows after its eval carve. So the test is `recorded <= cap`, and a
        shortfall is reported as an exhausted source rather than a failure.
      * On the ID cell the cap is `None` ("all of X's train rows"), so there is no count to check.

    Returns (ok, message). `message` is non-empty when there is something to report even on a pass.
    """
    got = _parse_train(recorded)
    want_srcs = [d for d, _ in expected]
    got_srcs = [d for d, _ in got]
    if got_srcs != want_srcs:
        return False, f"sources {got_srcs} != expected {want_srcs}"
    notes = []
    for (d, n_got), (_, cap) in zip(got, expected):
        if cap is None:                       # ID cell: all of the eval's own train rows
            continue
        if n_got > cap:
            return False, f"{d}: {n_got} rows exceeds the cap {cap}"
        if n_got < cap:
            notes.append(f"{d} exhausted at {n_got}/{cap}")
    return True, "; ".join(notes)


def _grid(sources, evals):
    """{(eval, rung): spec} for a given source pool."""
    return {(ev, tag): tuple(spec) for tag, ev, spec in cells_long(set(sources), sorted(evals))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--sens8-dir", default=None, help="default results/perex_wmodels_sens8/<slug>")
    ap.add_argument("--primary-dir", default=None, help="default results/perex_wmodels_lam3/<slug>")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max |Δ| tolerated per vector. Default 0 = bit-identical, which is what the "
                         "same computation on the same inputs gives. Raise it only with a stated reason.")
    ap.add_argument("--negative", action="store_true",
                    help="run on the CHANGED cells instead. They must DIFFER; agreement is the failure.")
    args = ap.parse_args()

    slug = _slug(args.model)
    s8_dir = Path(args.sens8_dir) if args.sens8_dir else ROOT / "results" / "perex_wmodels_sens8" / slug
    pri_dir = Path(args.primary_dir) if args.primary_dir else ROOT / "results" / "perex_wmodels_lam3" / slug

    g6, g8 = _grid(PANEL6, PANEL6), _grid(LONG_DATASETS, LONG_DATASETS)
    unchanged = sorted(k for k in g8 if k in g6 and g6[k] == g8[k])
    changed = sorted(k for k in g8 if k in g6 and g6[k] != g8[k])
    added = sorted(k for k in g8 if k not in g6)

    print("=== W-Models eight-dataset sensitivity: integrity checks (prereg M5 D6) ===")
    print(f"  model     {args.model}")
    print(f"  sens8     {s8_dir}")
    print(f"  primary   {pri_dir}")
    print(f"  grid      {len(g8)} cells at eight sources; {len(g6)} at six")
    print(f"            unchanged {len(unchanged)} | changed {len(changed)} | added {len(added)}")
    print(f"  mode      {'NEGATIVE control (cells must DIFFER)' if args.negative else 'free control'}")
    print(f"  tol       {args.tol:g}\n")

    if not s8_dir.is_dir():
        print(f"!!! sens8 sidecar directory not found: {s8_dir}", file=sys.stderr)
        return 2

    failures, not_run = [], []

    # ---------------------------------------------------------------- 1. pool semantics, all 40
    print("--- 1. pool semantics: recorded meta__train vs cells_long, all cells ---")
    n_spec = 0
    for (ev, rung) in sorted(g8):
        p = s8_dir / f"{ev}__{rung}__{slug}.npz"
        if not p.exists():
            not_run.append(f"{ev}/{rung}")
            continue
        recorded = str(np.load(p, allow_pickle=True)["meta__train"])
        expected = g8[(ev, rung)]
        n_spec += 1
        ok, note = _compare_spec(recorded, expected)
        if not ok:
            failures.append(f"{ev}/{rung}: meta__train '{recorded}' vs cells_long "
                            f"'{_spec_str(expected)}' -- {note}")
        else:
            print(f"  ok  {ev:15s} {rung:15s} {recorded}"
                  + (f"   [{note}]" if note else ""))
    print(f"  -> {n_spec}/{len(g8)} cells present and spec-checked\n")

    # The three named expectations of D6, each checked on its own so a miss is legible.
    print("--- 1b. the three named structural expectations ---")
    checks = [
        ("factscore/SameTask-long restored via expertqa",
         ("factscore", "SameTask-long") in g8 and g8[("factscore", "SameTask-long")] == (("expertqa", 1800),),
         "omitted at six (singleton factuality family), must exist at eight"),
        ("summarisation SameTask-long cells unmoved",
         all(g6.get((e, "SameTask-long")) == g8.get((e, "SameTask-long"))
             for e in ("xsum", "cnn_dailymail", "samsum")),
         "the summ family was already complete at six"),
        ("LOO-long caps 360 -> 257 on every existing eval",
         all(all(n == 257 for _, n in g8[(e, "LOO-long")]) for e in PANEL6),
         "LOO over 7 sources at a matched 1800 budget"),
    ]
    for name, ok, why in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}   ({why})")
        if not ok:
            failures.append(f"structural expectation failed: {name}")
    print()

    # ---------------------------------------------------------------- 2/3. vector comparison
    targets = changed if args.negative else unchanged
    label = "CHANGED (must differ)" if args.negative else "unchanged-pool (must be identical)"
    print(f"--- 2. vector comparison on the {len(targets)} {label} cells ---")

    n_cells = n_cmp = n_pool_dep = 0
    worst, worst_where = 0.0, "-"
    uncomparable = set()
    differed = []

    for (ev, rung) in targets:
        fname = f"{ev}__{rung}__{slug}.npz"
        s8_p, pri_p = s8_dir / fname, pri_dir / fname
        if not s8_p.exists():
            if f"{ev}/{rung}" not in not_run:
                not_run.append(f"{ev}/{rung}")
            continue
        if not pri_p.exists():
            failures.append(f"{ev}/{rung}: no PRIMARY sidecar at {pri_p.name} -- nothing to control against")
            continue

        s8 = np.load(s8_p, allow_pickle=True)
        pri = np.load(pri_p, allow_pickle=True)

        # The premise. In the free control the specs must match; in the negative control they must not.
        s8_train, pri_train = str(s8["meta__train"]), str(pri["meta__train"])
        if not args.negative and s8_train != pri_train:
            failures.append(f"{ev}/{rung}: TRAINING SPEC DIFFERS -- sens8 '{s8_train}' vs "
                            f"primary '{pri_train}'. This cell is not supposed to change.")
            continue
        if args.negative and s8_train == pri_train:
            failures.append(f"{ev}/{rung}: training spec IDENTICAL ('{s8_train}') on a cell the "
                            f"widening should have changed -- the negative control is vacuous here")
            continue

        if s8["y"].shape != pri["y"].shape or not np.array_equal(s8["y"], pri["y"]):
            failures.append(f"{ev}/{rung}: label vectors differ ({s8['y'].shape} vs "
                            f"{pri['y'].shape}) -- the eval split moved")
            continue
        if not np.array_equal(s8["seeds"], pri["seeds"]):
            failures.append(f"{ev}/{rung}: seed lists differ, {s8['seeds']} vs {pri['seeds']}")
            continue

        shared = sorted(_methods(s8) & _methods(pri))
        uncomparable |= (_methods(s8) ^ _methods(pri))
        if not shared:
            failures.append(f"{ev}/{rung}: no method present in BOTH sidecars")
            continue

        cell_worst, cell_worst_m, cell_pool_dep = 0.0, "-", 0
        for m in shared:
            a, b = s8[f"unc__{m}"], pri[f"unc__{m}"]
            if a.shape != b.shape:
                failures.append(f"{ev}/{rung}/{m}: shape {a.shape} vs {b.shape}")
                continue
            d = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
            n_cmp += 1
            if m in POOL_DEPENDENT:
                cell_pool_dep += 1
                n_pool_dep += 1
            if d > cell_worst:
                cell_worst, cell_worst_m = d, m
            if not args.negative and d > args.tol:
                failures.append(f"{ev}/{rung}/{m}: max|Δ| = {d:.3e} > tol {args.tol:g}")

        # A cell where only training-free floors were comparable says nothing about the pool.
        if cell_pool_dep == 0:
            failures.append(f"{ev}/{rung}: only training-pool-INDEPENDENT methods were comparable "
                            f"({shared}); this cell cannot control the widening")

        if args.negative:
            moved = cell_worst > 0.0
            differed.append(moved)
            if not moved:
                failures.append(f"{ev}/{rung}: every vector IDENTICAL on a cell whose pool changed "
                                f"-- the widening did not reach it")

        if cell_worst > worst:
            worst, worst_where = cell_worst, f"{ev}/{rung}/{cell_worst_m}"
        n_cells += 1
        print(f"  {ev:15s} {rung:15s} n={s8['y'].shape[0]:5d}  methods={len(shared)} "
              f"(pool-dep {cell_pool_dep})  max|Δ|={cell_worst:.3e}")

    print("\n--- summary ---")
    print(f"  cells compared          {n_cells}")
    print(f"  vector comparisons      {n_cmp}  (of which pool-dependent: {n_pool_dep})")
    if args.negative:
        print(f"  cells that moved        {sum(differed)}/{len(differed)}  (all of them must move)")
    else:
        print(f"  worst max|Δ|            {worst:.10f}   "
              f"{'(every vector bit-identical)' if worst == 0.0 else 'at ' + worst_where}")
    if uncomparable:
        print(f"  method-set mismatch     {sorted(uncomparable)} -- present in one sidecar only, so "
              f"this control says nothing about them")
    if not_run:
        print(f"  not yet run ({len(not_run):2d})        {not_run}")

    if failures:
        print(f"\n!!! FAILED -- {len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"    - {f}", file=sys.stderr)
        return 1
    if n_cells == 0:
        print("\n!!! nothing was compared -- this is NOT a pass", file=sys.stderr)
        return 1

    if args.negative:
        print(f"\nNEGATIVE CONTROL PASSED: all {n_cells} changed-pool cells moved, so the free "
              f"control above is not vacuous.")
    else:
        print(f"\nPASSED: the eight-dataset grid reproduces the six-dataset panel exactly on "
              f"{n_cells} unchanged-pool cells.")
    if not_run:
        print(f"  (partial: {len(not_run)} cell(s) have not run yet and are excluded by name, "
              f"not silently)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
