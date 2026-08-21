#!/usr/bin/env python
"""Assemble one W-Models population's per-eval ladder CSVs into a single master table.

The ladder writes ONE CSV per (stage, eval) -- twelve files for a full five-rung population -- which
is right for crash safety and wrong for reading. This concatenates them into
`results/wmodels_master__<slug>.csv`, one row per (eval, rung, method).

WHY NOT `qwen_pdl_master.py`. That reads `probedriftlong__<slug>__<eval>.csv` and is pinned to the
EIGHT-dataset grid. Pointing it at the reduced six-dataset panel would report the two dropped
datasets as missing cells rather than as out of scope -- its own docstring makes exactly this
argument about not bending an assembler onto a population it was not built for.

WHAT IT REFUSES TO DO
  * It never fills a cell. `factscore/SameTask-long` is genuinely absent from this panel (dropping
    `expertqa` leaves the `factuality` family a singleton, so `cells_long` omits the rung rather than
    emitting it empty). It is reported as EXPECTED-ABSENT, never as a gap and never as zero.
  * It fails loud on a duplicate (eval, rung, method) whose values DISAGREE. Identical duplicates are
    fine and are counted -- that is what a supserseding re-run looks like on its overlap.
  * It reports coverage UP FRONT, before the file is written, so a short grid cannot be mistaken for
    a complete one.

The PRR column is `prr_mean`, not `prr`. `fair_floor` rows carry `n_seeds = 0` -- it is a
derived max-of-three, not a trained method, so asserting `n_seeds == 3` over all rows is wrong.

    python scripts/checks/wmodels_master.py --model google/gemma-2-9b
    python scripts/checks/wmodels_master.py --model google/gemma-2-9b --strict
    python scripts/checks/wmodels_master.py --model google/gemma-2-9b --prefix wmodels_stage  # λ∈{0,2}
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

PANEL = ["pubmed_qa", "xsum", "cnn_dailymail", "samsum", "asqa", "factscore"]
# Reporting order: matched distribution -> same family -> opposite family -> everything else ->
# one opposite-family set. Matches results/pdl_master__Qwen_Qwen2.5-14B.csv so the two read alike.
RUNGS = ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]
# Genuinely out of scope for the reduced panel, not missing. See the module docstring.
EXPECTED_ABSENT = {("factscore", "SameTask-long")}
METHOD_ORDER = ["floor_sum", "floor_ppl", "floor_min", "fair_floor",
                "uniform", "attention", "saplma",
                "wmsp_norm", "wmsp_shrink1_5", "wmsp_shrink2"]
OUT_COLS = ["model", "eval", "rung", "method", "prr_mean", "prr_std", "n_seeds",
            "ci_lo", "ci_hi", "boot_p", "significant", "different_label_projection",
            "eval_med_len", "train", "carve", "git_sha", "cluster", "source"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemma-2-9b")
    ap.add_argument("--panel", choices=["six", "eight"], default="six",
                    help="'six' (default) = the primary reduced panel of prereg M5 section 2, with "
                         "factscore/SameTask-long expected-absent. 'eight' = the D6 comparability "
                         "sensitivity over the full ProbeDriftLong universe, where nothing is out of "
                         "scope: restoring expertqa is exactly what gives factscore its SameTask "
                         "partner back, so EXPECTED_ABSENT is empty and the grid is 40 cells. "
                         "Pair it with --prefix wmodels_sens8 and an explicit --out; the default "
                         "output name is the PRIMARY master and must never be written from an "
                         "eight-source run.")
    ap.add_argument("--prefix", default="wmodels_stage",
                    help="filename prefix before the A/B stage letter. Default picks up BOTH the "
                         "λ∈{0,2} and λ∈{0,1.5,2} runs; pass a more specific prefix to take one.")
    ap.add_argument("--lam3-only", action="store_true",
                    help="take only the λ∈{0,1.5,2} re-run (the superset). Recommended: its overlap "
                         "with the earlier run is bit-identical, proven by wmodels_lambda_control.py.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every in-scope cell is present")
    a = ap.parse_args()

    slug = a.model.replace("/", "_")

    # The panel decides both the expected grid and what counts as legitimately absent. Read the
    # eight from the library rather than transcribing it: probe_drift_long.dataset_configs is the one
    # definition, and a second copy here could silently drift from the rungs the ladder built.
    if a.panel == "eight":
        from probe_drift_long.dataset_configs import LONG_DATASETS
        panel, expected_absent = list(LONG_DATASETS), set()
    else:
        panel, expected_absent = list(PANEL), set(EXPECTED_ABSENT)

    # A run over eight sources must not be able to land on the primary master's filename. The
    # six-dataset panel is the registered primary (prereg M5 D6 item 1) and its file has to stay
    # byte-identical; an accidental overwrite here would be indistinguishable from a re-run.
    if a.panel == "eight" and a.out is None:
        print("!!! --panel eight requires an explicit --out. The default filename is the PRIMARY "
              "six-dataset master, which the D6 sensitivity must never overwrite.\n"
              f"    Suggested: --out results/wmodels_master8__{slug}.csv", file=sys.stderr)
        return 2

    pats = [f"{a.prefix}A_lam3__{slug}__*.csv", f"{a.prefix}B_lam3__{slug}__*.csv"]
    if not a.lam3_only:
        pats += [f"{a.prefix}A__{slug}__*.csv", f"{a.prefix}B__{slug}__*.csv"]

    cells, srcs, conflicts, dup_ok, quarantined = {}, [], [], 0, []
    for pat in pats:
        for p in sorted(RESULTS.glob(pat)):
            # DO_NOT_USE marks a QUARANTINED population -- generations that failed the §6 gate,
            # kept on disk as evidence, never as results. The stage wildcards can match them, so the
            # exclusion is explicit here rather than left to whoever writes the next glob.
            if "DO_NOT_USE" in p.name:
                quarantined.append(p.name)
                continue
            srcs.append(p.name)
            for r in csv.DictReader(open(p)):
                if r["method"].startswith("VERDICT"):
                    continue
                k = (r["eval"], r["rung"], r["method"])
                if k in cells:
                    if cells[k]["prr_mean"] != r["prr_mean"]:
                        conflicts.append(f"{k}: {cells[k]['prr_mean']} ({cells[k]['source']}) vs "
                                         f"{r['prr_mean']} ({p.name})")
                    else:
                        dup_ok += 1
                    continue                      # first file wins; _lam3 is globbed first
                r["source"] = p.name
                cells[k] = r

    print(f"=== W-Models master — {a.model} ===")
    print(f"  panel  {a.panel} ({len(panel)} evals x {len(RUNGS)} rungs"
          + (f", {len(expected_absent)} expected-absent)" if expected_absent else ")")
          + ("   [D6 SENSITIVITY, not the primary]" if a.panel == "eight" else ""))
    print(f"  source files ({len(srcs)}): {', '.join(srcs)}")
    if quarantined:
        print(f"  skipped {len(quarantined)} QUARANTINED file(s) (DO_NOT_USE): {quarantined}")
    print()

    if conflicts:
        print(f"!!! {len(conflicts)} CONFLICTING duplicate(s) -- the same cell with different "
              f"values:", file=sys.stderr)
        for c in conflicts[:10]:
            print(f"    - {c}", file=sys.stderr)
        print("    Refusing to write a master built from disagreeing runs.", file=sys.stderr)
        return 1

    present = {(e, r) for (e, r, m) in cells}
    missing, absent = [], []
    for ev in panel:
        for rung in RUNGS:
            if (ev, rung) in present:
                continue
            (absent if (ev, rung) in expected_absent else missing).append(f"{ev}/{rung}")

    n_expected = len(panel) * len(RUNGS) - len(expected_absent)
    print(f"  cells {len(present)}/{n_expected} in scope"
          f"{'' if dup_ok == 0 else f'   ({dup_ok} identical duplicate rows collapsed)'}")
    if absent:
        print(f"  expected-absent (out of scope, NOT a gap): {absent}")
    if missing:
        print(f"  MISSING ({len(missing)}): {missing}")
    print(f"  git_sha  {sorted({r['git_sha'][:12] for r in cells.values()})}")
    print(f"  carve    {sorted({r.get('carve', '') for r in cells.values()})}")
    print(f"  methods  {sorted({m for (_, _, m) in cells})}")

    def sort_key(item):
        (ev, rung, meth), _ = item
        return (panel.index(ev) if ev in panel else 99,
                RUNGS.index(rung) if rung in RUNGS else 99,
                METHOD_ORDER.index(meth) if meth in METHOD_ORDER else 99, meth)

    out = Path(a.out) if a.out else RESULTS / f"wmodels_master__{slug}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        for _, r in sorted(cells.items(), key=sort_key):
            r["model"] = a.model
            w.writerow(r)
    try:
        shown = out.relative_to(ROOT)
    except ValueError:                            # --out may point outside the repo
        shown = out
    print(f"\nwrote {shown}  ({len(cells)} rows)")

    if missing and a.strict:
        print("!!! --strict: grid incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
