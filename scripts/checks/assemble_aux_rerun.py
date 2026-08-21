"""Combine the per-eval auxiliary-loss CSVs into one grid, and state coverage cell by cell.

The re-run is split across one job per eval (a cell costs ~2.5h, so all four in one job overran the
wall). This joins them back and answers the only two questions that matter before anything is quoted:

  1. WHICH CELLS EXIST. The standing rule is the COMPLETE grid, every long eval x every cells_long rung.
     A partial grid reported as if complete is the failure this script exists to prevent, so missing
     cells are NAMED, not summarised as a count.
  2. WHICH ROWS TEST WHAT. `real_minus_shuffled` tests the LOSS and is valid on every cell. The
     held-out-source lambda SELECTION is only exercised where the training pool had >1 source, which is
     11 of 20 cells on the 4-eval grid. The two are reported separately and never pooled.

COLUMN UNION. Job 270751 started before `head_selection` / the three head-correlation columns were
added, so its CSV has 19 columns and the rest 23. The extra four are blank for every single-head (K=1)
row by definition, so the union is semantically identical -- but it is filled explicitly here rather
than left to whatever a reader's CSV tool does with a ragged join.

    python scripts/checks/assemble_aux_rerun.py
    python scripts/checks/assemble_aux_rerun.py --glob 'results/aux_rerun_orgad_*.csv'
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "checks"))

import probedriftlong as PDL  # noqa: E402

# Written by the driver; kept here so a missing column is filled explicitly rather than silently.
FIELDS = ["rung", "eval", "seed", "target", "K", "J_supervised", "head_selection",
          "head_attn_corr", "head_corr_sup_free", "head_corr_within_sup", "head_corr_within_free",
          "best_lambda", "lambda_selection", "held_out_dataset", "n_pool_sources",
          "drop_epoch", "prr_baseline", "prr_real", "prr_shuffled", "real_minus_shuffled",
          "real_minus_baseline", "n_target_fallback", "n_test"]


def fnum(r, k):
    """Parse a numeric field, returning None for a BLANK. A blank means not-measured and must never
    become 0.0 -- that is the distinction the driver went to trouble to preserve."""
    v = r.get(k, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/aux_rerun_*.csv")
    # The output must NOT be named so that it matches the input glob. It was
    # `results/aux_rerun_COMBINED.csv`, which `results/aux_rerun_*.csv` matches -- so a second run READ ITS
    # OWN OUTPUT and double-counted every row (198 reported for 99 real). Means survived that, but every
    # `n` was inflated, and an n is what a significance claim rests on. Renamed so the glob cannot reach
    # it, AND filtered explicitly below in case someone passes --out back into the glob's path.
    ap.add_argument("--out", default="results/COMBINED_aux_rerun.csv")
    ap.add_argument("--evals", default=",".join(PDL.LONG),
                    help="the grid to check coverage against (default: ALL long evals)")
    args = ap.parse_args()

    out_path = (ROOT / args.out).resolve()
    paths = [p for p in sorted(ROOT.glob(args.glob)) if p.resolve() != out_path]
    if not paths:
        raise SystemExit(f"no CSVs matched {args.glob}")
    # A combined file is never an input, whatever it is called -- self-inclusion is silent and doubles n.
    selfish = [p for p in paths if "COMBINED" in p.name.upper()]
    if selfish:
        raise SystemExit(f"refusing to read a combined file as input: {[p.name for p in selfish]}")

    rows, per_file = [], {}
    for p in paths:
        with open(p, newline="") as f:
            got = list(csv.DictReader(f))
        for r in got:
            r["_src"] = p.name
            for k in FIELDS:                       # explicit union -- see the header note
                r.setdefault(k, "")
        rows += got
        per_file[p.name] = len(got)

    print(f"read {len(paths)} file(s), {len(rows)} rows")
    for n, c in per_file.items():
        print(f"    {c:>5}  {n}")

    # ---------- 1. COVERAGE, named cell by cell ----------
    evals = [e.strip() for e in args.evals.split(",")]
    want = {(rung, X) for rung, X, _ in PDL.cells_long(set(PDL.LONG_SRC), evals)}
    have = {(r["rung"], r["eval"]) for r in rows}
    seeds_by_cell = defaultdict(set)
    for r in rows:
        seeds_by_cell[(r["rung"], r["eval"])].add(r["seed"])

    missing = sorted(want - have)
    print(f"\n=== COVERAGE: {len(want - set(missing))}/{len(want)} cells present "
          f"(grid = {len(evals)} evals x cells_long) ===")
    if missing:
        print("MISSING CELLS -- this grid is INCOMPLETE, do not report it as full:")
        for rung, X in missing:
            print(f"    {rung:16s} {X}")
    short = {c: s for c, s in seeds_by_cell.items() if len(s) < 3}
    if short:
        print("cells with fewer than 3 seeds:")
        for c, s in sorted(short.items()):
            print(f"    {c[0]:16s} {c[1]:<15} seeds={sorted(s)}")
    if not missing and not short:
        print("complete: every cell present at 3 seeds")

    # ---------- 2. WHAT EACH ROW TESTS ----------
    ho = [r for r in rows if r["lambda_selection"] == "held_out_source"]
    fb = [r for r in rows if r["lambda_selection"] == "same_dataset_slice"]
    print(f"\n=== lambda SELECTION: {len(ho)} rows held_out_source | {len(fb)} rows same_dataset_slice ===")
    print("The LOSS is tested on all rows. The held-out-source CRITERION is tested only on the first "
          "group;\nfallback rows are single-source pools (every ID and 1ds-Diff cell) and are not "
          "evidence about it.")

    # ---------- 3. THE DECISIVE COLUMN ----------
    print("\n=== real_minus_shuffled by target x rung (the decisive column; PRR is not) ===")
    print(f"{'target':<14} {'rung':<16} {'n':>4} {'mean':>8} {'pos/n':>8} {'lambda=0':>9}")
    print("-" * 66)
    for tname in sorted({r["target"] for r in rows}):
        for rung in ["ID", "SameTask-long", "DiffTask-long", "LOO-long", "1ds-Diff-long"]:
            sub = [r for r in rows if r["target"] == tname and r["rung"] == rung]
            if not sub:
                continue
            vals = [v for v in (fnum(r, "real_minus_shuffled") for r in sub) if v is not None]
            if not vals:
                continue
            npos = sum(1 for v in vals if v > 0)
            nzero_lam = sum(1 for r in sub if fnum(r, "best_lambda") == 0.0)
            print(f"{tname:<14} {rung:<16} {len(vals):>4} {sum(vals)/len(vals):>+8.4f} "
                  f"{npos:>4}/{len(vals):<3} {nzero_lam:>5}/{len(sub):<3}")

    # lambda=0 everywhere means selection kept choosing "no supervision" -- the original screen's
    # finding, and the thing this re-run set out to test under a better criterion. Say it explicitly.
    n0 = sum(1 for r in rows if fnum(r, "best_lambda") == 0.0)
    print(f"\nlambda=0 selected on {n0}/{len(rows)} rows"
          + ("  <-- selection is still choosing NO supervision" if n0 > len(rows) * 0.8 else ""))

    # ---------- 4. HEAD ARM ----------
    mh = [r for r in rows if r.get("head_selection") == "arbitrary_J"]
    if mh:
        print(f"\n=== head arm: {len(mh)} rows (selection = arbitrary_J, NEVER the paper's greedy) ===")
        sf = [v for v in (fnum(r, "head_corr_sup_free") for r in mh) if v is not None]
        if sf:
            m = sum(sf) / len(sf)
            print(f"supervised x free correlation: mean {m:+.4f} over {len(sf)} rows")
            if m > 0.95:
                print(">0.95: the free heads did NOT diverge from the supervised ones. This arm did "
                      "not test\n    the paper's structural claim -- it re-confirmed that the heads will "
                      "not diverge. Write it up\n    that way, NOT as 'subset-of-heads does not help'.")
        print("Caption: greedy per-head ranking was not implemented. At zero-init the heads are "
              "exchangeable,\nso first-J is an arbitrary-J draw; B.2 established the heads do not "
              "diverge under a shared\ncondition, so a ranking would have nothing to sort.")

    out = ROOT / args.out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["_src"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out}")
    print(f"POPULATION FOR THE CAPTION: ProbeDriftLong, {len(evals)} long evals x cells_long, "
          f"3 seeds, layer 15, Llama-3.1-8B; {len(want) - len(missing)}/{len(want)} cells realised.")


if __name__ == "__main__":
    main()
